# Licensed under the MIT License.

import os
import math
import random
import logging
import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn
import torch
import torchcde
from types import SimpleNamespace
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from torch.profiler import profile, record_function, ProfilerActivity
import torch.cuda.amp  # For automatic mixed precision
import gc  # For garbage collection

matplotlib.use('agg')

# Define args FIRST, before using them anywhere
args = SimpleNamespace(
    adjoint=False,
    visualize=True,
    niters=1000,
    lr=0.01,
    gpu=0,
    train_dir='./results',
    model_name='NCDETransformer',  # Use your model by default
    log_step=1,
    seed=27,
    noise_std=0.05,
    noise_a=0,
    cc=True,
    atol=1e-2,  # Relaxed from 0.1
    rtol=1e-2,  # Relaxed from 0.1
    method='euler',  # Changed from 'rk4' to faster 'euler'
    dropout=0
)

# Set up CUDA device AFTER args is defined
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create necessary directories
if not os.path.exists(args.train_dir):
    os.makedirs(args.train_dir)

# Setup logger after creating directories
def get_logger(name):
    logger = logging.getLogger(name)
    filename = f'{name}.log'
    fh = logging.FileHandler(filename, mode='a+', encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Create model-specific directories
model_dir = os.path.join(args.train_dir, args.model_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
# Common data directory for both models
data_dir = os.path.join(args.train_dir, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Set up logging in model directory
log = get_logger(os.path.join(model_dir, 'log'))

# Import odeint based on args
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# Define AttrDict
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      noise_a=.002,
                      a=0.,
                      b=1.):
    """Parametric formula for 2d spiral is `r = a + b * theta`.
    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check
    Returns:
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)  # [ntotal]
    aa = npr.randn(nspiral) * noise_a + a  # [nspiral]
    bb = npr.randn(nspiral) * noise_a + b  # [nspiral]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts  # [ntotal]
    rs_cw = aa.reshape(-1, 1) + bb.reshape(-1, 1) * 50. / zs_cw  # [nspiral, ntotal]
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=-1)  # [nspiral, ntotal, 2]
    orig_traj_cw = np.flip(orig_traj_cw, axis=1)

    zs_cc = orig_ts
    rw_cc = aa.reshape(-1, 1) + bb.reshape(-1, 1) * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=-1)

    # sample starting timestamps
    orig_trajs = []
    for _ in range(nspiral):
        if args.cc == 2:
            cc = bool(npr.rand() > .5)  # uniformly select rotation
        else:
            cc = args.cc
        orig_traj = orig_traj_cc[_] if cc else orig_traj_cw[_]
        orig_trajs.append(orig_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)

    samp_trajs = npr.randn(*orig_trajs.shape) * noise_std + orig_trajs

    return orig_trajs, samp_trajs, orig_ts


import torch
import torch.nn as nn
import torchcde

class NCDEVectorField(nn.Module):
    """Vector field for Neural Controlled Differential Equations.
    
    This implementation properly defines the vector field for a CDE where:
    dh(t)/dt = f(h(t)) * dX(t)/dt
    
    The vector field outputs a matrix that, when multiplied by dX/dt,
    gives the derivative of the hidden state.
    """
    def __init__(self, hidden_dim, control_dim, intermediate_dim=32):
        super(NCDEVectorField, self).__init__()
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        
        # Network to map hidden state to a matrix
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.Tanh(),
            nn.Linear(intermediate_dim, hidden_dim * control_dim)
        )
        
        # Initialize with small weights for stability
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, h):
        """
        Parameters:
            h: Current hidden state [batch_size, hidden_dim]
            
        Returns:
            Matrix of shape [batch_size, hidden_dim, control_dim] representing f(h)
            in the CDE equation: dh(t)/dt = f(h(t)) * dX(t)/dt
        """
        batch_size = h.shape[0]
        
        # Compute the matrix f(h) that will be multiplied by dX/dt
        matrix = self.net(h)
        
        # Reshape to appropriate dimensions for matrix-vector product with dX/dt
        return matrix.view(batch_size, self.hidden_dim, self.control_dim)


def cde_integrate(vector_field, initial_state, control_path, times, options=None):
    """
    Integrate a Neural CDE with proper handling of the control path.
    
    Parameters:
        vector_field: Function mapping h → f(h) [batch_size, hidden_dim, control_dim]
        initial_state: Initial hidden state h(t₀) [batch_size, hidden_dim]
        control_path: A CDE-compatible path object with methods:
            - evaluate(t): Get X(t) [batch_size, control_dim]
            - derivative(t): Get dX(t)/dt [batch_size, control_dim]
        times: Times at which to evaluate the solution [num_times]
        options: Options for the ODE solver
    
    Returns:
        The solution evaluated at the specified times [num_times, batch_size, hidden_dim]
    """
    options = options or {}
    
    # Actual ODE function that will be used by the solver
    def ode_func(t, h):
        # Compute the matrix f(h(t))
        f_h = vector_field(h)
        
        # Get the derivative of X at time t
        dXdt = control_path.derivative(t)
        
        # Batch matrix-vector multiplication: f(h(t)) × dX(t)/dt
        # For each item in batch, multiply [hidden_dim, control_dim] by [control_dim]
        # to get [hidden_dim]
        batch_size = h.shape[0]
        dh_dt = torch.bmm(f_h, dXdt.unsqueeze(-1)).squeeze(-1)
        
        return dh_dt
    
    # Use an ODE solver to integrate the system
    from torchdiffeq import odeint
    solution = odeint(
        ode_func, 
        initial_state, 
        times,
        atol=options.get('atol', 1e-4),
        rtol=options.get('rtol', 1e-3),
        method=options.get('method', 'dopri5')
    )
    
    return solution


class NCDEAttention(nn.Module):
    """
    Continuous-Time Multi-Head Attention using Neural CDEs
    
    This module implements attention where keys and values evolve as solutions to
    Neural Controlled Differential Equations, driven by the query path.
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, args_ode, dropout=0.1):
        super(NCDEAttention, self).__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        # Projections for Q, K, V
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        # Vector fields for key and value evolution
        # IMPORTANT FIX: Vector fields need to match dimensions correctly
        self.F_key = NCDEVectorField(d_k, d_k)  # Changed from d_model to d_k
        self.F_value = NCDEVectorField(d_v, d_k)  # Changed from d_model to d_k
        
        # Output projection
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # ODE solver settings
        self.args_ode = args_ode
        
        # For numerical integration
        self.quadrature_points = 10
    
    def forward(self, q, k, v, times, mask=None):
        """
        Computes continuous-time attention using NCDE-driven key and value trajectories
        
        Parameters:
            q: Query tensor [batch, seq_len, d_model]
            k: Key tensor [batch, seq_len, d_model]
            v: Value tensor [batch, seq_len, d_model]
            times: Timestamps for each position [batch, seq_len]
            mask: Attention mask [batch, seq_len, seq_len]
            
        Returns:
            output: Attended features [batch, seq_len, d_model]
        """
        batch_size, seq_len = q.size(0), q.size(1)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        # Apply linear projections
        q = self.w_qs(q).view(batch_size, seq_len, n_head, d_k)
        k = self.w_ks(k).view(batch_size, seq_len, n_head, d_k)
        v = self.w_vs(v).view(batch_size, seq_len, n_head, d_v)
        
        # Create continuous query path via interpolation
        # This becomes our control path X(t) for the NCDE
        q_permuted = q.permute(2, 0, 1, 3).contiguous().view(n_head * batch_size, seq_len, d_k)
        
        # Fix: Use the time vector that matches the sequence length
        t_1d = times[0, :seq_len].clone()  # Shape: [seq_len]
        
        from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
        
        # Create a path for each batch using the same time vector
        coeffs = natural_cubic_spline_coeffs(t_1d, q_permuted)
        q_path = NaturalCubicSpline(coeffs)
        
        # Use the optimized NCDE attention computation
        output = self.compute_ncde_attention(q_path, k, v, times, mask)
        
        # Project back to model dimensions and apply dropout
        output = self.fc(output)
        output = self.dropout(output)
        
        return output

    def compute_efficient_time_averaged_attention(self, q_path, k_trajectory, t_i, t, num_points=5):
        """
        Implementation using fixed-step RK4 integration for key evolution with corrected dimensions
        """
        device = k_trajectory.device
        dtype = k_trajectory.dtype
        batch_size = k_trajectory.size(0)  # This is batch_size * n_head
        seq_len = k_trajectory.size(1)
        d_k = k_trajectory.size(2)
        
        # Generate evaluation points
        tau_eval = torch.linspace(float(t_i), float(t), num_points, 
                                 device=device, dtype=dtype)
        
        # Define ODE function for key evolution
        def key_ode_func(t, k):
            """
            ODE function for key evolution
            k: [batch*n_head*seq_len, d_k]
            """
            # Get query at current time
            q_t = q_path.evaluate(t).to(dtype=dtype)  # [batch*n_head, d_k]
            dq_dt = q_path.derivative(t).to(dtype=dtype)  # [batch*n_head, d_k]
            
            # Reshape k to apply F_key
            k_reshaped = k.view(batch_size * seq_len, d_k)
            
            # Compute F_key for all positions at once
            f_k = self.F_key(k_reshaped)  # [batch*n_head*seq_len, d_k, d_k]
            
            # Expand dq_dt to match number of keys
            dq_dt_expanded = dq_dt.unsqueeze(1).expand(-1, seq_len, -1).contiguous()
            dq_dt_expanded = dq_dt_expanded.view(-1, d_k)
            
            # Compute derivative
            dk_dt = torch.bmm(f_k, dq_dt_expanded.unsqueeze(-1)).squeeze(-1)
            return dk_dt
        
        # Initialize storage for attention scores
        scores = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)
        
        # Process each evaluation point
        for i in range(num_points - 1):
            t_start = tau_eval[i]
            t_end = tau_eval[i + 1]
            
            # Generate sub-steps for RK4 integration
            M = 4  # Number of sub-steps
            t_points = torch.linspace(t_start, t_end, M + 1, device=device, dtype=dtype)
            
            # Initial key states
            k_init = k_trajectory.reshape(-1, d_k)
            
            # Integrate key trajectories using RK4
            k_solution = odeint(
                key_ode_func,
                k_init,
                t_points,
                method='rk4',
                options={'step_size': (t_end - t_start)/M}
            )  # [M+1, batch*n_head*seq_len, d_k]
            
            # Get query at evaluation point
            q_t = q_path.evaluate(t_end).to(dtype=dtype)  # [batch*n_head, d_k]
            
            # Reshape final key state for attention computation
            k_t = k_solution[-1].view(batch_size, seq_len, d_k)
            
            # Compute attention score
            q_t = q_t.view(batch_size, 1, d_k)
            score_t = torch.bmm(q_t, k_t.transpose(-2, -1)) / math.sqrt(d_k)
            scores += score_t
        
        # Average scores
        scores = scores / (num_points - 1)
        
        return scores  # [batch_size, seq_len, seq_len]

    def compute_ncde_attention(self, q_path, k, v, times, mask=None):
        """
        Modified attention computation with corrected dimensions
        """
        batch_size, seq_len = k.size(0), k.size(1)
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v
        dtype = k.dtype
        
        # Reshape k for multi-head processing
        k = k.contiguous().view(batch_size, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(batch_size * n_head, seq_len, d_k)
        
        # Get time sequence
        t_sequence = times[0, :seq_len].to(dtype=dtype)
        
        # Process in chunks
        chunk_size = 5
        attention_scores = torch.zeros(batch_size * n_head, seq_len, seq_len, 
                                     device=k.device, dtype=dtype)
        
        for t_idx in range(0, seq_len, chunk_size):
            chunk_end = min(t_idx + chunk_size, seq_len)
            t_start = t_sequence[t_idx].item()
            t_end = t_sequence[chunk_end-1].item()
            
            # Compute chunk scores with corrected dimensions
            chunk_scores = self.compute_efficient_time_averaged_attention(
                q_path,
                k,
                t_start,
                t_end,
            )  # [batch_size * n_head, seq_len, seq_len]
            
            # Update attention scores for this chunk
            attention_scores[:, t_idx:chunk_end, :] = chunk_scores[:, :chunk_end-t_idx, :]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
            attention_scores = attention_scores.masked_fill(mask, -1e9)
        
        # Compute attention weights
        attention_weights = torch.softmax(attention_scores / math.sqrt(d_k), dim=-1)
        
        # Reshape v for multi-head processing
        v = v.contiguous().view(batch_size, seq_len, n_head, d_v)
        v = v.permute(2, 0, 1, 3).contiguous().view(batch_size * n_head, seq_len, d_v)
        
        # Compute output
        output = torch.bmm(attention_weights, v)
        
        # Reshape output back to original dimensions
        output = output.view(n_head, batch_size, seq_len, d_v)
        output = output.permute(1, 2, 0, 3).contiguous()
        output = output.view(batch_size, seq_len, n_head * d_v)
        
        return output

    def solve_key_ncde_batch(self, k, q_path, times):
        """
        Solve key ODEs for all time steps in batch
        """
        batch_size, seq_len, d_k = k.shape
        
        # Flatten batch and sequence dimensions and ensure float32
        k_flat = k.reshape(-1, d_k).to(dtype=torch.float32)  # [batch*n_head*seq_len, d_k]
        times = times.to(dtype=torch.float32)
        
        def ode_func(t, h):
            # Get query derivative at time t
            dq_dt = q_path.derivative(t).to(dtype=torch.float32)  # [batch*n_head, d_k]
            
            # Expand dq_dt to match the number of keys
            dq_dt = dq_dt.repeat_interleave(seq_len, dim=0)  # [batch*n_head*seq_len, d_k]
            
            # Compute the vector field
            f_h = self.F_key(h)  # [batch*n_head*seq_len, d_k, d_k]
            
            # Compute the derivative
            dh_dt = torch.bmm(f_h, dq_dt.unsqueeze(-1)).squeeze(-1)
            return dh_dt
        
        # Solve the ODE system for all keys at once
        solution = odeint(
            ode_func,
            k_flat,
            times,
            method=self.args_ode.method,
            atol=self.args_ode.atol,
            rtol=self.args_ode.rtol
        )
        
        return solution  # [time, batch*n_head*seq_len, d_k]

    def solve_value_ncde_batch(self, v, q_path, times):
        """
        Solve value ODEs for all time steps in batch
        """
        batch_size, seq_len, d_v = v.shape
        
        # Flatten batch and sequence dimensions and ensure float32
        v_flat = v.reshape(-1, d_v).to(dtype=torch.float32)  # [batch*n_head*seq_len, d_v]
        times = times.to(dtype=torch.float32)
        
        def ode_func(t, h):
            # Get query derivative at time t
            dq_dt = q_path.derivative(t).to(dtype=torch.float32)  # [batch*n_head, d_k]
            
            # Expand dq_dt to match the number of values
            dq_dt = dq_dt.repeat_interleave(seq_len, dim=0)  # [batch*n_head*seq_len, d_k]
            
            # Compute the vector field
            f_h = self.F_value(h)  # [batch*n_head*seq_len, d_v, d_k]
            
            # Compute the derivative
            dh_dt = torch.bmm(f_h, dq_dt.unsqueeze(-1)).squeeze(-1)
            return dh_dt
        
        # Solve the ODE system for all values at once
        solution = odeint(
            ode_func,
            v_flat,
            times,
            method=self.args_ode.method,
            atol=self.args_ode.atol,
            rtol=self.args_ode.rtol
        )
        
        return solution  # [time, batch*n_head*seq_len, d_v]

class NCDETransformer(nn.Module):
    """
    Transformer with Neural CDE-based continuous-time attention
    """
    def __init__(self, obs_dim, device, batch_size=64, **kwargs):
        super(NCDETransformer, self).__init__()
        
        # ODE solver arguments
        args_ode = {
            'use_ode': True, 'actfn': 'tanh', 'layer_type': 'concat', 'zero_init': True,
            'atol': kwargs.get('atol', 0.1), 'rtol': kwargs.get('rtol', 0.1), 
            'method': kwargs.get('method', 'rk4'), 'regularize': False,
            'approximate_method': 'bilinear', 'nlinspace': 1, 'linear_type': 'before',
            'interpolate': 'cubic', 'itol': 1e-2
        }
        args_ode = AttrDict(args_ode)

        # Model dimensions
        d_model = 8  # Reduced from 16
        d_inner = 32  # Reduced from 64
        n_head = 2  # Reduced from 4
        d_k = d_v = 4
        dropout = kwargs.get('dropout', 0.1)

        # Input/output projections
        self.lin_in = nn.Linear(obs_dim, d_model)
        self.lin_out = nn.Linear(d_model, obs_dim)

        # Replace standard encoder with NCDE-based encoder
        self.ncde_attention = NCDEAttention(
            d_model, d_inner, n_head, d_k, d_v, args_ode, dropout
        )

        # Feed-forward network after attention
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Positional encoding
        self.position_vec = torch.tensor(
            [10000.0 ** (2.0 * (i // 2) / d_model) for i in range(d_model)]
        )

        self.batch_size = batch_size
        self.device = device

    def temporal_enc(self, time):
      """
      Input: batch×seq_len.
      Output: batch×seq_len×d_model.
      """
      # Make sure position_vec is on the same device as time
      self.position_vec = self.position_vec.to(time.device)
      
      result = time.unsqueeze(-1) / self.position_vec
      result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
      result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
      return result




    def to(self, device):
        """Move the entire model to the specified device and ensure all components are on the same device."""
        # Move the base model
        super(NCDETransformer, self).to(device)
        
        # Update the device attribute
        self.device = device
        
        # Ensure position_vec is on the correct device
        self.position_vec = self.position_vec.to(device)
        
        return self

    # Modified forward method with explicit device management
    def forward(self, samples, orig_ts, **kwargs):
      """
      Parameters:
          samples: Input time series data [batch, seq_len, feature_dim+1]
                  The last dimension contains the timestamp
          orig_ts: Original timestamps array

      Returns:
          output: Model predictions
      """
      # First ensure samples is on the correct device
      samples = samples.to(self.device)
      
      is_train = kwargs.get('is_train', False)
      time_idx = kwargs.get('idx', None)  # These are the time indices

      if is_train:
          bs = samples.shape[0]
          batch_idx = np.random.choice(bs, self.batch_size, replace=False)
          samples = samples[batch_idx, ...]
      else:
          bs = samples.shape[0]
          batch_idx = None  # No batch subsampling during testing

      # Extract times and features
      t0 = samples[..., -1]  # Last channel is time
      features = samples[..., :-1]  # Other channels are features

      # Apply input projection
      input_repr = self.lin_in(features)

      # Add temporal encoding
      # Ensure position_vec is on the correct device
      self.position_vec = self.position_vec.to(input_repr.device)
      result = t0.unsqueeze(-1) / self.position_vec
      result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
      result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
      temporal_encoding = result
      
      input_repr = input_repr + temporal_encoding

      # Convert timestamps to tensor and ensure it's on the correct device
      times = torch.tensor(orig_ts, device=self.device)
      times = times.unsqueeze(0).repeat(input_repr.shape[0], 1)

      # Apply NCDE-based attention
      attn_output = self.ncde_attention(
          input_repr, input_repr, input_repr, times
      )

      # Add & Norm (first residual connection)
      attn_output = self.norm1(attn_output + input_repr)

      # Feed-forward
      ff_output = self.feed_forward(attn_output)

      # Add & Norm (second residual connection)
      output = self.norm2(ff_output + attn_output)

      # Final projection to output space
      output = self.lin_out(output)

      # Return output and indices as a tuple in the format expected by calculate_loss
      return output, (batch_idx, time_idx)

    def calculate_loss(self, out, target):
        """
        Calculate MSE loss between predictions and targets.
        
        Parameters:
            out: Tuple of (predictions, (batch_indices, time_indices))
            target: Tuple of (target_x, pz0_mean, pz0_logvar)
        
        Returns:
            Loss value tensor
        """
        pred_x, (batch_idx, time_idx) = out
        target_x, _, _ = target
        
        # First handle batch subsampling if needed
        if batch_idx is not None:
            target_x = target_x[batch_idx]
        
        # Then handle time point subsampling
        if time_idx is not None:
            # Sample the same time points from the target
            target_subset = target_x[:, time_idx, :]
            
            # Verify shapes match
            if pred_x.shape[1] != target_subset.shape[1]:
                print(f"WARNING: Shape mismatch after resampling. pred_x: {pred_x.shape}, target: {target_subset.shape}")
                # This should not happen if time_idx is applied correctly
                min_dim = min(pred_x.shape[1], target_subset.shape[1])
                pred_x = pred_x[:, :min_dim, :]
                target_subset = target_subset[:, :min_dim, :]
            
            return ((pred_x - target_subset) ** 2).sum()
        else:
            # No time subsampling
            if pred_x.shape[1] != target_x.shape[1]:
                print(f"WARNING: Shape mismatch but no time_idx provided. Using truncation.")
                min_dim = min(pred_x.shape[1], target_x.shape[1])
                pred_x = pred_x[:, :min_dim, :]
                target_x = target_x[:, :min_dim, :]
            
            return ((pred_x - target_x) ** 2).sum()

class CDEInterpolation(NaturalCubicSpline):
    """
    Extension of NaturalCubicSpline with a derivative method.
    This demonstrates how a proper control path should work with NCDEs.
    """
    def derivative(self, t):
        """Return the derivative dX/dt at time t."""
        # Since NaturalCubicSpline already has a derivative method, 
        # we need a different name to avoid recursion
        return self.evaluate_derivative(t)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == '__main__':
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # Data parameters
    obs_dim = 2
    nspiral = 300
    start = 0.
    stop = 6 * np.pi
    noise_std = args.noise_std
    noise_a = args.noise_a
    a = 0.
    b = .3
    ntotal = 150
    nsample = 50
    ntrain = 200
    ntest = 100
    best_val = np.inf

    # Try to load existing data from common data directory
    data_path = os.path.join(data_dir, 'spiral_data.npz')
    if os.path.exists(data_path):
        data = np.load(data_path)
        orig_trajs = torch.from_numpy(data['orig_trajs']).float().to(device)
        samp_traj = torch.from_numpy(data['samp_traj']).float().to(device)
        orig_ts = data['orig_ts']
        test_idx = data['test_indices'].tolist()
        test_idx = sorted(test_idx)
        print("Loaded existing spiral data")
    else:
        print("Error: Please run Neural_ODE first to generate spiral data")
        exit(1)

    # Normalize trajectories
    trajs_min_x, trajs_min_y = torch.min(orig_trajs[:, :, 0]), torch.min(orig_trajs[:, :, 1])
    trajs_max_x, trajs_max_y = torch.max(orig_trajs[:, :, 0]), torch.max(orig_trajs[:, :, 1])
    orig_trajs[:, :, 0] = (orig_trajs[:, :, 0] - trajs_min_x) / (trajs_max_x - trajs_min_x)
    orig_trajs[:, :, 1] = (orig_trajs[:, :, 1] - trajs_min_y) / (trajs_max_y - trajs_min_y)
    samp_traj[:, :, 0] = (samp_traj[:, :, 0] - trajs_min_x) / (trajs_max_x - trajs_min_x)
    samp_traj[:, :, 1] = (samp_traj[:, :, 1] - trajs_min_y) / (trajs_max_y - trajs_min_y)

    # Split data
    train_trajs = samp_traj[:ntrain]
    test_trajs = samp_traj[ntrain:]
    train_target = orig_trajs[:ntrain]
    test_target = orig_trajs[ntrain:]

    # model
    if args.model_name == 'Neural_ODE':
        model = NeuralODE(obs_dim, device)
    elif args.model_name == 'Contiformer':
        model = ContiFormer(obs_dim, device)
    elif args.model_name == 'NCDETransformer':
        model = NCDETransformer(
            obs_dim=obs_dim,
            device=device,
            batch_size=64,
            atol=args.atol,
            rtol=args.rtol,
            method=args.method,
            dropout=args.dropout
        )
    else:
        raise NotImplementedError

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_meter = RunningAverageMeter()

    st = 0

    # Load checkpoint from model directory
    if args.train_dir is not None:
        ckpt_path = os.path.join(model_dir, f'ckpt_best.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            st = checkpoint['itr']
            best_val = checkpoint.get('best_val', np.inf)
            log.info('Loaded best checkpoint from {}'.format(ckpt_path))

    for itr in range(st + 1, args.niters + 1):
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        # Prepare data more efficiently
        idx = npr.choice(int(ntotal * 0.5), nsample, replace=False)
        idx = sorted(idx)
        
        # Use the new amp format and ensure float32 precision for ODE solver
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            # Training step
            samp_trajs = train_trajs[:, idx, :].contiguous()
            samp_ts = torch.tensor(orig_ts[idx], device=device, dtype=torch.float32).reshape(1, -1, 1)
            samp_ts = samp_ts.repeat(samp_trajs.shape[0], 1, 1)
            samp_trajs = torch.cat((samp_trajs, samp_ts), dim=-1)

            # Forward pass with explicit device placement
            out = model(samp_trajs, orig_ts, idx=idx, is_train=True)
            
            # Ensure loss computation tensors are on the correct device
            pz0_mean = torch.zeros(out[0].size(0), device=device, dtype=torch.float32)
            pz0_logvar = torch.zeros(out[0].size(0), device=device, dtype=torch.float32)
            loss = model.calculate_loss(out, (train_target, pz0_mean, pz0_logvar))

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Clear memory
        if itr % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        # Update loss meter
        loss_meter.update(loss.item())

        if itr % 10 == 0:  # Reduce logging frequency
            log.info(f'Iter: {itr}, running loss: {loss_meter.avg:.4f}')

        # Evaluation and Visualization (every 50 iterations)
        if itr % 50 == 0:
            with torch.no_grad(), torch.cuda.amp.autocast():
                # Test on validation set
                samp_trajs = test_trajs[:, test_idx, :]
                samp_ts = torch.tensor(orig_ts[test_idx], device=device).reshape(1, -1, 1)
                samp_ts = samp_ts.repeat(ntest, 1, 1)
                samp_trajs = torch.cat((samp_trajs, samp_ts), dim=-1).float()

                pred_x = model(samp_trajs, orig_ts, idx=test_idx)[0]
                target_subset = test_target[:, test_idx, :]
                
                # Calculate metrics
                mae = torch.abs(pred_x - target_subset).sum(dim=-1).mean()
                rmse = torch.sqrt(((pred_x - target_subset) ** 2).sum(dim=-1).mean())
                
                log.info(f'Iter: {itr}, MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}')

                # Save visualizations and predictions in model directory
                if args.visualize:
                    save_path = os.path.join(model_dir, f'vis_{itr}.png')
                    plt.savefig(save_path, dpi=500, bbox_inches='tight')
                    plt.close('all')

                    save_path = os.path.join(model_dir, f'pred_{itr}.pkl')
                    torch.save({
                        'pred': pred_x,
                        'target': test_target[:, test_idx, :],
                        'samp': samp_trajs
                    }, save_path)

                    log.info(f'Saved visualization and predictions at iteration {itr}')

                # Save best model in model directory
                if mae.item() < best_val:
                    best_val = mae.item()
                    ckpt_path = os.path.join(model_dir, f'ckpt_best.pth')
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'itr': itr,
                        'best_val': best_val
                    }, ckpt_path)
                    log.info(f'New best model saved at iteration {itr}')
                    
