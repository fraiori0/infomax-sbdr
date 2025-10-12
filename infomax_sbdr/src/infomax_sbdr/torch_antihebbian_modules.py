import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class AntiHebbianBaseTorch(nn.Module):
    """
    Base class for Anti-Hebbian learning modules.
    
    Implements the core anti-hebbian learning rule with forward and lateral connections.
    The network learns to produce sparse, distributed representations through competitive
    inhibition and anti-hebbian weight updates.
    """
    
    def __init__(
        self,
        n_features: int,
        p_target: float,
        momentum: float = 0.95,
        use_dropout: bool = False,
        dropout_rate: float = 0.2,
        bias_init_value: float = 0.0,
    ):
        """
        Args:
            n_features: Number of output features/units
            p_target: Target activation probability for each unit
            momentum: Momentum for running average of co-activations
            use_dropout: Whether to use dropout between forward and lateral layers
            dropout_rate: Dropout probability if use_dropout is True
            bias_init_value: Initial value for biases
        """
        super().__init__()
        
        self.n_features = n_features
        self.p_target = p_target
        self.momentum = momentum
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.bias_init_value = bias_init_value
        
        # These will be initialized by child classes
        self.w_f = None  # Forward weights
        self.w_l = None  # Lateral weights
        
        # Initialize running average of co-activations
        # mu[i,j] tracks the average co-activation of units i and j
        mu0 = torch.ones(n_features, n_features) * (p_target ** 2)
        mu0 += torch.eye(n_features) * (p_target - p_target ** 2)
        self.register_buffer('mu', mu0)
        
        # Dropout layer (if needed)
        if use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
    
    def _init_weights(self, module: nn.Module, init_variance: float):
        """Initialize weights with variance scaling (similar to JAX implementation)."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Variance scaling initialization (fan_in, truncated normal approximation)
            fan_in = module.weight.size(1)
            if isinstance(module, nn.Conv2d):
                fan_in *= module.weight.size(2) * module.weight.size(3)
            
            std = math.sqrt(init_variance / fan_in)
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, self.bias_init_value)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the anti-hebbian network.
        
        Args:
            x: Input tensor of shape (batch, *)
        
        Returns:
            Dictionary containing:
                - 'x': Input
                - 'y': Forward activation (before lateral inhibition)
                - 'z': Final activation (after lateral inhibition)
        """
        # Forward activation
        y = (self.w_f(x) > 0).float()
        
        # Apply dropout if enabled
        if self.use_dropout and self.training:
            y_dropped = self.dropout(y)
            z = (self.w_l(y_dropped) > 0).float()
        else:
            z = (self.w_l(y) > 0).float()
        
        # Final activation: AND between forward and lateral
        z = y * z
        
        return {
            'x': x,
            'y': y,
            'z': z,
        }
    
    @torch.no_grad()
    def _update_mu(self, z: torch.Tensor):
        """
        Update the running average of co-activations.
        
        Args:
            z: Output activations of shape (batch, n_features) or (batch, n_features, H, W)
        """
        # Flatten spatial dimensions if present (for conv layers)
        if z.ndim > 2:
            # For conv: (batch, channels, H, W) -> (batch*H*W, channels)
            batch_size = z.size(0)
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.n_features)
        else:
            z_flat = z
        
        # Compute co-activation matrix: z_i * z_j for all pairs
        # Shape: (batch, n_features, n_features)
        zi_zj = z_flat.unsqueeze(-1) * z_flat.unsqueeze(-2)
        
        # Update running average with momentum
        dmu = (1.0 - self.momentum) * (zi_zj.mean(dim=0) - self.mu)
        self.mu.add_(dmu)
    
    @torch.no_grad()
    def _compute_dw_f(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute updates for forward weights.
        
        The update rule encourages weights to match inputs that activate their units,
        scaled by the difference between target and actual co-activation.
        """
        raise NotImplementedError("Must be implemented by child class")
    
    @torch.no_grad()
    def _compute_dw_l(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute updates for lateral weights.
        
        The update rule implements anti-hebbian learning: weights between co-active
        units are adjusted to reduce their co-activation towards the target.
        """
        x = outputs['x']
        y = outputs['y']
        z = outputs['z']
        
        # Get current parameters
        mu = self.mu
        mu_self = torch.diag(mu)
        kernel = self.w_l.weight
        
        # Flatten spatial dimensions if present
        if z.ndim > 2:
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.n_features)
        else:
            z_flat = z
        
        # Co-activation matrix for this batch
        zi_zj = z_flat.unsqueeze(-1) * z_flat.unsqueeze(-2)  # (batch, n_features, n_features)
        
        # Anti-hebbian update: reduce co-activation towards target
        error = (self.p_target ** 2) - mu
        dkernel = (error.unsqueeze(0) * zi_zj).mean(dim=0)
        
        # Bias update
        dbias = self.p_target - mu_self
        
        return {
            'weight': dkernel,
            'bias': dbias if self.w_l.bias is not None else None,
        }
    
    @torch.no_grad()
    def _apply_updates(
        self,
        dw_f: Dict[str, torch.Tensor],
        dw_l: Dict[str, torch.Tensor],
        lr: float
    ):
        """Apply computed updates to the weights."""
        # Update forward weights
        self.w_f.weight.add_(dw_f['weight'], alpha=lr)
        if self.w_f.bias is not None and dw_f['bias'] is not None:
            self.w_f.bias.add_(dw_f['bias'], alpha=lr)
        
        # Update lateral weights
        self.w_l.weight.add_(dw_l['weight'], alpha=lr)
        if self.w_l.bias is not None and dw_l['bias'] is not None:
            self.w_l.bias.add_(dw_l['bias'], alpha=lr)
        
        # CRITICAL: Zero out diagonal of lateral weights (no self-connections)
        self.w_l.weight.fill_diagonal_(0)
    
    def update_params(self, outputs: Dict[str, torch.Tensor], lr: float = 0.1):
        """
        Update parameters using anti-hebbian learning rule.
        
        This method should be called after forward() in your training loop.
        
        Args:
            outputs: Dictionary from forward() containing 'x', 'y', 'z'
            lr: Learning rate for weight updates
        """
        # Update running average of co-activations
        self._update_mu(outputs['z'])
        
        # Compute weight updates
        dw_f = self._compute_dw_f(outputs)
        dw_l = self._compute_dw_l(outputs)
        
        # Apply updates
        self._apply_updates(dw_f, dw_l, lr)


class AntiHebbianDenseTorch(AntiHebbianBaseTorch):
    """
    Dense (fully-connected) Anti-Hebbian module.
    
    Example usage:
        model = AntiHebbianDense(
            in_features=784,
            n_features=256,
            p_target=0.05,
            momentum=0.95
        )
        
        # Training loop
        for inputs, labels in dataloader:
            outputs = model(inputs)
            model.update_params(outputs, lr=0.1)
    """
    
    def __init__(
        self,
        in_features: int,
        n_features: int,
        p_target: float,
        momentum: float = 0.95,
        init_variance_w_forward: float = 1.0,
        init_variance_w_lateral: float = 1.0,
        use_dropout: bool = False,
        dropout_rate: float = 0.2,
        bias_init_value: float = 0.0,
    ):
        """
        Args:
            in_features: Number of input features
            n_features: Number of output features/units
            p_target: Target activation probability
            momentum: Momentum for running average
            init_variance_w_forward: Variance scaling for forward weights
            init_variance_w_lateral: Variance scaling for lateral weights
            use_dropout: Whether to use dropout
            dropout_rate: Dropout probability
            bias_init_value: Initial bias value
        """
        super().__init__(
            n_features=n_features,
            p_target=p_target,
            momentum=momentum,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            bias_init_value=bias_init_value,
        )
        
        self.in_features = in_features
        
        # Forward layer: input -> units
        self.w_f = nn.Linear(in_features, n_features, bias=True)
        self._init_weights(self.w_f, init_variance_w_forward)
        
        # Lateral layer: units -> units (for inhibition)
        self.w_l = nn.Linear(n_features, n_features, bias=True)
        self._init_weights(self.w_l, init_variance_w_lateral)
        
        # Zero out diagonal (no self-connections)
        with torch.no_grad():
            self.w_l.weight.fill_diagonal_(0)
    
    @torch.no_grad()
    def _compute_dw_f(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute updates for forward weights in dense layer."""
        x = outputs['x']
        z = outputs['z']
        
        # Get current parameters
        kernel = self.w_f.weight  # (n_features, in_features)
        mu_self = torch.diag(self.mu)
        
        # Compute update: (p_target^2 - mu_self^2) * (x - kernel) * z
        # x: (batch, in_features)
        # z: (batch, n_features)
        # kernel: (n_features, in_features)
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1)  # (batch, 1, in_features)
        z_expanded = z.unsqueeze(-1)  # (batch, n_features, 1)
        kernel_expanded = kernel.unsqueeze(0)  # (1, n_features, in_features)
        
        # Compute xi_zj = (x - kernel) * z
        xi_zj = (x_expanded - kernel_expanded) * z_expanded  # (batch, n_features, in_features)
        
        # Scale by error term
        error = (self.p_target ** 2) - (mu_self ** 2)  # (n_features,)
        dkernel = (error.unsqueeze(-1) * xi_zj).mean(dim=0)  # (n_features, in_features)
        
        # Bias update
        dbias = self.p_target - mu_self
        
        return {
            'weight': dkernel,
            'bias': dbias if self.w_f.bias is not None else None,
        }


class AntiHebbianConv2dTorch(AntiHebbianBaseTorch):
    """
    Convolutional Anti-Hebbian module.
    
    Applies anti-hebbian learning to convolutional layers. Each output channel
    is treated as a "unit" for the purposes of co-activation tracking.
    
    Example usage:
        model = AntiHebbianConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            p_target=0.05,
            stride=1,
            padding=1
        )
        
        # Training loop
        for images, labels in dataloader:
            outputs = model(images)
            model.update_params(outputs, lr=0.1)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        p_target: float,
        stride: int = 1,
        padding: int = 0,
        momentum: float = 0.95,
        init_variance_w_forward: float = 1.0,
        init_variance_w_lateral: float = 1.0,
        use_dropout: bool = False,
        dropout_rate: float = 0.2,
        bias_init_value: float = 0.0,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (features)
            kernel_size: Size of convolutional kernel
            p_target: Target activation probability
            stride: Stride of convolution
            padding: Padding for convolution
            momentum: Momentum for running average
            init_variance_w_forward: Variance scaling for forward weights
            init_variance_w_lateral: Variance scaling for lateral weights
            use_dropout: Whether to use dropout
            dropout_rate: Dropout probability
            bias_init_value: Initial bias value
        """
        super().__init__(
            n_features=out_channels,
            p_target=p_target,
            momentum=momentum,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            bias_init_value=bias_init_value,
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Forward convolution: input -> feature maps
        self.w_f = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self._init_weights(self.w_f, init_variance_w_forward)
        
        # Lateral convolution: feature maps -> feature maps (1x1 conv for channel interactions)
        self.w_l = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,  # 1x1 convolution for lateral connections
            stride=1,
            padding=0,
            bias=True
        )
        self._init_weights(self.w_l, init_variance_w_lateral)
        
        # Zero out diagonal in lateral weights (no self-connections between channels)
        with torch.no_grad():
            # For Conv2d, weight shape is (out_channels, in_channels, H, W)
            # We want to zero out connections from channel i to channel i
            for i in range(out_channels):
                self.w_l.weight[i, i, :, :] = 0
    
    @torch.no_grad()
    def _compute_dw_f(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute updates for forward convolutional weights."""
        x = outputs['x']  # (batch, in_channels, H, W)
        z = outputs['z']  # (batch, out_channels, H', W')
        
        # Get current parameters
        kernel = self.w_f.weight  # (out_channels, in_channels, kH, kW)
        mu_self = torch.diag(self.mu)  # (out_channels,)
        
        # For convolutional layers, we need to compute the gradient w.r.t. the kernel
        # This is more complex than dense layers. We'll use a similar approach but
        # need to account for the spatial structure.
        
        # Unfold input to create patches
        # x_unfolded: (batch, in_channels*kH*kW, n_patches)
        x_unfolded = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        
        # Reshape for easier computation
        batch_size = x.size(0)
        n_patches = x_unfolded.size(-1)
        kernel_numel = self.in_channels * self.kernel_size * self.kernel_size
        
        # x_patches: (batch*n_patches, in_channels, kH, kW)
        x_patches = x_unfolded.transpose(1, 2).reshape(
            batch_size * n_patches,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )
        
        # z_flat: (batch*n_patches, out_channels)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        # Compute update similar to dense case
        # kernel: (out_channels, in_channels, kH, kW)
        x_patches = x_patches.unsqueeze(1)  # (batch*n_patches, 1, in_channels, kH, kW)
        kernel_expanded = kernel.unsqueeze(0)  # (1, out_channels, in_channels, kH, kW)
        z_expanded = z_flat.view(-1, self.out_channels, 1, 1, 1)  # (batch*n_patches, out_channels, 1, 1, 1)
        
        # xi_zj = (x - kernel) * z
        xi_zj = (x_patches - kernel_expanded) * z_expanded
        
        # Scale by error term and average over batch
        error = (self.p_target ** 2) - (mu_self ** 2)  # (out_channels,)
        error = error.view(1, self.out_channels, 1, 1, 1)
        
        dkernel = (error * xi_zj).mean(dim=0)  # (out_channels, in_channels, kH, kW)
        
        # Bias update
        dbias = self.p_target - mu_self
        
        return {
            'weight': dkernel,
            'bias': dbias if self.w_f.bias is not None else None,
        }
    
    @torch.no_grad()
    def _apply_updates(
        self,
        dw_f: Dict[str, torch.Tensor],
        dw_l: Dict[str, torch.Tensor],
        lr: float
    ):
        """Apply updates with special handling for conv lateral weights."""
        # Standard weight updates
        super()._apply_updates(dw_f, dw_l, lr)
        
        # For conv layers, zero out the diagonal channels in 1x1 lateral conv
        with torch.no_grad():
            for i in range(self.out_channels):
                self.w_l.weight[i, i, :, :] = 0


