import torch, os, sys, shutil, json
import numpy as np
# from sklearn.datasets import make_moons, make_s_curve, make_swiss_roll
import matplotlib.pyplot as plt
import torch.distributions as D
import torch.nn as nn
# from arguments import parse_arguments
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision.datasets import MNIST, CIFAR10, ImageNet
# from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torch.distributions as D
import matplotlib.animation as animation
import pylab
from scipy import interpolate
import plotly.graph_objects as go
import plotly.express as px
import torch.nn.functional as F
import math

device = torch.device('cuda')
# torch.set_default_tensor_type('torch.FloatTensor')


# Set NumPy print options for 3 decimal places
np.set_printoptions(precision=5, suppress=True)  # ,threshold=np.inf, linewidth=200)

# =================================================================================== #
#                      GMM                                                            #
# =================================================================================== #

def gmm_differentVariance_log_batch(means, weights, query_points, std, eps=1e-13):
    """
    A differentiable GMM that returns log-densities, with batch support.

    :param means:        shape (B, N, d) - positions of center of the Gaussian
    :param weights:      shape (B, N)    - weights for each Gaussian
    :param query_points: shape (B, M, d) - points at which to evaluate the GMM
    :param std:          shape (B, N, d)
    :param eps:          small offset to avoid log(0)
    :return:             shape (B, M) => log densities
    """
    if not (std > 0).all().item():
        print("WARNING!!! At least one entry in `std` is ≤ 0")

    d = means.shape[2]
    print(means.shape, weights.shape, query_points.shape, std.shape)
    # means       => (B, N, d)
    # query_points=> (B, M, d)
    # We'll broadcast pairwise diffs: (B, M, N, d)
    diff = query_points.unsqueeze(2) - means.unsqueeze(1)
    sq_dist = ((diff**2) / (std**2).unsqueeze(1)).sum(dim=-1)  # (B, M, N)
    det_sqrt = std.prod(dim=-1)  # (B, N)

    # When covariance matrix is cI, the density is
    # p(x) = (1/(2*pi*c)^{d/2}) * exp( -0.5*||x-mu||^2 / c )
    # Ignoring the shared constant (1/(2*pi)^{d/2}), the coefficient becomes std^d.
    kernel_vals = torch.exp(-0.5 * sq_dist) / det_sqrt.unsqueeze(1)  # (B, M, N)

    # Weighted by weights => shape (B, M, N)
    weighted_kernel = kernel_vals * weights.unsqueeze(1)

    # Sum across all discrete points => shape (B, M)
    densities = weighted_kernel.sum(dim=-1)

    # Return log density, adding eps is necessary, otherwise it's easy to blow up.
    log_densities = torch.log(densities + eps)  # shape (B, M)

    return log_densities


def gmm_differentVariance_log_batch_logsumexp(means, weights, query_points, std, eps=1e-13):
    """
    A differentiable GMM that returns log-densities, with batch support.

    :param means:        shape (B, N, d) - positions of center of the Gaussian
    :param weights:      shape (B, N)    - weights for each Gaussian
    :param query_points: shape (B, M, d) - points at which to evaluate the GMM
    :param std:          shape (B, N, d)
    :param eps:          small offset to avoid log(0)
    :return:             shape (B, M) => log densities
    """
    if not (std > 0).all().item():
        raise RuntimeError("WARNING!!! At least one entry in `std` is ≤ 0")
    row_sums = weights.sum(dim=1)
    max_error = (row_sums - 1).abs().max().item()
    assert max_error < 1e-5, f"Max row sum deviation: {max_error} (expected ≤1e-5)"

    d = means.shape[2]
    # means       => (B, N, d)
    # query_points=> (B, M, d)
    # We'll broadcast pairwise diffs: (B, M, N, d)
    diff = query_points.unsqueeze(2) - means.unsqueeze(1)
    sq_dist = ((diff**2) / (std**2).unsqueeze(1)).sum(dim=-1)  # (B, M, N)
    det_sqrt = std.prod(dim=-1)  # (B, N)

    # When covariance matrix is cI, the density is
    # p(x) = (1/(2*pi*c)^{d/2}) * exp( -0.5*||x-mu||^2 / c )
    # Ignoring the shared constant (1/(2*pi)^{d/2}), the coefficient becomes std^d.
    logCoeff = torch.log((weights / det_sqrt).unsqueeze(1)).repeat(
        1, query_points.shape[1], 1
    )
    X_ = -0.5 * sq_dist + logCoeff  # (B, M, N)

    log_2pi = torch.log(2 * torch.tensor(torch.pi, device=query_points.device, dtype=query_points.dtype))
    return torch.logsumexp(X_, 2) - (query_points.shape[2] / 2) * log_2pi


def LogGMM_2d_sameVar_batch(means, weights, query_points, std=0.03, eps=1e-30):
    """
    A differentiable GMM that returns log-densities, with batch support.

    :param means:        shape (B, N, 2) - positions of center of the Gaussian
    :param weights:      shape (B, N)    - weights for each Gaussian
    :param query_points: shape (B, M, 2) - points at which to evaluate the GMM
    :param std:          float or torch.Tensor (scalar) > 0
    :param eps:          small offset to avoid log(0)
    :return:             shape (B, M) => log densities
    """
    # means       => (B, N, 2)
    # query_points=> (B, M, 2)
    # We'll broadcast pairwise diffs: (B, M, N, 2)
    diff = query_points.unsqueeze(2) - means.unsqueeze(1)
    sq_dist = (diff**2).sum(dim=-1)  # (B, M, N)

    # When covariance matrix is cI, the density is
    # p(x) = (1/(2*pi*c)^{d/2}) * exp( -0.5*||x-mu||^2 / c )
    # Ignoring the shared constant (1/(2*pi)^{d/2}), the coefficient becomes std^d.
    kernel_vals = torch.exp(-0.5 * sq_dist / (std**2))  # (B, M, N)

    # Weighted by weights => shape (B, M, N)
    weighted_kernel = kernel_vals * weights.unsqueeze(1)

    # Sum across all discrete points => shape (B, M)
    densities = weighted_kernel.sum(dim=-1)

    # Return log density, adding eps is necessary, otherwise it's easy to blow up.
    log_densities = torch.log(densities + eps)  # shape (B, M)

    return log_densities


# using torch.logsumexp
def LogGMM_2d_sameVar_batch(means, weights, query_points, std=0.03, eps=1e-15):
    """
    A differentiable 2D GMM that returns log-densities, with batch support.

    :param means:        shape (B, N, 2) - positions of discrete points
    :param weights:      shape (B, N)    - weights for each point
    :param query_points: shape (B, M, 2) - points at which to evaluate the GMM
    :param std:          float or torch.Tensor (scalar) > 0
    :param eps:          small offset to avoid log(0)
    :return:             shape (B, M) => log densities
    """
    # means       => (B, N, 2)
    # query_points=> (B, M, 2)
    # We'll broadcast pairwise diffs: (B, M, N, 2)
    weights_log = torch.log(
        weights.unsqueeze(1).repeat(1, query_points.shape[1], 1) + eps
    )  # (B, M, N)
    diff = query_points.unsqueeze(2) - means.unsqueeze(1)
    sq_dist = (diff**2).sum(dim=-1)  # (B, M, N)

    log_constant = math.log(2 * math.pi) + 2 * math.log(std)
    weighted_kernel_log = -0.5 * sq_dist / (std**2) - log_constant + weights_log

    return torch.logsumexp(weighted_kernel_log, dim=2)  # shape (B, M)


# =================================================================================== #
#                                        MMD loss terms                               #
# FOR reference: MFG paper chose
# 1. --dataset gaussian_gaussian_mixture --MMD_kernel lap
# 2. --dataset MNIST  --MMD_kernel multiscale  --MNIST_noise 1e-2 --MNIST_single_digit false
# =================================================================================== #


def cdist_l1(X):
    # X: K x B x d
    X_pairs = X.unsqueeze(1) - X.unsqueeze(2)  # K x B x B x d
    dist = torch.abs(X_pairs).sum(-1)  # K x B x B

    return dist


class Lap_kernel(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, featurizer=None, d=2, c=1):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).to(device)
        self.bandwidth = bandwidth
        self.featurizer = featurizer

    def get_bandwidth(self, dist):
        if self.bandwidth is None:
            n_samples = dist.shape[1]
            return dist.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        # X: K x B x d
        K, B = X.shape[0], X.shape[1]
        X = X.view(K, B, -1)  # K x B x d. If X contains images, they're flattened to be vectors
        dist = cdist_l1(X)

        return torch.exp(
            -dist[None, ...]
            / (self.get_bandwidth(dist) * self.bandwidth_multipliers)[:, None, None, None]
        ).sum(dim=0)


class Multiscale_kernel(nn.Module):
    def __init__(self, bandwidth=None, featurizer=None, d=2, c=1):
        super().__init__()
        self.featurizer = featurizer
        if bandwidth is None:
            # self.bandwidth = torch.tensor([0.2, 0.5, 0.9, 1.3]).to(device)
            self.bandwidth = torch.tensor([1e-1, 3e-1, 1e0, 3e1, 1e2]).to(device)

    def forward(self, X):
        # X: K x B x d
        K, B = X.shape[0], X.shape[1]
        if self.featurizer is not None:
            X = self.featurizer(X)  # K x B x ...
        X = X.reshape(K, B, -1)  # K x B x d. If X contains images, they're flattened to be vectors
        n = len(X.shape)
        dist = torch.cdist(X, X) ** 2  # K x B x B

        return (self.bandwidth[(None,) * n] ** 2 * (dist[..., None] + self.bandwidth[(None,) * n] ** 2) ** -1).sum(
            dim=-1
        )


class MMDLoss(nn.Module):
    def __init__(self, mode='def', kernel='rbf', d=2, c=1, featurizer=None, n_kernels=5, bandwidth=None):
        super().__init__()

        if kernel == 'multiscale':
            self.kernel = Multiscale_kernel(d=d, c=c, featurizer=featurizer)
        elif kernel == 'lap':
            self.kernel = Lap_kernel(d=d, bandwidth=bandwidth, n_kernels=n_kernels)

        self.mode = mode

    def forward(self, X, Y, reduce=False):
        # X, Y: B x n x d
        # kernel: (B x n x d, B x n x d) -> B x 2*n x 2*n
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        if len(Y.shape) == 2:
            Y = Y.unsqueeze(0)

        K = self.kernel(torch.cat([X, Y], dim=1))

        n_x = X.shape[1]
        n_y = Y.shape[1]

        if self.mode == 'def':
            mask_x = 1 - torch.eye(n_x).repeat(1, 1, 1).to(device)
            mask_y = 1 - torch.eye(n_y).repeat(1, 1, 1).to(device)
            XX = (K[:, :n_x, :n_x] * mask_x).sum(dim=[1, 2]) / n_x / (n_x - 1)
            XY = K[:, :n_x, n_x:].mean(dim=[1, 2])
            YY = (K[:, n_x:, n_x:] * mask_y).sum(dim=[1, 2]) / n_y / (n_y - 1)

            MMDs = XX - 2 * XY + YY  # shape (B,)

            if reduce:
                return MMDs.mean()
                # return MMDs.sum()
            return MMDs
        else:
            XX = K[:, :n_x, :n_x].mean(dim=[1, 2])
            XY = K[:, :n_x, n_x:].mean(dim=[1, 2])
            YY = K[:, n_x:, n_x:].mean(dim=[1, 2])

            return XX - 2 * XY + YY

# MMD_dist = MMDLoss(kernel='multiscale')
# E = MMD_dist(X_1_pred, X_1)


# =================================================================================== #
#                      Different divergence computations                              #
# =================================================================================== #


def compute_porous_divergence(X, V, code, net, args):

    div_V = torch.zeros(X.shape[0], X.shape[1]).to(X.device)
    for j in range(args.dim):

        delta_x = args.Deltax * torch.ones_like(X[:, :, j].detach())
        random_sign = torch.randn_like(X[:, :, j].detach()).sign()
        random_sign[random_sign == 0] = 1
        delta_x *= random_sign
        delta_x *= (0.5 + 0.6 * torch.rand_like(delta_x))

        e = torch.zeros_like(X).to(X.device)
        e[:, :, j] = delta_x
        random_sign = torch.randn_like(X.detach()).sign()
        random_sign[random_sign == 0] = 1

        div_V += (net.decode(code, (X + e) * random_sign)[:, :, j]
                  - net.decode(code, X * random_sign)[:, :, j]) / delta_x

    return div_V


def compute_divergence_batch_central(X, V, code, net, args):

    with torch.no_grad():
        # 1. Compute delta_x per dimension [B, n, d]
        delta_x = args.Deltax * torch.ones_like(X)
        if args.Deltax_randomness:
            delta_x *= (0.5 + 1.5 * torch.rand_like(delta_x))

        # 2. Create +e and -e perturbations using repeat [B*d, n, d]
        X_expanded = X.repeat(args.dim, 1, 1)  # [B*d, n, d]

        # Initialize perturbations
        e = torch.zeros_like(X_expanded)

        # Create mask for each dimension
        for j in range(args.dim):
            # Get the batch indices corresponding to dimension j
            batch_indices = torch.arange(j * X.shape[0], (j + 1) * X.shape[0])
            e[batch_indices, :, j] = delta_x[:, :, j]

        # 3. Batch evaluate all perturbations (2 decoder calls)
        X_pos = X_expanded + e
        X_neg = X_expanded - e
        del X_expanded, e

    code_expanded = code.repeat(1, args.dim, 1)  # [B*d, latent_dim]

    # Decoder calls
    V_pos = net.decode(code_expanded, X_pos)  # [B*d, n, d]
    V_neg = net.decode(code_expanded, X_neg)  # [B*d, n, d]

    # 4. Compute divergence [B, n]
    div_V = torch.zeros(X.shape[0], X.shape[1], device=X.device)

    for j in range(args.dim):
        batch_indices = slice(j * X.shape[0], (j + 1) * X.shape[0])
        V_pos_j = V_pos[batch_indices, :, j]  # [B, n]
        V_neg_j = V_neg[batch_indices, :, j]  # [B, n]
        div_V += (V_pos_j - V_neg_j) / (2 * delta_x[:, :, j])

    return div_V


def compute_divergence_batch(X, V, code, net, args):

    with torch.no_grad():
        # 1. Compute delta_x per dimension [B, n, d]
        delta_x = args.Deltax * torch.ones_like(X.detach())
        if args.Deltax_randomness:
            delta_x *= (0.5 + 1.5 * torch.rand_like(delta_x))

        # 2. Create +e perturbations using repeat [B*d, n, d]
        X_expanded = X.repeat(args.dim, 1, 1)
        e_pos = torch.zeros_like(X_expanded)
        for j in range(args.dim):
            batch_indices = torch.arange(j * X.shape[0], (j + 1) * X.shape[0])
            e_pos[batch_indices, :, j] = delta_x[:, :, j]
        X_expanded += e_pos

    code_expanded = code.repeat(1, args.dim, 1)
    V_pos = net.decode(code_expanded, X_expanded)

    div_V = torch.zeros(X.shape[0], X.shape[1], device=X.device)
    for j in range(args.dim):
        # Select the batches for current dimension
        batch_indices = slice(j * X.shape[0], (j + 1) * X.shape[0])
        div_V += (V_pos[batch_indices, :, j] - V[:, :, j]) / (delta_x[:, :, j])

    return div_V


import torch

def compute_divergence(X5, V5, code, net, args):
    """
    X5: (B, N, C, H, W)
    V5: (B, N, C, H, W) = net.decode(code, X5)
    return: div_V (B, N)

    Compute the divergence in the "data space" using forward/central finite differences.
    """
    assert X5.dim() == 5, "compute_divergence(5D) expects X of shape (B,N,C,H,W)"
    device = X5.device
    B, N, C, H, W = X5.shape
    D = C * H * W

    # Flat view (only used to build e and pick the j-th component; not fed directly to the network)
    Xf = X5.view(B, N, D)
    Vf = V5.view(B, N, D)

    def decode_flat(xflat):
        """Reshape back to 5D -> decode -> flatten to (B, N, D)."""
        x5 = xflat.view(B, N, C, H, W)
        v5 = net.decode(code, x5)              # (B,N,C,H,W)
        return v5.view(B, N, D)                # (B,N,D)

    div_V = torch.zeros(B, N, device=device)

    use_central = (getattr(args, "divChoice", "forward") == "central")
    rand_mag = getattr(args, "Deltax_randomness", False)

    for j in range(D):
        delta_x = args.Deltax * torch.ones_like(Xf[:, :, j].detach())

        if rand_mag:
            delta_x = delta_x * (0.5 + 1.5 * torch.rand_like(delta_x))

        # For forward difference, use a random sign to reduce variance.
        if not use_central:
            rs = torch.sign(torch.randn_like(delta_x))
            rs[rs == 0] = 1
            delta_x = delta_x * rs

        e = torch.zeros_like(Xf, device=device)
        e[:, :, j] = delta_x

        if use_central:
            f_plus = decode_flat(Xf + e)[:, :, j]
            f_minus = decode_flat(Xf - e)[:, :, j]
            div_V += (f_plus - f_minus) / (2.0 * delta_x)
        else:
            f_plus = decode_flat(Xf + e)[:, :, j]
            div_V += (f_plus - Vf[:, :, j]) / delta_x

    return div_V


def adaptive_forwarddiff_divergence(X_k, V_k, code, net, base_delta=0.01):
    """
    Compute ∇·V using adaptive step sizes with consistent scaling.
    Args:
        X_k: Input coordinates [batch_size, num_points, dim]
        V_k: Precomputed vector field [batch_size, num_points, dim]
        code: Latent code [batch_size, code_dim]
        net: Decoder network
        device: torch device
        base_delta: Scaling factor for adaptive step
    Returns:
        div_V: Divergence field [batch_size, num_points]
    """
    device = X_k.device
    batch_size, num_points, dim = X_k.shape
    div_V = torch.zeros(batch_size, num_points, device=device)

    # Adaptive step size per sample in batch
    with torch.no_grad():
        delta_x = base_delta * X_k.abs().mean(dim=(1, 2))  # [batch_size]
        delta_x = torch.clamp(delta_x, 1e-5, 0.01)  # Prevent extreme values

    for j in range(dim):
        e = torch.zeros_like(X_k)
        e[..., j] = 1.0  # Unit direction

        # Scale perturbation by adaptive delta_x
        perturbation = delta_x.view(-1, 1, 1) * e  # [batch_size, 1, 1] * [bs, pts, dim]

        # Forward difference with consistent scaling
        V_plus = net.decode(code, X_k + perturbation)[..., j]
        div_V += (V_plus - V_k[..., j]) / delta_x.view(-1, 1)  # Same delta_x!

        # Optional: Central difference version
        # V_minus = net.decode(code, X_k - perturbation)[..., j]
        # div_V += (V_plus - V_minus) / (2 * delta_x.view(-1, 1))

    return div_V


def adaptive_higher_order_divergence(X_t, code, net, args, base_delta=0.01, min_delta=1e-5, max_delta=1e-1):
    """
    Verified version with adaptive step size and 4th-order finite differences.

    Args:
        X_t: Input coordinates (batch_size, num_points, dim)
        code: Latent code (batch_size, code_dim)
        net: Decoder network
        base_delta: Base step size (scales with input magnitude)
        dim: Spatial dimension
        min_delta: Minimum allowed step size
        max_delta: Maximum allowed step size

    Returns:
        div_V: Computed divergence (batch_size, num_points)
    """
    device = X_t.device
    batch_size = X_t.shape[0]
    div_V = torch.zeros(batch_size, X_t.shape[1], device=device)
    dim = args.dim

    # Adaptive Δx calculation (per sample in batch)
    with torch.no_grad():
        delta_x = base_delta * X_t.abs().mean(dim=(1, 2))  # shape [batch_size]
        delta_x = torch.clamp(delta_x, min_delta, max_delta)

    for j in range(dim):
        # Create perturbation vectors (correct broadcasting)
        e = torch.zeros_like(X_t)
        e[..., j] = 1.0

        # Scale perturbations per sample in batch
        delta = delta_x.view(-1, 1, 1) * e  # shape [batch_size, 1, 1] * [bs, pts, dim]
        delta2 = 2 * delta

        # Compute vector field components
        V_plus2 = net.decode(code, X_t + delta2)[..., j]
        V_plus = net.decode(code, X_t + delta)[..., j]
        V_minus = net.decode(code, X_t - delta)[..., j]
        V_minus2 = net.decode(code, X_t - delta2)[..., j]

        # 4th-order central difference (stable implementation)
        numerator = -V_plus2 + 8 * V_plus - 8 * V_minus + V_minus2
        div_V += numerator / (12 * delta_x.view(-1, 1))  # proper broadcasting

    # Numerical safety
    div_V = torch.nan_to_num(div_V, nan=0.0, posinf=0.0, neginf=0.0)
    return div_V


def DIV_finite_difference_2D(net, code, x, args, v=None):
    if v is None:
        v = net.decode(code, x)

    Deltax = args.Deltax
    # print("grid: ", Deltax)
    e0 = torch.zeros_like(x)
    e0[:, :, 0] += Deltax
    e1 = torch.zeros_like(x)
    e1[:, :, 1] += Deltax
    return (
        net.decode(code, (x + e0).to(device))[:, :, 0]
        - v[:, :, 0]
        + net.decode(code, (x + e1).to(device))[:, :, 1]
        - v[:, :, 1]
    ) / Deltax


import torch

def Div_Hutchinson(net, code, x5, args, v5=None,
                   num_e: int = 3, sigma: float = None,
                   chunk: int = None, use_amp: bool = True):
    """
    x5:  (B, N, C, H, W)
    v5:  (B, N, C, H, W) = net.decode(code, x5) (can be reused to save one decode)
    Returns: (B, N) divergence approximation  E_e[ e^T J(x) e ].
    """
    assert x5.dim() == 5, "Div_Hutchinson(5D) expects x of shape (B,N,C,H,W)"
    device = x5.device
    B, N, C, H, W = x5.shape
    D = C * H * W

    # Step size σ: scaled by dimension; if not provided, use default.
    if sigma is None:
        sigma = 0.01 / (D ** 0.5)

    # Need v(x) as baseline; reusing it saves one decode.
    if v5 is None:
        with torch.cuda.amp.autocast(enabled=use_amp):
            v5 = net.decode(code, x5)   # (B,N,C,H,W)

    def decode_5d(x):
        # Can be chunked along N dimension to avoid peak memory.
        if not chunk:
            with torch.cuda.amp.autocast(enabled=use_amp):
                return net.decode(code, x)
        outs = []
        for s in range(0, N, chunk):
            t = min(s + chunk, N)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outs.append(net.decode(code, x[:, s:t]))  # (B, t-s, C,H,W)
        return torch.cat(outs, dim=1)

    # Accumulate estimates over num_e random directions, then average.
    est_list = []
    for _ in range(num_e):
        e5 = torch.randn_like(x5, device=device)          # (B,N,C,H,W)
        v_e5 = decode_5d(x5 + sigma * e5)                 # (B,N,C,H,W)

        # Directional derivative ≈ (v(x+σe) - v(x)) / σ, then dot with e: e^T (J e).
        dir5 = (v_e5 - v5) / sigma                        # (B,N,C,H,W)
        est = (dir5.view(B, N, D) * e5.view(B, N, D)).sum(dim=-1)  # (B,N)
        est_list.append(est)

    approx_div = torch.stack(est_list, dim=-1).mean(dim=-1)  # (B,N)
    return approx_div


# =================================================================================== #
#                                      Different KL loss                              #
# =================================================================================== #

def compute_JKO_KL_Gaussianmix_NoDataGenerating(X, net, args, print_=True):
    device = args.device if hasattr(args, "device") else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    def _to_tensor(v):
        # v can be a tensor, or list/tuple[tensor, ...]
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, (list, tuple)):
            # Common case: list over time steps -> stack to (T+1, ...)
            return torch.stack(v, dim=0) if isinstance(v[0], torch.Tensor) else torch.tensor(v)
        # numpy / scalar
        return torch.as_tensor(v)

    # --- Key: convert each part of TrainingData into a tensor ---
    X0_path = _to_tensor(X[0]).detach().float().to(device)  # (T+1, N, d) or similar
    X1 = _to_tensor(X[1]).detach().float().to(device)       # target / current distribution

    if X0_path.dim() == 3 and X1.dim() == 3 and X0_path.shape[0] != X1.shape[0]:
        # Treat as (T+1, N, d): take P0 (0-th frame) and add batch dim B=1.
        path = X0_path[0].unsqueeze(0)                      # (1, N, d)
    else:
        path = X0_path                                      # (B, N, d) or (T+1, N, d) depending on usage

    path_X_1 = X1                                           # Keep naming consistent
    args.dim = path.shape[-1]

    # Generate data
    net.train()
    X_t = path

    code = net.encode(X_t)

    # code = net.encode(X_t,X_0_densityvalue=path_density, X_1=path_X_1,X_1_densityvalue=path_X_1_density)
    V_t = net.decode(code, X_t)
    path_T_pred = X_t + V_t

    div_V = Div_Hutchinson(net, code, X_t, args, V_t,)

    # KL divergence
    # X_t: (T+1,N,C,H,W) or (T+1,N,D)
    if path_T_pred.dim() == 5:
        B, N, C, H, W = path_T_pred.shape
        quad = (path_T_pred.view(B, N, -1)**2).sum(dim=2) / (2 * 0.1)  # -> (T+1, N)
    else:
        quad = (path_T_pred**2).sum(dim=2) / (2 * 0.1)                 # -> (T+1, N)

    E = -div_V.mean(dim=1) + quad.mean(dim=1)                         # -> (T+1,)

    # E = path_density_log.mean(dim=1) - div_V.mean(dim=1)  - gmm_differentVariance_log_batch_logsumexp( target_means, target_weights, path_T_pred,Std).mean(dim=1)

    # transport cost (L),    V.shape:(B,N,d)
    if V_t.dim() == 5:
        B, N, C, H, W = V_t.shape
        # Flatten (C,H,W) to D, then per particle compute squared norm and average over N -> (B,)
        W2 = (V_t.view(B, N, -1)**2).sum(dim=2).mean(dim=1)
    else:
        B, N, D = V_t.shape
        # Equivalent logic: per particle squared norm, average over N -> (B,)
        W2 = (V_t**2).sum(dim=2).mean(dim=1)

    # Debug prints for shapes
    print(f"Input shapes - path: {path.shape}")
    print(f"div_V shape: {div_V.shape}, E shape: {E.shape}")

    # In GenerateNewPathWithFixedInitial (debug print location kept here as in original)
    print(f"Before repeat - X_1: {X_t.shape}")
    print(f"After repeat - path_X_1: {path_X_1.shape}")

    if print_:
        print("E: ", E.detach().cpu().numpy().flatten())
        # print("(W2+2*args.deltat*E): ", (W2 + 2 * args.deltat * E).detach().cpu().numpy().flatten())

    return (W2 / (2 * args.deltat) + E).mean(), W2.detach().cpu(), E.detach().cpu(), path_T_pred[::args.B_dist].detach().cpu()


########################Other JKO model##################################

def compute_JKO_KL_MNIST_MMD(X, net, args, print_=True):

    X_0 = X[0].float().to(device)  # B x n x d; samples from P_0
    X_0_density = X[2].float().to(device)  # B x n x d; sample density values
    X_1 = X[1].float().to(device)  # B x n x d; samples from P_1
    X_1_density = X[3].float().to(device)  # B x n x d; sample density values
    grids = X[4].float().repeat(1 + args.jko_T, 1, 1).to(device)  # (B,d) --> (B,N,d)
    grid_values = X[5].float().repeat(1 + args.jko_T, 1).to(device)  # (B,1,1) --> (B,N)
    print(X_0.shape, grids.shape, grid_values.shape, 1 + args.jko_T, "The grid shape got boosted")

    # Generate the path of data
    path = []
    X_k = X_0

    path.append(X_k.clone())

    net.eval()
    with torch.no_grad():
        for i in range(args.jko_T):

            # code = net.encode(X_k)
            code = net.encode(X_k, X_1=X_1)
            V_t = net.decode(code, X_k)

            X_k = (X_k + V_t).detach()
            path.append(X_k.detach().clone())

    path = torch.cat(path, dim=0).detach().to(device)
    path_X_1 = X_1.repeat(1 + args.jko_T, 1, 1).detach().to(device)

    # Generate data
    net.train()
    X_t = path
    code = net.encode(X_t, X_1=path_X_1)
    V_t = net.decode(code, X_t)
    path_T_pred = X_t + V_t

    # MMD loss
    MMD_dist = MMDLoss(kernel='multiscale')
    E = MMD_dist(path_T_pred, path_X_1)
    print("MMD: ", E.shape)

    # transport cost (L),    V.shape:(B,N,d)
    W2 = (V_t**2).sum(dim=[1, 2]) / V_t.shape[1]

    if print_:
        print("W2: ", W2.detach().cpu().numpy().flatten())
        print("E: ", E.detach().cpu().numpy().flatten())
        print("(W2+2*args.deltat*E): ", (W2 + 2 * args.deltat * E).detach().cpu().numpy().flatten())

    # return (W2+2*args.deltat*E).mean(), W2, E, path_T_pred[::args.B_dist], None
    return (W2 / (2 * args.deltat) + E).mean(), W2, E, path_T_pred[::args.B_dist], None


def compute_JKO_porous_cost_NoDatagenerating(path, path_density, net, args):

    # Train the model
    net.train()
    X_t = path.detach().to(device)
    code = net.encode(X_t, path_density, None)
    V_t = net.decode(code, X_t)
    path_T_pred = X_t + V_t

    # Transport cost (L), V.shape:(B,N,d)
    W2 = (V_t**2).sum(dim=[1, 2]) / V_t.shape[1]

    # Compute divergence
    # Use finite difference to compute the divergence
    assert X_t.shape == V_t.shape, 'path.shape != V_t.shape'  # shape (B,n,d)

    div_V = compute_divergence(X_t, V_t, code, net, args)
    rho_new = path_density / (torch.exp(div_V) + 1e-10)

    # Compute porous medium energy, requires new density rho_new and X_t
    E = (rho_new**(args.porous_m - 1)).mean(dim=1) / (args.porous_m - 1)
    # print("m=2")

    # weights = torch.tensor([args.gamma**i for i in range( args.jko_T+1)]).to(device)
    # assert (weights.shape == E.shape), "weight shape does not match"

    print("Energy", W2 / (2 * args.deltat) + E)

    # return (W2+2*args.deltat*E).mean() , W2,  E, path_T_pred[::args.B_dist], rho_new[::args.B_dist]
    # return (W2/(2*args.deltat)+E).mean() , W2,  E, path_T_pred[::args.B_dist], rho_new[::args.B_dist]

    return (
        (W2 / (2 * args.deltat) + E).mean(),
        W2.detach().cpu(),
        E.detach().cpu(),
        path_T_pred.detach().cpu(),
        rho_new.detach().cpu(),
    )


def compute_JKO_aggregation_cost_NoDatagenerating(path, pq, net, args):

    # Train the model
    net.train()
    X_t = path.detach().to(device)
    code = net.encode(X_t, pq, None)
    V_t = net.decode(code, X_t)
    path_T_pred = X_t + V_t

    # Transport cost (L), V.shape:(B,N,d)
    W2 = (V_t**2).sum(dim=[1, 2]) / V_t.shape[1]

    # Compute aggregation energy
    distances = torch.cdist(path_T_pred, path_T_pred)
    print(distances.shape, distances.mean())

    p = (pq[:, 0, 0] + 1).view(-1, 1, 1)
    q = (pq[:, 0, 1] + 1).view(-1, 1, 1)
    E = (distances**q / q - distances**p / p).mean(dim=[1, 2])

    print("Energy", W2 / (2 * args.deltat) + E)

    return (
        (W2 / (2 * args.deltat) + E).mean(),
        W2.detach().cpu(),
        E.detach().cpu(),
        path_T_pred.detach().cpu(),
    )
