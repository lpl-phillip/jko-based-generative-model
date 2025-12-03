import torch, os, sys, shutil
import numpy as np
import matplotlib
import time
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.distributions as D
import torch.nn as nn
from arguments import parse_arguments
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from time import sleep
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from models import *
from utils import *
from loss_jko import *
from dataset_utils import *
import copy
from utils import _flatten_points_5d,_pca_fit_2d_concat

# =================================================================================== #
#                                        Meta                                         #
# =================================================================================== #
args = parse_arguments()

import utils as utils_mod  # Import utils again as a module

# Override Logger.close / __del__ so it does not close stdout
def _logger_safe_close(self):
    # Do not call self.console.close(); only close the log file
    if getattr(self, "file", None) is not None:
        try:
            self.file.close()
        except Exception:
            pass
        self.file = None

# Monkey-patch: from now on all Logger instances will use this safe close
utils_mod.Logger.close = _logger_safe_close
utils_mod.Logger.__del__ = _logger_safe_close
utils_mod.Logger.__exit__ = lambda self, *args: _logger_safe_close(self)

# For testing different substeps without overwriting the same log_dir
if args.exp_name == '':
    args.exp_name = f"substeps{args.substeps}"
else:
    args.exp_name = f"{args.exp_name}_substeps{args.substeps}"

device = torch.device('cuda')
log_dir, plot_dir, model_dir, txt_logger, tbx = prepare_loggers(args)

device = torch.device('cuda')
log_dir, plot_dir, model_dir, txt_logger, tbx = prepare_loggers(args)

loss_log_path = os.path.join(log_dir, 'loss_vector_log.txt')
with open(loss_log_path, 'w') as f:
    f.write("Iteration, LossVector\n") 

grad_log_path = os.path.join(log_dir, 'loss_vector_log.txt')
with open(grad_log_path, 'w') as f:
    f.write("Iteration, LossVector\n")

torch.manual_seed(args.seed)
np.random.seed(args.seed) 
# torch.set_default_tensor_type('torch.FloatTensor')
# =================================================================================== #
#                                        Data                                         #
# =================================================================================== #
def _flatten_points_5d(X):
    """
    Supported shapes: (1,B,C,H,W) / (B,C,H,W) / (B,H,W) / (B,D)
    Always return (B, D).
    """
    if X.dim() == 5:                   # (1,B,C,H,W)
        return X[0].reshape(X.size(1), -1).detach()
    if X.dim() == 4:                   # (B,C,H,W)
        return X.reshape(X.size(0), -1).detach()
    if X.dim() == 3:                   # (B,H,W) interpreted as grayscale
        return X.reshape(X.size(0), -1).detach()
    if X.dim() == 2:                   # (B,D) already flattened
        return X.detach()
    raise ValueError(f"Expected (1,B,C,H,W)/(B,C,H,W)/(B,H,W)/(B,D), got {tuple(X.shape)}")


def get_mnist_1_random_batch(
    img_size=24,
    root="./data",
    n_images=1024,
    device="cuda",
):
    """
    Each call randomly samples n_images digit-1 images from MNIST.
    No disk cache and no fixed seed (different sample every call).
    Returns a 5D tensor with shape [1, n_images, 1, img_size, img_size].
    """
    from torchvision import transforms, datasets
    import torch

    tfm = transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    targets = dataset.targets  # (60000,)

    # Only indices where digit == 1
    valid_indices = (targets == 1).nonzero(as_tuple=False).view(-1)

    # New random shuffle for each call (no manual seed here)
    perm = torch.randperm(len(valid_indices))
    chosen = valid_indices[perm[:n_images]]

    images = []
    for idx in chosen.tolist():
        img, _ = dataset[int(idx)]
        images.append(img)

    images_tensor = torch.stack(images, dim=0)      # [N, 1, H, W]
    images_5d = images_tensor.unsqueeze(0).to(device)  # [1, N, 1, H, W]
    return images_5d


def get_mnist_1024_images(
    img_size=24,
    seed=1234,
    root="./data",
    n_images=1024,
    digit = None,      # New: only take this digit; for you right now it's 1
):
    from torchvision import transforms, datasets
    import torch
    from pathlib import Path

    # Build a cache directory and filename based on n_images and digit
    digit_tag = f"digit{digit}" if digit is not None else "all"
    save_dir = Path(root) / f"mnist_{digit_tag}_{n_images}"
    save_dir.mkdir(parents=True, exist_ok=True)
    data_file = save_dir / f"mnist_{digit_tag}_{n_images}_images.pt"

    # If already cached, load directly
    if data_file.exists():
        print(f"Loading cached MNIST {digit_tag} {n_images} images...")
        images = torch.load(data_file)
        return images

    tfm = transforms.Compose([
        transforms.CenterCrop(img_size),  # use img_size for center crop
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = datasets.MNIST(root=root, train=True, download=True, transform=tfm)

    # First select all indices with label == digit
    targets = dataset.targets  # (60000,)
    if digit is None:
        valid_indices = torch.arange(len(dataset))
    else:
        valid_indices = (targets == digit).nonzero(as_tuple=False).view(-1)

    # Shuffle these indices with a fixed random seed
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(valid_indices), generator=g)
    fixed_indices = valid_indices[perm[:n_images]]  # use n_images here

    images = []
    for idx in fixed_indices.tolist():
        img, _ = dataset[idx]
        images.append(img)

    # Stack into tensor [n_images, 1, img_size, img_size]
    images_tensor = torch.stack(images, dim=0)

    # Convert to 5D format [1, n_images, 1, img_size, img_size]
    images_5d = images_tensor.unsqueeze(0)

    torch.save(images_5d, data_file)
    return images_5d

def sample_X1_standard_gaussian(B: int, N: int, D: int,
                            seed: int = 1234,
                            std: float = 1.0,
                            device: torch.device | str = "cpu",
                            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return X1 ∈ ℝ^{B×N×D} sampled from 𝒩(0, std^2 I).
    - std = 1.0  → standard Gaussian 𝒩(0, I)
    - std = (0.1)**0.5 → 𝒩(0, 0.1 I)
    Use a fixed seed to be fully reproducible.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, N, D, generator=g, dtype=dtype)  # sample on CPU first
    return (std * x).to(device)

def sample_X1_gaussian_5d_like(
    X_0_5d: torch.Tensor,
    seed: int = 1234,
    std: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Given a batch of images X_0_5d (B,N,C,H,W),
    generate Gaussian white noise X_1_5d with the same shape,
    with entries ~ N(0, std^2).
    """
    B, N, C, H, W = X_0_5d.shape
    D = C * H * W                      # flattened dimension

    # First sample in ℝ^{D} using the original 3D version
    X1_flat = sample_X1_standard_gaussian(
        B=B, N=N, D=D,
        seed=seed, std=std,
        device=device, dtype=dtype
    )                                  # shape (B,N,D)

    # Then reshape back to image shape
    X1_5d = X1_flat.view(B, N, C, H, W)
    return X1_5d


# =================================================================================== #
#                                       Network                                       #
# =================================================================================== #
images_5d = get_mnist_1024_images(img_size=24, seed=1234, root='./data',n_images=args.MNIST_n_images)  # [1,B,1,32,32]
B, C, H, W = images_5d.shape[1], images_5d.shape[2], images_5d.shape[3], images_5d.shape[4]
d = C * H * W
args.img_C = C
args.img_h = H
args.img_w = W
has_t = getattr(args, "concat_t", False)
args.cond_ch = args.img_C + (1 if has_t else 0)

print("Computing global PCA for consistent visualization...")
X_0_flat =_flatten_points_5d(images_5d.to(device))
X_1_flat = sample_X1_gaussian_5d_like(
    X_0_5d=images_5d.to(device),
    seed=1234,
    std=1,
    device=device,
)
X_1_flat = _flatten_points_5d(X_1_flat)

# Compute global PCA
args.global_pca_mu, args.global_pca_V2 = _pca_fit_2d_concat(X_0_flat, X_1_flat)
print(f"Global PCA computed - mu shape: {args.global_pca_mu.shape}, V2 shape: {args.global_pca_V2.shape}")

if args.savedModelName != '':
    print("Loading saved Model")
    saved_model_path = os.path.join(log_dir+'_saved', 'model',args.savedModelName)
    model = torch.load(saved_model_path,weights_only=False)
    net = model.to(device)
    args.WarmUpEpoch = 0
else:
    print("creating model from scratch")
    model = eval(args.model)
    net = model(d, args).to(device)
if hasattr(net, "args"):
    net.args.substeps = args.substeps
    print(f"[DEBUG] override net.args.substeps -> {net.args.substeps}")

net.train()
net_modules = [module for k, module in net._modules.items()]
txt_logger.write('There are {} trainable parameters in the network.'.format(get_num_parameters(net)))
print("activation function:  ", args.nn_act)


# =================================================================================== #
#                                       Optimizer                                     #
# =================================================================================== #

decay, no_decay = [], []
for n, p in net.named_parameters():
    if not p.requires_grad:
        continue
    if any(nd in n for nd in ["bias", "LayerNorm.weight", "embeddings"]):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW(
    [
      {"params": decay,    "weight_decay": 0.05},
      {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=args.lr,
)
#optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# scheduler
if args.lr_scheduler == 'cyclic':
    scheduler = CosineAnnealingLR(optimizer, args.N_iter_final, 1e-5, last_epoch=-1)
elif args.lr_scheduler == 'adaptive':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
else:
    print("No schduler is applied")
    scheduler = None


# =================================================================================== #
#                                       Training                                      #
# =================================================================================== #


# =================================================================================== #
#                                Final Test on New Distribution                       #
# =================================================================================== #

print("Start Final Test ")

X_0  = get_mnist_1024_images(img_size=24, seed=1234, root='./data',n_images=args.MNIST_n_images,digit = 1).to(device)

X_1_target  =  sample_X1_gaussian_5d_like(
    X_0_5d=X_0,
    seed=1234,
    std=(0.1) ** 0.5,
    device=device,
)
args.TrainingData = [X_0,X_1_target]

model_name = type(net).__name__
print("Loaded model class:", model_name)

X_k  =X_1_target.clone()

path = [X_k.clone()]
with torch.no_grad():
    if model_name == "JKO_op_net_time":

        for _ in range(args.jko_T_test):
            code = net.encode(X_k)
            dX = net.reverse(code, X_k)
            X_k = (X_k+ dX).detach()
            path.append(X_k.clone())
    else:
        for _ in range(args.jko_T_test):
            code = net.encode(X_k)
            V    = net.decode(code, X_k)
            X_k  = (X_k - V).detach()  # reverse
            path.append(X_k.clone())

# Turn list[(B,n,d)] into (T+1, n, d) to match the plotting signature used in training
path_tensor = torch.stack(path, dim=0)        # (T+1, B, n, d)
if path_tensor.dim() == 4:
    # This case is typically (T+1, B, N, D); keep as is, _flatten_path_5d will handle it
    pass

elif path_tensor.dim() == 6:
    # MNIST image trajectories: (T+1, B, N, C, H, W)
    T1, B, N, C, H, W = path_tensor.shape
    # Merge batch and N, but keep C,H,W → (T+1, B*N, C, H, W)
    path_tensor = path_tensor.view(T1, B * N, C, H, W)
path_density_dummy = torch.ones(path_tensor.shape[0], path_tensor.shape[1], device=device)


# ============================================================
# Only save the first max_points points from standard Gaussian
# to digit, at a few key time steps, and separate folders by substeps.
# ============================================================

if path_tensor.dim() == 5:  # (T+1, BN, C, H, W)
    T1, BN, C, H, W = path_tensor.shape

    # ===== Only save the first max_points points =====
    max_points = 100                     # originally set for 100 points
    num_points = min(BN, max_points)     # will not exceed max_points

    # ===== Different substeps go to different folders =====
    # args.substeps is defined in arguments.py
    substeps_tag = getattr(args, "substeps", None)

    if substeps_tag is None:
        # If --substeps is not passed, use a generic folder name
        base_dir = os.path.join(plot_dir, f"reverse_{max_points}points")
    else:
        # Attach substeps to distinguish different tests
        base_dir = os.path.join(
            plot_dir,
            f"reverse_{max_points}points_substeps{substeps_tag}"
        )

    os.makedirs(base_dir, exist_ok=True)

import numpy as np

for idx in range(num_points):
    # Each point has its own folder: sample_000, sample_001, ...
    sample_dir = os.path.join(base_dir, f"sample_{idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    # ===== Save 10 frames that are evenly spaced in time =====
    desired_frames = 10

    if T1 <= desired_frames:
        # If the total number of time steps is less than or equal to 10,
        # just save every time step.
        selected_ts = list(range(T1))
    else:
        # Otherwise, sample 10 time indices evenly from [0, T1 - 1].
        # np.linspace returns float values, so we cast them to int indices.
        selected_ts = np.linspace(0, T1 - 1, desired_frames, dtype=int)

    # Remove possible duplicate indices and sort them
    selected_ts = sorted(set(int(t) for t in selected_ts))

    for k, t in enumerate(selected_ts):
        t = int(t)
        # path_tensor shape: (T1, num_points, C, H, W)
        # We take the trajectory of the current point at time t.
        img = path_tensor[t, idx:idx+1].cpu()  # (1, C, H, W)

        # Save frame as step_000.png, step_001.png, ...
        save_path = os.path.join(sample_dir, f"step_{k:03d}.png")
        save_image(img, save_path, normalize=True)

print(f"Saved reverse trajectories of {num_points} points to {base_dir}")


if args.plot_jko_gif:
    plot_kl_jko_gif(
        X_1_target[0].cpu().float(),      # initial (n,d)
        path_tensor.cpu().float(),        # trajectory (T+1,n,d)
        X_0[0].cpu().float(),             # target (n,d)
        save_dir=os.path.join(plot_dir, 'jko_test.gif'),
        d=d, net=net, args=args
    )

print("Final test completed and GIF saved.")
