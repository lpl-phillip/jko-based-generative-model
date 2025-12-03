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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import _flatten_points_5d, _pca_fit_2d_concat

# python main_kl_Gaussiancombo_withMonitor.py --dataset mnist --jko_T 10 --jko_T_test 10 --B_dist 1 --B_sample 1024 --B_dist_test 1 --B_sample_test 1024 --h 512 --dropout_p 0 --N_iter 10006 --max_stall_epochs 50 --deltat 0.5 --Deltax 0.01 --exp_name kl_h1024_8g_mnist --log_interval 200 --divChoice central --concat_densityvalue false --substeps 1 --lr 1e-4 --lr_scheduler cyclic --scheduler_last_iter 0 --plot_jko_gif True --model JKO_op_net_time
# =================================================================================== #
#                                        Meta                                         #
# =================================================================================== #
args = parse_arguments()

device = torch.device('cuda')
log_dir, plot_dir, model_dir, txt_logger, tbx = prepare_loggers(args)

loss_log_path = os.path.join(log_dir, 'loss_vector_log.txt')
with open(loss_log_path, 'w') as f:
    f.write("Iteration, LossVector\n") 

grad_log_path = os.path.join(log_dir, 'loss_log.txt')
with open(grad_log_path, 'w') as f:
    f.write("Iteration, LossVector\n")

torch.manual_seed(args.seed)
np.random.seed(args.seed) 
# torch.set_default_tensor_type('torch.FloatTensor')
# =================================================================================== #
#                                        Data                                         #
# =================================================================================== #

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

    # If cached already, load directly
    if data_file.exists():
        print(f"Loading cached MNIST {digit_tag} {n_images} images...")
        images = torch.load(data_file)
        return images

    tfm = transforms.Compose([
        transforms.CenterCrop(img_size),  # <= center-crop to img_size
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

def get_mnist_1_random_batch(
    img_size=24,
    root="./data",
    n_images=1024,
    device="cuda",
):
    """
    Each call randomly samples n_images MNIST digit-1 images
    (no disk caching, no fixed seed).
    Returns a 5D tensor of shape [1, n_images, 1, img_size, img_size].
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

    # Only keep indices for digit = 1
    valid_indices = (targets == 1).nonzero(as_tuple=False).view(-1)

    # New random shuffle each call (no manual seed here)
    perm = torch.randperm(len(valid_indices))
    chosen = valid_indices[perm[:n_images]]

    images = []
    for idx in chosen.tolist():
        img, _ = dataset[int(idx)]
        images.append(img)

    images_tensor = torch.stack(images, dim=0)          # [N, 1, H, W]
    images_5d = images_tensor.unsqueeze(0).to(device)   # [1, N, 1, H, W]
    return images_5d


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

    # First sample in R^D using the original 3D version
    X1_flat = sample_X1_standard_gaussian(
        B=B, N=N, D=D,
        seed=seed, std=std,
        device=device, dtype=dtype
    )                                  # shape (B,N,D)

    # Then reshape back to image shape
    X1_5d = X1_flat.view(B, N, C, H, W)
    return X1_5d

def sample_X1_standard_gaussian(B: int, N: int, D: int,
                                seed: int = 1234,
                                std: float = 1.0,
                                device: torch.device | str = "cpu",
                                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Return X1 ∈ ℝ^{B×N×D} sampled from 𝒩(0, std^2 I).
    - std = 1.0  → standard Gaussian 𝒩(0, I)
    - std = (0.1)**0.5 → 𝒩(0, 0.1 I)
    A fixed seed is used for full reproducibility.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, N, D, generator=g, dtype=dtype)  # sample on CPU first
    return (std * x).to(device)
# =================================================================================== #
#                                       Network                                       #
# =================================================================================== #
images_5d = get_mnist_1024_images(img_size=24, seed=1234, root='./data', n_images=args.MNIST_n_images)  # [1,B,1,32,32]
B, C, H, W = images_5d.shape[1], images_5d.shape[2], images_5d.shape[3], images_5d.shape[4]
d = C * H * W
args.img_C = C
args.img_h = H
args.img_w = W
has_t = getattr(args, "concat_t", False)
args.cond_ch = args.img_C + (1 if has_t else 0)
print("Computing global PCA for consistent visualization...")
X_0_flat = _flatten_points_5d(images_5d.to(device))
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
    saved_model_path = os.path.join(log_dir+'_saved', 'model', args.savedModelName)
    model = torch.load(saved_model_path, weights_only=False)
    net = model.to(device)
    args.WarmUpEpoch = 0
else:
    print("creating model from scratch")
    model = eval(args.model)
    net = model(d, args).to(device)
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
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

# scheduler
if args.lr_scheduler == 'cyclic':
    scheduler = CosineAnnealingLR(optimizer, args.N_iter_final, 1e-5, last_epoch=-1)
elif args.lr_scheduler == 'adaptive':
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
else:
    print("No scheduler is applied")
    scheduler = None


# =================================================================================== #
#                                       Training                                      #
# =================================================================================== #


loss_hist = {'loss':[], 'W2':[], 'E':[]}
loss_hist_all = {'loss':[], 'W2':[], 'E':[]}

# Fixed MNIST images (B=1, N=100, C=1, H=32, W=32)
fixed_X_0 = get_mnist_1024_images(img_size=24, seed=1234, root='./data', n_images=args.MNIST_n_images).to(device)

# Generate standard Gaussian noise with the same shape as fixed_X_0
fixed_X_1 = sample_X1_gaussian_5d_like(
    X_0_5d=fixed_X_0,
    seed=1234,
    std=1.0,
    device=device,
)

# Training data: keep consistent with test.py
args.TrainingData = [fixed_X_0, fixed_X_1]


def GenerateNewPathWithFixedInitial_MNIST(net, args):
    """
    Generate an evolution path for MNIST images,
    keeping the same structure and interface as the original code.
    """
    # Get fixed initial and target MNIST data
    X_0 = fixed_X_0.to(device)

    X_1 = fixed_X_1.to(device)

    # X_0 = get_mnist_1_random_batch(
    #     img_size=args.img_h,              # 24
    #     root="./data",
    #     n_images=args.MNIST_n_images,
    #     device=device,
    # )                                    # [1, N, 1, H, W]

    # # 2. Generate the corresponding Gaussian noise as the "target"
    # # If you later change the sigma in the KL term to sqrt(0.1),
    # # here std should also be updated accordingly.
    # X_1 = sample_X1_gaussian_5d_like(
    #     X_0_5d=X_0,
    #     seed=np.random.randint(0, 10**9),    # different seed each time
    #     std=1.0,                             # or std=(0.1)**0.5 depending on your KL setup
    #     device=device,
    # )

    # Generate the path
    path = []
    X_k = X_0[0:1]  # take the first batch, shape [1, N, C, H, W]

    path.append(X_k.clone())

    net.eval()
    with torch.no_grad():
        for i in range(args.jko_T):

            code = net.encode(X_k)

            V_k = net.decode(code, X_k)

            # div_V = compute_divergence(X_k, V_k, code, net, args)

            X_k = (X_k + V_k).detach()

            path.append(X_k.detach().clone())

    path = torch.cat(path, dim=0).detach().to(device)  # shape [1+args.jko_T, N, C, H, W]
    args.TrainingData = [path, X_1]

    # Note: we no longer repeat X_1 over time,
    # because it already has a time dimension
    path_X_1 = X_1  # shape [1+args.jko_T, N, C, H, W]

    args.ReferenceLoss = None


for i in tqdm(range(args.N_iter)):
    # save model
    if (i+1) % args.save_interval == 0:
        torch.save(net, os.path.join(model_dir, '{}.pth'.format(i)))

    loss, loss_W2, loss_E, X_1_pred = compute_JKO_KL_Gaussianmix_NoDataGenerating(args.TrainingData, net,  args)

    writeLossToFile(loss_W2 + 2 * args.deltat * loss_E,  i, loss_log_path, args)

    with torch.no_grad():
        if should_generate_new_data(i, (loss_W2 / (2 * args.deltat) + loss_E).detach(), args):

            plot_kl_jko_gif(args.TrainingData[0][0].cpu().float(), X_1_pred.cpu(),
                            args.TrainingData[1][0].cpu().float(),
                            save_dir=os.path.join(plot_dir, 'jko_i={}.gif'.format(i)), d=d, net=net, args=args)

            with open(loss_log_path, 'a') as f:
                f.write("New data generated..\n")

            # Explicit cleanup before generating new data
            del loss, loss_W2, loss_E, X_1_pred
            # torch.cuda.empty_cache()
            args.current_stall_count = 0

            if args.dataset in ['mnist', 'MNIST_SAMPLE']:
                GenerateNewPathWithFixedInitial_MNIST(net, args)

            args.ChangePoint.append(len(loss_hist_all['loss']))
            print("NEW DATA GENERATED With fixed initial")
            continue

    args.loss_E  = loss_E[::args.B_dist]
    args.loss_W2 = loss_W2[::args.B_dist]

    # optimize
    optimizer.zero_grad()
    loss.backward()

    g_norm, w_norm, r_val = check_gradUpdate(net, loss, i)
    with open(grad_log_path, 'a') as f:
        f.write(f"# Iter {i}: {g_norm:3f}   {w_norm:3f}   {r_val:3f}\n")

    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

    g_norm, w_norm, r_val = check_gradUpdate(net, loss, i)
    with open(grad_log_path, 'a') as f:
        f.write(f"# Iter {i}: {g_norm:3f}   {w_norm:3f}   {r_val:3f}\n\n")

    optimizer.step()
    args.current_stall_count += 1
    if args.lr_scheduler == 'cyclic' and i < args.N_iter_final:
        scheduler.step()  # LR decays smoothly to eta_min by i=2000
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current LR: {current_lr:.6f}")

    # log
    loss_hist['loss'].append(loss.detach().cpu().flatten())
    loss_hist['E'].append(loss_E.detach().cpu().flatten())
    loss_hist['W2'].append(loss_W2.detach().cpu().flatten())

    loss_hist_all['loss'].append(loss.detach().cpu().flatten())
    loss_hist_all['E'].append(loss_E.flatten())
    loss_hist_all['W2'].append(loss_W2.flatten())

    tbx.add_scalar(tag='loss', scalar_value=float(loss), global_step=i)
    tbx.add_scalar(tag='E', scalar_value=float(loss_E.mean()), global_step=i)
    tbx.add_scalar(tag='W2', scalar_value=float(loss_W2.mean()), global_step=i)

    if i % args.log_interval == 0:
        with torch.no_grad():
            # plotting
            X_0 = args.TrainingData[0][0].float() #  n x d; samples from P_0
            X_1 = args.TrainingData[1][0].float() #  n x d; samples from P_1

            loss_mean = log_loss_hist(loss_hist, txt_logger, iter=i)
            if args.lr_scheduler == 'adaptive':
                scheduler.step(loss_mean)

            # plot the first step and the whole trajectory
            plot_gaussian(X_0.cpu(), X_1_pred[0].cpu(), X_1.cpu(), obs=None, net=net, args=args, save_dir=os.path.join(plot_dir, 'fig_i={}'.format(i)))
            if args.plot_jko_gif:
                plot_kl_jko_gif(X_0.cpu(), X_1_pred.cpu(), X_1.cpu(),   save_dir=os.path.join(plot_dir, 'jko_i={}.gif'.format(i)), d=d, net=net, args=args)
        loss_hist = reset_loss_hist(loss_hist)

print(f"Attempting to save plot in: {log_dir}")
print(f"Directory exists: {os.path.exists(log_dir)}")
print(f"Directory writable: {os.access(log_dir, os.W_OK)}")
if loss_hist_all is not None and len(loss_hist_all) > 0:
    plot_loss(loss_hist_all, args, log_dir)
else:
    print("[plot_loss] skip: empty loss history (test-only run)")

print('Training is Done!')

# # =================================================================================== #
# #                                Final Test on New Distribution                       #
# # =================================================================================== #

# print("Start Final Test ")

# X_0         = args.TrainingData[0].to(device)  # mix (initial in training)
# X_1_target  = args.TrainingData[1].to(device)  # standard (target in training)

# model_name = type(net).__name__
# print("Loaded model class:", model_name)

# X_k  = X_1_target.clone()

# path = [X_k.clone()]
# with torch.no_grad():
#     if model_name == "JKO_op_net_time":

#         for _ in range(args.jko_T_test):
#             code = net.encode(X_k)
#             dX = net.reverse(code, X_k)
#             X_k = (X_k+dX).detach()
#             path.append(X_k.clone())
#     else:
#         for _ in range(args.jko_T_test):
#             code = net.encode(X_k)
#             V    = net.decode(code, X_k)
#             X_k  = (X_k - args.deltat * V).detach()  # reverse
#             path.append(X_k.clone())

# # — Turn list[(B,n,d)] into (T+1, n, d) to match the plotting signature used in training —
# path_tensor = torch.stack(path, dim=0)        # (T+1, B, n, d)
# if path_tensor.dim() == 4:
#     path_tensor = path_tensor[:, 0, :, :]     # take batch 0 → (T+1, n, d)

# if args.plot_jko_gif:
#     plot_kl_jko_gif(
#         X_1_target[0].cpu().float(),      # initial (n,d)
#         path_tensor.cpu().float(),        # trajectory (T+1,n,d)
#         X_0[0].cpu().float(),
#         save_dir=os.path.join(plot_dir, 'jko_test.gif'),
#         d=d, net=net, args=args
#     )

# print("Final test completed and GIF saved.")
