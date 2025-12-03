import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda')


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2D_NoNorm(nn.Module):
    """
    Simple 2D residual block without normalization:
        out = act( conv2(act(conv1(x))) + skip(x) )
    """
    def __init__(self, in_ch, out_ch, act='relu'):
        super().__init__()
        if act == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif act == 'gelu':
            act_fn = nn.GELU()
        elif act == 'mish':
            act_fn = nn.Mish()
        else:
            act_fn = nn.ReLU(inplace=True)

        self.act = act_fn
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # skip connection to match channels if needed
        if in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


class _CNNDecoder2D_ResNoNorm(nn.Module):
    """
    Flow-friendly decoder:
      - no BatchNorm / GroupNorm
      - ResBlocks + nearest-neighbor upsampling
      - same interface as your original _CNNDecoder2D

    z:    (B,h) or (B,N,h)
    cond: None or (B,C,H,W) or (B,N,C,H,W)
    returns:
      (B,C,H,W) or (B,N,C,H,W)
    """
    def __init__(self,
                 out_channels=1,
                 base_channels=64,
                 latent_dim=256,
                 img_size=(64, 64),
                 act='relu',
                 cond_channels=0):
        super().__init__()
        H, W = img_size
        self.H, self.W, self.out_channels = H, W, out_channels
        ch = base_channels
        proj_ch = ch * 8

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'mish':
            self.act = nn.Mish()
        else:
            self.act = nn.ReLU(inplace=True)

        # low-res spatial size
        self.fh, self.fw = H // 8, W // 8

        # latent -> low-res feature map
        self.fc = nn.Linear(latent_dim, proj_ch * self.fh * self.fw)

        # upsampling path with ResBlocks, no norms
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock2D_NoNorm(proj_ch, ch * 4, act=act),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock2D_NoNorm(ch * 4, ch * 2, act=act),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResBlock2D_NoNorm(ch * 2, ch, act=act),
        )

        # tail: combine with cond and predict velocity / image
        in_ch_tail = ch + cond_channels
        self.tail = nn.Sequential(
            ResBlock2D_NoNorm(in_ch_tail, ch, act=act),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
            # optional: bound velocity
            # nn.Tanh()
        )

    def _forward_4d(self, z2d, cond4d=None):
        """
        z2d:   (B, latent_dim)
        cond4d:(B, cond_channels, H, W) or None
        return:(B, out_channels, H, W)
        """
        B = z2d.size(0)

        x = self.fc(z2d)                           # (B, proj_ch*fh*fw)
        x = x.view(B, -1, self.fh, self.fw)        # (B, proj_ch, fh, fw)

        x = self.up1(x)                            # (B, 4ch, H/4, W/4)
        x = self.up2(x)                            # (B, 2ch, H/2, W/2)
        x = self.up3(x)                            # (B, ch,  H,   W)

        if cond4d is not None:
            # cond4d: (B, cond_channels, H, W)
            x = torch.cat([x, cond4d], dim=1)      # (B, ch+cond_ch, H, W)

        out = self.tail(x)                         # (B, out_channels, H, W)
        return out

    def forward(self, z, cond=None):
        """
        z: (B,h) or (B,N,h)
        cond:
          - None
          - (B,C,H,W)
          - (B,N,C,H,W)
        """
        if z.dim() == 2:
            # (B,h)
            if cond is None or cond.dim() == 4:
                cond4d = cond
            elif cond.dim() == 5:
                cond4d = cond.view(-1, *cond.shape[-3:])
            else:
                raise ValueError("cond must be None, (B,C,H,W) or (B,N,C,H,W)")
            return self._forward_4d(z, cond4d)

        elif z.dim() == 3:
            # (B,N,h)
            B, N, h = z.shape
            z2d = z.reshape(B * N, h)

            if cond is None:
                cond4d = None
            elif cond.dim() == 5:  # (B,N,C,H,W)
                cond4d = cond.view(B * N, *cond.shape[-3:])
            elif cond.dim() == 4:  # (B,C,H,W) -> broadcast per frame
                cond4d = cond.unsqueeze(1).expand(B, N, *cond.shape[1:]).reshape(
                    B * N, *cond.shape[1:]
                )
            else:
                raise ValueError("cond must be None, (B,C,H,W) or (B,N,C,H,W)")

            out4d = self._forward_4d(z2d, cond4d)  # (B*N, C, H, W)
            return out4d.view(B, N, self.out_channels, self.H, self.W)

        else:
            raise ValueError(f"Decoder expects z as (B,h) or (B,N,h), got {z.shape}")




class _CNNEncoder2D(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, latent_dim=256, img_size=(64,64), act='relu'):
        super().__init__()
        H, W = img_size
        # Now only downsample twice, overall scale factor 4x
        assert H % 4 == 0 and W % 4 == 0
        ch = base_channels
        act_fn = nn.ReLU(inplace=True) if act == 'relu' else nn.GELU()

        # Stem block unchanged
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, 1, 1), nn.BatchNorm2d(ch), act_fn,
            nn.Conv2d(ch, ch, 3, 1, 1), nn.BatchNorm2d(ch), act_fn
        )
        # First downsampling: H/2, W/2, channels 2ch
        self.down1 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.BatchNorm2d(ch * 2), act_fn,
            nn.Conv2d(ch * 2, ch * 2, 3, 1, 1), nn.BatchNorm2d(ch * 2), act_fn
        )
        # Second downsampling: H/4, W/4, channels 4ch
        self.down2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.BatchNorm2d(ch * 4), act_fn,
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.BatchNorm2d(ch * 4), act_fn
        )

        # No more down3/down4, directly do pooling + FC
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch * 4, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def _forward_4d(self, x):  # x: (B, C, H, W)
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        z = self.head(x)  # (B, latent_dim)
        return z

    def forward(self, x):
        if x.dim() == 4:  # (B, C, H, W)
            return self._forward_4d(x)  # -> (B, latent_dim)
        elif x.dim() == 5:  # (B, N, C, H, W)
            B, N, C, H, W = x.shape
            z = self._forward_4d(x.view(B * N, C, H, W))  # (B*N, latent_dim)
            return z.view(B, N, -1)  # (B, N, latent_dim)
        else:
            raise ValueError(f"Encoder expects 4D or 5D, got {x.shape}")


class _CNNDecoder2D(nn.Module):
    def __init__(self, out_channels=1, base_channels=64, latent_dim=256, img_size=(64,64), act='relu', cond_channels=0):
        super().__init__()
        H,W = img_size
        self.H, self.W, self.out_channels = H, W, out_channels
        ch = base_channels; proj_ch = ch*8
        act_fn = nn.ReLU(inplace=True) if act=='relu' else nn.GELU()
        self.fh, self.fw = H//8, W//8
        self.fc   = nn.Sequential(nn.Linear(latent_dim, proj_ch*self.fh*self.fw), act_fn)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(proj_ch, ch * 4, 3, 1, 1),
                                 nn.BatchNorm2d(ch * 4), act_fn)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ch * 4, ch * 2, 3, 1, 1),
                                 nn.BatchNorm2d(ch * 2), act_fn)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ch * 2, ch, 3, 1, 1),
                                 nn.BatchNorm2d(ch), act_fn)
        # self.up4  = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ch*2, ch, 3,1,1),   nn.BatchNorm2d(ch),   act_fn)
        self.tail = nn.Sequential(nn.Conv2d(ch + cond_channels, ch, 3,1,1), nn.BatchNorm2d(ch), act_fn,
                                  nn.Conv2d(ch, out_channels, 3,1,1))

    def _forward_4d(self, z2d, cond4d=None):  # z2d: (B,h)
        B = z2d.size(0)
        x = self.fc(z2d).view(B, -1, self.fh, self.fw)
        x = self.up1(x); x = self.up2(x); x = self.up3(x)
        if cond4d is not None:
            x = torch.cat([x, cond4d], dim=1)
        return self.tail(x)                     # (B,C,H,W)

    def forward(self, z, cond=None):
        if z.dim() == 2:                        # (B,h)
            cond4d = cond if (cond is None or cond.dim()==4) else cond.view(-1, *cond.shape[-3:])
            return self._forward_4d(z, cond4d)  # (B,C,H,W)
        elif z.dim() == 3:                      # (B,N,h)
            B,N,h = z.shape
            z2d = z.reshape(B*N, h)
            if cond is None:
                cond4d = None
            elif cond.dim() == 5:               # (B,N,C,H,W)
                cond4d = cond.view(B*N, *cond.shape[-3:])
            elif cond.dim() == 4:               # (B,C,H,W) -> broadcast to each frame
                cond4d = cond.unsqueeze(1).expand(B, N, *cond.shape[1:]).reshape(B*N, *cond.shape[1:])
            else:
                raise ValueError("cond must be (B,C,H,W) or (B,N,C,H,W)")
            out4d = self._forward_4d(z2d, cond4d)     # (B*N,C,H,W)
            return out4d.view(B, N, self.out_channels, self.H, self.W)
        else:
            raise ValueError(f"Decoder expects z as (B,h) or (B,N,h), got {z.shape}")


class MHT_Block(nn.Module):
    def __init__(self, args):
        super(MHT_Block, self).__init__()
        self.args = args
        if args.nn_act == 'relu':
            self.act = nn.ReLU()
        elif args.nn_act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif args.nn_act == 'gelu':
            self.act = nn.GELU()
        elif args.nn_act == 'mish':
            self.act = nn.Mish()

        # Note: embed_dim = args.h, heads = args.n_MHT_heads
        self.MHT_1 = nn.MultiheadAttention(args.h, self.args.n_MHT_heads)
        self.norm_1 = nn.LayerNorm(args.h)
        self.MLP_1 = nn.Sequential(
            nn.Conv1d(args.h, args.h, 1),
            self.act,
            nn.Conv1d(args.h, args.h, 1),
            self.act,
        )
        self.norm_2 = nn.LayerNorm(args.h)

    # Input/output shapes: Q/K/V are all (N, B, h)
    def forward(self, Q, K, V):
        out = self.MHT_1(Q, K, V, need_weights=False)[0]      # (N,B,h)
        out = self.norm_1(out.permute(1, 0, 2)).permute(0, 2, 1)  # -> (B,h,N)
        if self.args.MHT_res_link:
            out = self.MLP_1(out) + out
        else:
            out = self.MLP_1(out)
        out = self.norm_2(out.permute(0, 2, 1))               # -> (B,N,h)
        return out.permute(1, 0, 2)                           # -> (N,B,h)


class JKO_op_net_time(nn.Module):
    def __init__(self, d, args):
        super(JKO_op_net_time, self).__init__()
        self.args = args
        self.img_C = getattr(args, "img_C", 1)
        self.img_H = getattr(args, "img_h", 24)
        self.img_W = getattr(args, "img_w", 24)
        self.h = getattr(args, "h", 512)
        act_name = getattr(args, "act", "relu")
        self.cond_ch = getattr(args, "cond_ch", 0)
        self.MHT_list = nn.ModuleList([MHT_Block(args) for _ in range(self.args.n_MHT)])
        if args.nn_act == 'relu':
            self.act = nn.ReLU()
        elif args.nn_act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif args.nn_act == 'gelu':
            self.act = nn.GELU()
        elif args.nn_act == 'mish':
            self.act = nn.Mish()
        base_ch = 64

        self.mlp_first_P0 = _CNNEncoder2D(in_channels=self.img_C, base_channels=base_ch,
                                          latent_dim=self.h, img_size=(self.img_H, self.img_W), act=act_name)
        self.mlp_first_Pt = _CNNEncoder2D(in_channels=self.img_C, base_channels=base_ch,
                                          latent_dim=self.h, img_size=(self.img_H, self.img_W), act=act_name)
        #self.mlp_last = _CNNDecoder2D(out_channels=self.img_C, base_channels=base_ch,  latent_dim=self.h, img_size=(self.img_H, self.img_W),
                                     # act=act_name, cond_channels=self.cond_ch)
        self.mlp_last = _CNNDecoder2D_ResNoNorm(
                                    out_channels=self.img_C,
                                    base_channels=base_ch,
                                    latent_dim=self.h,
                                    img_size=(self.img_H, self.img_W),
                                    act=act_name,
                                    cond_channels=self.cond_ch,
                                )

        self.decoder = MHT_Block(args) 

    # X has shape: (B, n,d ), t has shape: (B, 1)
    def forward(self, X_0,  X_t=None, t=None):

        if X_t is None:
            X_t = X_0
        if t is None:
            t = torch.zeros( ( X_t.shape[0], 1,X_t.shape[1]    )  ).to(device)
        if t.dim() == 2: 
            t = t[:, None].repeat(1, 1, X_t.shape[1])  # B x 1 x n_t

        code = self.encode(X_0)
        V_t = self.decode(code,X_t, t)
        return V_t

    def encode(self, X_0):

        B, N, C, H, W = X_0.shape
        z_BNh = self.mlp_first_P0(X_0)
        X = z_BNh.permute(1, 0, 2).contiguous()

        for i, MHT in enumerate(self.MHT_list):
            X = MHT(X, X, X) + X if self.args.MHT_res_link else MHT(X, X, X)
        return X

    def decode(self, code, X_t, t=None):
        # X_t: (B,N,C,H,W), code:(N,B,h)
        substeps = getattr(self.args, "substeps", 1)
        B, N, C, H, W = X_t.shape
        if t is None:
            t = torch.zeros((B, N, 1), device=X_t.device)
        elif t.dim() == 2:  # (B,N) -> (B,N,1)
            t = t[:, :, None]

        X_current = X_t
        dt = 1.0 / substeps

        def _make_cond(Xc, tb):
            # Xc: (B,N,C,H,W); tb: (B,N,1) or (B,N,H,W)
            if getattr(self.args, "concat_t", True):
                if tb.dim() == 3 and tb.size(-1) == 1:
                    t_img = tb.reshape(B * N, 1, 1, 1).expand(B * N, 1, H, W)
                elif tb.dim() == 4 and tb.shape[-2:] == (H, W):
                    t_img = tb.reshape(B * N, 1, H, W) if tb.size(-3) == 1 else tb.reshape(B * N, tb.size(-3), H, W)
                else:
                    raise ValueError("t must be (B,N,1) or (B,N,H,W)")
                return torch.cat([Xc.reshape(B * N, C, H, W), t_img], dim=1)  # (B*N, C+1, H, W)
            else:
                return Xc.reshape(B * N, C, H, W)  # (B*N, C, H, W)

        def _vel(Xc, tb):

            Xc_enc_BNh = self.mlp_first_Pt(Xc)  # CNN encoder on 5D

            # 2) Into decoder attention: needs shape (N,B,h)
            Xc_enc_NBh = Xc_enc_BNh.permute(1, 0, 2).contiguous()  # (N,B,h)
            if self.args.MHT_res_link:
                Y = self.decoder(Xc_enc_NBh, code, code) + Xc_enc_NBh  # (N,B,h)
            else:
                Y = self.decoder(Xc_enc_NBh, code, code)

            # 3) CNN decode to pixel-wise increment
            Y_BNh = Y.permute(1, 0, 2).contiguous()  # (B,N,h)
            cond4d = _make_cond(Xc, tb)  # (B*N, C(+1), H, W)
            cond5d = cond4d.view(B, N, cond4d.size(1), H, W)  # (B,N,C(+1),H,W)
            delta = self.mlp_last(Y_BNh, cond=cond5d)  # (B,N,C,H,W)
            return delta

        # --------- Training: standard Euler substep: X <- X + dt * v(X,t) ---------
        for s in range(substeps):
            t0 = t + s * dt
            k1 = _vel(X_current, t0)  # here k1 is v(X_current, t0)
            X_current = X_current + dt * k1  # x = x + v * dt

        V_total = X_current - X_t
        return V_total  # (B,N,C,H,W)

    def reverse(self, code, X_t, t=None):
        # X_t: (B,N,C,H,W), code:(N,B,h)
        substeps = getattr(self.args, "substeps", 1)
        B, N, C, H, W = X_t.shape
        if t is None:
            t = torch.ones((B, N, 1), device=X_t.device)
        elif t.dim() == 2:  # (B,N) -> (B,N,1)
            t = t[:, :, None]

        X_current = X_t
        dt = 1.0 / substeps

        def _make_cond(Xc, tb):

            if getattr(self.args, "concat_t", True):
                if tb.dim() == 3 and tb.size(-1) == 1:
                    t_img = tb.reshape(B * N, 1, 1, 1).expand(B * N, 1, H, W)
                elif tb.dim() == 4 and tb.shape[-2:] == (H, W):
                    t_img = tb.reshape(B * N, 1, H, W) if tb.size(-3) == 1 else tb.reshape(B * N, tb.size(-3), H, W)
                else:
                    raise ValueError("t must be (B,N,1) or (B,N,H,W)")
                return torch.cat([Xc.reshape(B * N, C, H, W), t_img], dim=1)  # (B*N, C+1, H, W)
            else:
                return Xc.reshape(B * N, C, H, W)  # (B*N, C, H, W)

        def _vel(Xc, tb):

            Xc_enc_BNh = self.mlp_first_Pt(Xc)  # CNN encoder on 5D

            Xc_enc_NBh = Xc_enc_BNh.permute(1, 0, 2).contiguous()  # (N,B,h)
            if self.args.MHT_res_link:
                Y = self.decoder(Xc_enc_NBh, code, code) + Xc_enc_NBh  # (N,B,h)
            else:
                Y = self.decoder(Xc_enc_NBh, code, code)

            Y_BNh = Y.permute(1, 0, 2).contiguous()  # (B,N,h)
            cond4d = _make_cond(Xc, tb)  # (B*N, C(+1), H, W)
            cond5d = cond4d.view(B, N, cond4d.size(1), H, W)  # (B,N,C(+1),H,W)
            delta = self.mlp_last(Y_BNh, cond=cond5d)  # (B,N,C,H,W)
            return delta

        for s in range(substeps):
            t0 = t - s * dt
            k1 = _vel(X_current, t0)
            k2 = _vel(X_current + 0.5 * dt * k1, t0 - 0.5 * dt)
            k3 = _vel(X_current + 0.5 * dt * k2, t0 - 0.5 * dt)
            k4 = _vel(X_current + dt * k3, t0 - dt)
            X_current = X_current - (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        V_total = X_current - X_t
        return V_total  # (B,N,C,H,W)
