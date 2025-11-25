import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda')



class JKO_op_net(nn.Module):
    def __init__(self, d, args):
        super(JKO_op_net, self).__init__()
        self.args = args
        if args.nn_act == 'relu':
            self.act = nn.ReLU()
        elif args.nn_act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif args.nn_act == 'gelu':
            self.act = nn.GELU()
        elif args.nn_act == 'mish':
            self.act = nn.Mish()

        self.mlp_first_P0 = nn.Sequential(
            nn.Conv1d(  d+2 if args.dataset == 'aggregation' else d + int(args.concat_densityvalue), 
                        args.h, 1), 
            nn.Dropout(p=args.dropout_p),
            self.act,
            nn.Conv1d(args.h, args.h, 1),
            self.act,
        )
        self.mlp_first_P1 = nn.Sequential(
            nn.Conv1d(d + int(args.concat_densityvalue) , args.h, 1),
            nn.Dropout(p=args.dropout_p),
            self.act,
            nn.Conv1d(args.h, args.h, 1),
            self.act,
        )
        self.mlp_first_Pt = nn.Sequential(
                nn.Conv1d(d, args.h, 1),
                self.act,
                nn.Conv1d(args.h, args.h, 1),
                self.act,
            )
        '''    
        if args.concat_densityvalue: 
            print("density concatenated")
            
        else: 
            print("SAME MLP FOR DECODER but no drop out")
            # Get all layers except dropout
            layers = []
            for layer in self.mlp_first_P0:
                if not isinstance(layer, nn.Dropout):
                    layers.append(layer)
            self.mlp_first_Pt = nn.Sequential(*layers)
            #self.mlp_first_Pt = self.mlp_first_P0
        '''

        self.MHT_list = nn.ModuleList([ MHT_Block(args)  for i in range(self.args.n_MHT) ])
        self.decoder = MHT_Block(args) 

        self.mlp_last = nn.Sequential(
                nn.Conv1d(args.h + d, args.h, 1),
                self.act,
                nn.Conv1d(args.h, d, 1)
            )


    #X has shape: (B, n,d )
    def forward(self, X_0, X_0_densityvalue=None, X_1=None, X_1_densityvalue=None):    

        code = self.encode(X_0,X_0_densityvalue, X_1, X_1_densityvalue)
        V_t = self.decode(code,X_0)

        return V_t

    def encode(self, X_0,X_0_densityvalue=None, X_1=None, X_1_densityvalue=None):    

        if self.args.concat_densityvalue:
            if self.args.dataset == 'aggregation':
                X_0 = torch.cat( (X_0, X_0_densityvalue), dim=-1  )
            else:
                assert X_0_densityvalue.ndim == 2 and X_0.shape[:2] == X_0_densityvalue.shape, "wrong density shape"
                X_0 = torch.cat( (X_0, X_0_densityvalue.unsqueeze(-1)), dim=-1  )
            if X_1 is not None:
                X_1 = torch.cat( (X_1, X_1_densityvalue.unsqueeze(-1)), dim=-1  )

        #reshape to (batch_size, d, n) and projection (batch_size, h, n)
        X_0_encode = self.mlp_first_P0(X_0.permute(0, 2, 1) )
        if X_1 is not None:
            X_1_encode = self.mlp_first_P1(X_1.permute(0, 2, 1) )
            X = torch.cat([X_0_encode, X_1_encode], dim=-1).permute(2, 0, 1) #(n,b,h)
        else: 
            X = X_0_encode.permute(2, 0, 1)

        for i, MHT in enumerate(self.MHT_list):
            if self.args.MHT_res_link:
                X = MHT(X, X, X) + X
            else:
                X = MHT(X, X, X)

        return X

    def decode(self, code, X_t): #X_t has shape: (B, n,d ) return V_t (B,n,d)


        #reshape to (batch_size, d, n) and projection (batch_size, h, n)        
        X_t_encode = self.mlp_first_Pt( X_t.permute(0, 2, 1)).permute(2, 0, 1) #(n,b,h)

        #decode X_t
        if self.args.MHT_res_link:
                Y = self.decoder(X_t_encode, code, code) + X_t_encode #(n,b,h)
        else:
                Y = self.decoder(X_t_encode, code, code) #(n,b,h)

        Y = torch.cat([Y, X_t.permute(1, 0, 2)], dim=-1) #(n,b,h+d)
            
        V_t = self.mlp_last(Y.permute(1, 2, 0)).permute(0, 2, 1)  #(b,h+d,n)  --> (b,d,n) --> (b,n,d)
        
        return V_t
        #X_t_pred = torch.stack([net(X_0, X_1, t_i) for t_i in t_vec], dim=2) # B x n x K x d, #Shape of t_i: (B, 1), and have the same value




#MHT_Block contains Self_Attention + normalizatio + MLP
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

        self.MHT_1 = nn.MultiheadAttention(args.h, self.args.n_MHT_heads)
        self.norm_1 = nn.LayerNorm(args.h)
        self.MLP_1 = nn.Sequential(
            nn.Conv1d(args.h, args.h, 1),
            #nn.Dropout(args.dropout_p),
            self.act,
            nn.Conv1d(args.h, args.h, 1),
            self.act,
        )
        self.norm_2 = nn.LayerNorm(args.h)

    #self attention pass Q=K=V=X, cross attention pass Q as query, K V as the data to learn from
    def forward(self, Q, K, V):
        
        out = self.MHT_1(Q, K, V, need_weights=False)[0]
        out = self.norm_1(out.permute(1, 0, 2)).permute(0, 2, 1)
        if self.args.MHT_res_link:
            out = self.MLP_1(out) + out
        else:
            out = self.MLP_1(out)
        out = self.norm_2(out.permute(0, 2, 1))

        return out.permute(1, 0, 2)

class JKO_op_net_time(nn.Module):
    def __init__(self, d, args):
        super(JKO_op_net_time, self).__init__()
        self.args = args
        if args.nn_act == 'relu':
            self.act = nn.ReLU()
        elif args.nn_act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif args.nn_act == 'gelu':
            self.act = nn.GELU()
        elif args.nn_act == 'mish':
            self.act = nn.Mish()

        self.mlp_first_P0 = nn.Sequential(
            nn.Conv1d(d , args.h, 1),  #torch.nn.Conv1d(in_channels, out_channels, kernel_size) with input tensor of shape (batch_size, in_channels, sequence_length)
            nn.Dropout(p=args.dropout_p),
            self.act,
            nn.Conv1d(args.h, args.h, 1),
            self.act,
        )
        self.mlp_first_P1 = nn.Sequential(
            nn.Conv1d(d, args.h, 1),
            nn.Dropout(p=args.dropout_p),
            self.act,
            nn.Conv1d(args.h, args.h, 1),
            self.act,
        )
        self.mlp_first_Pt = nn.Sequential(
            nn.Conv1d(d + 1, args.h, 1),
            nn.Dropout(p=args.dropout_p),
            self.act,
            nn.Conv1d(args.h, args.h, 1),
            self.act,
        )

        self.MHT_list = nn.ModuleList([ MHT_Block(args)  for i in range(self.args.n_MHT) ])
        self.decoder = MHT_Block(args) 

        if self.args.concat_t:
            self.mlp_last = nn.Sequential(
                nn.Conv1d(args.h + d + 1, args.h, 1),
                nn.Dropout(p=args.dropout_p),
                self.act,
                nn.Conv1d(args.h, d, 1)
            )
        else:
            self.mlp_last = nn.Sequential(
                nn.Conv1d(args.h + d, args.h, 1),
                nn.Dropout(p=args.dropout_p),
                self.act,
                nn.Conv1d(args.h, d, 1)
            )


    #X has shape: (B, n,d ), t has shape: (B, 1)
    def forward(self, X_0, X_1, X_t=None, t=None):

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
         #reshape to (batch_size, d, n) and projection (batch_size, h, n)
        X_0_encode = self.mlp_first_P0(X_0.permute(0, 2, 1) )

        X = X_0_encode.permute(2, 0, 1)

        for i, MHT in enumerate(self.MHT_list):
            if self.args.MHT_res_link:
                X = MHT(X, X, X) + X
            else:
                X = MHT(X, X, X)

        return X



    # def decode(self, code, X_t, t=None):
    #     substeps = getattr(self.args, "substeps", 1)
    #     if t is None:
    #         t = torch.ones((X_t.shape[0], 1, X_t.shape[1])).to(device)
    #     if t.dim() == 2:
    #         t = t[:, None].repeat(1, 1, X_t.shape[1])  # B x 1 x n
    #
    #     X_current = X_t.clone()
    #     dt = 1.0 / substeps
    #
    #     for s in range(substeps):
    #         t_block = t - s * dt
    #         X_cur_encode = self.mlp_first_Pt(
    #             torch.cat([X_current.permute(0, 2, 1), t_block], dim=1)
    #         ).permute(2, 0, 1)
    #
    #         if self.args.MHT_res_link:
    #             Y = self.decoder(X_cur_encode, code, code) + X_cur_encode
    #         else:
    #             Y = self.decoder(X_cur_encode, code, code)
    #
    #         if self.args.concat_t:
    #             Y = torch.cat([Y, X_current.permute(1, 0, 2), t_block.permute(2, 0, 1)], dim=-1)
    #         else:
    #             Y = torch.cat([Y, X_current.permute(1, 0, 2)], dim=-1)
    #
    #         V_inc = self.mlp_last(Y.permute(1, 2, 0)).permute(0, 2, 1)  # (b,n,d)
    #         X_current = X_current + V_inc / substeps
    #
    #     V_total = X_current - X_t
    #     return V_total
    def decode(self, code, X_t, t=None):
        substeps = getattr(self.args, "substeps", 1)
        if t is None:
            t = torch.zeros((X_t.shape[0], 1, X_t.shape[1])).to(device)
        if t.dim() == 2:
            t = t[:, None].repeat(1, 1, X_t.shape[1])  # B x 1 x n

        X_current = X_t
        dt = 1.0 / substeps

        def _vel(Xc, tb):
            Xc_enc = self.mlp_first_Pt(
                torch.cat([Xc.permute(0, 2, 1), tb], dim=1)
            ).permute(2, 0, 1)  # (n,B,h)

            if self.args.MHT_res_link:
                Y = self.decoder(Xc_enc, code, code) + Xc_enc
            else:
                Y = self.decoder(Xc_enc, code, code)

            if self.args.concat_t:
                Y = torch.cat([Y, Xc.permute(1, 0, 2), tb.permute(2, 0, 1)], dim=-1)
            else:
                Y = torch.cat([Y, Xc.permute(1, 0, 2)], dim=-1)

            V = self.mlp_last(Y.permute(1, 2, 0)).permute(0, 2, 1)  # (B,n,d)
            return V

        for s in range(substeps):
            t0 = t + s * dt

            # k1 = f(t0, X)
            k1 = _vel(X_current, t0)

            k2 = _vel(X_current + 0.5 * dt * k1, t0 + 0.5 * dt)

            k3 = _vel(X_current + 0.5 * dt * k2, t0 + 0.5 * dt)

            k4 = _vel(X_current + dt * k3, t0 + dt)

            X_current = X_current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        V_total = X_current - X_t
        return V_total

    def reverse(self, code, X_t, t=None):
        substeps = getattr(self.args, "substeps", 1)
        if t is None:
            t = torch.ones((X_t.shape[0], 1, X_t.shape[1])).to(device)
        if t.dim() == 2:
            t = t[:, None].repeat(1, 1, X_t.shape[1])  # B x 1 x n

        X_current = X_t
        dt = 1.0 / substeps

        def _vel(Xc, tb):
            Xc_enc = self.mlp_first_Pt(
                torch.cat([Xc.permute(0, 2, 1), tb], dim=1)
            ).permute(2, 0, 1)  # (n,B,h)

            if self.args.MHT_res_link:
                Y = self.decoder(Xc_enc, code, code) + Xc_enc
            else:
                Y = self.decoder(Xc_enc, code, code)

            if self.args.concat_t:
                Y = torch.cat([Y, Xc.permute(1, 0, 2), tb.permute(2, 0, 1)], dim=-1)
            else:
                Y = torch.cat([Y, Xc.permute(1, 0, 2)], dim=-1)

            V = self.mlp_last(Y.permute(1, 2, 0)).permute(0, 2, 1)  # (B,n,d)
            return V

        for s in range(substeps):
            t0 = t - s * dt

            # k1 = f(t0, X)
            k1 = _vel(X_current, t0)

            k2 = _vel(X_current + 0.5 * dt * k1, t0 - 0.5 * dt)

            k3 = _vel(X_current + 0.5 * dt * k2, t0 - 0.5 * dt)

            k4 = _vel(X_current + dt * k3, t0 - dt)

            X_current = X_current - (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        V_total = X_current - X_t
        return V_total