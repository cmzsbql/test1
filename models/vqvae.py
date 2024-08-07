import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset,LFQ,QuantizeEMAReset2
import math
import torch
import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        output_emb_width = self.code_dim
        if args.quantizer == "lfq":
            output_emb_width = math.ceil(math.log2(self.num_code))
            self.code_dim = output_emb_width
        elif args.dataname == 'stock':
            input_emb_width = 6
        elif args.dataname == 'energy':
            input_emb_width = 28
        elif args.dataname == 'sine':
            input_emb_width = 5
        elif args.dataname == 'etth':
            input_emb_width = 7
        elif args.dataname == 'fmri':
            input_emb_width = 50
        elif args.dataname == 'mujoco':
            input_emb_width = 14
        self.encoder = Encoder(input_emb_width, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, quantizer=args.quantizer)
        self.decoder = Decoder(input_emb_width, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        elif args.quantizer == "lfq":
            self.quantizer = LFQ(self.num_code, self.code_dim, args)
        elif args.quantizer == "ema_reset2":
            self.quantizer = QuantizeEMAReset2(nb_code, code_dim, args)


    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x

    def encode2(self, x):

        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        N, _, _ = x_encoder.shape
        x_encoder = self.postprocess(x_encoder)
        # x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        return x_encoder
        # x0 =x
        # self.center = torch.tensor(self.center).to(x.device).float()
        # x = x.reshape(-1,24,4)
        # distances = torch.cdist(x, self.center)
        # nearest_indices = torch.argmin(distances, dim=-1)
        # a = torch.min(distances, dim=-1).values
        # aa = self.forward_decoder(nearest_indices)
        # # distances2 = torch.cdist(aa.reshape(-1,24,4), self.center)
        # # nearest_indices2 = torch.argmin(distances, dim=-1)
        # return nearest_indices


    def encode(self, x):

        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        N, _, _ = x_encoder.shape
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx
        # x0 =x
        # self.center = torch.tensor(self.center).to(x.device).float()
        # x = x.reshape(-1,24,4)
        # distances = torch.cdist(x, self.center)
        # nearest_indices = torch.argmin(distances, dim=-1)
        # a = torch.min(distances, dim=-1).values
        # aa = self.forward_decoder(nearest_indices)
        # # distances2 = torch.cdist(aa.reshape(-1,24,4), self.center)
        # # nearest_indices2 = torch.argmin(distances, dim=-1)
        # return nearest_indices

    def forward(self, x):
        # feature_vim = x.shape[-1]a.values
        #
        # x_in = self.preprocess(x)
        # # Encode
        # if feature_vim == 251:
        #     x_encoder = self.encoder251(x_in)
        # else:
        #     x_encoder = self.encoder263(x_in)
        # ## quantization
        # x_quantized, loss, perplexity = self.quantizer(x_encoder)
        #
        # ## decoder
        # if feature_vim == 251:
        #     x_decoder = self.decoder251(x_quantized)
        # else:
        #     x_decoder = self.decoder263(x_quantized)
        # x_out = self.postprocess(x_decoder)
        # return x_out, loss, perplexity

        # a = self.encode(x)
        # aa = self.forward_decoder(a)
        # x = x.reshape(-1,96,1)

        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)###shared2 x.shape[-1]
        x_out = self.postprocess(x_decoder)

        # x_out = x_out.reshape(-1, 96, 321)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(x.shape[0], -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out

        # self.center = torch.tensor(self.center).to(x.device).float()
        # selected_vectors = self.center[x]
        # selected_vectors = selected_vectors.reshape(-1,96,321)
        # return selected_vectors

class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(args, nb_code, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def encode2(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode2(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss, perplexity = self.vqvae(x)
        
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        