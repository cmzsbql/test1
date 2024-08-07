import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding

class TSG_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.trans_base = TransBase(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.trans_head = TransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs):
        feat = self.trans_base(idxs)
        logits = self.trans_head(feat)
        return logits

    def sample(self, input_idx, if_categorial=False):
        B, L = input_idx.shape
        T = L
        for t in range(T):
            # 前向传播，预测词汇
            logits = self.forward(input_idx)  # 假设 forward 方法返回 logits (B, L, vocab_size)
            probs = torch.softmax(logits, dim=-1)


            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                sampled_probs = torch.gather(probs, -1, idx.unsqueeze(-1)).squeeze(-1)

            else:
                sampled_probs, idx = torch.topk(probs, k=1, dim=-1)
                sampled_probs = sampled_probs.squeeze(-1)
                idx = idx.squeeze(-1)

            non_mask = input_idx != self.num_vq
            sampled_probs = torch.where(non_mask, torch.ones_like(sampled_probs), sampled_probs)
            # new_mask_num = torch.ceil(torch.cos(torch.tensor((t+1)/T*torch.pi / 2))*L).int()
            new_mask_num = T - 1 - t
            if new_mask_num == L:
                new_mask_num = L-1
            if t<T-1:
                topk, _ = torch.topk(sampled_probs, new_mask_num, dim=-1, largest=False)
                thresholds = topk[:, -1]
                mask = sampled_probs <= thresholds.unsqueeze(-1)
                idx = torch.where(mask, torch.full_like(idx, self.num_vq), idx)

            input_idx = torch.where(non_mask, input_idx,idx)
        return input_idx

    def sample_cond(self, input_idx, if_categorial=False,non_mask=None):
        B, L = input_idx.shape
        T = L-torch.sum(non_mask[0])
        special_mask = (input_idx != self.num_vq)^non_mask
        for t in range(T):
            # 前向传播，预测词汇
            logits = self.forward(input_idx)  # 假设 forward 方法返回 logits (B, L, vocab_size)
            probs = torch.softmax(logits, dim=-1)


            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                sampled_probs = torch.gather(probs, -1, idx.unsqueeze(-1)).squeeze(-1)

            else:
                sampled_probs, idx = torch.topk(probs, k=1, dim=-1)
                sampled_probs = sampled_probs.squeeze(-1)
                idx = idx.squeeze(-1)

            # non_mask = input_idx != self.num_vq
            sampled_probs = torch.where(non_mask, torch.ones_like(sampled_probs), sampled_probs)
            # new_mask_num = torch.ceil(torch.cos(torch.tensor((t+1)/T*torch.pi / 2))*L).int()
            new_mask_num = T - 1 - t
            if new_mask_num == L:
                new_mask_num = L-1

            topk, _ = torch.topk(sampled_probs, new_mask_num+1, dim=-1, largest=False)
            thresholds = topk[:, -1]
            mask = sampled_probs < thresholds.unsqueeze(-1)
            idx = torch.where(mask,torch.full_like(idx, self.num_vq), idx)
            idx = torch.where(mask&special_mask,input_idx, idx)
            input_idx = torch.where(non_mask, input_idx,idx)
            non_mask = ~mask
        return input_idx#[:,-T:]
    # def sample_cond(self, input_idx, if_categorial=False,non_mask=None):
    #     B, L = input_idx.shape
    #
    #     logits = self.forward(input_idx)
    #     probs = torch.softmax(logits, dim=-1)
    #     sampled_probs, idx = torch.topk(probs, k=1, dim=-1)
    #     sampled_probs = sampled_probs.squeeze(-1)
    #     pre_probs = torch.gather(probs, -1, input_idx.unsqueeze(-1)).squeeze(-1)
    #     delta_probs = sampled_probs - pre_probs
    #     topk, _ = torch.topk(delta_probs, 11, dim=-1, largest=False)
    #     thresholds = topk[:, -1]
    #     non_mask = delta_probs < thresholds.unsqueeze(-1)
    #
    #     T = L-torch.sum(non_mask,1).min()
    #     special_mask = (input_idx != self.num_vq)^non_mask
    #     for t in range(T):
    #         # 前向传播，预测词汇
    #         logits = self.forward(input_idx)  # 假设 forward 方法返回 logits (B, L, vocab_size)
    #         probs = torch.softmax(logits, dim=-1)
    #
    #         if if_categorial:
    #             dist = Categorical(probs)
    #             idx = dist.sample()
    #             sampled_probs = torch.gather(probs, -1, idx.unsqueeze(-1)).squeeze(-1)
    #
    #         else:
    #             sampled_probs, idx = torch.topk(probs, k=1, dim=-1)
    #             sampled_probs = sampled_probs.squeeze(-1)
    #             idx = idx.squeeze(-1)
    #
    #         # non_mask = input_idx != self.num_vq
    #         sampled_probs = torch.where(non_mask, torch.ones_like(sampled_probs), sampled_probs)
    #         # new_mask_num = torch.ceil(torch.cos(torch.tensor((t+1)/T*torch.pi / 2))*L).int()
    #         new_mask_num = T - 1 - t
    #         if new_mask_num == L:
    #             new_mask_num = L-1
    #
    #         topk, _ = torch.topk(sampled_probs, new_mask_num+1, dim=-1, largest=False)
    #         thresholds = topk[:, -1]
    #         mask = sampled_probs < thresholds.unsqueeze(-1)
    #         idx = torch.where(mask,torch.full_like(idx, self.num_vq), idx)
    #         idx = torch.where(mask&special_mask,input_idx, idx)
    #         input_idx = torch.where(non_mask, input_idx,idx)
    #         non_mask = ~mask
    #     return input_idx#[:,-T:]
    #     # T = 1#L-non_mask.sum(1).min()
    #     # special_mask = (input_idx != self.num_vq)^non_mask
    #     # for t in range(T):
    #     #     # 前向传播，预测词汇
    #     #     logits = self.forward(input_idx)  # 假设 forward 方法返回 logits (B, L, vocab_size)
    #     #     probs = torch.softmax(logits, dim=-1)
    #     #
    #     #     if if_categorial:
    #     #         dist = Categorical(probs)
    #     #         idx = dist.sample()
    #     #         sampled_probs = torch.gather(probs, -1, idx.unsqueeze(-1)).squeeze(-1)
    #     #
    #     #     else:
    #     #         sampled_probs, idx = torch.topk(probs, k=1, dim=-1)
    #     #         sampled_probs = sampled_probs.squeeze(-1)
    #     #         idx = idx.squeeze(-1)
    #     #     pre_probs = torch.gather(probs, -1, input_idx.unsqueeze(-1)).squeeze(-1)
    #     #     sampled_probs = sampled_probs - pre_probs
    #     #     # non_mask = input_idx != self.num_vq
    #     #     sampled_probs = torch.where(non_mask, torch.ones_like(sampled_probs), sampled_probs)
    #     #     # new_mask_num = torch.ceil(torch.cos(torch.tensor((t+1)/T*torch.pi / 2))*L).int()
    #     #     new_mask_num = T - 1 - t
    #     #     if new_mask_num == L:
    #     #         new_mask_num = L-1
    #     #
    #     #     topk, _ = torch.topk(sampled_probs, new_mask_num+1, dim=-1, largest=False)
    #     #     thresholds = topk[:, -1]
    #     #     mask = sampled_probs < thresholds.unsqueeze(-1)
    #     #     idx = torch.where(mask,torch.full_like(idx, self.num_vq), idx)
    #     #     idx = torch.where(mask&special_mask,input_idx, idx)
    #     #     input_idx = torch.where(non_mask, input_idx,idx)
    #     #     non_mask = ~mask
    #     # return input_idx#[:,-T:]

class SelfAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512,
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq+1, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        # ## lstm block
        # self.blocks = StackedLSTM(embed_dim, embed_dim, num_layers,batch_first=True, bidirectional=True)#lstm

        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # forward the Trans model
        token_embeddings = self.tok_emb(idx)

        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x


class TransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        ###lstm block
        # self.blocks = StackedLSTM(embed_dim, embed_dim, num_layers,batch_first=True, bidirectional=True)#lstm


        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)#num_vq+1
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x) #lstm
        x = self.ln_f(x)
        logits = self.head(x)
        return logits



class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,batch_first=True, bidirectional=False):
        super(StackedLSTM, self).__init__()
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size, hidden_size, batch_first=batch_first, bidirectional=bidirectional) for _ in range(num_layers)])
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for lstm_layer, linear_layer in zip(self.lstm_layers, self.linear_layers):
            x, _ = lstm_layer(x)
            x = linear_layer(x)
        return x


        

