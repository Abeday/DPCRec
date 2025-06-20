import torch.nn as nn
import torch.nn.functional as f
from utils import *
from torch.nn import Dropout


class Capsule(torch.nn.Module):

    def __init__(self, emb_dim, poi_num, route_num, low_capsule_num, drop_fusion):

        super(Capsule, self).__init__()

        init_n = 1
        self.h_caps_dim = 64
        self.l_caps_dim = emb_dim
        self.output_dim = poi_num

        self.h_caps_num = poi_num//20  # solve the problem of insufficient calculating power
        self.l_caps_num = low_capsule_num

        self.rn = route_num

        # weights initialization
        self.dropout = Dropout(p=drop_fusion)
        self.dim_linear_1 = torch.nn.Linear(self.h_caps_dim, 1)
        self.part_linear_all = torch.nn.Linear(self.h_caps_num, self.output_dim)

        self.l_map_h = nn.Parameter(init_n * torch.randn(self.l_caps_dim, self.h_caps_dim),
                                    requires_grad=True)
        self.route_weights = nn.Parameter(init_n * torch.randn(1, self.h_caps_num, self.l_caps_num),
                                          requires_grad=False)

    def forward(self, u):

        h_caps = None
        u_map = torch.matmul(u, self.l_map_h)
        for iteration in range(self.rn):

            b = self.route_weights
            c = f.softmax(b, dim=-1)
            s = torch.matmul(c, u_map)
            v = squash(s)

            u_v = torch.sum(
                torch.matmul(v, u_map.transpose(1, 2)),
                dim=0, keepdim=True
            )
            self.route_weights.data = self.route_weights.data + u_v.data
            h_caps = v
        # mlp

        out_put = self.dropout(self.dim_linear_1(h_caps).squeeze(-1))
        out_put = self.dropout(self.part_linear_all(out_put))

        #  l2 norm
        # caps = caps / (th.norm(caps, dim=2).view(bs, self.K, 1) + 1e-9)

        return out_put


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, head_num):
        super(MultiHeadAttention, self).__init__()

        self.qkv_d = embed_dim
        self.head_n = head_num

        self.W_Q = nn.Linear(self.qkv_d, self.qkv_d * self.head_n, bias=False)
        self.W_K = nn.Linear(self.qkv_d, self.qkv_d * self.head_n, bias=False)
        self.W_V = nn.Linear(self.qkv_d, self.qkv_d * self.head_n, bias=False)
        self.fc = nn.Linear(self.qkv_d, self.qkv_d)

    def forward(self, q_origin, k_origin, mask=None):

        # q_origin: [bs, short_len, emb_dim]
        # k_origin: [bs, short_len, emb_dim]
        bs, q_l, q_d = q_origin.shape
        _, kv_l, kv_d = k_origin.shape

        q = self.W_Q(q_origin)  # q: [bs, len, num * dim]
        k = self.W_K(k_origin)  # k: [bs, len, num * dim]
        v = self.W_V(k_origin)  # v: [bs, len, num * dim]

        # (bs, input len, head num, dim) --> (head num * bs, num, dim)
        q_head = q.view(bs, q_l, self.head_n, self.qkv_d).permute(2, 0, 1, 3).contiguous().view(-1, q_l, self.qkv_d)
        k_head = k.view(bs, kv_l, self.head_n, self.qkv_d).permute(2, 0, 1, 3).contiguous().view(-1, kv_l, self.qkv_d)
        v_head = v.view(bs, kv_l, self.head_n, self.qkv_d).permute(2, 0, 1, 3).contiguous().view(-1, kv_l, self.qkv_d)

        simi = torch.matmul(q_head, k_head.transpose(-1, -2))
        if mask is not None:
            mask = mask.repeat(self.head_n, 1, 1)  # (bs * head num, kv_l, kv_l)
            simi = simi.masked_fill(mask, -torch.inf)

        attention = f.softmax(simi / (self.qkv_d ** (1 / 2)), dim=-1)

        attention = torch.nan_to_num(attention)
        out = torch.matmul(attention, v_head)

        out = out.view(self.head_n, bs, q_l, kv_d).permute(1, 2, 0, 3).contiguous().view(bs, -1, q_d)
        out_f = f.tanh(self.fc(out))

        return out_f


class ASE_Cell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(ASE_Cell, self).__init__()
        self.hidden_size = hidden_size

        # mapping matrices
        self.w_p = nn.Linear(input_size, 3 * hidden_size)
        self.w_d = nn.Linear(input_size, 2 * hidden_size)
        self.w_t = nn.Linear(input_size, 2 * hidden_size)
        self.w_h = nn.Linear(hidden_size, 7 * hidden_size)

    def forward(self, p, d, t, h):
        # calculate all mappings
        wp_p, wp_z, wp_hk = self.w_p(p).chunk(3, dim=1)
        wd_d, wd_z = self.w_d(d).chunk(2, dim=1)
        wt_t, wt_z = self.w_t(t).chunk(2, dim=1)
        wh_p, wh_d, wh_t, wh_z, wh_hq_d, wh_hq_t, wh_hk = self.w_h(h).chunk(7, dim=1)

        # gates and hidden states
        p_t = f.sigmoid(wh_p + wp_p)
        d_t = f.sigmoid(wh_d + wd_d)
        t_t = f.sigmoid(wh_t + wt_t)
        z_t = f.sigmoid(wh_z + wp_z + wd_z + wt_z)
        hq = f.tanh(d_t * wh_hq_d + t_t * wh_hq_t)
        hk = f.tanh(wp_hk + p_t * wh_hk)
        simi = torch.matmul(hq, hk.transpose(-1, -2))
        alpha = f.softmax(simi / (self.hidden_size ** (1 / 2)), dim=-1)
        h_tilde = torch.matmul(alpha, hk)

        # calculate the current output
        h_new = (1 - z_t) * h + z_t * h_tilde

        return h_new


class ASE(nn.Module):  # Advanced Short-Term Interest Feature Extractor
    def __init__(self, input_size, hidden_size, output_size, drop_short):
        super(ASE, self).__init__()
        self.cell = ASE_Cell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = Dropout(p=drop_short)

    def forward(self, x, s, t):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.cell.hidden_size).to(x.device)

        out = []
        # 分时间步处理输入序列
        for step in range(seq_len):
            h = self.cell(x[:, step, :], s[:, step, :], t[:, step, :], h)
            h = self.dropout(h)
            out.append(f.tanh(self.fc(h)).unsqueeze(1))

        return out


class ALE(nn.Module):  # Advanced Long-Term Interest Feature Extractor
    def __init__(self, input_dim, head_num, layer_num, dropout=0.1):
        super(ALE, self).__init__()
        self.pre_layer = ALE_MultiHeadDeepAttention(input_dim, head_num, dropout)
        self.layers = nn.ModuleList([ALE_Layer(input_dim, head_num, dropout) for _ in range(layer_num)])

    def forward(self, q, k, mask):

        output = self.pre_layer(q, k, mask)  # get the input of self-attention
        for layer in self.layers:
            output = layer(output, output)

        return output


class ALE_Layer(nn.Module):
    def __init__(self, embed_dim, heads_num, dropout=0.1):
        super(ALE_Layer, self).__init__()
        self.attn_layer = ALE_MultiHeadDeepAttention(embed_dim, heads_num, dropout)
        self.linear_layer = ALE_FeedForward(embed_dim, dropout)

        # 层归一化和残差连接
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):

        # attn part
        out_attn = self.attn_layer(q, k, mask)  # attn
        out_residual1 = k + self.dropout1(out_attn)  # residual connection
        out_norm1 = self.norm1(out_residual1)  # norm

        # forward part
        out_forward = self.linear_layer(out_norm1)  # linear
        out_residual2 = out_norm1 + self.dropout2(out_forward)  # residual connection
        out_norm2 = self.norm2(out_residual2)  # norm

        return out_norm2


class ALE_FeedForward(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(ALE_FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim*4)
        self.fc2 = nn.Linear(input_dim*4, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ALE_MultiHeadDeepAttention(nn.Module):
    def __init__(self, embed_dim, head_num, drop_long):
        super(ALE_MultiHeadDeepAttention, self).__init__()
        self.qkv_d = embed_dim
        self.head_n = head_num

        self.W_Q = nn.Linear(self.qkv_d, self.qkv_d * self.head_n, bias=False)
        self.W_K = nn.Linear(self.qkv_d, self.qkv_d * self.head_n, bias=False)
        self.W_V = nn.Linear(self.qkv_d, self.qkv_d * self.head_n, bias=False)

        self.linear1 = nn.Linear(self.qkv_d * 4, self.qkv_d * 2)
        self.linear2 = nn.Linear(self.qkv_d * 2, self.qkv_d)
        self.linear3 = nn.Linear(self.qkv_d, 1)

        self.dropout = Dropout(p=drop_long)

    def forward(self, q_origin, k_origin, mask=None):
        # mask : (batch size, k length, k length)
        # q : (batch size, q length, embedding dim)
        # k : (batch size, k length, embedding dim)
        bs, q_l, q_d = q_origin.shape
        _, kv_l, kv_d = k_origin.shape  # Sequence length

        # 通过线性层生成q, k, v
        q = self.W_Q(q_origin)
        k = self.W_K(k_origin)
        v = self.W_V(k_origin)

        # (bs, input len, head num * dim) --> (head num * bs, num, dim)
        q_head = q.view(bs, q_l, self.head_n, self.qkv_d).permute(2, 0, 1, 3).contiguous().view(-1, q_l, self.qkv_d)
        k_head = k.view(bs, kv_l, self.head_n, self.qkv_d).permute(2, 0, 1, 3).contiguous().view(-1, kv_l, self.qkv_d)
        v_head = v.view(bs, kv_l, self.head_n, self.qkv_d).permute(2, 0, 1, 3).contiguous().view(-1, kv_l, self.qkv_d)

        # 计算注意力权重simi
        dw0 = torch.cat((q_head, k_head, q_head-k_head, q_head*k_head), dim=-1)
        dw1 = torch.sigmoid(self.linear1(dw0))
        dw2 = torch.sigmoid(self.linear2(dw1))
        simi = self.linear3(dw2).squeeze(-1)

        if mask is not None:
            mask = mask.repeat(self.head_n, 1)  # (bs * head num, kv_l)
            simi = simi.masked_fill(mask, -torch.inf)

        attention = f.softmax(simi / (self.qkv_d ** (1 / 2)), dim=1)
        attention = self.dropout(attention)
        # attention_non = torch.nan_to_num(attention, nan=0.0)

        # 加权求和
        out = torch.bmm(attention.unsqueeze(1), v_head)
        out = out.view(self.head_n, bs, 1, kv_d).permute(1, 2, 0, 3).contiguous().view(bs, -1, kv_d)

        return out
