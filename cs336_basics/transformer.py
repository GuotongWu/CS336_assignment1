import torch
import math
import torch.nn as nn
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                device: torch.device | None = None, 
                dtype: torch.dtype | None = None):
        super().__init__()
        w = torch.empty((out_features, in_features), dtype=dtype, device=device)
        std = math.sqrt(2 / (in_features + out_features))

        w = nn.init.trunc_normal_(w, mean=0, std=std, a=-3*std, b=3*std)
        self.w = nn.Parameter(w, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.t()
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        embed_matrix = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        embed_matrix = nn.init.trunc_normal_(embed_matrix, mean=0, std=1, a=-3, b=3)
        self.embed_matrix = nn.Parameter(embed_matrix, requires_grad=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_matrix[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, 
                 device: torch.device | None=None, 
                 dtype: torch.dtype | None=None):
        super().__init__()
        g = torch.ones(d_model, dtype=dtype, device=device)
        self.g = nn.Parameter(g, requires_grad=True)
        self.d_model = d_model
        self.eps = eps
    
    def forward(self, x: torch.Tensor):
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(x**2, dim=-1) / self.d_model + self.eps)
        result = x / rms.unsqueeze(dim=-1) * self.g
        return result.to(in_type)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, 
                 device: torch.device | None=None, 
                 dtype: torch.dtype | None=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)        
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, Theta: float, d_k: int, max_seq_len: int, 
                 device: torch.device | None=None):
        super().__init__()
        assert d_k % 2 == 0, f"Make sure d_k is an even number."
        k = d_k // 2
        denominator = Theta ** (torch.arange(k, device=device) * (-2.0 / d_k))
        i_vec = torch.arange(max_seq_len, device=device).unsqueeze(dim=-1)
        theta = i_vec * denominator.unsqueeze(dim=0)

        sin_mat = torch.sin(theta)
        self.register_buffer("sin_mat", sin_mat, persistent=False)

        cos_mat = torch.cos(theta)
        self.register_buffer("cos_mat", cos_mat, persistent=False)

        self.k = k
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None):
        if token_positions is None:
            token_positions = torch.arange(x.shape[2], device=self.device)
            token_positions = token_positions.unsqueeze(0).expand(x.shape[0],-1)

        cos_select = self.cos_mat[token_positions]
        sin_select = self.sin_mat[token_positions]
        
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rotate_even = x_even * cos_select - x_odd * sin_select
        rotate_odd = x_even * sin_select + x_odd * cos_select

        return torch.stack((rotate_even, rotate_odd), dim=-1).flatten(start_dim=-2)

def softmax(x: torch.Tensor, dim: int):
    exp_subtract = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_subtract / torch.sum(exp_subtract, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch. Tensor, V: torch.Tensor, 
                                 mask: torch.BoolTensor | None=None):
    d_k = Q.size(-1)
    scores = torch.einsum("...qd, ...kd -> ...qk", Q, K) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(~mask, -torch.inf)
    attn_weights = softmax(scores, dim=-1)
    return torch.einsum("...qk, ...kv -> ...qv", attn_weights, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, is_rope: bool=False,
                 max_seq_len: int | None=None, theta: float | None=None, 
                 token_positions: torch.Tensor=None, 
                 device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.device = device
        
        self.wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.wo = Linear(d_model, d_model, device=device, dtype=dtype)

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions
        self.is_rope = is_rope

        if is_rope:
            self.rope = RoPE(self.theta, self.d_k, self.max_seq_len, self.device)

    def forward(self, x: torch.Tensor):
        """x.shape = (batch, seq_len, d_model)"""
        batch, seq_len, _ = x.shape
        q = self.wq(x).view(batch, seq_len, self.num_heads, self.d_k).permute(2, 0, 1, 3)
        k = self.wk(x).view(batch, seq_len, self.num_heads, self.d_k).permute(2, 0, 1, 3)
        v = self.wv(x).view(batch, seq_len, self.num_heads, self.d_k).permute(2, 0, 1, 3)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        if self.is_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)

        attn = scaled_dot_product_attention(q, k, v, mask) # (num_heads, batch, seq_len, d_k)
        attn = rearrange(attn, "h b s d_k -> b s (h d_k)") # (batch, seq_len, d_model)
        return self.wo(attn)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads:int, d_ff: int,
                 is_rope: bool, max_seq_len: int, theta: float, 
                 device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.mha = MultiHeadAttention(d_model, num_heads, is_rope,
                                      max_seq_len, theta, token_positions=None,
                                      device=device, dtype=dtype)

        self.rmsnorm2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.mha(self.rmsnorm1(x))
        
        return x + self.ffn(self.rmsnorm2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, 
                 d_model: int, num_heads: int, d_ff: int, rope_theta: float,
                 device: torch.device | None=None, dtype: torch.dtype | None=None):

        super().__init__()
        self.embed = Embedding(vocab_size, d_model, device, dtype)
        self.transformer_block_list = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, True,
                              context_length, rope_theta, device, dtype) 
            for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ln = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: torch.Tensor):
        x = self.embed(x)
        for sub_net in self.transformer_block_list:
            x = sub_net(x)
        return self.ln(self.norm(x))

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # l1 = Linear(32, 64, device, dtype=torch.float32)
    # x = torch.rand(8, 32, dtype=torch.float32, device=device)
    # print(l1(x).shape)

    # rmsnorm = RMSNorm(512, 1e-5, device, dtype=torch.float16)
    # x = torch.rand(8, 64, 512, dtype=torch.float16, device=device)
    # print(rmsnorm(x).shape)

    # swiglu = SwiGLU(512, 512*8//3, device)
    # x = torch.rand(8, 64, 512, dtype=torch.float32, device=device)
    # print(swiglu(x).shape)

    # rope = RoPE(1e-5, 64, 512, device)
    # x = torch.rand(8, 512, 64)
    # token_positions = torch.randint(0, 512, (8, 512))
    # print(rope(x, token_positions).shape)

    # x = torch.rand(8, 512, 64)
    # print(softmax(x, dim=-1).shape)

    mha = MultiHeadAttention(64, 8)
    rope = RoPE(1e-5, 64, 512, device)
    x = torch.rand(8, 512, 64)
    token_positions = torch.randint(0, 512, (8, 512))
    mha.rope_init(1e-5, 512)
    print(mha(x, True, token_positions).shape)