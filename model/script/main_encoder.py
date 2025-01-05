import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    pos: (batch, seq_len) or (..., n)
    dim: must be even
    theta: base frequency
    Returns: a "complex form" embedding of shape (..., dim//2, 2, 2) or expanded with i=j=2
    """
    assert dim % 2 == 0
    # scale in [0, ..., dim/2) => we step by 2
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)   # (dim/2)
    # outer product of pos and omega => shape: (..., n) x (dim/2) => (..., n, dim/2)
    out = torch.einsum("...n,d->...nd", pos, omega)
    # Build the 2x2 "complex" rotation matrix [cos, -sin; sin, cos]
    out = torch.stack([torch.cos(out), -torch.sin(out),
                       torch.sin(out),  torch.cos(out)], dim=-1)
    # Rearrange last dimension for easier application to Q/K
    out = rearrange(out, "... n d (i j) -> ... n d i j", i=2, j=2)
    return out.float()

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    xq, xk: (batch, seq_len, hidden_dim)
    freqs_cis: shape broadcastable to xq
               basically the "complex" representation of cos/sin from rope(...)
    Returns xq_out, xk_out with rotary embedding applied.
    """
    # Reshape [b, s, hdim] -> [b, s, hdim/2, 1, 2]
    # for an easy matmul with the [cos, -sin; sin, cos] blocks
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

    # We do something like complex multiplication:
    #   [cos, -sin] [q_re]
    #   [sin,  cos] [q_im]
    # for each chunk.  This is exactly what's stored in freqs_cis[..., 0..1].
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    # Reshape back to [b, s, hidden_dim]
    xq_out = xq_out.reshape(*xq.shape).type_as(xq)
    xk_out = xk_out.reshape(*xk.shape).type_as(xk)
    return xq_out, xk_out


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=2304, n_heads=16, dropout_prob=0.1, rope_theta=10000):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads."
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.rope_theta = rope_theta

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        returns: (batch, seq_len, hidden_dim)
        """
        b, s, d = x.shape
        # Compute Q, K, V
        q = self.q_proj(x)  # (b, s, d)
        k = self.k_proj(x)  # (b, s, d)
        v = self.v_proj(x)  # (b, s, d)

        # Build position ids => shape (b, s), each row is [0,1,2..., s-1]
        #   Or we could do per-sample if desired, but typical is same positions for entire batch
        position_ids = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)

        # Get the rotary embeddings:
        #   rope(...) returns shape (b, s, d//2, 2, 2) if we're rotating the entire dimension
        #   but typically we use half-d or so. We'll keep it simple and rotate the entire dimension.
        freqs_cis = rope(position_ids, d, self.rope_theta)  # (b, s, d/2, 2, 2) -> rearranged

        # Apply RoPE to Q, K => shape still (b, s, d)
        q, k = apply_rope(q, k, freqs_cis)  # (b, s, d), (b, s, d)

        # Reshape for multi‐head attn: => (b, s, n_heads, head_dim)
        q = q.reshape(b, s, self.n_heads, self.head_dim)
        k = k.reshape(b, s, self.n_heads, self.head_dim)
        v = v.reshape(b, s, self.n_heads, self.head_dim)

        # Transpose to (b, n_heads, s, head_dim) for scaled dot‐product
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        #  q shape: (b, nH, s, hD)
        #  k shape: (b, nH, s, hD)
        # => attn logits: (b, nH, s, s)
        scale = 1.0 / (self.head_dim ** 0.5)
        attn_logits = torch.einsum("bnhd,bmhd->bnhm", q, k) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum => shape (b, nH, s, hD)
        out = torch.einsum("bnhm,bmhd->bnhd", attn_weights, v)
        # Back to (b, s, nH, hD) => (b, s, d)
        out = out.transpose(1, 2).reshape(b, s, d)

        # Final linear projection
        out = self.o_proj(out)
        out = self.dropout(out)
        # Residual + LN
        x = self.ln(x + out)
        return x

class RoPETransformerBlock(nn.Module):
    def __init__(self, hidden_dim=2304, n_heads=16, dropout_prob=0.1, rope_theta=10000):
        super().__init__()
        self.mha = RoPEMultiHeadAttention(hidden_dim, n_heads, dropout_prob, rope_theta)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 1) MHA + residual
        x = self.mha(x)
        # 2) FFN + residual
        y = self.ffn(x)
        x = self.ln(x + self.dropout(y))
        return x

class SigLiPTransformer(nn.Module):
    """
    A simpler Transformer that consumes image+text tokens (each 2304-dim),
    applies rotary positional embeddings in attention, and produces a
    4096-dim [CLS] embedding for SigLiP-style contrastive loss.
    """
    def __init__(
        self,
        hidden_dim=2304,
        out_dim=4096,
        num_layers=6,
        n_heads=16,
        dropout_prob=0.1,
        rope_theta=10000,
        tokenizer=None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.tokenizer = tokenizer  # so we can freeze it if desired

        # A learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Stack multiple RoPETransformerBlocks
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout_prob=dropout_prob,
                rope_theta=rope_theta
            ) for _ in range(num_layers)
        ])

        # Final projection to 4096
        self.proj_to_out = nn.Linear(hidden_dim, out_dim)

        # SigLiP learnable temperature and bias
        self.t_prime = nn.Parameter(torch.zeros(1))  # log-temperature
        self.bias = nn.Parameter(torch.zeros(1))     # bias

    def freeze_tokenizer(self):
        """
        Disable gradient updates for the tokenizer (e.g. if you want it fixed).
        """
        if self.tokenizer is not None:
            for param in self.tokenizer.parameters():
                param.requires_grad = False
        else:
            print("No tokenizer provided; nothing to freeze.")

    def forward(self, img_tokens: torch.Tensor, txt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tokens: (B, N_img, 2304)
            txt_tokens: (B, N_txt, 2304)
        Returns: (B, 4096) representing the [CLS] embedding after the last layer
        """
        B, N_img, H_img = img_tokens.shape
        B2, N_txt, H_txt = txt_tokens.shape
        assert B == B2 and H_img == H_txt == self.hidden_dim

        # Concat image + text
        tokens = torch.cat([img_tokens, txt_tokens], dim=1)  # (B, N_img+N_txt, 2304)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.repeat(B, 1, 1)          # (B, 1, 2304)
        x = torch.cat([cls_tokens, tokens], dim=1)           # (B, 1+N_img+N_txt, 2304)

        # Pass through each RoPETransformerBlock
        for block in self.blocks:
            x = block(x)

        # Take final [CLS]
        cls_out = x[:, 0, :]  # (B, 2304)
        # Project to 4096
        out = self.proj_to_out(cls_out)  # (B, 4096)
        return out

    def compute_siglip_loss(self, anchor_emb: torch.Tensor, model_emb: torch.Tensor) -> torch.Tensor:
        """
        anchor_emb: (B, 4096) from anchored embedding model
        model_emb:  (B, 4096) from this SigLiPTransformer
        Returns a scalar SigLiP loss.
        """
        B, D = anchor_emb.shape
        B2, D2 = model_emb.shape
        assert B == B2 and D == D2, "Mismatched shapes for anchor/model embeddings."

        # L2 normalize
        anchor_norm = F.normalize(anchor_emb, dim=1)  # (B, 4096)
        model_norm  = F.normalize(model_emb,  dim=1)  # (B, 4096)

        # Temperature
        t = self.t_prime.exp()  # scalar

        # Pairwise logits => (B, B)
        logits = anchor_norm @ model_norm.transpose(0, 1) * t + self.bias

        # Labels => +1 on diag, -1 off diag
        labels = 2 * torch.eye(B, device=logits.device) - 1  # (B, B)

        # Sigmoid loss for all pairs
        #   L = - 1/B * sum_{i,j} [ log sigma( z_ij * logits_ij ) ]
        loss = -F.logsigmoid(labels * logits).sum() / B
        return loss
