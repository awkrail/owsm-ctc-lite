import torch.nn as nn

from owsm_ctc_lite.layer_norm import LayerNorm

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        qk_norm = False,
        use_flash_attn = False,
        causal = False,
        cross_attn = False,
        use_sdpa = False,
    ):
        super(MultiheadAttention, self).__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(dropout_rate) if not use_flash_attn else nn.Identity()
        self.dropout_rate = dropout_rate

        # LayerNorm for q and k
        self.q_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()
        self.k_norm = LayerNorm(self.d_k) if qk_norm else nn.Identity()

        self.use_flash_attn = use_flash_attn
        self.causal = causal
        self.use_sdpa = use_sdpa
