"""
Winnougan Anima CLIP Loader
────────────────────────────
Standalone loader for the Qwen 3.5 4B hybrid (Mamba2 + Attention) text encoder
used with the Anima 2B diffusion model (cosmos-qwen3.5).

Architecture: 32 layers, hidden=2560, vocab=248320
- SSM layers  : 0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30
- Attn layers : 3, 7, 11, 15, 19, 23, 27, 31
- Output norm : Linear(2560→1024) + ExpRMSNorm + SiLU + Linear(1024→1024)

CRITICAL: Requires the Qwen3.5 tokenizer (vocab=248320), NOT the Qwen3
tokenizer (vocab=151936). Place tokenizer files in:
  <node_dir>/qwen35_tokenizer/
Or they will be auto-downloaded from Qwen/Qwen3.5-4B on HuggingFace.

Source: GumGum10/comfyui-qwen35-anima
"""

import os
import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.sd
import comfy.ops
import comfy.utils
import comfy.model_management
import comfy.sd1_clip
import safetensors.torch as safetensors_torch
from comfy.ldm.common_dit import rms_norm
from comfy.supported_models_base import ClipTarget

log = logging.getLogger("Winnougan")

NODE_NAME  = "Winnougan Anima CLIP Loader"
NODE_DIR   = os.path.dirname(os.path.realpath(__file__))
QWEN35_TOKENIZER_DIR = os.path.join(NODE_DIR, "qwen35_tokenizer")


# ── Norms ─────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class ExpRMSNorm(nn.Module):
    """
    RMSNorm with exp(weight) parameterisation.
    The late norm's learned weights are near-zero (~-0.003). Standard RMSNorm
    would collapse all token information; exp(weight) ≈ 1 preserves it.
    """
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, torch.exp(self.weight.float()).to(x.dtype), self.eps)


# ── SSM Block (Mamba2) ────────────────────────────────────────────────────────

class SSMBlock(nn.Module):
    """
    Mamba2-style Selective State Space Model block.
    d_ssm=4096, nheads=32, head_dim=128, ngroups=32, d_state=64
    conv_dim = d_ssm + 2*ngroups*d_state = 8192
    """
    def __init__(self, hidden_size=2560, d_inner=8192, n_groups=32,
                 d_gate=4096, conv_kernel=4, norm_dim=128,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.d_inner  = d_inner
        self.n_groups = n_groups
        self.d_ssm    = d_gate
        self.head_dim = d_gate // n_groups         # 128
        self.d_state  = (d_inner - d_gate) // (2 * n_groups)  # 64

        self.in_proj_qkv = ops.Linear(hidden_size, d_inner,   bias=False, device=device, dtype=dtype)
        self.in_proj_z   = ops.Linear(hidden_size, d_gate,    bias=False, device=device, dtype=dtype)
        self.in_proj_a   = ops.Linear(hidden_size, n_groups,  bias=False, device=device, dtype=dtype)
        self.in_proj_b   = ops.Linear(hidden_size, n_groups,  bias=False, device=device, dtype=dtype)
        self.conv1d      = ops.Conv1d(d_inner, d_inner, conv_kernel, groups=d_inner,
                                      padding=conv_kernel - 1, bias=False, device=device, dtype=dtype)
        self.out_proj    = ops.Linear(d_gate, hidden_size, bias=False, device=device, dtype=dtype)
        self.norm        = RMSNorm(norm_dim, device=device, dtype=dtype)

        self.A_log    = nn.Parameter(torch.zeros(n_groups, device=device, dtype=dtype))
        self.dt_bias  = nn.Parameter(torch.zeros(n_groups, device=device, dtype=dtype))

    def _ssm_scan(self, x, B_state, C_state, dt_input, D_input):
        batch, seq_len, nheads, head_dim = x.shape
        d_state = B_state.shape[-1]
        device  = x.device

        A       = -torch.exp(self.A_log.to(device).float())
        dt_bias = self.dt_bias.to(device).float()

        h       = torch.zeros(batch, nheads, head_dim, d_state, device=device, dtype=torch.float32)
        outputs = []

        x_f  = x.float();   B_f = B_state.float()
        C_f  = C_state.float(); dt_f = dt_input.float(); D_f = D_input.float()

        for t in range(seq_len):
            dt_t  = F.softplus(dt_f[:, t] + dt_bias)
            dA_t  = torch.exp(dt_t * A.unsqueeze(0))
            dBx   = dt_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum('bnh,bns->bnhs', x_f[:, t], B_f[:, t])
            h     = dA_t.unsqueeze(-1).unsqueeze(-1) * h + dBx
            y_t   = torch.einsum('bnhs,bns->bnh', h, C_f[:, t])
            y_t   = y_t + D_f[:, t].unsqueeze(-1) * x_f[:, t]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(x.dtype)

    def forward(self, hidden_states):
        batch, seq_len, _ = hidden_states.shape

        z        = self.in_proj_z(hidden_states)
        xBC      = self.in_proj_qkv(hidden_states)
        dt_input = self.in_proj_b(hidden_states)
        D_input  = self.in_proj_a(hidden_states)

        xBC_conv = self.conv1d(xBC.transpose(1, 2))[..., :seq_len]
        xBC_conv = F.silu(xBC_conv.transpose(1, 2))

        x, B_conv, C_conv = torch.split(
            xBC_conv,
            [self.d_ssm, self.n_groups * self.d_state, self.n_groups * self.d_state],
            dim=-1,
        )

        x       = x.reshape(batch, seq_len, self.n_groups, self.head_dim)
        B_state = B_conv.reshape(batch, seq_len, self.n_groups, self.d_state)
        C_state = C_conv.reshape(batch, seq_len, self.n_groups, self.d_state)

        y = self._ssm_scan(x, B_state, C_state, dt_input, D_input)
        y = self.norm(y).reshape(batch, seq_len, -1) * F.silu(z)
        return self.out_proj(y)


# ── Self-Attention Block ──────────────────────────────────────────────────────

class GatedSelfAttention(nn.Module):
    """16-head GQA (4 KV heads, head_dim=256) with gated Q projection."""
    def __init__(self, hidden_size=2560, num_heads=16, num_kv_heads=4,
                 head_dim=256, rope_theta=1000000.0,
                 device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.gqa_ratio    = num_heads // num_kv_heads
        self.inner_dim    = num_heads * head_dim  # 4096

        self.q_proj = ops.Linear(hidden_size, 2 * self.inner_dim,       bias=False, device=device, dtype=dtype)
        self.k_proj = ops.Linear(hidden_size, num_kv_heads * head_dim,  bias=False, device=device, dtype=dtype)
        self.v_proj = ops.Linear(hidden_size, num_kv_heads * head_dim,  bias=False, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_dim, hidden_size,           bias=False, device=device, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, device=device, dtype=dtype)

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        B, L, _ = hidden_states.shape

        q, gate = self.q_proj(hidden_states).chunk(2, dim=-1)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q.view(B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)

        k = k.repeat_interleave(self.gqa_ratio, dim=1)
        v = v.repeat_interleave(self.gqa_ratio, dim=1)

        attn_mask = attention_mask.to(q.dtype) if attention_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              is_causal=(attn_mask is None))
        out = out.transpose(1, 2).reshape(B, L, self.inner_dim) * F.silu(gate)
        return self.o_proj(out)


def _apply_rotary_emb(x, cos, sin):
    d  = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)


def _precompute_freqs_cis(head_dim, seq_len, theta=1000000.0, device=None, dtype=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t     = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos   = freqs.cos().unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    sin   = freqs.sin().unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    return (cos.to(dtype), sin.to(dtype)) if dtype else (cos, sin)


# ── MLP ───────────────────────────────────────────────────────────────────────

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=9216, device=None, dtype=None, ops=None):
        super().__init__()
        ops = ops or nn
        self.gate_proj = ops.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj   = ops.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Hybrid Block ──────────────────────────────────────────────────────────────

class HybridBlock(nn.Module):
    def __init__(self, hidden_size=2560, intermediate_size=9216,
                 use_ssm=True, has_mlp=True,
                 device=None, dtype=None, ops=None):
        super().__init__()
        self.use_ssm = use_ssm
        self.has_mlp = has_mlp

        self.input_layernorm = RMSNorm(hidden_size, device=device, dtype=dtype)
        if use_ssm:
            self.linear_attn = SSMBlock(hidden_size=hidden_size, device=device, dtype=dtype, ops=ops)
        else:
            self.self_attn = GatedSelfAttention(hidden_size=hidden_size, device=device, dtype=dtype, ops=ops)
        if has_mlp:
            self.post_attention_layernorm = RMSNorm(hidden_size, device=device, dtype=dtype)
            self.mlp = SwiGLUMLP(hidden_size=hidden_size, intermediate_size=intermediate_size,
                                  device=device, dtype=dtype, ops=ops)

    def forward(self, x, attention_mask=None, freqs_cis=None):
        residual = x
        x_norm   = self.input_layernorm(x)
        x = residual + (self.linear_attn(x_norm) if self.use_ssm
                        else self.self_attn(x_norm, attention_mask=attention_mask, freqs_cis=freqs_cis))
        if self.has_mlp:
            residual = x
            x = residual + self.mlp(self.post_attention_layernorm(x))
        return x


# ── Full Qwen 3.5 Hybrid Backbone ─────────────────────────────────────────────

class Qwen35HybridModel(nn.Module):
    """
    Qwen 3.5 4B text backbone (confirmed against Qwen/Qwen3.5-4B config.json).
    32 layers alternating SSM/attention every 4 layers; layer 31 has no MLP.
    Output norm: Linear(2560→1024) + ExpRMSNorm + SiLU + Linear(1024→1024)
    """
    SELF_ATTN_LAYERS  = {3, 7, 11, 15, 19, 23, 27, 31}
    NUM_LAYERS        = 32
    HIDDEN_SIZE       = 2560
    INTERMEDIATE_SIZE = 9216
    VOCAB_SIZE        = 248320
    OUTPUT_DIM        = 1024
    HEAD_DIM          = 256
    ROPE_THETA        = 1000000.0

    def __init__(self, config_dict=None, dtype=None, device=None, operations=None):
        super().__init__()
        cfg = config_dict or {}
        ops = operations or comfy.ops.disable_weight_init

        self.embed_tokens = ops.Embedding(self.VOCAB_SIZE, self.HIDDEN_SIZE, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            HybridBlock(
                hidden_size       = self.HIDDEN_SIZE,
                intermediate_size = self.INTERMEDIATE_SIZE,
                use_ssm = (i not in self.SELF_ATTN_LAYERS),
                has_mlp = (i != 31),
                device=device, dtype=dtype, ops=ops,
            )
            for i in range(self.NUM_LAYERS)
        ])

        # Output projection: 2560 → 1024
        self.norm = nn.Sequential(
            ops.Linear(self.HIDDEN_SIZE, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
            ExpRMSNorm(self.OUTPUT_DIM, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(self.OUTPUT_DIM, self.OUTPUT_DIM, bias=True, device=device, dtype=dtype),
        )

        self._output_scale = cfg.get("output_scale", 1.0)

        # Optional calibration (per-dim affine, from calibrate.py)
        self._calibration_scale = None
        self._calibration_bias  = None
        self._use_calibration   = cfg.get("use_calibration", False)

        # Optional Procrustes alignment (4B → 0.6B concept space, from compute_alignment.py)
        self._rotation_matrix   = None
        self._rotation_mean_4b  = None
        self._rotation_mean_06b = None
        self._use_alignment         = cfg.get("use_alignment", False)
        self._alignment_strength    = cfg.get("alignment_strength", 1.0)

        if self._use_calibration:
            self._load_calibration()
        if self._use_alignment:
            self._load_alignment()

    # ── Optional calibration / alignment loaders ──────────────────────────────

    def _load_calibration(self):
        path = os.path.join(NODE_DIR, "calibration_params.safetensors")
        if not os.path.exists(path):
            log.warning(f"[{NODE_NAME}] Calibration file not found: {path}")
            self._use_calibration = False
            return
        try:
            cal = safetensors_torch.load_file(path)
            self._calibration_scale = cal["scale"].float()
            self._calibration_bias  = cal["bias"].float()
            log.info(f"[{NODE_NAME}] Calibration loaded.")
        except Exception as e:
            log.warning(f"[{NODE_NAME}] Calibration load failed: {e}")
            self._use_calibration = False

    def _load_alignment(self):
        path = os.path.join(NODE_DIR, "rotation_matrix.safetensors")
        if not os.path.exists(path):
            log.warning(f"[{NODE_NAME}] Alignment file not found: {path}")
            self._use_alignment = False
            return
        try:
            data = safetensors_torch.load_file(path)
            self._rotation_matrix   = data["rotation"].float()
            self._rotation_mean_4b  = data["mean_4b"].float()
            self._rotation_mean_06b = data["mean_06b"].float()
            log.info(f"[{NODE_NAME}] Procrustes alignment loaded.")
        except Exception as e:
            log.warning(f"[{NODE_NAME}] Alignment load failed: {e}")
            self._use_alignment = False

    # ── Forward ───────────────────────────────────────────────────────────────

    def get_input_embeddings(self):  return self.embed_tokens
    def set_input_embeddings(self, e): self.embed_tokens = e

    def forward(self, input_ids, attention_mask=None, embeds=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, **kwargs):

        x       = embeds if embeds is not None else self.embed_tokens(input_ids, out_dtype=dtype or torch.float32)
        seq_len = x.shape[1]

        freqs_cis = _precompute_freqs_cis(self.HEAD_DIM, seq_len,
                                           theta=self.ROPE_THETA, device=x.device, dtype=x.dtype)

        # Build causal + padding mask for attention layers
        attn_mask = None
        fill = torch.finfo(x.dtype).min / 4
        causal = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(fill).triu_(1)
        if attention_mask is not None:
            pad = 1.0 - attention_mask.to(x.dtype).reshape(
                attention_mask.shape[0], 1, -1, attention_mask.shape[-1]
            ).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            attn_mask = causal + pad.masked_fill(pad.bool(), fill)
        elif seq_len > 1:
            attn_mask = causal

        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask=attn_mask, freqs_cis=freqs_cis)
            if intermediate_output is not None:
                if isinstance(intermediate_output, int) and i == intermediate_output:
                    intermediate = x.clone()
                elif isinstance(intermediate_output, list) and i in intermediate_output:
                    intermediate = intermediate or {}
                    intermediate[i] = x.clone()

        x = self.norm(x)

        # Procrustes alignment
        if self._use_alignment and self._rotation_matrix is not None:
            R    = self._rotation_matrix.to(device=x.device, dtype=x.dtype)
            m4b  = self._rotation_mean_4b.to(device=x.device, dtype=x.dtype)
            m06b = self._rotation_mean_06b.to(device=x.device, dtype=x.dtype)
            a    = self._alignment_strength
            x    = torch.einsum('ij,...j->...i', R, x - m4b) + (1.0 - a) * m4b + a * m06b

        # Per-dimension affine calibration
        if self._use_calibration and self._calibration_scale is not None:
            x = x * self._calibration_scale.to(x.device, x.dtype) \
                  + self._calibration_bias.to(x.device, x.dtype)

        if self._output_scale != 1.0:
            x = x * self._output_scale

        return x, intermediate


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class Qwen35Tokenizer:
    """
    Qwen 3.5 tokenizer (vocab=248320).
    DO NOT substitute the Qwen3 tokenizer (vocab=151936) — different BPE rules
    mean every token maps to the wrong embedding row.
    """
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.tokenizer       = self._load()
        self.embedding_size  = 1024
        self.embedding_key   = "qwen35_4b"
        self.max_length      = 1024
        self.pad_token_id    = 151643   # <|endoftext|>
        self.eos_token_id    = 248044
        self.tokens_to_skip  = set(self.tokenizer.encode(""))

    def _load(self):
        from transformers import AutoTokenizer
        # 1. Bundled tokenizer directory
        if os.path.isdir(QWEN35_TOKENIZER_DIR):
            has_files = (
                os.path.exists(os.path.join(QWEN35_TOKENIZER_DIR, "vocab.json")) or
                os.path.exists(os.path.join(QWEN35_TOKENIZER_DIR, "tokenizer.json"))
            )
            if has_files:
                log.info(f"[{NODE_NAME}] Using bundled Qwen3.5 tokenizer.")
                return AutoTokenizer.from_pretrained(QWEN35_TOKENIZER_DIR, trust_remote_code=False)

        # 2. Auto-download from HuggingFace
        log.info(f"[{NODE_NAME}] Downloading Qwen3.5 tokenizer from Qwen/Qwen3.5-4B...")
        try:
            return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=False)
        except Exception as e:
            raise RuntimeError(
                f"[{NODE_NAME}] Could not load Qwen3.5 tokenizer: {e}\n"
                f"Download vocab.json + tokenizer.json from https://huggingface.co/Qwen/Qwen3.5-4B "
                f"and place them in: {QWEN35_TOKENIZER_DIR}"
            )

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        ids   = self.tokenizer.encode(text, add_special_tokens=False)
        pairs = [
            (tid, 1.0, i) if return_word_ids else (tid, 1.0)
            for i, tid in enumerate(ids)
        ]
        pad = (self.pad_token_id, 1.0, len(pairs)) if return_word_ids else (self.pad_token_id, 1.0)
        while len(pairs) < self.max_length:
            pairs.append(pad)
        return [pairs[:self.max_length]]

    def untokenize(self, token_weight_pair):
        ids = [t[0] for t in token_weight_pair if t[0] != self.pad_token_id]
        return self.tokenizer.decode(ids)

    def state_dict(self): return {}

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)


class AnimaQwen35Tokenizer:
    """Dual tokenizer: Qwen3.5 (main encoder) + T5 (LLM adapter target IDs)."""
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.qwen35_4b = Qwen35Tokenizer(embedding_directory=embedding_directory,
                                          tokenizer_data=tokenizer_data)
        from comfy.text_encoders.anima import T5XXLTokenizer
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory,
                                     tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        qwen_ids = self.qwen35_4b.tokenize_with_weights(text, return_word_ids, **kwargs)
        return {
            "qwen35_4b": [
                [(k[0], 1.0, k[2]) if return_word_ids else (k[0], 1.0) for k in chunk]
                for chunk in qwen_ids
            ],
            "t5xxl": self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs),
        }

    def untokenize(self, token_weight_pair): return self.t5xxl.untokenize(token_weight_pair)
    def state_dict(self): return {}
    def decode(self, token_ids, **kwargs): return self.qwen35_4b.decode(token_ids, **kwargs)


# ── CLIP wrappers ──────────────────────────────────────────────────────────────

class Qwen35_4BClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None,
                 attention_mask=True, model_options={}):
        super().__init__(
            device=device, layer=layer, layer_idx=layer_idx,
            textmodel_json_config={}, dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=Qwen35HybridModel,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )


class AnimaQwen35TEModel(comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen35_4b",
                         clip_model=Qwen35_4BClipModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        if "t5xxl" in token_weight_pairs:
            out[2]["t5xxl_ids"] = torch.tensor(
                [a[0] for a in token_weight_pairs["t5xxl"][0]], dtype=torch.int
            )
            out[2]["t5xxl_weights"] = torch.tensor(
                [a[1] for a in token_weight_pairs["t5xxl"][0]]
            )
        return out


def _te_factory(dtype_llama=None, llama_quantization_metadata=None):
    """Return a concrete AnimaQwen35TEModel subclass with pinned dtype / quant."""
    class _AnimaQwen35TEModel(AnimaQwen35TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = {**model_options, "quantization_metadata": llama_quantization_metadata}
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return _AnimaQwen35TEModel


# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class WinnouganAnimaCLIPLoader:
    NAME     = NODE_NAME
    CATEGORY = "Winnougan"

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        te_dir = os.path.join(folder_paths.models_dir, "text_encoders")
        files  = []
        if os.path.isdir(te_dir):
            files = sorted(f for f in os.listdir(te_dir) if f.endswith(".safetensors"))
        if not files:
            files = ["qwen35_4b.safetensors"]

        return {
            "required": {
                "clip_name": (files,),
            },
            "optional": {
                "use_calibration": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply per-dimension affine calibration (requires calibration_params.safetensors).",
                }),
                "use_alignment": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Procrustes rotation to align 4B → 0.6B concept space (requires rotation_matrix.safetensors).",
                }),
                "alignment_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Bias-shift blend: 0 = keep 4B magnitude, 1 = shift to 0.6B magnitude. The rotation is always applied when alignment is on.",
                }),
                "output_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "Uniform scale applied after calibration. Leave at 1.0 when calibration is on.",
                }),
            },
        }

    RETURN_TYPES  = ("CLIP",)
    RETURN_NAMES  = ("clip",)
    FUNCTION      = "load_clip"

    def load_clip(self, clip_name, use_calibration=False, use_alignment=False,
                  alignment_strength=0.0, output_scale=1.0):
        import folder_paths

        clip_path = os.path.join(folder_paths.models_dir, "text_encoders", clip_name)
        if not os.path.exists(clip_path):
            raise FileNotFoundError(f"[{NODE_NAME}] Text encoder not found: {clip_path}")

        log.info(f"[{NODE_NAME}] Loading {clip_name} "
                 f"[calibration={'on' if use_calibration else 'off'}, "
                 f"alignment={'on' if use_alignment else 'off'}, "
                 f"alignment_strength={alignment_strength}, "
                 f"output_scale={output_scale}]")

        sd = safetensors_torch.load_file(clip_path)

        # Detect dtype from norm weights
        detected_dtype = None
        for key in ("model.norm.weight", "layers.0.input_layernorm.weight",
                    "norm.1.weight", "model.layers.0.input_layernorm.weight"):
            if key in sd:
                detected_dtype = sd[key].dtype
                break

        quant = comfy.utils.detect_layer_quantization(sd, "")
        te_cls = _te_factory(
            dtype_llama                  = detected_dtype,
            llama_quantization_metadata  = quant,
        )

        clip_target = ClipTarget(AnimaQwen35Tokenizer, te_cls)
        param_count = sum(torch.tensor(v.shape).prod().item() for v in sd.values())

        clip = comfy.sd.CLIP(target=clip_target, state_dict=[sd], parameters=param_count)

        # Apply runtime settings to loaded model
        try:
            inner = clip.cond_stage_model.qwen35_4b.transformer
            inner._output_scale      = output_scale
            inner._use_calibration   = use_calibration
            inner._use_alignment     = use_alignment
            inner._alignment_strength = alignment_strength
            if use_calibration:
                inner._load_calibration()
            if use_alignment:
                inner._load_alignment()
        except AttributeError as e:
            log.warning(f"[{NODE_NAME}] Could not apply settings: {e}")

        log.info(f"[{NODE_NAME}] Loaded {param_count:,} parameters.")
        return (clip,)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "WinnouganAnimaCLIPLoader": WinnouganAnimaCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WinnouganAnimaCLIPLoader": "Winnougan Anima CLIP Loader",
}
