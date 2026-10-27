import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)

from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
from vllm_ascend.utils import enable_sp


def tensor_parallel_wrap(func):
    def wrap(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        if deepstack_input_embeds is None or not enable_sp():
            return deepstack_input_embeds
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        deepstack_input_embeds.tensors = {
            k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()
        }
        return deepstack_input_embeds

    return wrap


def forward_with_split_qkv_rmsnorm_mrope(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    qkv, _ = self.qkv_proj(hidden_states)
    if isinstance(self.rotary_emb, AscendMRotaryEmbedding):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        if cos_sin.device != qkv.device:
            cos_sin = cos_sin.to(qkv.device)
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(qkv.dtype)
        q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            mrope_section=self.rotary_emb.mrope_section,
            is_interleaved=self.rotary_emb.mrope_interleaved,
            rope_dim=self.rotary_emb.rotary_dim,
        )
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = tensor_parallel_wrap(
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
)


def _fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
    outputs = []
    for t, h, w in grid_thw:
        outputs.append(
            pos_embed_interpolate_native(
                self.pos_embed.weight,
                t,
                h,
                w,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.dtype,
            )
        )
    return torch.cat(outputs, dim=0)


Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate


def patch_qwen3_vl_moe_pp_layer_range():
    try:
        from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM
    except Exception:
        return

    if not hasattr(Qwen3MoeLLMForCausalLM, "start_layer"):
        Qwen3MoeLLMForCausalLM.start_layer = property(lambda self: self.model.start_layer)

    if not hasattr(Qwen3MoeLLMForCausalLM, "end_layer"):
        Qwen3MoeLLMForCausalLM.end_layer = property(lambda self: self.model.end_layer)


patch_qwen3_vl_moe_pp_layer_range()


# ----------------------------------------------------------------------------
# EVS (Efficient Video Sampling) device fix for recompute_mrope_positions.
# mrope_positions is built from numpy in get_mrope_input_positions (CPU) while
# multimodal_embeddings live on the NPU. The EVS arithmetic inside
# recompute_mrope_positions (mm_pos[...] + base) mixes the two and raises a
# cross-device error on NPU. Move mrope_positions to the embeddings' device
# before delegating to the original implementation. (No-op when already
# co-located, e.g. on GPU.)
# ----------------------------------------------------------------------------
def _mask_outer_vision_markers(input_ids, vision_start_token_id, vision_end_token_id):
    # Newer HF processors retain an outer video wrapper around the per-frame
    # wrappers. It is text for position reconstruction, not another frame.
    # Mask only the outer start in a copy used by the position parser; keep
    # sequence offsets and the actual prompt token IDs unchanged.
    if isinstance(input_ids, torch.Tensor):
        marker_positions = torch.where((input_ids == vision_start_token_id) | (input_ids == vision_end_token_id))[0]
        markers = input_ids[marker_positions]
        outer_positions = marker_positions[:-1][
            (markers[:-1] == vision_start_token_id) & (markers[1:] == vision_start_token_id)
        ]
        masked_ids = input_ids.clone()
        masked_ids[outer_positions] = vision_end_token_id
        return masked_ids

    marker_positions = [
        i for i, token in enumerate(input_ids) if token in (vision_start_token_id, vision_end_token_id)
    ]
    outer_positions = [
        current
        for current, following in zip(marker_positions, marker_positions[1:])
        if input_ids[current] == input_ids[following] == vision_start_token_id
    ]
    if not outer_positions:
        return input_ids
    masked_ids = list(input_ids)
    for position in outer_positions:
        masked_ids[position] = vision_end_token_id
    return masked_ids


def _evs_recompute_mrope_positions_wrap(orig):
    def wrap(self, input_ids, multimodal_embeddings, mrope_positions, num_computed_tokens):
        if mrope_positions is not None and len(multimodal_embeddings):
            mrope_positions = mrope_positions.to(multimodal_embeddings[0].device)
        config = getattr(self, "config", None)
        if config is not None:
            input_ids = _mask_outer_vision_markers(input_ids, config.vision_start_token_id, config.vision_end_token_id)
        return orig(self, input_ids, multimodal_embeddings, mrope_positions, num_computed_tokens)

    return wrap


Qwen3VLForConditionalGeneration.recompute_mrope_positions = _evs_recompute_mrope_positions_wrap(
    Qwen3VLForConditionalGeneration.recompute_mrope_positions
)
