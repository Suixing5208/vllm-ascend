from types import SimpleNamespace

from vllm_ascend.models.qwen3_5_vl import _AscendVLPreprocessMixin


class _UpstreamModel:
    def __init__(self, *, vllm_config, prefix="model"):
        self.multimodal_config = vllm_config.multimodal_config
        self.deepstack_num_level = vllm_config.deepstack_num_level
        self.model_config = SimpleNamespace(multimodal_config=self.multimodal_config)
        self.visual = SimpleNamespace(input_norm=None)
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(
                in_channels=3, patch_size=2, temporal_patch_size=2, spatial_merge_size=2
            )
        )


class _Model(_AscendVLPreprocessMixin, _UpstreamModel):
    pass


def _config(method, deepstack_num_level):
    return SimpleNamespace(
        multimodal_config=SimpleNamespace(
            get_video_pruning_spec=lambda: None if method is None else (method, 0.75),
            mm_device_do_normalize=False,
        ),
        deepstack_num_level=deepstack_num_level,
    )


def test_qwen35_vl_mixin_restores_pruning_method_and_empty_deepstack_state():
    model = _Model(vllm_config=_config("evs", 0))
    assert model.video_pruning_method == "evs"
    assert model.use_deepstack is False


def test_qwen35_vl_mixin_preserves_nonempty_deepstack_state():
    model = _Model(vllm_config=_config(None, 2))
    assert model.video_pruning_method is None
    assert model.use_deepstack is True
