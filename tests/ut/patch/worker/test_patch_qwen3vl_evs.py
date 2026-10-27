#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from vllm_ascend.patch.worker import patch_qwen3vl


def test_evs_recompute_mrope_positions_moves_positions_to_embedding_device():
    original = MagicMock(return_value="positions")
    moved_positions = object()
    mrope_positions = MagicMock()
    mrope_positions.to.return_value = moved_positions
    embedding = MagicMock(device="npu:0")
    wrapper = patch_qwen3vl._evs_recompute_mrope_positions_wrap(original)

    result = wrapper(object(), [1, 2], [embedding], mrope_positions, 7)

    assert result == "positions"
    mrope_positions.to.assert_called_once_with("npu:0")
    original.assert_called_once()
    assert original.call_args.args[3] is moved_positions


def test_evs_recompute_mrope_positions_preserves_cpu_positions_without_embeddings():
    original = MagicMock(return_value="positions")
    mrope_positions = MagicMock()
    wrapper = patch_qwen3vl._evs_recompute_mrope_positions_wrap(original)

    wrapper(object(), [1], [], mrope_positions, 0)

    mrope_positions.to.assert_not_called()
    assert original.call_args.args[3] is mrope_positions


@pytest.mark.parametrize("tensor_input", [False, True])
def test_evs_recompute_handles_outer_video_wrapper(tensor_input):
    # An outer HF wrapper precedes the timestamp and the actual frame start.
    # Treating it as a frame subtracts the timestamp width from index zero,
    # producing an empty destination slice in the upstream position helper.
    input_ids = [10, 1, 2, 10, 12, 12, 11, 11, 3]
    if tensor_input:
        input_ids = torch.tensor(input_ids)
    original_ids = input_ids.clone() if tensor_input else list(input_ids)
    model = SimpleNamespace(
        config=SimpleNamespace(vision_start_token_id=10, vision_end_token_id=11),
    )
    positions = torch.arange(9).expand(3, -1).clone()
    mm_positions = torch.zeros(6, 5)
    mm_positions[:, :3] = torch.arange(6).unsqueeze(1)
    mm_positions[2, 3] = 1
    mm_positions[3:5, 4] = 1
    embedding = torch.cat([torch.ones(6, 2), mm_positions], dim=1)

    def original(self, ids, embeddings, mrope, num_computed):
        return Qwen3VLForConditionalGeneration._recompute_mrope_positions(
            input_ids=ids,
            multimodal_embeddings=embeddings,
            mrope_positions=mrope,
            num_computed_tokens=num_computed,
            vision_start_token_id=10,
            image_token_id=13,
            video_token_id=12,
        )

    with pytest.raises(RuntimeError, match="expanded size"):
        original(model, input_ids, [embedding], positions, 0)

    wrapper = patch_qwen3vl._evs_recompute_mrope_positions_wrap(original)
    clean, new_positions, delta = wrapper(model, input_ids, [embedding], positions, 0)

    assert clean[0].shape == (6, 2)
    assert new_positions.shape == (3, 9)
    assert delta == int(new_positions.max()) + 1 - 9
    if tensor_input:
        assert torch.equal(input_ids, original_ids)
    else:
        assert input_ids == original_ids


def test_evs_keeps_image_and_legacy_video_markers():
    input_ids = [10, 13, 11, 1, 10, 12, 11]
    assert patch_qwen3vl._mask_outer_vision_markers(input_ids, 10, 11) == input_ids
