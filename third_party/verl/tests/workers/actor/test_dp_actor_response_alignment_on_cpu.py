# Copyright 2026 THU-BPM
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

import torch

from verl.workers.actor.dp_actor import DataParallelPPOActor


def _actor_without_initialization() -> DataParallelPPOActor:
    return object.__new__(DataParallelPPOActor)


def _compact_teacher_layout():
    # Both responses begin at column 6 in a tensor whose width was fixed by a
    # longer response outside this micro-batch. Prompts are left-padded and
    # responses are right-padded, matching _build_teacher_batch_from_prompts.
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        ]
    )
    response_lengths = torch.tensor([2, 3])
    return attention_mask, response_lengths


def test_align_compact_response_tensor_uses_attended_span_not_micro_batch_max():
    actor = _actor_without_initialization()
    attention_mask, response_lengths = _compact_teacher_layout()
    tensor = torch.arange(11).repeat(2, 1)

    aligned = actor._align_compact_response_tensor(
        tensor,
        response_length=5,
        response_lengths=response_lengths,
        attention_mask=attention_mask,
    )

    # Column 5 predicts the first response token for both rows. Deriving the
    # start from width=11 and local max response length=3 would incorrectly
    # start at column 7.
    expected = torch.tensor(
        [
            [5, 6, 0, 0, 0],
            [5, 6, 7, 0, 0],
        ]
    )
    torch.testing.assert_close(aligned, expected)


def test_build_full_topk_indices_uses_the_same_compact_response_positions():
    actor = _actor_without_initialization()
    attention_mask, response_lengths = _compact_teacher_layout()
    response_mask = torch.arange(5).unsqueeze(0) < response_lengths.unsqueeze(1)
    topk_indices = torch.tensor(
        [
            [[10, 11], [12, 13], [0, 0], [0, 0], [0, 0]],
            [[20, 21], [22, 23], [24, 25], [0, 0], [0, 0]],
        ]
    )

    full = actor._build_full_topk_indices(
        topk_indices,
        batch_size=2,
        seqlen=11,
        response_length=5,
        response_mask=response_mask,
        attention_mask=attention_mask,
    )

    torch.testing.assert_close(full[0, 5:7], topk_indices[0, :2])
    torch.testing.assert_close(full[1, 5:8], topk_indices[1, :3])
    assert torch.count_nonzero(full[:, :5]) == 0
    assert torch.count_nonzero(full[:, 8:]) == 0
