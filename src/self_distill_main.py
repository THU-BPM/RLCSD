"""
Entry point for self-distillation training using verl's infrastructure.

Self-distillation flow per rollout step:
1. Student rollout: vLLM generates responses (verl standard)
2. Privileged text extraction: read GT solution/answer from the dataset batch
3. Teacher forward: compute log probs with privileged context on the same response tokens
4. Actor update with distillation loss

Usage:
    python -m src.self_distill_main <verl hydra config overrides>
"""

import os
import sys
import json
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party", "verl"))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import open_dict

from src.opsd_format import (
    build_rollout_messages,
    build_teacher_messages,
    RLCSD_TEACHER_TRANSITION_PROMPT,
    extract_boxed_answer,
    normalize_privileged_text_mode,
    strip_legacy_math_prompt)
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.metric_utils import compute_data_metrics

RLCSD_LOSS_MODES = {"rlcsd","rlsd_ectr", "opsd_ectr"}

# rlcsd_5 averages teacher_wrong_lp over up to K non-self negative siblings.
# 4 covers the typical group_size=8 case (3-7 non-self negatives) without
# ballooning per-rollout teacher-forward cost.
RLCSD_DEFAULT_K_MAX = 4
DISTILL_LOSS_MODES = {"opsd", "sdpo", "rlsd", "srpo", *RLCSD_LOSS_MODES}
MATH_DATA_SOURCES = {
    "dapo_math_17k",
    "openthoughts_114k_math_filtered",
    "deepmath_filtered",
    "deepmath_filtered_level5_7",
    "deepmath_filtered_level6_8",
    "deepmath_filtered_level7_10",
    "amc23",
    "aime24",
    "aime25",
}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for segment in content:
            if isinstance(segment, dict):
                if segment.get("type") == "text":
                    parts.append(str(segment.get("text", "")))
                elif "content" in segment:
                    parts.append(str(segment.get("content", "")))
            elif segment is not None:
                parts.append(str(segment))
        return " ".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _raw_prompt_to_problem(raw_prompt):
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, list):
        user_parts = []
        fallback_parts = []
        for msg in raw_prompt:
            if not isinstance(msg, dict):
                continue
            text = _content_to_text(msg.get("content", ""))
            if not text:
                continue
            fallback_parts.append(text)
            if msg.get("role") == "user":
                user_parts.append(text)
        parts = user_parts or fallback_parts
        return strip_legacy_math_prompt("\n".join(parts).strip())

    if raw_prompt is None:
        return ""
    return strip_legacy_math_prompt(str(raw_prompt).strip())


def _policy_loss_cfg(self):
    return self.config.actor_rollout_ref.actor.policy_loss


def _rollout_custom_cfg(self):
    rollout_cfg = self.config.actor_rollout_ref.get("rollout", {})
    custom_cfg = rollout_cfg.get("custom", None)
    return custom_cfg if custom_cfg is not None else {}


def _cfg_get(self, key, default):
    return _policy_loss_cfg(self).get(key, default)


def _custom_cfg_get(self, key, default):
    custom_cfg = _rollout_custom_cfg(self)
    if key in custom_cfg:
        return custom_cfg.get(key)
    return _cfg_get(self, key, default)


def _cfg_bool(self, key, default=False):
    value = _cfg_get(self, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _custom_cfg_bool(self, key, default=False):
    value = _custom_cfg_get(self, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _teacher_total_token_budget(self) -> int:
    actor_cfg = self.config.actor_rollout_ref.actor
    budget = actor_cfg.get("ppo_max_token_len_per_gpu", None)
    if budget is not None:
        return int(budget)
    return int(self.config.data.max_prompt_length) + int(self.config.data.max_response_length)


def _teacher_chat_template_kwargs(self):
    kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
    kwargs["enable_thinking"] = _custom_cfg_bool(self, "teacher_enable_thinking", True)
    return kwargs


def _compute_prefilter_metric_overrides(self, batch: DataProto) -> dict[str, float]:
    """Capture trainer-side metrics that should stay on the pre-filter rollout batch."""
    required_keys = {"token_level_scores", "token_level_rewards", "advantages", "returns", "response_mask"}
    if any(key not in batch.batch.keys() for key in required_keys):
        return {}
    prefilter_metrics = compute_data_metrics(batch=batch, use_critic=getattr(self, "use_critic", True))
    kept_prefixes = ("critic/", "response_length/", "response_length_non_aborted/")
    kept_exact_keys = {"response/aborted_ratio"}
    return {
        key: value
        for key, value in prefilter_metrics.items()
        if key.startswith(kept_prefixes) or key in kept_exact_keys
    }


def _get_actor_dp_size(self) -> int:
    """Best-effort lookup for actor data-parallel size during training."""
    try:
        if hasattr(self, "_get_dp_size") and getattr(self, "actor_rollout_wg", None) is not None:
            return int(self._get_dp_size(self.actor_rollout_wg, "actor"))
    except Exception:
        pass
    return 1


def _prepare_actor_update_batch(self, batch: DataProto) -> DataProto:
    """Pad rlcsd actor batches to DP divisibility with zero-loss replicas."""
    if _cfg_get(self, "loss_mode", "vanilla") not in RLCSD_LOSS_MODES:
        return batch

    batch_size = len(batch)
    actor_dp_size = _get_actor_dp_size(self)
    if batch_size == 0 or actor_dp_size <= 1 or batch_size % actor_dp_size == 0:
        metrics = getattr(self, "_rlcsd_step_metrics", {})
        metrics["rlcsd/actor_pad_count"] = 0.0
        metrics["rlcsd/actor_padded_batch_size"] = float(batch_size)
        self._rlcsd_step_metrics = metrics
        return batch

    pad_size = actor_dp_size - (batch_size % actor_dp_size)
    pad_indices = [idx % batch_size for idx in range(pad_size)]
    padded_tail = batch.select_idxs(pad_indices)
    padded_tail.meta_info = dict(batch.meta_info)

    if "response_mask" in padded_tail.batch.keys():
        padded_tail.batch["response_mask"] = torch.zeros_like(padded_tail.batch["response_mask"])
    if "uid" in padded_tail.non_tensor_batch:
        padded_tail.non_tensor_batch["uid"] = np.asarray(
            [f"__rlcsd_pad__{i}" for i in range(pad_size)],
            dtype=object)
    metrics = getattr(self, "_rlcsd_step_metrics", {})
    metrics["rlcsd/actor_pad_count"] = float(pad_size)
    metrics["rlcsd/actor_padded_batch_size"] = float(batch_size + pad_size)
    self._rlcsd_step_metrics = metrics
    actor_batch = DataProto.concat([batch, padded_tail])
    actor_batch.meta_info = dict(batch.meta_info)
    if "global_token_num" in batch.meta_info:
        original_token_num = list(batch.meta_info.get("global_token_num", []))
        actor_batch.meta_info["global_token_num"] = original_token_num + [original_token_num[i] for i in pad_indices]
    return actor_batch


def _collapse_repeated_to_batch(values, batch_size):
    if values is None:
        return None

    values = list(values)
    if len(values) == batch_size:
        return values
    if batch_size <= 0 or len(values) % batch_size != 0:
        raise AssertionError(f"Cannot align {len(values)} items to batch size {batch_size}")

    repeat_factor = len(values) // batch_size
    return [values[i * repeat_factor] for i in range(batch_size)]


def _extract_problems_answers_solutions(self, batch: DataProto, raw_prompts=None, reward_models=None, extra_infos=None):
    """Extract problem texts, GT answers, and GT solutions from the dataset batch."""
    batch_size = len(batch)
    if raw_prompts is None:
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    if reward_models is None:
        reward_models = batch.non_tensor_batch.get("reward_model")
    if extra_infos is None:
        extra_infos = batch.non_tensor_batch.get("extra_info")

    raw_prompts = _collapse_repeated_to_batch(raw_prompts, batch_size) if raw_prompts is not None else None
    reward_models = _collapse_repeated_to_batch(reward_models, batch_size) if reward_models is not None else None
    extra_infos = _collapse_repeated_to_batch(extra_infos, batch_size) if extra_infos is not None else None

    problems = []
    ground_truths = []
    solutions = []
    for i in range(batch_size):
        raw_prompt = raw_prompts[i] if raw_prompts is not None else None
        reward_model = reward_models[i] if reward_models is not None else {}
        answer = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        ground_truths.append(str(answer))

        extra_info = extra_infos[i] if extra_infos is not None else {}
        problem = _content_to_text(extra_info.get("problem", "")) if isinstance(extra_info, dict) else ""
        if not problem:
            problem = _raw_prompt_to_problem(raw_prompt)
        if not problem and "prompts" in batch.batch:
            problem = strip_legacy_math_prompt(
                self.tokenizer.decode(batch.batch["prompts"][i], skip_special_tokens=True).strip()
            )
        problems.append(problem)

        solution = extra_info.get("solution", "") if isinstance(extra_info, dict) else ""
        solutions.append(_content_to_text(solution).strip())

    return problems, ground_truths, solutions


def _build_teacher_prompt(problem, answer, solution, mode, transition_prompt=None,
                          wrapper_variant="neutral", thinking=False):
    privileged_text, messages, _ = build_teacher_messages(
        problem=problem,
        answer=answer,
        solution=solution,
        mode=mode,
        transition_prompt=transition_prompt,
        wrapper_variant=wrapper_variant,
        thinking=thinking)
    return privileged_text, messages


def _materialize_teacher_context(self, problem, answer, solution):
    """Build a teacher context for the rlcsd_* methods.

    rlcsd_* uses the lighter RLCSD_TEACHER_TRANSITION_PROMPT (does not ask
    the model to derive a separate answer). opsd/sdpo/rlsd/srpo continue
    to go through _build_teacher_context_payloads and keep the original
    TEACHER_TRANSITION_PROMPT.

    The "neutral" wrapper variant reproduces this lighter wording.  Setting
    teacher_wrapper_variant=verbose / terse in the YAML config swaps both
    the framing line and the transition prompt for the robustness probe.
    """
    use_thinking = _custom_cfg_bool(self, "thinking_system_prompt", False)
    wrapper_variant = _custom_cfg_get(self, "teacher_wrapper_variant", "neutral")
    privileged_text, messages = _build_teacher_prompt(
        problem=problem,
        answer=answer,
        solution=solution,
        mode="solution_answer",
        wrapper_variant=wrapper_variant,
        thinking=use_thinking)
    prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **_teacher_chat_template_kwargs(self))
    return {
        "prompt": prompt,
        "text": privileged_text,
        "answer": answer}


def _sample_other_rollout(indices, current_idx):
    candidate_indices = [idx for idx in indices if idx != current_idx]
    if not candidate_indices:
        return None
    return random.choice(candidate_indices)


def _expand_to_rollout(values, batch_size, rollout_n):
    num_values = len(values)
    if num_values == batch_size:
        return values
    if num_values * rollout_n != batch_size:
        raise AssertionError(f"Mismatch: cached={num_values}, rollout_n={rollout_n}, batch={batch_size}")

    expanded = []
    for value in values:
        for _ in range(rollout_n):
            expanded.append(value)
    return expanded


def _build_teacher_context_payloads(self, problems, ground_truths, solutions):
    requested_mode = normalize_privileged_text_mode(_custom_cfg_get(self, "privileged_text_mode", "solution_answer"))
    privileged_texts = []
    effective_modes = []
    teacher_prompts = []

    if requested_mode == "solution_answer":
        missing_indices = [i for i, solution in enumerate(solutions) if not solution]
        if missing_indices:
            preview = ", ".join(str(i) for i in missing_indices[:8])
            if len(missing_indices) > 8:
                preview = f"{preview}, ..."
            raise ValueError(
                "privileged_text_mode=solution_answer requires every sample to have a non-empty GT solution, "
                f"but {len(missing_indices)}/{len(solutions)} samples in the current batch are missing it "
                f"(batch indices: {preview})."
            )

    use_thinking = _custom_cfg_bool(self, "thinking_system_prompt", False)
    wrapper_variant = _custom_cfg_get(self, "teacher_wrapper_variant", "neutral")
    for problem, answer, solution in zip(problems, ground_truths, solutions):
        privileged_text, messages, effective_mode = build_teacher_messages(
            problem=problem,
            answer=answer,
            solution=solution,
            mode=requested_mode,
            wrapper_variant=wrapper_variant,
            thinking=use_thinking)
        privileged_texts.append(privileged_text)
        effective_modes.append(effective_mode)
        teacher_prompts.append(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **_teacher_chat_template_kwargs(self))
        )

    return {
        "privileged_texts": privileged_texts,
        "teacher_prompts": teacher_prompts,
        "effective_modes": effective_modes,
        "rationales": privileged_texts}


def _generate_rationales(self, batch: DataProto, gen_output: DataProto) -> dict | None:
    """Cache GT privileged text from the dataset batch for teacher forwarding."""
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    if loss_mode not in DISTILL_LOSS_MODES:
        return None
    if loss_mode in RLCSD_LOSS_MODES:
        return None

    raw_prompts = batch.non_tensor_batch.get("raw_prompt")
    if raw_prompts is None:
        raw_prompts = gen_output.non_tensor_batch.get("raw_prompt")

    problems, ground_truths, solutions = self._extract_problems_answers_solutions(
        batch,
        raw_prompts=raw_prompts,
        reward_models=batch.non_tensor_batch.get("reward_model"),
        extra_infos=batch.non_tensor_batch.get("extra_info"))
    payloads = _build_teacher_context_payloads(self, problems, ground_truths, solutions)
    payloads.update({"problems": problems, "ground_truths": ground_truths, "solutions": solutions})
    return payloads


def _align_resp_dim(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate the response dimension (dim=1) to match target_len."""
    current_len = tensor.shape[1]
    if current_len == target_len:
        return tensor
    if current_len < target_len:
        return F.pad(tensor, (0, target_len - current_len), value=0.0)
    return tensor[:, :target_len]


def _build_teacher_batch_from_prompts(self, teacher_prompts, responses, prefix="teacher") -> DataProto:
    """Pack teacher prompts and sampled responses into model-ready tensors."""
    batch_size = len(teacher_prompts)
    if batch_size == 0:
        return DataProto.from_dict(tensors={})

    pad_id = self.tokenizer.pad_token_id or 0
    teacher_prompt_ids_list = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in teacher_prompts]

    teacher_input_ids_list = []
    teacher_prompt_lens = []
    teacher_total_budget = _teacher_total_token_budget(self)
    for i in range(batch_size):
        prompt_ids = teacher_prompt_ids_list[i]
        response_ids = [token for token in responses[i].tolist() if token != pad_id]
        max_prompt_tokens = max(teacher_total_budget - len(response_ids), 0)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[-max_prompt_tokens:] if max_prompt_tokens > 0 else []
        teacher_input_ids_list.append(prompt_ids + response_ids)
        teacher_prompt_lens.append(len(prompt_ids))

    max_prompt_len = max(teacher_prompt_lens)
    response_len = max(len(ids) - teacher_prompt_lens[i] for i, ids in enumerate(teacher_input_ids_list))
    total_len = max_prompt_len + response_len

    teacher_input_ids = torch.full((batch_size, total_len), pad_id, dtype=torch.long)
    teacher_attention_mask = torch.zeros((batch_size, total_len), dtype=torch.long)
    teacher_position_ids = torch.zeros((batch_size, total_len), dtype=torch.long)

    for i, ids in enumerate(teacher_input_ids_list):
        prompt_len = teacher_prompt_lens[i]
        prompt_offset = max_prompt_len - prompt_len
        response_ids = ids[prompt_len:]
        response_count = len(response_ids)

        teacher_input_ids[i, prompt_offset:max_prompt_len] = torch.tensor(ids[:prompt_len], dtype=torch.long)
        if response_count > 0:
            teacher_input_ids[i, max_prompt_len:max_prompt_len + response_count] = torch.tensor(
                response_ids, dtype=torch.long
            )

        teacher_attention_mask[i, prompt_offset:max_prompt_len + response_count] = 1
        teacher_position_ids[i, prompt_offset:max_prompt_len + response_count] = torch.arange(
            prompt_len + response_count
        )

    return DataProto.from_dict(
        tensors={
            f"{prefix}_input_ids": teacher_input_ids,
            f"{prefix}_attention_mask": teacher_attention_mask,
            f"{prefix}_position_ids": teacher_position_ids}
    )


def _build_teacher_multi_batch_from_prompt_groups(self, teacher_prompt_groups, responses, prefix="teacher") -> DataProto:
    """Pack per-sample teacher prompt groups into [batch, num_ctx, seq] tensors."""
    batch_size = len(teacher_prompt_groups)
    if batch_size == 0:
        return DataProto.from_dict(tensors={})

    group_sizes = [len(group) for group in teacher_prompt_groups]
    max_group_size = max(group_sizes, default=0)
    if max_group_size <= 0:
        raise ValueError(f"{prefix} received no teacher prompts to pack.")

    # Replicate the first prompt of each sample to fill empty slots so every
    # (sample, ctx) carries a non-empty input. This lets the actor's
    # multi-context teacher forward run unconditionally on every column,
    # avoiding NCCL collective desync when DP ranks see different valid_mask
    # column populations. Replicas are marked invalid in valid_mask so the
    # downstream loss ignores their outputs.
    flat_prompts = []
    flat_responses = []
    flat_indices = []
    for sample_idx, prompt_group in enumerate(teacher_prompt_groups):
        if not prompt_group:
            raise ValueError(f"{prefix} sample {sample_idx} has an empty prompt group.")
        for group_idx in range(max_group_size):
            real = group_idx < len(prompt_group)
            prompt = prompt_group[group_idx] if real else prompt_group[0]
            flat_prompts.append(prompt)
            flat_responses.append(responses[sample_idx])
            flat_indices.append((sample_idx, group_idx, real))

    flat_responses = torch.stack(flat_responses, dim=0)
    flat_prefix = f"{prefix}_flat"
    flat_batch = _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=flat_prompts,
        responses=flat_responses,
        prefix=flat_prefix)

    flat_input_ids = flat_batch.batch[f"{flat_prefix}_input_ids"]
    flat_attention_mask = flat_batch.batch[f"{flat_prefix}_attention_mask"]
    flat_position_ids = flat_batch.batch[f"{flat_prefix}_position_ids"]
    seq_len = flat_input_ids.shape[1]

    input_ids = flat_input_ids.new_zeros((batch_size, max_group_size, seq_len))
    attention_mask = flat_attention_mask.new_zeros((batch_size, max_group_size, seq_len))
    position_ids = flat_position_ids.new_zeros((batch_size, max_group_size, seq_len))
    valid_mask = torch.zeros((batch_size, max_group_size), dtype=torch.bool)

    for flat_idx, (sample_idx, group_idx, real) in enumerate(flat_indices):
        input_ids[sample_idx, group_idx] = flat_input_ids[flat_idx]
        attention_mask[sample_idx, group_idx] = flat_attention_mask[flat_idx]
        position_ids[sample_idx, group_idx] = flat_position_ids[flat_idx]
        if real:
            valid_mask[sample_idx, group_idx] = True

    return DataProto.from_dict(
        tensors={
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attention_mask": attention_mask,
            f"{prefix}_position_ids": position_ids,
            f"{prefix}_valid_mask": valid_mask},
        num_batch_dims=1)


def _compute_rlcsd_teacher_inputs(self, batch: DataProto) -> DataProto | None:
    """Build positive/negative teacher prompts from rollout groups identified by uid."""
    self._data_metric_overrides = _compute_prefilter_metric_overrides(self, batch)
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    if loss_mode not in RLCSD_LOSS_MODES:
        raise ValueError(f"_compute_rlcsd_teacher_inputs called with unsupported loss_mode={loss_mode}")
    batch_size = batch.batch.batch_size[0]
    if batch_size == 0:
        batch.meta_info["skip_actor_update"] = True
        self._aligned_teacher_rollout_details = {}
        self._rlcsd_step_metrics = {"rlcsd/skip_actor_update": 1.0}
        return None

    uid_values = batch.non_tensor_batch.get("uid")
    if uid_values is None:
        raise ValueError("rlcsd_* requires uid in non_tensor_batch to recover rollout groups.")

    problems, ground_truths, _ = self._extract_problems_answers_solutions(batch)
    responses = batch.batch["responses"]
    responses_text = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
    reward_tensor = batch.batch["token_level_scores"] if "token_level_scores" in batch.batch.keys() else batch.batch["rm_scores"]
    if reward_tensor.dim() > 1:
        rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()
    else:
        rewards = reward_tensor.detach().cpu().tolist()

    groups = {}
    for idx, uid in enumerate(uid_values.tolist()):
        groups.setdefault(str(uid), []).append(idx)

    num_aborted = 0
    all_same_outcome_group_count = 0
    self_negative_missing_boxed_count = 0
    self_excluded_sampling_skip_count = 0
    valid_mask = np.zeros(batch_size, dtype=bool)
    correct_prompt_by_idx = {}
    wrong_prompt_by_idx = {}
    correct_text_by_idx = {}
    wrong_text_by_idx = {}
    correct_answer_by_idx = {}
    wrong_answer_by_idx = {}
    correct_prompt_group_by_idx = {}
    wrong_prompt_group_by_idx = {}
    correct_text_group_by_idx = {}
    wrong_text_group_by_idx = {}
    correct_answer_group_by_idx = {}
    wrong_answer_group_by_idx = {}
    correct_pool_sizes = []
    wrong_pool_sizes = []
    valid_group_count = 0
    for group_indices in groups.values():
        positives = [idx for idx in group_indices if rewards[idx] > 0]
        negatives = [idx for idx in group_indices if rewards[idx] <= 0]
        if not positives or not negatives:
            num_aborted += 1
            all_same_outcome_group_count += 1
            continue

        problem = problems[group_indices[0]]
        positive_contexts = {}
        for idx in positives:
            correct_answer = extract_boxed_answer(responses_text[idx]) or ground_truths[idx]
            positive_contexts[idx] = _materialize_teacher_context(
                self,
                problem=problem,
                answer=correct_answer,
                solution=responses_text[idx])

        negative_contexts = {}
        for idx in negatives:
            wrong_answer = extract_boxed_answer(responses_text[idx])
            if wrong_answer is None:
                continue
            negative_contexts[idx] = _materialize_teacher_context(
                self,
                problem=problem,
                answer=wrong_answer,
                solution=responses_text[idx])

        valid_positive_indices = list(positive_contexts)
        valid_negative_indices = list(negative_contexts)
        if not valid_positive_indices or not valid_negative_indices:
            num_aborted += 1
            continue

        group_valid_count = 0
        for idx in group_indices:
            if idx in positive_contexts:
                positive_idx = _sample_other_rollout(valid_positive_indices, idx)
                if positive_idx is None:
                    self_excluded_sampling_skip_count += 1
                    continue
                correct_context = positive_contexts[positive_idx]
                wrong_idx = random.choice(valid_negative_indices)
                wrong_context = negative_contexts[wrong_idx]
                rlcsd_wrong_pool = list(valid_negative_indices)
            else:
                negative_idx = _sample_other_rollout(valid_negative_indices, idx)
                if negative_idx is None:
                    self_excluded_sampling_skip_count += 1
                    continue
                wrong_context = negative_contexts.get(negative_idx)
                if wrong_context is None:
                    self_negative_missing_boxed_count += 1
                    continue
                wrong_idx = negative_idx
                correct_context = positive_contexts[random.choice(valid_positive_indices)]
                rlcsd_wrong_pool = [c for c in valid_negative_indices if c != idx]

            valid_mask[idx] = True
            correct_prompt_by_idx[idx] = correct_context["prompt"]
            wrong_prompt_by_idx[idx] = wrong_context["prompt"]
            correct_text_by_idx[idx] = correct_context["text"]
            wrong_text_by_idx[idx] = wrong_context["text"]
            correct_answer_by_idx[idx] = correct_context["answer"]
            wrong_answer_by_idx[idx] = wrong_context["answer"]
            if loss_mode == "rlcsd":
                # K = min(rlcsd_k_max, |non-self negatives|). Always include the
                # chosen single-pair wrong sibling first so the marginal mean
                # covers it; fill remaining slots with random other non-self
                # negatives.
                k_max = int(_cfg_get(self, "rlcsd_k_max", RLCSD_DEFAULT_K_MAX) or RLCSD_DEFAULT_K_MAX)
                rest = [c for c in rlcsd_wrong_pool if c != wrong_idx]
                random.shuffle(rest)
                chosen = [wrong_idx] + rest[: max(k_max - 1, 0)]
                wrong_prompt_group_by_idx[idx] = [negative_contexts[c]["prompt"] for c in chosen]
                wrong_text_group_by_idx[idx]   = [negative_contexts[c]["text"]   for c in chosen]
                wrong_answer_group_by_idx[idx] = [negative_contexts[c]["answer"] for c in chosen]
                wrong_pool_sizes.append(len(chosen))
            group_valid_count += 1

        if group_valid_count > 0:
            valid_group_count += 1
        else:
            num_aborted += 1

    num_groups = len(groups)
    valid_sample_count = int(valid_mask.sum())
    step_metrics = {
        "rlcsd/aborted_ratio": float(num_aborted / max(num_groups, 1)),
        "rlcsd/valid_group_ratio": float(valid_group_count / max(num_groups, 1)),
        "rlcsd/valid_group_count": float(valid_group_count),
        "rlcsd/valid_sample_count": float(valid_sample_count),
        "rlcsd/group_count": float(num_groups),
        "rlcsd/all_same_outcome_group_count": float(all_same_outcome_group_count),
        "rlcsd/self_negative_missing_boxed_count": float(self_negative_missing_boxed_count),
        "rlcsd/self_excluded_sampling_skip_count": float(self_excluded_sampling_skip_count),
        "rlcsd/skipped_sample_count": float(batch_size - valid_sample_count),
        "rlcsd/actor_pad_count": 0.0,
        "rlcsd/actor_padded_batch_size": float(valid_sample_count),
        "rlcsd/correct_pool_size_mean": float(np.mean(correct_pool_sizes)) if correct_pool_sizes else 0.0,
        "rlcsd/wrong_pool_size_mean": float(np.mean(wrong_pool_sizes)) if wrong_pool_sizes else 0.0}

    if not valid_mask.any():
        filtered_batch = batch.select_idxs(valid_mask)
        batch.batch = filtered_batch.batch
        batch.non_tensor_batch = filtered_batch.non_tensor_batch
        batch.meta_info["skip_actor_update"] = True
        step_metrics["rlcsd/skip_actor_update"] = 1.0
        self._aligned_teacher_rollout_details = {
            "teacher_correct_prompts": [],
            "teacher_wrong_prompts": [],
            "privileged_texts_correct": [],
            "privileged_texts_wrong": [],
            "correct_answers": [],
            "wrong_answers": []}
        self._rlcsd_step_metrics = step_metrics
        return None

    kept_indices = [idx for idx in range(batch_size) if valid_mask[idx]]
    filtered_batch = batch.select_idxs(valid_mask)
    batch.batch = filtered_batch.batch
    batch.non_tensor_batch = filtered_batch.non_tensor_batch

    if loss_mode == "rlcsd":
        teacher_correct_prompts = [correct_prompt_by_idx[idx] for idx in kept_indices]
        teacher_wrong_prompts = [wrong_prompt_by_idx[idx] for idx in kept_indices]
        privileged_texts_correct = [correct_text_by_idx[idx] for idx in kept_indices]
        privileged_texts_wrong = [wrong_text_by_idx[idx] for idx in kept_indices]
        correct_answers = [correct_answer_by_idx[idx] for idx in kept_indices]
        wrong_answers = [wrong_answer_by_idx[idx] for idx in kept_indices]
        # rlcsd_5 only marginalizes on the wrong side; correct stays single.
        teacher_correct_prompt_groups = []
        privileged_text_groups_correct = []
        correct_answer_groups = []
        teacher_wrong_prompt_groups = [wrong_prompt_group_by_idx[idx] for idx in kept_indices]
        privileged_text_groups_wrong = [wrong_text_group_by_idx[idx] for idx in kept_indices]
        wrong_answer_groups = [wrong_answer_group_by_idx[idx] for idx in kept_indices]
    else:
        teacher_correct_prompts = [correct_prompt_by_idx[idx] for idx in kept_indices]
        teacher_wrong_prompts = [wrong_prompt_by_idx[idx] for idx in kept_indices]
        privileged_texts_correct = [correct_text_by_idx[idx] for idx in kept_indices]
        privileged_texts_wrong = [wrong_text_by_idx[idx] for idx in kept_indices]
        correct_answers = [correct_answer_by_idx[idx] for idx in kept_indices]
        wrong_answers = [wrong_answer_by_idx[idx] for idx in kept_indices]
        teacher_correct_prompt_groups = []
        teacher_wrong_prompt_groups = []
        privileged_text_groups_correct = []
        privileged_text_groups_wrong = []
        correct_answer_groups = []
        wrong_answer_groups = []

    self._aligned_teacher_rollout_details = {
        "teacher_correct_prompts": teacher_correct_prompts,
        "teacher_wrong_prompts": teacher_wrong_prompts,
        "privileged_texts_correct": privileged_texts_correct,
        "privileged_texts_wrong": privileged_texts_wrong,
        "correct_answers": correct_answers,
        "wrong_answers": wrong_answers,
        "teacher_correct_prompt_groups": teacher_correct_prompt_groups,
        "teacher_wrong_prompt_groups": teacher_wrong_prompt_groups,
        "privileged_text_groups_correct": privileged_text_groups_correct,
        "privileged_text_groups_wrong": privileged_text_groups_wrong,
        "correct_answer_groups": correct_answer_groups,
        "wrong_answer_groups": wrong_answer_groups,
        "effective_modes": ["solution_answer"] * len(teacher_correct_prompts)}
    self._rlcsd_step_metrics = step_metrics

    if loss_mode == "rlcsd":
        teacher_correct_pair = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_correct_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_correct")
        teacher_wrong_multi = _build_teacher_multi_batch_from_prompt_groups(
            self,
            teacher_prompt_groups=teacher_wrong_prompt_groups,
            responses=batch.batch["responses"],
            prefix="teacher_wrong_multi")
        return teacher_correct_pair.union(teacher_wrong_multi)
    else:
        teacher_correct = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_correct_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_correct")
        teacher_wrong = _build_teacher_batch_from_prompts(
            self,
            teacher_prompts=teacher_wrong_prompts,
            responses=batch.batch["responses"],
            prefix="teacher_wrong")
        return teacher_correct.union(teacher_wrong)


# ------------------------------------------------------------------ #
# Step 3: Compute teacher log probs (called during training phase)
# ------------------------------------------------------------------ #

def _compute_teacher_log_probs(self, batch: DataProto) -> DataProto | None:
    """Build teacher-side inputs from cached GT privileged prompts."""
    loss_mode = _cfg_get(self, "loss_mode", "vanilla")
    self._data_metric_overrides = {}
    if loss_mode not in DISTILL_LOSS_MODES:
        return None
    if loss_mode in RLCSD_LOSS_MODES:
        return _compute_rlcsd_teacher_inputs(self, batch)

    self._rlcsd_step_metrics = {}

    batch_size = batch.batch.batch_size[0]
    problems, ground_truths, solutions = self._extract_problems_answers_solutions(batch)
    aligned_context = _build_teacher_context_payloads(self, problems, ground_truths, solutions)
    self._aligned_teacher_rollout_details = aligned_context
    teacher_prompts = aligned_context["teacher_prompts"]

    if len(teacher_prompts) != batch_size:
        rollout_n = self.config.actor_rollout_ref.rollout.n
        teacher_prompts = _expand_to_rollout(teacher_prompts, batch_size, rollout_n)

    responses = batch.batch["responses"]
    pad_id = self.tokenizer.pad_token_id or 0

    teacher_prompt_ids_list = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in teacher_prompts]
    teacher_prompt_lens = []
    teacher_prompt_lens_raw = []
    teacher_prompt_tokens_truncated = []
    teacher_prompt_truncated = []
    teacher_total_budget = _teacher_total_token_budget(self)
    for i in range(batch_size):
        raw_prompt_len = len(teacher_prompt_ids_list[i])
        response_ids = [token for token in responses[i].tolist() if token != pad_id]
        max_prompt_tokens = max(teacher_total_budget - len(response_ids), 0)
        kept_prompt_len = min(raw_prompt_len, max_prompt_tokens)
        truncated_tokens = raw_prompt_len - kept_prompt_len
        teacher_prompt_lens.append(kept_prompt_len)
        teacher_prompt_lens_raw.append(raw_prompt_len)
        teacher_prompt_tokens_truncated.append(truncated_tokens)
        teacher_prompt_truncated.append(1.0 if truncated_tokens > 0 else 0.0)

    batch.non_tensor_batch["teacher_prompt_truncated"] = np.asarray(teacher_prompt_truncated, dtype=np.float32)
    batch.non_tensor_batch["teacher_prompt_truncated_tokens"] = np.asarray(
        teacher_prompt_tokens_truncated, dtype=np.float32
    )
    batch.non_tensor_batch["teacher_prompt_tokens_before_trunc"] = np.asarray(
        teacher_prompt_lens_raw, dtype=np.float32
    )
    batch.non_tensor_batch["teacher_prompt_tokens_after_trunc"] = np.asarray(
        teacher_prompt_lens, dtype=np.float32
    )
    return _build_teacher_batch_from_prompts(
        self,
        teacher_prompts=teacher_prompts,
        responses=responses,
        prefix="teacher")


def _judge_rationales(self) -> dict:
    """Return per-step teacher-context metrics when available."""
    metrics = getattr(self, "_rlcsd_step_metrics", {})
    self._rlcsd_step_metrics = {}
    return dict(metrics)


def _get_data_metric_overrides(self) -> dict:
    """Return per-step metric overrides that should win over post-filter stats."""
    metrics = getattr(self, "_data_metric_overrides", {})
    self._data_metric_overrides = {}
    return dict(metrics)


_ORIGINAL_RLHF_BUILD_MESSAGES = None


def _patched_rlhf_build_messages(self, example: dict):
    """Format stored raw problems into official OPSD train/val rollout prompts."""
    messages = _ORIGINAL_RLHF_BUILD_MESSAGES(self, example)

    data_source = str(example.get("data_source", "")).lower()
    if data_source not in MATH_DATA_SOURCES:
        return messages

    extra_info = example.get("extra_info") or {}
    problem = _content_to_text(extra_info.get("problem", "")) if isinstance(extra_info, dict) else ""
    if not problem:
        problem = _raw_prompt_to_problem(messages)

    # Read thinking_system_prompt from data config (self = RLHFDataset).
    use_thinking = False
    cfg = getattr(self, "config", None)
    if cfg is not None:
        val = cfg.get("thinking_system_prompt", False)
        use_thinking = str(val).strip().lower() in ("1", "true", "yes", "on") if isinstance(val, str) else bool(val)
    return build_rollout_messages(problem, data_source=data_source, thinking=use_thinking)


def _save_train_rollout(self, batch: DataProto):
    """Save train rollout details with teacher context aligned to the current batch order."""
    log_dir = self._get_rollout_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"train_step_{self.global_steps}.jsonl")

    batch_size = batch.batch.batch_size[0]
    prompts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
    responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
    score_tensor = batch.batch["token_level_scores"]
    if score_tensor.dim() > 1:
        scores = score_tensor.sum(dim=-1).cpu().tolist()
    else:
        scores = score_tensor.cpu().tolist()

    ground_truths = []
    for i in range(batch_size):
        item = batch[i]
        reward_model = item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
        ground_truths.append(str(ground_truth))

    aligned_context = getattr(self, "_aligned_teacher_rollout_details", None)
    privileged_texts = aligned_context.get("privileged_texts", []) if aligned_context is not None else []
    teacher_prompts = aligned_context.get("teacher_prompts", []) if aligned_context is not None else []
    effective_modes = aligned_context.get("effective_modes", []) if aligned_context is not None else []
    privileged_texts_correct = aligned_context.get("privileged_texts_correct", []) if aligned_context is not None else []
    privileged_texts_wrong = aligned_context.get("privileged_texts_wrong", []) if aligned_context is not None else []
    teacher_correct_prompts = aligned_context.get("teacher_correct_prompts", []) if aligned_context is not None else []
    teacher_wrong_prompts = aligned_context.get("teacher_wrong_prompts", []) if aligned_context is not None else []
    teacher_correct_prompt_groups = aligned_context.get("teacher_correct_prompt_groups", []) if aligned_context is not None else []
    teacher_wrong_prompt_groups = aligned_context.get("teacher_wrong_prompt_groups", []) if aligned_context is not None else []
    privileged_text_groups_correct = aligned_context.get("privileged_text_groups_correct", []) if aligned_context is not None else []
    privileged_text_groups_wrong = aligned_context.get("privileged_text_groups_wrong", []) if aligned_context is not None else []
    correct_answers = aligned_context.get("correct_answers", []) if aligned_context is not None else []
    wrong_answers = aligned_context.get("wrong_answers", []) if aligned_context is not None else []
    correct_answer_groups = aligned_context.get("correct_answer_groups", []) if aligned_context is not None else []
    wrong_answer_groups = aligned_context.get("wrong_answer_groups", []) if aligned_context is not None else []

    lines = []
    for i in range(batch_size):
        entry = {
            "step": self.global_steps,
            "sample_idx": i,
            "prompt": prompts[i],
            "response": responses[i],
            "score": scores[i],
            "ground_truth": ground_truths[i]}
        if i < len(privileged_texts):
            entry["privileged_text"] = privileged_texts[i]
        if i < len(teacher_prompts):
            entry["teacher_prompt"] = teacher_prompts[i]
        if i < len(effective_modes):
            entry["privileged_text_mode"] = effective_modes[i]
        if i < len(privileged_texts_correct):
            entry["privileged_text_correct"] = privileged_texts_correct[i]
        if i < len(privileged_texts_wrong):
            entry["privileged_text_wrong"] = privileged_texts_wrong[i]
        if i < len(teacher_correct_prompts):
            entry["teacher_correct_prompt"] = teacher_correct_prompts[i]
        if i < len(teacher_wrong_prompts):
            entry["teacher_wrong_prompt"] = teacher_wrong_prompts[i]
        if i < len(teacher_correct_prompt_groups):
            entry["teacher_correct_prompt_pool"] = teacher_correct_prompt_groups[i]
        if i < len(teacher_wrong_prompt_groups):
            entry["teacher_wrong_prompt_pool"] = teacher_wrong_prompt_groups[i]
        if i < len(privileged_text_groups_correct):
            entry["privileged_text_correct_pool"] = privileged_text_groups_correct[i]
        if i < len(privileged_text_groups_wrong):
            entry["privileged_text_wrong_pool"] = privileged_text_groups_wrong[i]
        if i < len(correct_answers):
            entry["teacher_correct_answer"] = correct_answers[i]
        if i < len(wrong_answers):
            entry["teacher_wrong_answer"] = wrong_answers[i]
        if i < len(correct_answer_groups):
            entry["teacher_correct_answer_pool"] = correct_answer_groups[i]
        if i < len(wrong_answer_groups):
            entry["teacher_wrong_answer_pool"] = wrong_answer_groups[i]
        lines.append(json.dumps(entry, ensure_ascii=False))

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved train rollout details to {filename} ({batch_size} samples)")


# ------------------------------------------------------------------ #
# Apply monkey patches
# ------------------------------------------------------------------ #

def _apply_patches():
    global _ORIGINAL_RLHF_BUILD_MESSAGES
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if _ORIGINAL_RLHF_BUILD_MESSAGES is None:
        _ORIGINAL_RLHF_BUILD_MESSAGES = RLHFDataset._build_messages
    RLHFDataset._build_messages = _patched_rlhf_build_messages

    RayPPOTrainer._extract_problems_answers_solutions = _extract_problems_answers_solutions
    RayPPOTrainer._generate_rationales = _generate_rationales
    RayPPOTrainer._compute_teacher_log_probs = _compute_teacher_log_probs
    RayPPOTrainer._prepare_actor_update_batch = _prepare_actor_update_batch
    RayPPOTrainer._judge_rationales = _judge_rationales
    RayPPOTrainer._get_data_metric_overrides = _get_data_metric_overrides
    RayPPOTrainer._save_train_rollout = _save_train_rollout


_apply_patches()


def _ensure_legacy_worker_impl(config):
    worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
    if worker_impl == "disable":
        with open_dict(config):
            config.trainer.use_legacy_worker_impl = "enable"
        print(
            "[Self-Distill] Overriding trainer.use_legacy_worker_impl=disable -> enable. "
            "Privileged teacher prompting and full-logit distillation are currently implemented "
            "only in the legacy actor worker path."
        )
    return config


# ------------------------------------------------------------------ #
# Custom TaskRunner that applies patches inside the Ray actor
# ------------------------------------------------------------------ #

from verl.trainer.main_ppo import TaskRunner


class SelfDistillTaskRunner(TaskRunner):
    def run(self, config):
        from src.self_distill_main import _apply_patches as apply_sd_patches

        apply_sd_patches()
        return super().run(config)


if __name__ == "__main__":
    import hydra
    import ray
    from verl.trainer.main_ppo import migrate_legacy_reward_impl, run_ppo
    from verl.utils.device import auto_set_device

    @hydra.main(config_path="../third_party/verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
    def main(config):
        auto_set_device(config)
        config = migrate_legacy_reward_impl(config)
        config = _ensure_legacy_worker_impl(config)
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SelfDistillTaskRunner))

    main()
