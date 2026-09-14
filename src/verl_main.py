"""
Multi-GPU training with vLLM generation + DDP training via torchrun.

Each GPU runs one process with:
- Its own vLLM instance (TP=1) for fast generation
- DDP-wrapped training model for gradient sync
- Sleep/wake alternation: only one of vLLM or training model on GPU at a time

Usage:
    torchrun --nproc_per_node=8 src/verl_main.py --config configs/verl_opsd_solution_answer.yaml
"""

import gc
import json
import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset as TorchDataset, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

import yaml
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import (
    prepare_training_data, prepare_eval_data,
    extract_answer_from_boxed, check_answer,
)
from src.prompts import (
    STUDENT_SYSTEM_MESSAGE,
    TEACHER_SYSTEM_MESSAGE,
    TEACHER_PROMPT_TEMPLATE_ANSWER_ONLY,
    TEACHER_PROMPT_TEMPLATE_SOLUTION_ANSWER,
)
from src.opsd_format import _answer_instruction
from src.losses import (
    generalized_jsd_loss, sdpo_loss, rlsd_loss,
    compute_entropy, compute_grpo_advantages, compute_rollout_is_weights,
)
from src.models import TeacherContext, EMATeacher
from src.verl_reward import compute_reward


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

@dataclass
class VerlConfig:
    method: str = "opsd"
    model_path: str = "Qwen/Qwen3-1.7B"
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128

    learning_rate: float = 5e-6
    num_epochs: int = 30
    per_device_batch_size: int = 4
    group_size: int = 4
    train_micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    max_train_samples: int = 17000

    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    val_max_completion_length: int = 38912
    temperature: float = 1.1
    top_p: float = 0.95
    top_k_sampling: int = 20
    student_enable_thinking: bool = False
    val_enable_thinking: bool = True
    teacher_enable_thinking: bool = True
    thinking_system_prompt: bool = False

    privileged_text_mode: str = "solution_answer"

    vllm_gpu_memory_utilization: float = 0.6
    vllm_tensor_parallel_size: int = 1

    beta: float = 0.0
    jsd_token_clip: float = 0.05
    teacher_mode: str = "fixed"
    alpha: float = 0.5
    top_k_distill: int = 100
    is_clip: float = 2.0
    ema_decay: float = 0.95
    epsilon: float = 0.2
    epsilon_w: float = 0.2
    lam: float = 0.5
    rollout_is: str = ""
    rollout_is_threshold: str = "2.0"
    rollout_is_batch_normalize: bool = False
    teacher_sync_interval: int = 10

    logging_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 100
    val_before_train: bool = False
    output_dir: str = "./outputs/verl_opsd_solution_answer"
    use_swanlab: bool = True
    project_name: str = "rlcsd"
    experiment_name: str = ""
    data_dir: str = "./data"
    train_dataset: str = "openthoughts_114k_math_filtered"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

class SimpleDataset(TorchDataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def apply_chat(tokenizer, system, user, enable_thinking=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)


def build_student_input(tokenizer, problem, enable_thinking, thinking=False):
    content = f"Problem: {problem}\n\n{_answer_instruction(thinking)}"
    return apply_chat(tokenizer, STUDENT_SYSTEM_MESSAGE, content, enable_thinking=enable_thinking)

def normalize_privileged_text_mode(mode):
    normalized = str(mode or "solution_answer").strip().lower().replace("-", "_")
    if normalized in {"solution+answer", "solution_and_answer"}:
        normalized = "solution_answer"
    if normalized in {"answer", "answer_only"}:
        normalized = "answer_only"
    if normalized not in {"solution_answer", "answer_only"}:
        raise ValueError(
            f"Unsupported privileged_text_mode={mode!r}. "
            "Supported values: solution_answer (alias: solution+answer), answer_only."
        )
    return normalized


def build_teacher_input(tokenizer, problem, answer, solution, privileged_text_mode, enable_thinking, thinking=False):
    mode = normalize_privileged_text_mode(privileged_text_mode)
    if mode == "solution_answer":
        if not solution:
            raise ValueError("privileged_text_mode=solution_answer requires a non-empty GT solution")
        content = TEACHER_PROMPT_TEMPLATE_SOLUTION_ANSWER.format(
            problem=problem,
            answer=answer,
            solution=solution,
        )
    else:
        content = TEACHER_PROMPT_TEMPLATE_ANSWER_ONLY.format(problem=problem, answer=answer)
    if thinking:
        from src.opsd_format import BOXED_ANSWER_INSTRUCTION, THINKING_BOXED_ANSWER_INSTRUCTION
        content = content.replace(BOXED_ANSWER_INSTRUCTION, THINKING_BOXED_ANSWER_INSTRUCTION)
    return apply_chat(tokenizer, TEACHER_SYSTEM_MESSAGE, content, enable_thinking=enable_thinking)


def normalize_rollout_is_mode(mode):
    normalized = str(mode or "").strip().lower()
    if normalized in {"", "none", "null", "false"}:
        return None
    if normalized not in {"token", "sequence"}:
        raise ValueError(
            f"Unsupported rollout_is={mode!r}. Supported values: token, sequence, none."
        )
    return normalized


def cast_config_value(current_value, new_value):
    if isinstance(current_value, bool):
        if isinstance(new_value, bool):
            return new_value
        normalized = str(new_value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value from {new_value!r}")

    target_type = type(current_value)
    if target_type is str:
        return str(new_value)

    try:
        return target_type(new_value)
    except (ValueError, TypeError):
        return new_value


# ------------------------------------------------------------------ #
# Trainer
# ------------------------------------------------------------------ #

class VerlTrainer:
    def __init__(self, cfg: VerlConfig):
        self.cfg = cfg
        self.global_step = 0
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.rank == 0
        self.device = torch.device(f"cuda:{self.local_rank}")
        # NOTE: don't call torch.cuda.set_device here — it initializes CUDA
        # which prevents vLLM from forking. Set device after vLLM is created.

        self._run_timestamp = time.strftime('%Y%m%d_%H%M%S')
        self._run_dir = os.path.join(cfg.output_dir, self._run_timestamp)

        self.log_file = None
        self.swanlab_run = None
        if self.is_main:
            os.makedirs(self._run_dir, exist_ok=True)
            self.log_file = open(os.path.join(self._run_dir, "train.log"), "a")
            self.log(f"Config: {vars(cfg)}")
            self.log(f"World size: {self.world_size}, Rank: {self.rank}")

            if cfg.use_swanlab:
                try:
                    import swanlab
                    swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY", ""))
                    self.swanlab_run = swanlab.init(
                        project=cfg.project_name,
                        experiment_name=cfg.experiment_name or f"{cfg.method}_{self._run_timestamp}",
                        config=vars(cfg),
                    )
                except Exception as e:
                    self.log(f"Warning: SwanLab init failed: {e}")

    def log(self, msg):
        if self.is_main:
            print(msg)
            if self.log_file:
                self.log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
                self.log_file.flush()

    def log_metrics(self, metrics, step=None):
        if self.swanlab_run:
            import swanlab
            swanlab.log(metrics, step=step or self.global_step)

    def _save_rollout(self, problems, answers, privileged_texts, responses, rewards, epoch, batch_idx):
        if not self.is_main:
            return
        rollout_dir = os.path.join(self._run_dir, "rollouts")
        os.makedirs(rollout_dir, exist_ok=True)
        path = os.path.join(rollout_dir, f"epoch{epoch}.jsonl")
        gs = self.cfg.group_size
        for i in range(0, len(problems), gs):
            entry = {
                "step": self.global_step, "batch_idx": batch_idx,
                "problem": problems[i], "answer": answers[i],
                "privileged_texts": [privileged_texts[i]],
                "samples": [
                    {"response": responses[i+j], "reward": rewards[i+j]}
                    for j in range(gs) if i+j < len(responses)
                ],
            }
            with open(path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _save_checkpoint(self, model, tokenizer, final=False):
        if not self.is_main:
            return
        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self._run_dir, f"checkpoint-{suffix}")
        os.makedirs(path, exist_ok=True)
        unwrapped = model.module if isinstance(model, DDP) else model
        unwrapped.save_pretrained(path)
        tokenizer.save_pretrained(path)
        with open(os.path.join(path, "training_state.json"), "w") as f:
            json.dump({"global_step": self.global_step}, f)
        self.log(f"Checkpoint saved: {path}")

    # ------------------------------------------------------------------ #
    # vLLM generation (per-GPU instance)
    # ------------------------------------------------------------------ #

    def _create_vllm(self):
        """Create a vLLM instance on the local GPU.

        Must isolate vLLM from torchrun's distributed env to prevent
        vLLM subprocesses from trying to join the training process group.
        """
        from vllm import LLM

        # Save and clear torchrun distributed env vars
        # so vLLM subprocesses don't try to join the training process group
        dist_env_keys = [
            "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",
            "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
            "TORCHELASTIC_RUN_ID", "OMP_NUM_THREADS",
        ]
        saved_env = {}
        for key in dist_env_keys:
            if key in os.environ:
                saved_env[key] = os.environ.pop(key)

        # Pin vLLM to local GPU via CUDA_VISIBLE_DEVICES
        # vLLM will see it as GPU 0, but that's fine since it runs isolated
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.local_rank)

        llm = LLM(
            model=self.cfg.model_path,
            gpu_memory_utilization=self.cfg.vllm_gpu_memory_utilization,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max(
                self.cfg.max_prompt_length + self.cfg.max_completion_length,
                self.cfg.val_max_completion_length,
            ),
            enforce_eager=True,
        )

        # Restore env vars (vLLM is initialized, its subprocesses already spawned)
        del os.environ["CUDA_VISIBLE_DEVICES"]
        for key, val in saved_env.items():
            os.environ[key] = val

        return llm

    def _vllm_generate(self, llm, prompts, max_tokens, temperature=1.0,
                       top_p=0.95, top_k=20, return_logprobs=False):
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
            logprobs=1 if return_logprobs else None,
        )
        outputs = llm.generate(prompts, params)
        if not return_logprobs:
            return [out.outputs[0].text for out in outputs]
        texts = []
        all_logprobs = []
        for out in outputs:
            texts.append(out.outputs[0].text)
            token_logprobs = []
            if out.outputs[0].logprobs:
                for step_logprobs in out.outputs[0].logprobs:
                    # step_logprobs is a dict {token_id: Logprob}; take the sampled token's logprob
                    if step_logprobs:
                        token_logprobs.append(next(iter(step_logprobs.values())).logprob)
                    else:
                        token_logprobs.append(0.0)
            all_logprobs.append(token_logprobs)
        return texts, all_logprobs

    def _extract_merged_weights(self, model):
        """Extract merged (base + LoRA) weights."""
        unwrapped = model.module if isinstance(model, DDP) else model
        if isinstance(unwrapped, PeftModel):
            unwrapped.merge_adapter()
            base = unwrapped.base_model.model
            state_dict = {
                name: param.data.cpu().clone()
                for name, param in base.named_parameters()
            }
            unwrapped.unmerge_adapter()
        else:
            state_dict = {
                name: param.data.cpu().clone()
                for name, param in unwrapped.named_parameters()
            }
        return state_dict

    def _sync_weights_to_vllm(self, llm, weights):
        """Push updated weights to vLLM."""
        try:
            def _load_fn(worker, weights):
                model = worker.model_runner.model
                for name, param in model.named_parameters():
                    if name in weights:
                        param.data.copy_(weights[name].to(param.device))
            llm.collective_rpc(_load_fn, args=(weights,))
        except Exception as e:
            self.log(f"Warning: vLLM weight sync failed: {e}, using stale weights")

    # ------------------------------------------------------------------ #
    # Training micro-step
    # ------------------------------------------------------------------ #

    def _pad_rollout_logprobs(self, rollout_logprobs, max_len, device):
        """Pad variable-length rollout logprobs to (B, max_len) tensor."""
        B = len(rollout_logprobs)
        padded = torch.zeros(B, max_len, device=device)
        for i, lps in enumerate(rollout_logprobs):
            L = min(len(lps), max_len)
            if L > 0:
                padded[i, :L] = torch.tensor(lps[:L], device=device)
        return padded

    def _compute_old_logprobs(self, model, tokenizer, student_prompts, responses, cfg, micro_batch_size):
        """Cache old-policy token log probs under the training engine before any updates."""
        device = self.device
        cached = []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for mb_start in range(0, len(responses), micro_batch_size):
                mb_end = min(mb_start + micro_batch_size, len(responses))
                mb_student_prompts = student_prompts[mb_start:mb_end]
                mb_responses = responses[mb_start:mb_end]

                s_enc = tokenizer(
                    mb_student_prompts, padding=True, truncation=True,
                    max_length=cfg.max_prompt_length, return_tensors="pt",
                ).to(device)
                r_enc = tokenizer(
                    mb_responses, padding=True, truncation=True,
                    max_length=cfg.max_completion_length, return_tensors="pt",
                ).to(device)

                s_prompt_len = s_enc["input_ids"].shape[1]
                resp_ids = r_enc["input_ids"]
                resp_mask = (resp_ids != tokenizer.pad_token_id).long()
                R = resp_ids.shape[1]

                s_full = torch.cat([s_enc["input_ids"], resp_ids], dim=1)
                s_mask = torch.cat([s_enc["attention_mask"], resp_mask], dim=1)

                s_out = model(input_ids=s_full, attention_mask=s_mask)
                s_logits = s_out.logits[:, s_prompt_len - 1: s_prompt_len - 1 + R, :]
                s_lp_all = F.log_softmax(s_logits, dim=-1)
                token_lp = torch.gather(s_lp_all, -1, resp_ids.unsqueeze(-1)).squeeze(-1)

                for i in range(token_lp.shape[0]):
                    valid = resp_mask[i].bool()
                    cached.append(token_lp[i][valid].cpu().tolist())

                del s_out, s_logits, s_lp_all

        if was_training:
            model.train()
        return cached

    def _training_micro_step(self, model, tokenizer, teacher_ctx,
                             mb_student_prompts, mb_teacher_prompts,
                             mb_responses, mb_advantages, cfg,
                             mb_rollout_logprobs=None, mb_old_logprobs=None):
        """Forward+backward for one micro-batch."""
        device = self.device

        s_enc = tokenizer(
            mb_student_prompts, padding=True, truncation=True,
            max_length=cfg.max_prompt_length, return_tensors="pt",
        ).to(device)
        t_enc = tokenizer(
            mb_teacher_prompts, padding=True, truncation=True,
            max_length=cfg.max_prompt_length, return_tensors="pt",
        ).to(device)
        r_enc = tokenizer(
            mb_responses, padding=True, truncation=True,
            max_length=cfg.max_completion_length, return_tensors="pt",
        ).to(device)

        s_prompt_len = s_enc["input_ids"].shape[1]
        t_prompt_len = t_enc["input_ids"].shape[1]
        resp_ids = r_enc["input_ids"]
        resp_mask = (resp_ids != tokenizer.pad_token_id).long()
        R = resp_ids.shape[1]

        if resp_mask.sum() == 0:
            return None

        s_full = torch.cat([s_enc["input_ids"], resp_ids], dim=1)
        s_mask = torch.cat([s_enc["attention_mask"], resp_mask], dim=1)
        t_full = torch.cat([t_enc["input_ids"], resp_ids], dim=1)
        t_mask = torch.cat([t_enc["attention_mask"], resp_mask], dim=1)

        # Student forward (with grad, through DDP)
        s_out = model(input_ids=s_full, attention_mask=s_mask)
        s_logits = s_out.logits[:, s_prompt_len - 1: s_prompt_len - 1 + R, :]

        rollout_is_mode = normalize_rollout_is_mode(cfg.rollout_is)

        # Old log probs: prefer cached training-engine old policy, then rollout engine, then current detach.
        if mb_old_logprobs is not None:
            old_lp = self._pad_rollout_logprobs(mb_old_logprobs, R, device)
        elif mb_rollout_logprobs is not None:
            old_lp = self._pad_rollout_logprobs(mb_rollout_logprobs, R, device)
        else:
            s_lp_all = F.log_softmax(s_logits, dim=-1)
            old_lp = torch.gather(
                s_lp_all, -1, resp_ids.unsqueeze(-1)
            ).squeeze(-1).detach()

        rollout_is_weights = None
        rollout_is_metrics = {}
        if rollout_is_mode is not None and mb_old_logprobs is not None and mb_rollout_logprobs is not None:
            rollout_lp = self._pad_rollout_logprobs(mb_rollout_logprobs, R, device)
            rollout_is_weights, rollout_is_metrics = compute_rollout_is_weights(
                old_log_probs=old_lp,
                rollout_log_probs=rollout_lp,
                mask=resp_mask,
                rollout_is=rollout_is_mode,
                rollout_is_threshold=cfg.rollout_is_threshold,
                rollout_is_batch_normalize=cfg.rollout_is_batch_normalize,
            )

        # Teacher forward (no grad, unwrapped model)
        unwrapped = model.module if isinstance(model, DDP) else model
        with torch.no_grad(), teacher_ctx:
            t_out = unwrapped(input_ids=t_full, attention_mask=t_mask)
        t_logits = t_out.logits[:, t_prompt_len - 1: t_prompt_len - 1 + R, :]

        # Loss
        if cfg.method == "opsd":
            loss = generalized_jsd_loss(
                s_logits, t_logits, resp_mask,
                beta=cfg.beta, temperature=cfg.temperature,
                jsd_token_clip=cfg.jsd_token_clip,
            )
        elif cfg.method == "sdpo":
            dist_mask = torch.ones(s_logits.shape[0], device=device)
            loss = sdpo_loss(
                s_logits, t_logits, old_lp, resp_mask, dist_mask,
                alpha=cfg.alpha, top_k=cfg.top_k_distill,
                is_clip=cfg.is_clip, temperature=cfg.temperature,
            )
        elif cfg.method == "rlsd":
            loss = rlsd_loss(
                s_logits, t_logits, old_lp, resp_ids, resp_mask,
                mb_advantages.to(device),
                epsilon=cfg.epsilon, epsilon_w=cfg.epsilon_w, lam=cfg.lam,
                rollout_is_weights=rollout_is_weights,
            )
        else:
            raise ValueError(f"Unknown method: {cfg.method}")

        with torch.no_grad():
            ent = compute_entropy(s_logits, resp_mask).item()
            t_ent = compute_entropy(t_logits, resp_mask).item()
            resp_len = resp_mask.sum(dim=1).float().mean().item()

        del s_out, t_out, s_logits, t_logits
        result = {"loss": loss, "entropy": ent, "teacher_entropy": t_ent,
                  "avg_response_length": resp_len}
        result.update(rollout_is_metrics)
        return result

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def train(self):
        cfg = self.cfg

        # --- Tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, trust_remote_code=True, padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # --- vLLM FIRST (before CUDA/NCCL init to avoid fork issues) ---
        self.log("Initializing vLLM...")
        llm = self._create_vllm()
        self.log("vLLM initialized")

        # --- Init distributed (after vLLM, since NCCL init touches CUDA) ---
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend="nccl")
        self.log(f"Distributed initialized: rank={self.rank}/{self.world_size}")

        # --- Training model (CPU first, will move to GPU after vLLM sleeps) ---
        self.log("Loading training model to CPU...")
        # Only rank 0 loads first to avoid disk contention
        if self.rank == 0:
            train_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", trust_remote_code=True,
            )
        dist.barrier()
        if self.rank != 0:
            train_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", trust_remote_code=True,
            )
        dist.barrier()

        if cfg.use_lora:
            lora_config = LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            )
            train_model = get_peft_model(train_model, lora_config)
            if self.is_main:
                train_model.print_trainable_parameters()
        train_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.log("Training model loaded (CPU)")

        # Teacher
        ema_teacher = None
        if cfg.teacher_mode == "ema":
            ema_teacher = EMATeacher(train_model, decay=cfg.ema_decay)
        elif cfg.teacher_mode == "snapshot":
            ema_teacher = EMATeacher(train_model, decay=1.0)

        # --- Data ---
        self.log("Loading data...")
        train_data = prepare_training_data(cfg.data_dir, dataset_name=cfg.train_dataset)
        if cfg.max_train_samples > 0 and len(train_data) > cfg.max_train_samples:
            random.seed(42)
            train_data = random.sample(train_data, cfg.max_train_samples)
        self.log(f"Training data: {len(train_data)} problems")

        eval_data = None
        if self.is_main:
            eval_data = prepare_eval_data(cfg.data_dir)
            for name, data in eval_data.items():
                self.log(f"  eval/{name}: {len(data)} problems")

        dataset = SimpleDataset(train_data)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=cfg.per_device_batch_size,
            sampler=sampler, collate_fn=lambda x: x,
        )

        # --- Optimizer & Scheduler ---
        optimizer = torch.optim.AdamW(
            train_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        total_steps = len(dataloader) * cfg.num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
        )

        mbs = cfg.train_micro_batch_size
        grad_accum = cfg.gradient_accumulation_steps

        self.log(f"Training: method={cfg.method}, epochs={cfg.num_epochs}, "
                 f"bs={cfg.per_device_batch_size}, group_size={cfg.group_size}, "
                 f"micro_bs={mbs}, grad_accum={grad_accum}, "
                 f"lr={cfg.learning_rate}, total_steps={total_steps}")

        # --- Validation before training ---
        if cfg.val_before_train and eval_data and self.is_main:
            self.log("Running validation before training...")
            eval_metrics = self._evaluate_vllm(llm, tokenizer, eval_data)
            self.log_metrics(eval_metrics, step=0)
            self.log(f"[Eval Step 0 (pre-train)] {eval_metrics}")

        # --- Training loop ---
        for epoch in range(cfg.num_epochs):
            sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, batch in enumerate(dataloader):
                problems = [item["problem"] for item in batch]
                answers = [item["answer"] for item in batch]
                solutions = [item.get("solution", "") for item in batch]
                privileged_text_mode = normalize_privileged_text_mode(cfg.privileged_text_mode)

                if privileged_text_mode == "solution_answer":
                    missing_indices = [i for i, solution in enumerate(solutions) if not str(solution).strip()]
                    if missing_indices:
                        preview = ", ".join(str(i) for i in missing_indices[:8])
                        if len(missing_indices) > 8:
                            preview = f"{preview}, ..."
                        raise ValueError(
                            "privileged_text_mode=solution_answer requires every sample to have a non-empty GT "
                            f"solution, but {len(missing_indices)}/{len(solutions)} samples in the current batch "
                            f"are missing it (batch indices: {preview})."
                        )

                # ============================================
                # PHASE 1: ROLLOUT (vLLM on each GPU)
                # ============================================

                # 1a. Generate student responses
                use_thinking = cfg.thinking_system_prompt
                all_student_prompts, all_problems, all_answers, all_solutions = [], [], [], []
                for p, a, s in zip(problems, answers, solutions):
                    for _ in range(cfg.group_size):
                        all_student_prompts.append(
                            build_student_input(tokenizer, p, enable_thinking=cfg.student_enable_thinking, thinking=use_thinking)
                        )
                        all_problems.append(p)
                        all_answers.append(a)
                        all_solutions.append(s)

                need_logprobs = cfg.method == "rlsd"
                if need_logprobs:
                    responses, rollout_logprobs = self._vllm_generate(
                        llm, all_student_prompts, cfg.max_completion_length,
                        temperature=cfg.temperature, top_p=cfg.top_p,
                        top_k=cfg.top_k_sampling, return_logprobs=True,
                    )
                else:
                    responses = self._vllm_generate(
                        llm, all_student_prompts, cfg.max_completion_length,
                        temperature=cfg.temperature, top_p=cfg.top_p,
                        top_k=cfg.top_k_sampling,
                    )
                    rollout_logprobs = None

                # 1b. Rewards
                rewards = compute_reward(responses, all_answers)
                if isinstance(rewards, list):
                    rewards = torch.tensor(rewards)

                # 1c. Save rollout
                if self.global_step % cfg.logging_steps == 0:
                    self._save_rollout(
                        all_problems, all_answers, all_solutions,
                        responses, rewards.tolist(), epoch, batch_idx,
                    )

                # 1d. Teacher prompts
                all_teacher_prompts = [
                    build_teacher_input(
                        tokenizer, p, a, s, privileged_text_mode,
                        enable_thinking=cfg.teacher_enable_thinking,
                        thinking=use_thinking,
                    )
                    for p, a, s in zip(all_problems, all_answers, all_solutions)
                ]

                # ============================================
                # PHASE 2: SWITCH — vLLM sleep, training model to GPU
                # ============================================
                self.log("Sleeping vLLM...")
                llm.sleep(level=2)
                gc.collect()
                torch.cuda.empty_cache()
                self.log("vLLM asleep")

                train_model = train_model.to(self.device)

                # Wrap with DDP (first time or after CPU offload)
                ddp_model = DDP(train_model, device_ids=[self.local_rank])
                self.log("Training model on GPU (DDP)")

                # ============================================
                # PHASE 3: TRAINING
                # ============================================
                ddp_model.train()
                unwrapped = ddp_model.module
                teacher_ctx = TeacherContext(unwrapped, cfg.teacher_mode, ema_teacher)

                cached_old_logprobs = None
                if rollout_logprobs is not None:
                    cached_old_logprobs = self._compute_old_logprobs(
                        ddp_model, tokenizer, all_student_prompts, responses, cfg, mbs
                    )

                advantages = None
                if cfg.method == "rlsd":
                    advantages = compute_grpo_advantages(rewards, group_size=cfg.group_size)

                B = len(responses)
                micro_step = 0
                agg_metrics = {}
                num_updates = 0

                for mb_start in range(0, B, mbs):
                    mb_end = min(mb_start + mbs, B)
                    sl = slice(mb_start, mb_end)

                    mb_rlp = rollout_logprobs[mb_start:mb_end] if rollout_logprobs is not None else None
                    mb_old_lp = cached_old_logprobs[mb_start:mb_end] if cached_old_logprobs is not None else None

                    result = self._training_micro_step(
                        ddp_model, tokenizer, teacher_ctx,
                        mb_student_prompts=all_student_prompts[sl],
                        mb_teacher_prompts=all_teacher_prompts[sl],
                        mb_responses=responses[sl],
                        mb_advantages=advantages[sl] if advantages is not None else None,
                        cfg=cfg,
                        mb_rollout_logprobs=mb_rlp,
                        mb_old_logprobs=mb_old_lp,
                    )

                    if result is None:
                        continue

                    loss = result["loss"]
                    scaled = loss / grad_accum
                    scaled.backward()

                    for k, v in result.items():
                        if k != "loss":
                            agg_metrics[k] = agg_metrics.get(k, 0.0) + v
                    agg_metrics["loss"] = agg_metrics.get("loss", 0.0) + loss.item()

                    del loss, scaled, result
                    gc.collect()
                    torch.cuda.empty_cache()

                    micro_step += 1

                    if micro_step % grad_accum == 0:
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        num_updates += 1
                        if cfg.teacher_mode == "ema" and ema_teacher is not None:
                            ema_teacher.update(unwrapped)

                # Flush remaining
                if micro_step % grad_accum != 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    num_updates += 1
                    if cfg.teacher_mode == "ema" and ema_teacher is not None:
                        ema_teacher.update(unwrapped)

                self.global_step += num_updates

                if (cfg.teacher_mode == "snapshot" and ema_teacher is not None
                    and self.global_step % cfg.teacher_sync_interval == 0):
                    ema_teacher._init_shadow(unwrapped)

                # Metrics
                n_micro = max(micro_step, 1)
                for k in agg_metrics:
                    agg_metrics[k] /= n_micro
                agg_metrics["avg_reward"] = rewards.mean().item()
                agg_metrics["correct_rate"] = (rewards > 0).float().mean().item()

                epoch_loss += agg_metrics.get("loss", 0.0)
                epoch_steps += 1

                if self.global_step % cfg.logging_steps == 0:
                    agg_metrics["epoch"] = epoch
                    agg_metrics["lr"] = scheduler.get_last_lr()[0]
                    self.log_metrics(agg_metrics)
                    self.log(
                        f"[Step {self.global_step}] "
                        f"loss={agg_metrics.get('loss',0):.4f} "
                        f"reward={agg_metrics['avg_reward']:.3f} "
                        f"correct={agg_metrics['correct_rate']:.3f} "
                        f"resp_len={agg_metrics.get('avg_response_length',0):.0f} "
                        f"entropy={agg_metrics.get('entropy',0):.3f} "
                        f"teacher_entropy={agg_metrics.get('teacher_entropy',0):.3f}"
                    )

                if self.global_step % cfg.save_steps == 0:
                    self._save_checkpoint(ddp_model, tokenizer)

                # ============================================
                # PHASE 4: SWITCH — training model to CPU, wake vLLM
                # ============================================
                self.log("Syncing weights to vLLM...")
                updated_weights = self._extract_merged_weights(ddp_model)

                # Unwrap DDP before moving to CPU
                del ddp_model
                train_model = train_model.cpu()
                gc.collect()
                torch.cuda.empty_cache()

                llm.wake_up()
                self._sync_weights_to_vllm(llm, updated_weights)
                del updated_weights
                gc.collect()
                self.log("vLLM awake with updated weights")

                # Evaluation (vLLM is awake, only main rank)
                if self.global_step % cfg.eval_steps == 0 and eval_data and self.is_main:
                    eval_metrics = self._evaluate_vllm(llm, tokenizer, eval_data)
                    self.log_metrics(eval_metrics)
                    self.log(f"[Eval Step {self.global_step}] {eval_metrics}")

                dist.barrier()

            avg = epoch_loss / max(epoch_steps, 1)
            self.log(f"Epoch {epoch + 1}/{cfg.num_epochs} -- avg_loss: {avg:.4f}")

        # Final checkpoint
        llm.sleep(level=2)
        gc.collect(); torch.cuda.empty_cache()
        train_model = train_model.to(self.device)
        self._save_checkpoint(train_model, tokenizer, final=True)
        train_model = train_model.cpu()

        if self.swanlab_run:
            import swanlab
            swanlab.finish()
        dist.destroy_process_group()

    def _evaluate_vllm(self, llm, tokenizer, eval_data):
        results = {}
        for name, data in eval_data.items():
            use_thinking = self.cfg.thinking_system_prompt
            prompts = [
                build_student_input(tokenizer, d["problem"], enable_thinking=self.cfg.val_enable_thinking, thinking=use_thinking)
                for d in data
            ]
            responses = self._vllm_generate(
                llm, prompts, self.cfg.val_max_completion_length,
                temperature=0.0, top_p=1.0,
            )
            correct = total = 0
            for resp, item in zip(responses, data):
                pred = extract_answer_from_boxed(resp)
                ok = check_answer(pred, item["answer"]) if pred else False
                correct += int(ok)
                total += 1
            results[f"eval/{name}/accuracy"] = correct / max(total, 1)
        return results


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, unknown = parser.parse_known_args()

    cli_overrides = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise ValueError(f"Unrecognized CLI token {token!r}. Expected --key value pairs.")
        if i + 1 >= len(unknown):
            raise ValueError(f"Missing value for CLI override {token!r}.")
        key = token[2:].replace("-", "_")
        value = unknown[i + 1]
        cli_overrides[key] = value
        i += 2

    return args, cli_overrides


def main():
    args, cli_overrides = parse_args()
    cfg = VerlConfig()
    with open(args.config) as f:
        yaml_config = yaml.safe_load(f) or {}
    for key, value in yaml_config.items():
        if hasattr(cfg, key):
            try:
                value = cast_config_value(getattr(cfg, key), value)
            except ValueError:
                pass
            setattr(cfg, key, value)
    for key, value in cli_overrides.items():
        if hasattr(cfg, key):
            try:
                value = cast_config_value(getattr(cfg, key), value)
            except ValueError:
                pass
            setattr(cfg, key, value)

    trainer = VerlTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
