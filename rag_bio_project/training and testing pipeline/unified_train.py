from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, Literal
import json
import os
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


TrainObjective = Literal["sft", "dpo", "pg"]


# ======================= 配置 ======================= #

@dataclass
class TrainConfig:
    model_name: str = "gpt2"
    ref_model_name: Optional[str] = None
    trust_remote_code: bool = True

    # 训练目标：sft / dpo / pg
    train_objective: TrainObjective = "sft"

    # 数据路径
    sft_file: str = "data/sft.jsonl"
    dpo_file: str = "data/dpo.jsonl"
    rl_file: str = "data/rl.jsonl"

    # 长度
    max_length: int = 512
    max_prompt_length: int = 256

    # 训练参数
    batch_size: int = 4
    num_epochs: int = 1
    lr: float = 5e-5
    beta: float = 0.1               # DPO 温度
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.03

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 输出
    output_dir: str = "outputs/unified"
    log_every: int = 50
    save_every_epochs: int = 1

    # 微调方式：lora / full
    tuning_strategy: str = "lora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    # 量化加载（仅在 LoRA 时有意义）
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # 随机种子
    seed: int = 42


# ======================= 工具函数 ======================= #

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_api_only_model(name: str) -> bool:
    """
    简单黑名单：这些认为是 API-only，不能在本脚本里直接 finetune
    """
    name = name.lower()
    bad_prefixes = [
        "gpt-3.5",
        "gpt-4",
        "gpt-5",
        "o3-",
        "o4-",
        "chatgpt",
    ]
    return any(name.startswith(p) for p in bad_prefixes)


def assert_model_trainable(cfg: TrainConfig):
    # 1) API-only 基座
    if is_api_only_model(cfg.model_name):
        raise ValueError(
            f"Base model '{cfg.model_name}' is API-only / not directly finetunable in this script."
        )

    # 2) 策略合法
    if cfg.tuning_strategy not in {"lora", "full"}:
        raise ValueError(f"Unknown tuning_strategy: {cfg.tuning_strategy}")

    # 3) 全参 + 8bit/4bit 禁止
    if cfg.tuning_strategy == "full":
        if cfg.load_in_8bit or cfg.load_in_4bit:
            raise ValueError(
                "Full-parameter finetuning is not compatible with 8bit/4bit loading in this script."
            )

    # 4) LoRA 但没装 peft
    if cfg.tuning_strategy == "lora" and not PEFT_AVAILABLE:
        raise ImportError(
            "tuning_strategy is 'lora' but `peft` is not installed. "
            "Install it with `pip install peft` or switch to 'full' finetuning."
        )


# ======================= 数据集定义 ======================= #

class SFTDataset(Dataset):
    """
    行格式：
    - {"prompt": "...", "response": "..."}
    或
    - {"prompt": "...", "chosen": "..."}
    """
    def __init__(self, path: str):
        self.samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class PreferenceDataset(Dataset):
    """
    DPO 数据：
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    def __init__(self, path: str):
        self.samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class RLDataset(Dataset):
    """
    PG 数据：
    至少 {"prompt": "..."}
    """
    def __init__(self, path: str):
        self.samples: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ======================= collate functions ======================= #

def sft_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    max_prompt_length: int,
):
    prompts = [ex.get("prompt", "") for ex in batch]
    responses = [ex.get("response", ex.get("chosen", "")) for ex in batch]

    input_ids, attention_masks, labels = [], [], []

    for p, r in zip(prompts, responses):
        p_ids = tokenizer(
            p,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]

        r_ids = tokenizer(
            r,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - len(p_ids),
        )["input_ids"]

        ids = (p_ids + r_ids)[:max_length]

        # prompt 区域 label = -100
        label = [-100] * min(len(p_ids), len(ids)) + ids[min(len(p_ids), len(ids)):]

        input_ids.append(ids)
        attention_masks.append([1] * len(ids))
        labels.append(label)

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def pad(seqs: List[List[int]], pad_val: int):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [pad_val] * (max_len - len(s)) for s in seqs],
            dtype=torch.long,
        )

    batch_dict = {
        "input_ids": pad(input_ids, pad_token_id),
        "attention_mask": pad(attention_masks, 0),
        "labels": pad(labels, -100),
    }
    return batch_dict


def dpo_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    max_prompt_length: int,
):
    prompts = [ex.get("prompt", "") for ex in batch]
    chosens = [ex.get("chosen", "") for ex in batch]
    rejecteds = [ex.get("rejected", "") for ex in batch]

    chosen_input_ids, chosen_attention_mask, chosen_answer_mask = [], [], []
    rejected_input_ids, rejected_attention_mask, rejected_answer_mask = [], [], []

    for p, yc, yr in zip(prompts, chosens, rejecteds):
        p_ids = tokenizer(
            p,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_length,
        )["input_ids"]

        yc_ids = tokenizer(
            yc,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - len(p_ids),
        )["input_ids"]

        yr_ids = tokenizer(
            yr,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - len(p_ids),
        )["input_ids"]

        c_ids = (p_ids + yc_ids)[:max_length]
        r_ids = (p_ids + yr_ids)[:max_length]

        c_mask_ans = [0] * min(len(p_ids), len(c_ids)) + [1] * max(0, len(c_ids) - len(p_ids))
        r_mask_ans = [0] * min(len(p_ids), len(r_ids)) + [1] * max(0, len(r_ids) - len(p_ids))

        chosen_input_ids.append(c_ids)
        chosen_attention_mask.append([1] * len(c_ids))
        chosen_answer_mask.append(c_mask_ans)

        rejected_input_ids.append(r_ids)
        rejected_attention_mask.append([1] * len(r_ids))
        rejected_answer_mask.append(r_mask_ans)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def pad_long(seqs: List[List[int]], pad_val: int):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [pad_val] * (max_len - len(s)) for s in seqs],
            dtype=torch.long,
        )

    def pad_float(seqs: List[List[int]], pad_val: float):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [pad_val] * (max_len - len(s)) for s in seqs],
            dtype=torch.float32,
        )

    return {
        "chosen_input_ids": pad_long(chosen_input_ids, pad_id),
        "chosen_attention_mask": pad_long(chosen_attention_mask, 0),
        "chosen_answer_mask": pad_float(chosen_answer_mask, 0.0),
        "rejected_input_ids": pad_long(rejected_input_ids, pad_id),
        "rejected_attention_mask": pad_long(rejected_attention_mask, 0),
        "rejected_answer_mask": pad_float(rejected_answer_mask, 0.0),
    }


def rl_collate_fn(batch: List[Dict[str, Any]], tokenizer, max_prompt_length: int):
    prompts = [ex.get("prompt", "") for ex in batch]
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "prompts": prompts,
    }


# ======================= log-prob 计算 ======================= #

def compute_answer_logps(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算答案区域的平均 log p，返回 [batch]
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, L, V]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_answer_mask = answer_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    shift_labels = shift_labels.unsqueeze(-1)
    token_logps = torch.gather(log_probs, dim=-1, index=shift_labels).squeeze(-1)
    token_logps = token_logps * shift_answer_mask

    lengths = shift_answer_mask.sum(dim=-1) + 1e-8
    seq_logps = token_logps.sum(dim=-1) / lengths
    return seq_logps


# ======================= 模型构建 ======================= #

def build_model_and_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}
    if cfg.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    if cfg.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)

    if cfg.tuning_strategy == "lora":
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)

    model.to(cfg.device)
    return model, tokenizer


def build_ref_model(cfg: TrainConfig):
    ref_name = cfg.ref_model_name or cfg.model_name
    model_kwargs: Dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}
    if cfg.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    if cfg.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    ref = AutoModelForCausalLM.from_pretrained(ref_name, **model_kwargs)
    ref.to(cfg.device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


def make_optimizer_and_scheduler(
    cfg: TrainConfig,
    model: torch.nn.Module,
    num_steps_per_epoch: int,
):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.lr)
    t_total = math.ceil(num_steps_per_epoch * cfg.num_epochs)
    warmup_steps = int(cfg.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total,
    )
    return optimizer, scheduler


# ======================= SFT 训练 ======================= #

def train_sft(cfg: TrainConfig):
    dataset = SFTDataset(cfg.sft_file)
    model, tokenizer = build_model_and_tokenizer(cfg)
    collate = lambda b: sft_collate_fn(b, tokenizer, cfg.max_length, cfg.max_prompt_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    num_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model, num_steps_per_epoch)

    global_step = 0
    model.train()
    running_loss = 0.0

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.log_every
                    print(f"[SFT] epoch={epoch+1}/{cfg.num_epochs} step={global_step} loss={avg_loss:.4f}")
                    running_loss = 0.0

        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_dir = os.path.join(cfg.output_dir, f"sft-epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[SFT] saved to {save_dir}")

    final_dir = os.path.join(cfg.output_dir, "sft-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[SFT] final model saved to {final_dir}")


# ======================= DPO 训练 ======================= #

def train_dpo(cfg: TrainConfig):
    dataset = PreferenceDataset(cfg.dpo_file)
    model, tokenizer = build_model_and_tokenizer(cfg)
    ref_model = build_ref_model(cfg)

    collate = lambda b: dpo_collate_fn(b, tokenizer, cfg.max_length, cfg.max_prompt_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    num_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model, num_steps_per_epoch)

    global_step = 0
    model.train()
    running_loss = 0.0

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            policy_chosen_logps = compute_answer_logps(
                model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_answer_mask"],
            )
            policy_rejected_logps = compute_answer_logps(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_answer_mask"],
            )

            with torch.no_grad():
                ref_chosen_logps = compute_answer_logps(
                    ref_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_answer_mask"],
                )
                ref_rejected_logps = compute_answer_logps(
                    ref_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_answer_mask"],
                )

            pi_diff = policy_chosen_logps - policy_rejected_logps
            ref_diff = ref_chosen_logps - ref_rejected_logps
            advantages = cfg.beta * (pi_diff - ref_diff)

            loss = -F.logsigmoid(advantages).mean()
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.log_every
                    print(f"[DPO] epoch={epoch+1}/{cfg.num_epochs} step={global_step} loss={avg_loss:.4f}")
                    running_loss = 0.0

        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_dir = os.path.join(cfg.output_dir, f"dpo-epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[DPO] saved to {save_dir}")

    final_dir = os.path.join(cfg.output_dir, "dpo-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[DPO] final model saved to {final_dir}")


# ======================= 策略梯度 PG 训练 ======================= #

def train_pg(
    cfg: TrainConfig,
    reward_fn: Optional[Callable[[List[str], List[str]], torch.Tensor]] = None,
):
    """
    单模型 REINFORCE：
      - 从 prompt 采样 response
      - 调 reward_fn(prompt, response)
      - loss = - E[ reward * log π(a|s) ]
    """
    if reward_fn is None:
        raise ValueError(
            "train_objective='pg' requires a reward_fn(prompts, responses) -> Tensor; got None."
        )

    dataset = RLDataset(cfg.rl_file)
    model, tokenizer = build_model_and_tokenizer(cfg)

    collate = lambda b: rl_collate_fn(b, tokenizer, cfg.max_prompt_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    num_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model, num_steps_per_epoch)

    global_step = 0
    model.train()
    running_loss = 0.0

    generation_kwargs = dict(
        max_new_tokens=32,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            prompts = batch["prompts"]

            # 1) 采样响应
            with torch.no_grad():
                gen_out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

            responses: List[str] = []
            answer_masks: List[torch.Tensor] = []

            for i in range(gen_out.size(0)):
                prompt_len = int(attention_mask[i].sum().item())
                full_ids = gen_out[i]
                resp_ids = full_ids[prompt_len:]
                text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                responses.append(text)

                ans_mask = torch.zeros_like(full_ids, dtype=torch.float32)
                ans_mask[prompt_len:] = 1.0
                answer_masks.append(ans_mask)

            # 2) reward
            rewards = reward_fn(prompts, responses)
            if not torch.is_tensor(rewards):
                rewards = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)
            rewards = rewards.to(cfg.device)

            # 3) log π(a|s)
            gen_out = gen_out.to(cfg.device)
            answer_masks = torch.stack(answer_masks).to(cfg.device)
            attn_mask_full = torch.ones_like(gen_out, dtype=torch.long, device=cfg.device)

            logps = compute_answer_logps(
                model,
                gen_out,
                attn_mask_full,
                answer_masks,
            )

            loss = -(rewards * logps).mean()
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item() * cfg.gradient_accumulation_steps

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_every == 0:
                    avg_loss = running_loss / cfg.log_every
                    print(f"[PG] epoch={epoch+1}/{cfg.num_epochs} step={global_step} loss={avg_loss:.4f}")
                    running_loss = 0.0

        if (epoch + 1) % cfg.save_every_epochs == 0:
            save_dir = os.path.join(cfg.output_dir, f"pg-epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"[PG] saved to {save_dir}")

    final_dir = os.path.join(cfg.output_dir, "pg-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[PG] final model saved to {final_dir}")


# ======================= 统一入口 ======================= #

def train_unified(
    cfg: TrainConfig,
    reward_fn: Optional[Callable[[List[str], List[str]], torch.Tensor]] = None,
):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    assert_model_trainable(cfg)

    if cfg.train_objective == "sft":
        train_sft(cfg)
    elif cfg.train_objective == "dpo":
        train_dpo(cfg)
    elif cfg.train_objective == "pg":
        train_pg(cfg, reward_fn=reward_fn)
    else:
        raise ValueError(f"Unknown train_objective: {cfg.train_objective}")


if __name__ == "__main__":
    # 简单 demo：SFT tiny-gpt2
    cfg = TrainConfig(
        model_name="sshleifer/tiny-gpt2",
        train_objective="sft",
        sft_file="data/sft.jsonl",
        dpo_file="data/dpo.jsonl",
        rl_file="data/rl.jsonl",
        output_dir="outputs/unified_example",
        tuning_strategy="full",
    )

    def dummy_reward(prompts: List[str], responses: List[str]) -> torch.Tensor:
        vals = [len(r) / 50.0 for r in responses]
        return torch.tensor(vals, dtype=torch.float32)

    if cfg.train_objective == "pg":
        train_unified(cfg, reward_fn=dummy_reward)
    else:
        train_unified(cfg)
