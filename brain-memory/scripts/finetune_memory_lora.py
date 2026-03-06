"""Fine-tune local LLM with LoRA to interpret memory injection vectors.

Anti-overfitting measures:
- Validation split with early stopping
- Lower learning rate + weight decay
- Smaller LoRA rank (8 instead of 16)
- Higher dropout (0.1)
- Gradient accumulation for smoother updates
- Print both train and val loss each epoch

Usage::

    python -m scripts.finetune_memory_lora [data_path] [model_name] [output_dir] [max_epochs]
    python -m scripts.finetune_memory_lora data/memory_training_data.json Qwen/Qwen2.5-3B-Instruct checkpoints/lora_memory 30
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class MemoryInjectionDataset(Dataset):
    """Dataset of (question_tokens, memory_vector, answer_tokens) triples."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 256) -> None:
        with open(data_path) as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        messages = [{"role": "user", "content": ex["question"]}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            input_text = f"<|user|>\n{ex['question']}\n<|assistant|>\n"

        full_text = input_text + ex["answer"]

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_length = input_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:input_length] = -100  # Only train on the answer portion

        memory_vector = torch.tensor(ex["memory_vector"], dtype=torch.float32)
        energy = torch.tensor(ex["energy"], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "memory_vector": memory_vector,
            "energy": energy,
        }


def _get_layer(model, idx: int):
    """Get the transformer layer module by index, handling PEFT wrapping."""
    if hasattr(model, "model") and hasattr(model.model, "model"):
        # PEFT wraps: model.model.model.layers[i]
        inner = model.model.model
        if hasattr(inner, "layers"):
            return inner.layers[idx]
    if hasattr(model, "base_model"):
        inner = model.base_model.model
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            return inner.model.layers[idx]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[idx]
    raise ValueError("Cannot find transformer layers")


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    memory_vector_holder: list,
    projection,
) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)
        mem_vec = batch["memory_vector"].to(model.device)
        energy = batch["energy"].to(model.device)

        batch_projected = []
        for b in range(mem_vec.shape[0]):
            p = projection(mem_vec[b], energy[b])
            batch_projected.append(F.normalize(p, dim=0))
        memory_vector_holder[0] = torch.stack(batch_projected, dim=0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        num_batches += 1
        memory_vector_holder[0] = None

    model.train()
    return total_loss / max(num_batches, 1)


def main() -> None:
    # Force line-buffered stdout so log files update in real time
    sys.stdout.reconfigure(line_buffering=True)

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/memory_training_data.json"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-3B-Instruct"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/lora_memory"
    max_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    val_path = data_path.replace(".json", "_val.json")

    print(f"Fine-tuning {model_name} with LoRA (anti-overfit)")
    print(f"Train: {data_path}")
    print(f"Val:   {val_path}")
    print(f"Max epochs: {max_epochs}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    config = model.config
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    injection_layer = num_layers // 2

    print(f"Model: {num_layers} layers, hidden={hidden_dim}, inject@{injection_layer}")

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    # Only target layers AFTER injection point
    target_modules = []
    for i in range(injection_layer, num_layers):
        target_modules.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj",
        ])

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,              # Smaller rank = less memorization capacity
        lora_alpha=16,
        lora_dropout=0.1,  # Higher dropout = more regularization
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Projection (frozen) ---
    from memory.neural_bridge import MemoryProjection

    with open(data_path) as f:
        sample = json.load(f)
    brain_dim = len(sample[0]["memory_vector"])

    projection = MemoryProjection(
        brain_dim=brain_dim, llm_dim=hidden_dim,
    ).to(model.device)

    proj_path = Path("checkpoints/projection/projection.pt")
    if proj_path.exists():
        projection.load_state_dict(
            torch.load(proj_path, map_location=model.device),
        )
        print(f"Loaded projection (brain_dim={brain_dim})")

    for p in projection.parameters():
        p.requires_grad = False

    # --- Injection hook ---
    memory_vector_holder: list[torch.Tensor | None] = [None]
    injection_scale = 0.3

    def injection_hook(module, input, output):
        if memory_vector_holder[0] is None:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        mem = memory_vector_holder[0].to(
            hidden_states.device, dtype=hidden_states.dtype,
        )
        # mem shape: [batch, llm_dim] — each sample has its own memory vector
        hidden_norm = hidden_states[:, -1, :].norm(dim=-1, keepdim=True)  # [batch, 1]

        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] += mem * hidden_norm * injection_scale

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    layer = _get_layer(model, injection_layer)
    hook_handle = layer.register_forward_hook(injection_hook)

    # --- Datasets ---
    train_dataset = MemoryInjectionDataset(data_path, tokenizer)
    train_batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    val_loader = None
    if Path(val_path).exists():
        val_dataset = MemoryInjectionDataset(val_path, tokenizer)
        val_loader = DataLoader(
            val_dataset, batch_size=4, shuffle=False,
            num_workers=0, pin_memory=True,
        )
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        print(f"Train: {len(train_dataset)}, No val set found")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=0.05,
    )

    # --- Training with early stopping ---
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    grad_accum_steps = 2  # Effective batch size = 4 * 2 = 8

    model.train()

    for epoch in range(max_epochs):
        total_train_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            mem_vec = batch["memory_vector"].to(model.device)
            energy = batch["energy"].to(model.device)

            with torch.no_grad():
                # Project each sample's memory vector individually then stack
                batch_projected = []
                for b in range(mem_vec.shape[0]):
                    p = projection(mem_vec[b], energy[b])
                    batch_projected.append(F.normalize(p, dim=0))
                projected = torch.stack(batch_projected, dim=0)  # [batch, llm_dim]

            memory_vector_holder[0] = projected

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += outputs.loss.item()
            num_batches += 1
            memory_vector_holder[0] = None

            if (step + 1) % 50 == 0:
                running_avg = total_train_loss / num_batches
                print(
                    f"    step {step + 1}/{len(train_loader)}  "
                    f"loss={running_avg:.4f}",
                    flush=True,
                )

        # Flush leftover gradients
        if num_batches % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train = total_train_loss / max(num_batches, 1)

        # --- Validation + early stopping ---
        if val_loader is not None:
            avg_val = evaluate(model, val_loader, memory_vector_holder, projection)
            print(
                f"  Epoch {epoch + 1}/{max_epochs}: "
                f"train={avg_train:.4f}  val={avg_val:.4f}  best={best_val_loss:.4f}"
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"  Early stopping at epoch {epoch + 1} "
                        f"(val loss didn't improve for {patience} epochs)"
                    )
                    break
        else:
            print(f"  Epoch {epoch + 1}/{max_epochs}: train={avg_train:.4f}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    hook_handle.remove()

    # Final save if no val set was used
    if val_loader is None:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print(f"\nLoRA weights saved to {output_dir}")
    if val_loader is not None:
        print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
