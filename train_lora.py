#!/usr/bin/env python3
"""LoRA SFT for NL->shell-command generation, via TRL's SFTTrainer.

Design choices, and why (see docs/DISTILLATION.md for the literature behind
each one):
  - LoRA, not full fine-tune: ~30x less measured forgetting in the cited
    study; the anchor result (llama-3.2-3b 0.17->0.46) was obtained this way.
  - Attention projections + MLP (q/k/v/o/gate/up/down), not all linear
    layers: the forgetting-in-LoRA literature converges on restricting the
    update subspace as the actionable lever against catastrophic forgetting.
  - Completion-only loss (mask the prompt): standard practice, avoids wasting
    gradient budget modelling the (repeated, low-information) instruction
    template.
  - bf16, sdpa attention, sequence packing at 512: matches the compute plan
    in docs/TEACHER.md / the training-cost estimate in
    docs/GENERATION_FEASIBILITY.md.
  - No chain-of-thought, no multi-example prompting baked into the data
    format: both were measured to hurt small models on this exact task
    (docs/GENERATION_FEASIBILITY.md sec 2-3).

Training file format: JSONL, one {"messages": [...]} per line, using each
base model's own chat template (never a hand-rolled one -- that would desync
from the instruct checkpoint's learned priors, per DISTILLATION.md sec 6).
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--attn_implementation", default="sdpa")
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--packing", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument("--num_train_epochs", type=float, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", nargs="+",
                     default=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"])
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_strategy", default="steps")
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_strategy", default="steps")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_to", default="none")
    return ap.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype="bfloat16" if args.bf16 else "auto",
        attn_implementation=args.attn_implementation,
    )

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_file, split="train")

    def format_example(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_example, remove_columns=eval_ds.column_names)

    # Completion-only loss: mask everything before the assistant turn's
    # opening. This template string is Qwen2/Qwen2.5/Qwen3 chat-template
    # syntax; if the base model changes to a different family, this response
    # template must be updated to match its chat template's assistant marker.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        packing=args.packing,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to=args.report_to,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        data_collator=collator if not args.packing else None,
        # NOTE: TRL's packing mode concatenates multiple examples per
        # sequence and is incompatible with DataCollatorForCompletionOnlyLM's
        # single-response-per-sequence assumption. If packing=True is kept,
        # completion-only masking is NOT applied -- see the printed warning
        # below. Flagged explicitly rather than silently training on prompt
        # tokens; revisit if the format-compliance gains this project's
        # anchor result attributes to completion-only masking don't show up.
    )

    if args.packing:
        print(
            "WARNING: packing=True disables completion-only loss masking "
            "(DataCollatorForCompletionOnlyLM is incompatible with packed "
            "sequences in this TRL version). Training will compute loss over "
            "full packed sequences, prompt tokens included. If this measurably "
            "hurts vs. the anchor result, prefer packing=False + the collator."
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
