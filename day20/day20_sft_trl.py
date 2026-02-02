import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


"""
SFT는 [질문, 답변] 쌍으로 구성된 정제된 데이터를 사용
모델을 특정 목적(채팅, 코드 작성, 요약 등)에 맞게 최적화하여 "어시스턴트"로 변모

출력 포메팅도 여기서 조절함
"""


def main():
    model_id = "HuggingFaceTB/SmolLM2-135M"
    dataset_name = "HuggingFaceTB/smoltalk"
    output_dir = "day20/day20_output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16
    )

    dataset = load_dataset(dataset_name, "everyday-conversations", split="train[:100]")

    def formatting_prompts_func(example):
        text = ""
        for msg in example['messages']:
            role = "Human" if msg['role'] == 'user' else "Assistant"
            text += f"### {role}: {msg['content']}\n"
        return {"text": text}

    dataset = dataset.map(formatting_prompts_func, remove_columns=dataset.column_names)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=10,
        logging_steps=1,
        save_steps=10,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        max_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    main()
