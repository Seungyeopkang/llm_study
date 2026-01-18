import torch
from transformers import (
    AutoModelForCausalLM, #모델 불러오기
    AutoTokenizer, # 토크나이징
    TrainingArguments, #학습 설정 시트
    Trainer,
    DataCollatorForLanguageModeling #토큰 포장지
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training #로라 관련 키트
from datasets import load_dataset
import bitsandbytes as bnb #PagedAdamW8bit 구현 위함

def create_qlora_config():
    return LoraConfig(
        r=16, #랭크
        lora_alpha=32, #로라 알파값 (스케일링)
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def load_model_and_tokenizer(model_id):
    from transformers import BitsAndBytesConfig #양자화
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, #4비트 양자화
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained( #모델 불러오기
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model.gradient_checkpointing_enable() #gradient checkpoint
    model = prepare_model_for_kbit_training(model) 
    
    lora_config = create_qlora_config() #qrola 불러오기
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def prepare_dataset(tokenizer, max_length=256):
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]") #데이터 불러오기
    
    def format_instruction(example):
        if example["input"]:
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}
    
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    def tokenize_function(examples):  #변환 규칙
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"] #처리한 columns 은 제거
    )
    
    return tokenized_dataset

def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return
    
    model_id = "HuggingFaceTB/SmolLM2-135M"
    
    print(f"Loading {model_id} with QLoRA...")
    model, tokenizer = load_model_and_tokenizer(model_id)
    
    print("Preparing dataset...")
    train_dataset = prepare_dataset(tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./day11_output",
        num_train_epochs=1,
        per_device_train_batch_size=2, #물리적 배치
        gradient_accumulation_steps=4, #묶음 => 배치사이즈 8 효과
        learning_rate=2e-4,
        fp16=True, #숫자 하나를 16비트로 저장, loss같은 예민한 것은 32비트로, 아니면 16비트로 => qlora것
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit", #ram 땡겨쓰기
        warmup_steps=10,
        max_steps=50,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) # mlm True 는 bert 스타일(빈칸채우기)
    )
    
    print("\n" + "="*50)
    print("Starting Fine-tuning (RTX 4060 Optimized)")
    print("="*50)
    print(f"Model: {model_id}")
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")
    print("="*50 + "\n")
    
    trainer.train()
    
    print("\n" + "="*50)
    print("Fine-tuning Complete!")
    print("="*50)
    
    model.save_pretrained("./day11_output/final_model")
    tokenizer.save_pretrained("./day11_output/final_model")
    
    print("Model saved to ./day11_output/final_model")

if __name__ == "__main__":
    main()



"""
학습 가능 파라미터: 921,600개 (전체의 1.1%)
학습 시간: 37초 (50 steps)
Loss 감소: 2.49 → 2.15 (13% 개선)
메모리 사용: 8GB 이내 ✅
상태: OOM 없이 완료 ✅
"""

"""
1. 모델에 qlora 적용 => 모델 불러올때 적용
2. fp16 => qlora 파라미터
3. paged_adamw_8bit => 옵티마이저 계산할 때
4. 4비트 양자화 => 전체적인 모델 구성할 때
5. gradient checkpoint => forward, backward
6. gradient accumulation: 작은 배치를 여러 번 모아서 한 번에 업데이트


4비트로 압축된 원본 모델 가중치를 계산 직전에 잠깐 16비트로 해제(De-quantize).
이것과 16비트인 LoRA 가중치를 더해서 계산을 수행.
계산이 끝나면 다시 원본 모델은 4비트 상태로 돌아가고, 업데이트는 오직 16비트인 LoRA에만 적용.

"""