import torch
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb
import time
import gc

"""
adamW : 모델 업데이트 옵티마이저, 과거 이력 사용

Paged : OOM 시에 RAM을 빌려서 사용
    gpu 메모리가 부족해지면 옵티마이저 상태값들을 cpu ram으로 이동시킴
    업데이트 차례 시에 다시 gpu로 이동
    => oom을 막아준다
    => 속도가 느려질 수 있음, 용량 한계, cuda 내에서만 가능

PagedAdamW8bit : 아담의 모멘텀, 분산을 8비트로 저장하고, oom시에 cpu ram으로 이동시킴 (비선형 양자화라서 성능저하가 거의 없음)

"""



def get_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def reset_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

def train_loop(optimizer_name, model, optimizer):
    print(f"\n[{optimizer_name}] Training Start...")
    print(f"Initial Memory: {get_memory_mb():.2f} MB")
    
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    
    dummy_input = torch.randint(0, 1000, (4, 128)).cuda()
    dummy_target = torch.randint(0, 1000, (4, 128)).cuda()
    
    start_time = time.time()
    for step in range(3): #일반적인 메모리 체크
        optimizer.zero_grad()
        outputs = model(dummy_input)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        loss = criterion(logits, dummy_target.view(-1))
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1} Peak Memory: {get_memory_mb():.2f} MB")
    
    duration = time.time() - start_time
    peak_mem = get_memory_mb()
    print(f"[{optimizer_name}] Done. Peak VRAM: {peak_mem:.2f} MB, Time: {duration:.4f}s")
    return peak_mem

def main():
    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    print(f"Loading {model_id}...")
    
    # 1. Test Standard AdamW
    reset_memory() #기본적 모델 불러오기
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    # Enable gradient checkpointing to focus on Optimizer overhead
    model.gradient_checkpointing_enable() 
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mem_adamw = train_loop("Standard AdamW", model, optimizer)
    
    del model, optimizer
    reset_memory()

    # 2. Test PagedAdamW8bit
    print("-" * 50)
    reset_memory()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    model.gradient_checkpointing_enable()
    
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=1e-4) #PagedAdamW8bit
    mem_bnb = train_loop("PagedAdamW8bit", model, optimizer)
    
    print("=" * 50)
    print(f"AdamW VRAM:        {mem_adamw:.2f} MB")
    print(f"PagedAdamW8bit VRAM: {mem_bnb:.2f} MB")
    print(f"Savings:           {mem_adamw - mem_bnb:.2f} MB")
    print("=" * 50)

if __name__ == "__main__":
    main()



"""
Standard AdamW (기본):
메모리: 16.4 GB (8GB VRAM 초과 -> 스왑 발생)
시간: 24.8초 (느림)
PagedAdamW8bit (8비트):
메모리: 7.0 GB (8GB 안에 쏙!)
시간: 4.55초 (약 5.5배 빠름)
절약량: 약 9.3 GB
"""

"""
OOM 방지 순서(RAM 충분)
1. PagedAdamW8bit
2. Batch Size 줄이기
3. Gradient Checkpointing
4. 모델 파라미터 줄이기(lora 등)

OOM 방지 순서(RAM 부족)
1. AdamW8bit 사용 (Paged ❌)
2. Gradient Checkpointing
3. Batch Size 줄이기
4. 모델 파라미터 줄이기(lora 등)

"""