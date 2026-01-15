import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import time


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def reset_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def measure_case(model_id, text, use_checkpointing):
    reset_memory()
    print(f"\n[{'CHECKPOINTING ON' if use_checkpointing else 'CHECKPOINTING OFF'}] Loading Model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    

    #gradient checkpointing
    #메모리 사용을 줄이기 위해 forward 계산 중간값을 저장하지 않음
    #대신 backward시에 다시 계산하여 gradient를 계산함
    #메모리 사용량이 절반정도 줄어듬
    #하지만 backward 계산 시간이 증가함


    if use_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # kv캐시 역할을 하는 부분, kv는 효율을 위해 중간 계산값 저장해두는건데, 체크포인트와 충돌되는 개념
    
    model.train()
    
    inputs = tokenizer([text] * 4, return_tensors="pt").to("cuda")
    print(f"Input Shape: {inputs['input_ids'].shape}")
    
    print("Warming up...")## reset
    _ = model(**inputs, labels=inputs["input_ids"]).loss.backward() 
    model.zero_grad()
    reset_memory()
    
    print("Measuring...")
    start_time = time.time()
    
    outputs = model(**inputs, labels=inputs["input_ids"])##학습시작, **input은 딕셔너리 언패킹
    loss = outputs.loss ## 로스 계산
    print(f"Forward done. Memory: {get_peak_memory():.2f} MB")
    
    loss.backward() ##backward 계산
    end_time = time.time()
    
    peak_mem = get_peak_memory()
    duration = end_time - start_time
    
    print(f"Backward done. Peak Memory: {peak_mem:.2f} MB")
    print(f"Duration: {duration:.4f} s")
    
    del model, inputs, outputs, loss
    reset_memory()
    
    return peak_mem, duration

def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    text = "Hello world " * 500
    
    mem_off, time_off = measure_case(model_id, text, False)
    
    mem_on, time_on = measure_case(model_id, text, True)
    
    print("\n" + "="*40)
    print(f"Results Summary ({model_id})")
    print("="*40)
    print(f"VRAM Usage:  {mem_off:.2f} MB -> {mem_on:.2f} MB (Saved: {mem_off - mem_on:.2f} MB)")
    print(f"Compute Time: {time_off:.4f} s -> {time_on:.4f} s")
    print("="*40)
    print("Interpretation: Trades computation time for memory.")

if __name__ == "__main__":
    main()


"""
사용처
1. Full Fine-Tuning 
2. LoRA
3. 긴 문맥 처리

batch 를 줄이다가 1배치도 안되면 이때 진행

안켰을 때
Forward done. Memory: 16588.96 MB
Backward done. Peak Memory: 16588.96 MB

켰을 때
Forward done. Memory: 6348.38 MB
Backward done. Peak Memory: 7441.22 MB

바로 쓰고 버리기 때문에 기존보다는 메모리 크기가 작긴함
근데 2번 계산해야해서 시간은 오래걸림
"""