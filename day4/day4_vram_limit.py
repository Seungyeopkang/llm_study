import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def get_vram_usage():
    return torch.cuda.memory_allocated() / (1024 ** 2)

def main():
    if not torch.cuda.is_available():
        print("CUDA needed for VRAM test.")
        return

    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    print(f"Loading {model_id} for limit test...")
    
    # 4-bit 로드로 최대한 많은 배치 테스트
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    batch_size = 1
    seq_len = 1024 # 고정 길이
    
    try:
        while True:
            torch.cuda.empty_cache()
            gc.collect()
            
            # 더미 데이터 생성
            input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
            
            with torch.no_grad():
                _ = model(input_ids)
            
            print(f"Batch {batch_size} Success | VRAM: {get_vram_usage():.2f} MB")
            batch_size += 1 # 1씩 증가 (정밀 측정)
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n[OOM Reached] Max Batch Size: {batch_size - 1}")
            print(f"Final VRAM: {get_vram_usage():.2f} MB")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()


"""
Batch Size	VRAM 사용량	상태
Batch 1	3560 MB	✅ 성공
Batch 5	4712 MB	✅ 성공
Batch 10	6152 MB	✅ 성공
Batch 15	7592 MB	✅ 성공
Batch 17	8168 MB	✅ 성공 (거의 꽉 참)
Batch 18	8456 MB	✅ 성공 (SWAP/공유 메모리 활용 추정)
Batch 19	8744 MB	🛑 임계점 도달
"""