import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc
### bitAnyBytesConfig : 양자화 설정 도구
def get_vram_usage():
    return torch.cuda.memory_allocated() / (1024 ** 2)

def load_and_measure(model_id, load_type):
    print(f"\n[{load_type}] Loading...")
    torch.cuda.empty_cache()
    gc.collect()
    before_mem = get_vram_usage()
    

    ###load_in_8bit=True : 8비트 양자화
    ###load_in_4bit=True : 4비트 양자화
    if load_type == "FP16 (Half)":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    elif load_type == "8-bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=quant_config, 
            device_map="auto"
        )
    elif load_type == "4-bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=quant_config, 
            device_map="auto"
        )
        
    after_mem = get_vram_usage()
    print(f"Model VRAM: {after_mem - before_mem:.2f} MB")
    
    del model
    torch.cuda.empty_cache()
    gc.collect() ## 가비지콜렉터 제거

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping VRAM test.")
        return

    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    print(f"Testing Model: {model_id}")
    print(f"Initial VRAM: {get_vram_usage():.2f} MB")
    
    load_and_measure(model_id, "FP16 (Half)")
    load_and_measure(model_id, "8-bit")
    load_and_measure(model_id, "4-bit")

if __name__ == "__main__":
    main()
