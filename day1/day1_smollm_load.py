import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
## 트랜스포머 허깅페이스에서 가져온 모델


model_id = "HuggingFaceTB/SmolLM-135M"

tokenizer = AutoTokenizer.from_pretrained(model_id)
#tokenizer.json이나 tokenizer_config.json 파일을 다운로드하여 로드
model = AutoModelForCausalLM.from_pretrained(model_id)
#토크나이저와 모델 로드


print(f"Model ID: {model_id}")
print(f"Total: {sum(p.numel() for p in model.parameters()):,}") # 전체 파라미터
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}") # 학습 가능 파라미터만

print("-" * 30)
for name, param in model.named_parameters(): # 계층별 파라미터
    print(f"{name}: {param.size()}")

print("-" * 30)
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
#데이터를 다운받는게 아니라 straming을 통해 필요 부분만 가져오겠다
sample = next(iter(dataset))
#streaming 모드이기 때문에 인덱스처럼 접근하는게 불가능하고, iter처럼 반복자 만들고 next로 넘기면서 읽어와야 함
print(f"Dataset Sample: {sample['text'][:200]}...")

