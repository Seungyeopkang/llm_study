import json
import os
from transformers import GPT2TokenizerFast
from itertools import chain

## 토큰 패킹 : 여러 문서를 하나의 시퀀스로 합쳐서 GPU가 한 번에 처리할 수 있도록 하는 과정
## 의미가 중간에 끊길 수 있음 -> eos_token_id를 추가해서 끊어줌


def main():
    input_file = "day15/day15_output/history_10k.jsonl"
    block_size = 1024
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        
    print(f"Loaded {len(data)} documents.")
    
    tokenized_datasets = []
    for doc in data:
        text = doc.get("text", "")
        tokens = tokenizer(text)["input_ids"]
        tokens.append(tokenizer.eos_token_id)
        tokenized_datasets.append(tokens)
        
    concatenated = list(chain(*tokenized_datasets))
    total_length = len(concatenated)
    
    print(f"Total tokens: {total_length}")
    
    valid_length = (total_length // block_size) * block_size
    concatenated = concatenated[:valid_length]
    
    packed_data = [concatenated[i : i + block_size] for i in range(0, valid_length, block_size)]
    
    print(f"Packed into {len(packed_data)} blocks of size {block_size}.")
    print(f"Efficiency: {len(data)} documents -> {len(packed_data)} packed blocks.")

if __name__ == "__main__":
    main()



"""
1. Raw Data 수집: 웹, 책, 코드 등 테라바이트(TB) 단위의 데이터 긁어오기.

2. Cleaning 
Heuristic Filtering: 짧은 글, 저질 글 필터링.
Exact/Fuzzy Dedup: 중복된 글 제거 (MinHash 등).
Model-based Filtering: fastText 등으로 교육적 가치 확인.

3. Data Construction & Mix :
특정 주제 데이터 추출 
비율에 맞게 데이터 섞기

4. Tokenization & Packing :
글자를 숫자(ID)로 바꾸고 꽉꽉 눌러 담기.
여기까지 하면 최종 훈련용 데이터셋(.bin 또는 .jsonl 형태)이 완성
"""