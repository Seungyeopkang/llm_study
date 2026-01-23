import re
from datasets import load_dataset
import json
import os
from tqdm import tqdm


"""
고품질 데이터를 가져오기 위한 과정
"""


def main():
    output_dir = "day15_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, "history_10k.jsonl")
    
    #키워드를 통해서 확실한 데이터 뽑기 위함
    keywords = ["history", "ancient", "war", "century", "empire", "civilization", "dynasty", "archaeology"]
    #비슷한 단어들을 제거하기 위해 정규표현식 사용
    patterns = [re.compile(rf"\b{k}\b") for k in keywords]
    
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    
    count = 0
    target_count = 10000
    
    print(f"Collecting {target_count} history samples...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset):
            text = example.get("text", "").lower()
            
            if any(p.search(text) for p in patterns): #키워드 확인
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
                count += 1
                
                if count >= target_count:
                    break
    
    print(f"Done. Saved {count} samples to {output_file}")

if __name__ == "__main__":
    main()


"""
1단계: 룰 기반 필터링 (Rule-based)
2단계: 경량 모델 분류
3단계: LLM 기반 필터링 & 합성


위 필터링은 철자만 보기 때문에 동음이의어 같은 지능이 없음 (딱 1차 필터링으로 적합)
"""
