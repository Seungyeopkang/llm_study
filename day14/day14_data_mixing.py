from datasets import load_dataset, interleave_datasets
from collections import Counter

def main():
    print("Loading datasets...")
    
    ds_web = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True) # 웹 데이터
    ds_knowledge = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True) # 지식 데이터
    ds_code = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True) # 코드 데이터

    print("Datasets loaded.")

    probabilities = [0.5, 0.3, 0.2] # 웹 50%, 지식 30%, 코드 20%
    print(f"Mixing Ratios: Web={probabilities[0]}, Knowledge={probabilities[1]}, Code={probabilities[2]}")

    # Stream 모드는 전체 개수를 알 수 없기 때문에, 임의로 100만개씩 잘라서 source 컬럼 추가
    ds_web = ds_web.add_column("source", ["web"] * 1000000) # 웹 데이터에 source 컬럼 추가
    ds_knowledge = ds_knowledge.add_column("source", ["knowledge"] * 1000000) # 지식 데이터에 source 컬럼 추가
    ds_code = ds_code.add_column("source", ["code"] * 1000000) # 코드 데이터에 source 컬럼 추가
    
    # 데이터 섞기 (확률적으로)
    mixed_dataset = interleave_datasets(
        [ds_web, ds_knowledge, ds_code],
        probabilities=probabilities,
        seed=42,
        stopping_strategy="first_exhausted" # 하나라도 다 떨어지면 멈춤
    )

    shuffled_dataset = mixed_dataset.shuffle(seed=42, buffer_size=1000)

    print("\nVerifying distribution (Sampling 1000 examples)...")
    
    source_counter = Counter()

    for i, example in enumerate(shuffled_dataset):
        if i >= 1000: break
        
        source = example.get('source')
        if source:
            source_counter[source] += 1

    print("\nResulting Distribution (per 1000 samples):")
    for source, count in source_counter.items():
        print(f"{source}: {count} ({count/10.0}%)")

    print("\nSample Data:")
    for i, example in enumerate(shuffled_dataset.take(3)):
        print(f"--- Example {i+1} ---")
        text = example.get('text', "")
        if text:
            print(text[:200] + "...")
        else:
            print("[No Text Found]")

if __name__ == "__main__":
    main()

#어떤 능력을 키울지 결정하는 설계 과정
#Moe도 섞어서 준다. 라우터가 학습을 통해 알아서 분배