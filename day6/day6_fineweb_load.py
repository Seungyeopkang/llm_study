import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def main():##15조 토큰 일부만 가져옴
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2") ##gpt2 토크나이저, BPE 표준
    ##BPE : 자주 나오는 글자 쌍을 반복적으로 합치는 것, OOV에 강점을 보임
    ## 최근발전 : BBPE (바이트 단위), WordPiece(BERT, 확률을 좀 더 따짐), Unigram(gemma, 불필요한 조각 버리기)

    samples = []
    for i, example in enumerate(dataset):
        samples.append(example["text"])
        if i >= 9:
            break
            
    print(f"Loaded {len(samples)} samples")
    
    token_counts = []
    for text in samples:
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))
        
    print(f"Average Token Count: {sum(token_counts) / len(token_counts):.2f}")
    print(f"Max Token Count: {max(token_counts)}")
    print(f"Min Token Count: {min(token_counts)}")
    
    print("\nSample Snippet (First 100 chars):")
    print("-" * 20)
    print(samples[0][:100] + "...")

if __name__ == "__main__":
    main()

"""
Average Token Count: 944.80
Max Token Count: 3479
Min Token Count: 136

Sample Snippet (First 100 chars):
--------------------
The Independent Jane
For all the love, romance and scandal in Jane Austen’s books, what they are rea...


max_seq_length을 설정할 때 토큰의 길이를 생각해야 함
토큰 히스토 그리고, Vram과 데이터 경향을 종합해서 판단함

상적인 대화/요약 모델: 2,048 ~ 4,096 정도면 충분하다고 판단.
코드 생성/긴 문서 분석 모델: 32k, 128k까지 억지로라도 늘리려고 시도 (이때는 Flash Attention 같은 기술을 총동원).
"""