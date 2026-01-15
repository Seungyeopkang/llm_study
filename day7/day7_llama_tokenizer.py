import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "HuggingFaceTB/SmolLM2-135M"
    ##모델 토크나이저 불러옴 (BBPE사용)
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = "안녕하세요, Llama 3.2의 토크나이저를 테스트합니다. Hello, world!"
    ## 토크나이징
    
    tokens = tokenizer.encode(text)
    print(f"\n[Encoding Results]")
    print(f"Original Text: {text}")
    print(f"Token IDs: {tokens}")
    print(f"Token Count: {len(tokens)}")

    # 디코딩
    # 한글은 대부분 깨져있음 (바이트 단위로 토크나이징을 함, 이때 자주 쓰이는건 직접 출력됨)
    print(f"\n[Token Segmentation]")
    for token_id in tokens:
        decoded_token = tokenizer.decode([token_id])
        print(f"ID: {token_id:<6} -> Token: '{decoded_token}'")
    

    #모델 로드
    print(f"\nLoading model {model_id} to check weight sharing...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu" 
    )
    

    ## 가중치 공유
    # input : 토크나이저 -> 벡터
    # output : 벡터 -> 토크나이저
    # 이 때 입 출력 단어 표가 똑같아야 함

    # 입출력 요구사항 동시 만족을 위해 모델이 어정쩡 해짐
    # 앞 뒤가 소통을 해야해서 병목이 심해짐
    # 파인튜닝 시 성능이 잘 안나옴

    # 그럼에도 작은 모델에 대해서, 적은 데이터셋에서는 성능이 잘나오거나 효율이 좋음
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    
    print(f"\n[Weight Sharing Check]")
    print(f"Input Embedding Address : {input_embeddings.data_ptr()}")
    print(f"Output Embedding Address: {output_embeddings.data_ptr()}")
    
    is_tied = input_embeddings.data_ptr() == output_embeddings.data_ptr()
    print(f"Are weights tied? : {is_tied}")
    

    if not is_tied:
        print("\nNote: Llama 3.2 does not use Tied Embeddings.")
        print(f"Input Embedding Shape : {input_embeddings.shape}")
        print(f"Output Embedding Shape: {output_embeddings.shape}")

if __name__ == "__main__":
    main()
