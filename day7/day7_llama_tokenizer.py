import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "HuggingFaceTB/SmolLM2-135M"

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = "안녕하세요, Llama 3.2의 토크나이저를 테스트합니다. Hello, world!"
    

    tokens = tokenizer.encode(text)
    print(f"\n[Encoding Results]")
    print(f"Original Text: {text}")
    print(f"Token IDs: {tokens}")
    print(f"Token Count: {len(tokens)}")

    print(f"\n[Token Segmentation]")
    for token_id in tokens:
        decoded_token = tokenizer.decode([token_id])
        print(f"ID: {token_id:<6} -> Token: '{decoded_token}'")
    

    print(f"\nLoading model {model_id} to check weight sharing...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu" 
    )
    

    
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
