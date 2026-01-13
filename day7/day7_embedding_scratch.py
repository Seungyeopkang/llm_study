import torch

class SimpleEmbedding:
    def __init__(self, vocab_size, embed_dim):
        # 1. 거대한 숫자 표(Lookup Table) 생성
        # nn.Embedding 대신 직접 랜덤 matrix 선언
        self.weight = torch.randn(vocab_size, embed_dim)
        # 랜덤 초기화 후 학습되는구조
        # 보통 input_ids가 원핫 벡터로 들어옴, 이것을 x^T W 를 통해서 학습 가능한 구조

    
    #어차피 수학적으로 같으니깐, back 할때만 wx 구조로 설계
    def forward(self, input_ids):
        # 2. 인덱스로 해당 줄(Row)만 가로채기
        # 이 연산이 사실상 임베딩의 전부!
        return self.weight[input_ids]

def main():
    vocab_size = 10
    embed_dim = 4
    
    # 임베딩 레이어 생성
    emb = SimpleEmbedding(vocab_size, embed_dim)
    
    # 입력 토큰 아이디 (예: [3, 5, 0])
    input_ids = torch.tensor([3, 5, 0])
    
    # 임베딩 수행
    vectors = emb.forward(input_ids)
    
    print("=== Embedding Weight Matrix ===")
    print(emb.weight)
    
    print("\n=== Input IDs ===")
    print(input_ids)
    
    print("\n=== Output Vectors (Lookup Results) ===")
    print(vectors)
    
    # 주소 확인 (공부용)
    print(f"\n3번 토큰 벡터가 웨이트 행렬의 3번째 줄과 일치하는가? {torch.equal(vectors[0], emb.weight[3])}")

if __name__ == "__main__":
    main()
