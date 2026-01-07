import torch
import torch.nn as nn
"""
단어 벡터를 2개씩 짝지어서 2차원 평면의 화살표(벡터)로 보고, 위치(n)만큼 회전(Rotation)시킴.
첫 번째 단어는 1도 돌리고, 두 번째 단어는 2도 돌리는 식.
장점: 두 단어의 내적(Attention)을 구할 때, "절대적 위치"는 사라지고 "상대적 거리(각도 차이)"만 남게 됨. (이게 Long Context 확장에 유리함!)

거리에만 집중하기 때문에 관계가 명확해짐
"""

## 차원마다 회전 속도 정의
#앞 차원은 빠르게, 뒷 차원은 느리게 회전

def compute_theta(dim, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return theta


def compute_rotation_matrix(seq_len, dim, theta):
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, theta) ## 각도 계산
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) #복소수 회전 행렬
    return freqs_cis

def apply_rope(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) ##2개씩 짝짓기
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x_complex.size(-1)) # 회전 행렬 모양 맞추기  [1, seq_len, 1, head_dim/2]
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3) # 복소수 곱셈을 통해서 회전
    return x_rotated.type_as(x)

def run_rope_check():
    bs, seq_len, n_heads, head_dim = 1, 10, 4, 16
    
    x_q = torch.randn(bs, seq_len, n_heads, head_dim)
    x_k = torch.randn(bs, seq_len, n_heads, head_dim)
    
    theta = compute_theta(head_dim)
    freqs_cis = compute_rotation_matrix(seq_len, head_dim, theta)
    
    x_q_rotated = apply_rope(x_q, freqs_cis)
    x_k_rotated = apply_rope(x_k, freqs_cis)
    
    print(f"Input Shape: {x_q.shape}")
    print(f"Rotated Shape: {x_q_rotated.shape}")
    
    diff = (x_q - x_q_rotated).abs().mean()
    print(f"Mean Difference (Original vs Rotated): {diff.item():.4f}")

    sample_original = x_q[0, 0, 0, :4]
    sample_rotated = x_q_rotated[0, 0, 0, :4]
    
    print("\nSample Vector (First 4 elements):")
    print(f"Original: {sample_original}")
    print(f"Rotated : {sample_rotated}")

if __name__ == "__main__":
    torch.manual_seed(42)
    run_rope_check()
