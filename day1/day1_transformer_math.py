import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

model_id = "HuggingFaceTB/SmolLM-135M"
model = AutoModelForCausalLM.from_pretrained(model_id)
layer = model.model.layers[0] ## layer0에 대해서만 구조화한거임

input_ids = torch.tensor([[1, 2, 3]])
hidden_states = model.model.embed_tokens(input_ids)
## 임베딩됨, [1, 3, 576] (1배치, 3토큰, 576차원)


#layer norm보다 효율적인 버전
def rms_norm(x, weight, eps=1e-5):
    variance = x.pow(2).mean(-1, keepdim=True) #모든 요소 제곱 후 평균
    x_normed = x * torch.rsqrt(variance + eps) #제곱근 구하고 역수 구하고 곱하기
    return x_normed * weight #weight에 곱해서 정규화된 값 반환


#pre normalization 구조, 임베딩 이후 바로 정규화
norm_1 = rms_norm(hidden_states, layer.input_layernorm.weight)

# q k v (wx) 계산 과정
#GCA 적용함
q = F.linear(norm_1, layer.self_attn.q_proj.weight) # [1, 3, 576] -> [1, 3, 576]
k = F.linear(norm_1, layer.self_attn.k_proj.weight) # [1, 3, 576] -> [1, 3, 192]
v = F.linear(norm_1, layer.self_attn.v_proj.weight) # [1, 3, 576] -> [1, 3, 192]

#view와 transpose를 통해 3차원으로 변환
"""PyTorch의 행렬 곱셈(matmul)은 보통 마지막 두 차원을 행렬로 보고 곱합니다. 
만약 차원이 [Batch, Head, Token, Dim] 순서라면 
독립된 연산 구역: PyTorch는 앞의 Batch와 Head를 일종의 '묶음(Batch)'으로 취급합니다.
병렬 처리: 즉, 9개의 헤드가 각각 가지고 있는 [3, 64] (토큰 수, 헤드 차원) 행렬들을 동시에 각각 따로 행렬 곱셈을 수행하여 헤드별 점수판을 만들 수 있게 됩니다."""

q = q.view(1, 3, 9, 64).transpose(1, 2) #헤드 구성, 576을 9개 헤드로 바꿈. [1, 9, 3, 64] (배치, 헤드, 토큰, 차원)
k = k.view(1, 3, 3, 64).transpose(1, 2) #헤드 구성, 192을 3개 헤드로 바꿈. [1, 3, 3, 64] (배치, 헤드, 토큰, 차원)
v = v.view(1, 3, 3, 64).transpose(1, 2) #헤드 구성, 192을 3개 헤드로 바꿈. [1, 3, 3, 64] (배치, 헤드, 토큰, 차원)


#헤드 수를 맞추기 위함. kv헤드 각각을 3번씩 연속해서 복사, dim=1은 head 차원을 복사하겠다
#[1, 9, 3, 64]
k = k.repeat_interleave(3, dim=1)
v = v.repeat_interleave(3, dim=1)


d_k = 64
scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5) #[1, 9, 3, 64] * [1, 9, 64, 3] -> [1, 9, 3, 3]
attn_weights = F.softmax(scores, dim=-1)
context = torch.matmul(attn_weights, v) #[1, 9, 3, 3] * [1, 9, 3, 64] -> [1, 9, 3, 64]


context = context.transpose(1, 2).reshape(1, 3, 576) # 헤드 합치기
attn_output = F.linear(context, layer.self_attn.o_proj.weight) # Attention 결과 통합
hidden_states = hidden_states + attn_output # residual connection


#[1, 3, 576]
##feed forward
norm_2 = rms_norm(hidden_states, layer.post_attention_layernorm.weight) #mlp 전 정규화
gate = F.linear(norm_2, layer.mlp.gate_proj.weight) # SILU GATE 생성, [1, 3, 1536]
up = F.linear(norm_2, layer.mlp.up_proj.weight) #데이터의 내용, [1, 3, 1536]
intermediate = F.silu(gate) * up #SILU 결과물, [1, 3, 1536]
mlp_output = F.linear(intermediate, layer.mlp.down_proj.weight) # intermediate차원을 576으로 다시 압축, [1, 3, 576]
hidden_states = hidden_states + mlp_output # residual connection

final_norm = rms_norm(hidden_states, model.model.norm.weight) #마지막 정규화
logits = F.linear(final_norm, model.lm_head.weight) #logits

print(f"Input IDs shape: {input_ids.shape}")
print(f"Hidden States shape: {hidden_states.shape}")
print(f"Final Norm shape: {final_norm.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Tied Weight Check: {id(model.model.embed_tokens.weight) == id(model.lm_head.weight)}")
