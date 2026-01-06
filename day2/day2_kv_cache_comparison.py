import torch
import torch.nn as nn
import time


##n_head : 쿼리의 헤드 수
##n_kv_heads : 키와 밸류의 헤드 수
##head_dim : 헤드의 차원
def calculate_kv_cache_mem(batch_size, seq_len, n_heads, n_kv_heads, head_dim, dtype=torch.float16):
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    
    k_cache_shape = (batch_size, n_kv_heads, seq_len, head_dim) #[batch_size, n_kv_heads, seq_len, head_dim]
    v_cache_shape = (batch_size, n_kv_heads, seq_len, head_dim) #[batch_size, n_kv_heads, seq_len, head_dim]
    

    ## mem계산 = 요소 x 파라미터
    num_elements = (batch_size * n_kv_heads * seq_len * head_dim) * 2 
    mem_bytes = num_elements * bytes_per_param
    mem_mb = mem_bytes / (1024 ** 2)
    
    return mem_mb

def compare_attention_mechanisms(config, batch_size=1, seq_len=4096):
    n_heads = config["n_heads"]
    head_dim = config["head_dim"]
    
    configs = [
        ("MHA", n_heads),
        (f"GQA (Your Config)", config["n_kv_heads"]),
        ("MQA", 1) ## kv 헤드 1개
    ]
    
    print(f"{'Type':<20} | {'KV Heads':<10} | {'Memory (MB)':<15}")
    print("-" * 50)
    
    for name, n_kv_heads in configs:
        mem = calculate_kv_cache_mem(batch_size, seq_len, n_heads, n_kv_heads, head_dim)
        print(f"{name:<20} | {n_kv_heads:<10} | {mem:>15.2f} MB")

class LlamaGQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.head_dim = config["head_dim"]
        self.n_groups = self.n_heads // self.n_kv_heads  ## 그룹 계산방법 = 총 헤드 / kv 헤드
        
        # gca 같은 방식은 차원이랑 kv헤드 x head_dim 이 다르기 때문에 직접 사용을 한다
        # kv는 결국 차원 축소를 여기서 진행하게 된다
        self.q_proj = nn.Linear(config["dim"], self.n_heads * self.head_dim, bias=False) 
        self.k_proj = nn.Linear(config["dim"], self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config["dim"], self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config["dim"], bias=False) ## 콜라보 시키는 부분

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        #배치, 헤드, 토큰, 차원 순으로 변경까지 진행한 것
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.n_groups > 1: ## 헤드 수 맞추는 역할
            # [Batch, 8, Seq, Dim] 원래모양 => none으로 [Batch, 8, 1, Seq, Dim] 모양으로 만듬
            # expend로 [Batch, 8, 4, Seq, Dim] 모양으로 만듬. 4배 복사하는거랑 같음
            # reshape로 [Batch, 32, Seq, Dim] 모양으로 만듬
            k = k[:, :, None, :, :].expand(batch_size, self.n_kv_heads, self.n_groups, seq_len, self.head_dim).reshape(batch_size, self.n_heads, seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(batch_size, self.n_kv_heads, self.n_groups, seq_len, self.head_dim).reshape(batch_size, self.n_heads, seq_len, self.head_dim)
        
        #토치가 뒤에 2개만 보기 때문에 transpose로 [Batch, 32, Seq, Dim] 모양으로 변환
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(out)

if __name__ == "__main__":
    
    config = {
        "dim": 8192,
        "n_heads": 64,
        "n_kv_heads": 16,
        "head_dim": 256
    }
    
    compare_attention_mechanisms(config)
    
    model = LlamaGQA(config)
    x = torch.randn(1, 10, 8192)
    output = model(x)
    print(f"\nModel Output Shape: {output.shape}")


""" baseline
config = {
        "dim": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 128
    }

Type                 | KV Heads   | Memory (MB)    
--------------------------------------------------    
MHA                  | 32         |           64.00 MB
GQA (Your Config)    | 8          |           16.00 MB
MQA                  | 1          |            2.00 MB

Model Output Shape: torch.Size([1, 10, 4096])
"""



""" dim 2배
config = {
        "dim": 8192,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 128
    }
Type                 | KV Heads   | Memory (MB)    
--------------------------------------------------    
MHA                  | 32         |           64.00 MB
GQA (Your Config)    | 8          |           16.00 MB
MQA                  | 1          |            2.00 MB


"""

"""## 헤드 2배
config = {
        "dim": 8192,
        "n_heads": 64,
        "n_kv_heads": 8,
        "head_dim": 128
    }

Type                 | KV Heads   | Memory (MB)
--------------------------------------------------
MHA                  | 64         |          128.00 MB
GQA (Your Config)    | 8          |           16.00 MB
MQA                  | 1          |            2.00 MB
Model Output Shape: torch.Size([1, 10, 8192])
"""


"""kv 헤드 2배
config = {
        "dim": 8192,
        "n_heads": 64,
        "n_kv_heads": 16,
        "head_dim": 128
    }
Type                 | KV Heads   | Memory (MB)    
--------------------------------------------------
MHA                  | 64         |          128.00 MB
GQA (Your Config)    | 16         |           32.00 MB
MQA                  | 1          |            2.00 MB

Model Output Shape: torch.Size([1, 10, 8192])
"""



"""head 차원 2배
config = {
        "dim": 8192,
        "n_heads": 64,
        "n_kv_heads": 16,
        "head_dim": 256
    }
Type                 | KV Heads   | Memory (MB)    
--------------------------------------------------
MHA                  | 64         |          256.00 MB
GQA (Your Config)    | 16         |           64.00 MB
MQA                  | 1          |            4.00 MB

"""

#결론 1 : dim 자체는 전체 부분의 영향이지, 국소적으로 kv 메모리에는 영향이 없음
#결론 2 : head가 늘어나면 확장성으로 mha만 늘어남
#결론 3 : kv 헤드가 늘어나면 확장성으로 gqa만 늘어남
#결론 4 : head 차원이 늘어나면 확장성으로 mha, gqa 모두 늘어남