import torch
import torch.nn as nn
import math
"""
저차원 근사
기존 가중치 고정하고, 병렬로 추가 길을 만들어서 저차원으로 근사
전략적으로 선택해서 주입
"""




class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        # rank : 저차원 임베딩 차원
        # alpha : 스케일링 (영향력의 크기)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        #차원을 줄였다가 복원하는 구조
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        # 학습 시작 전(Step 0)에는 LoRA가 모델에 아무런 영향도 끼치지 않게 하기 위해서, 1스텝 이후 업데이트 됨
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling ## 전체 스케일링 조절

class LinearWithLoRA(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(
            original_linear.in_features, 
            original_linear.out_features, 
            rank, 
            alpha
        )
        #원래 가중치는 학습하지 않음   
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x): #원래 가중치 + LoRA 가중치
        return self.original_linear(x) + self.lora(x)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def inject_lora(model, rank=8, alpha=16):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            print(f"Injecting LoRA into: {name}")
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            inject_lora(module, rank, alpha)

def main():
    torch.manual_seed(42)
    
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    print("=== Before LoRA ===")
    before_total, before_trainable = count_parameters(model)
    print(f"Total Params: {before_total:,}")
    print(f"Trainable Params: {before_trainable:,}")
    
    print("\n=== Injecting LoRA... ===")
    inject_lora(model, rank=8, alpha=16)
    
    print("\n=== After LoRA ===")
    after_total, after_trainable = count_parameters(model)
    print(f"Total Params: {after_total:,}")
    print(f"Trainable Params: {after_trainable:,}")
    
    reduction = (after_trainable / before_total) * 100
    print(f"\nTrainable Ratio: {reduction:.2f}%")
    
    x = torch.randn(1, 512)
    output = model(x)
    print(f"\nForward Pass Check: {output.shape} (Success)")

if __name__ == "__main__":
    main()



"""
Total Params: 1,055,242
Trainable Params: 1,055,242

=== Injecting LoRA... ===
Injecting LoRA into: 0
Injecting LoRA into: 2
Injecting LoRA into: 4

=== After LoRA ===
Total Params: 1,083,994
Trainable Params: 28,752

Trainable Ratio: 2.72%
"""


#LoRA는 99.8점 이상을 기록
#특히 튜닝 부분에서 뛰어난 성능