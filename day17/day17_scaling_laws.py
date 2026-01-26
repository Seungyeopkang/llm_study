import numpy as np
import matplotlib.pyplot as plt
import os

#모델 사이즈 vs 데이터 사이즈 체크 손실 함수



A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28
L0 = 1.69

def chinchilla_loss(N, D):
    return A / (N**alpha) + B / (D**beta) + L0 # 위와 같은 계산으로 체크 가능함

def main():
    output_dir = "day17_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_sizes = np.array([1e7, 1e8, 1e9, 1e10]) # 10M, 100M, 1B, 10B => 모델 사이즈
    tokens = np.logspace(8, 12, 100) # 100M to 1T tokens => 데이터 사이즈
    
    plt.figure(figsize=(10, 6))
    
    for N in model_sizes:
        losses = [chinchilla_loss(N, D) for D in tokens]
        plt.plot(tokens, losses, label=f"N = {N/1e6:.0f}M")
        
    plt.xscale("log")
    plt.xlabel("Tokens (D)")
    plt.ylabel("Cross Entropy Loss (L)")
    plt.title("Chinchilla Scaling Laws: Loss vs. Data Size")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plot_path = os.path.join(output_dir, "scaling_law_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    budget_flops = 1e21 
    print(f"\nTarget Compute Budget: {budget_flops:.0e} FLOPs")
    
    a = 0.5 
    b = 0.5 
    G = (budget_flops / 6)
    

    # 예산 연산으로 최대의 지능을 뽑아내는 방식
    N_opt = (G / ( (A*alpha)/(B*beta) )**(-1/(alpha+beta)) ) ** (beta/(alpha+beta)) #최적의 모델 사이즈 탐색
    # Approximation from Chinchilla paper: N_opt \propto C^0.5, D_opt \propto C^0.5
    
    C = budget_flops
    N_chinchilla = 0.079 * (C**0.5)
    D_chinchilla = 2.1 * (C**0.5)
    
    print(f"Optimal Model Size (N): {N_chinchilla/1e6:.1f}M parameters")
    print(f"Optimal Data Size (D): {D_chinchilla/1e9:.1f}B tokens")
    print(f"Predicted Loss: {chinchilla_loss(N_chinchilla, D_chinchilla):.4f}")

if __name__ == "__main__":
    main()

