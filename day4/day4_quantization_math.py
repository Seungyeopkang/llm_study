import torch

#기본적인 양자화
#최대값 기준으로 대칭으로 자르기 때문에 데이터 고려 안하고, 낭비되는 부분이 생김
def absmax_quantize(x):
    scale = x.abs().max() / 127 # 최대값을 127로 나눔
    x_quant = (x / scale).round().clamp(-127, 127).to(torch.int8) #모든 숫자를 scale로 나눔. 그리고 -127 ~ 127 사이로 가두기
    return x_quant, scale


def zeropoint_quantize(x):
    x_range = x.max() - x.min() # 범위 설정
    x_range = 1 if x_range == 0 else x_range ## 0 방지
    scale = x_range / 255
    zero_point = (-x.min() / scale).round() ##최솟값 0으로 양자화 매핑
    x_quant = (x / scale + zero_point).round().clamp(0, 255).to(torch.uint8) ## 0~ 255 값으로 양자화, 이때 zero_point를 추가하여 양자화
    return x_quant, scale, zero_point

##복원과정
def dequantize_absmax(x_quant, scale): 
    return x_quant.float() * scale

def dequantize_zeropoint(x_quant, scale, zero_point):
    return (x_quant.float() - zero_point) * scale

def main():
    torch.manual_seed(42)
    original_weights = torch.randn(10) * 10
    
    print("=== Original Weights (Float32) ===")
    print(original_weights)

    # 1. AbsMax (Symmetric)
    q_abs, scale_abs = absmax_quantize(original_weights)
    recon_abs = dequantize_absmax(q_abs, scale_abs)
    
    print("\n=== AbsMax Quantization (Int8) ===")
    print(f"Scale: {scale_abs.item():.6f}")
    print(f"Quantized: {q_abs}")
    print(f"Reconstructed: {recon_abs}")
    print(f"Error (MSE): {(original_weights - recon_abs).pow(2).mean().item():.6f}")

    # 2. Zero-Point (Asymmetric)
    q_zp, scale_zp, zp = zeropoint_quantize(original_weights)
    recon_zp = dequantize_zeropoint(q_zp, scale_zp, zp)
    
    print("\n=== Zero-Point Quantization (UInt8) ===")
    print(f"Scale: {scale_zp.item():.6f}, Zero-Point: {zp.item()}")
    print(f"Quantized: {q_zp}")
    print(f"Reconstructed: {recon_zp}")
    print(f"Error (MSE): {(original_weights - recon_zp).pow(2).mean().item():.6f}")

if __name__ == "__main__":
    main()
