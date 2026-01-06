# Day 1: Core Concepts & SmolLM Analysis

본 문서는 1주차 1일차 학습 내용인 Transformer의 핵심 구조 분석과 SmolLM 모델 분석을 정리한 결과입니다.

## 1. 학습 요약
- **SmolLM-135M 분석**: 허깅페이스의 소형 모델을 로드하여 전체 파라미터 구조와 레이어별 구성을 파악함.
- **Transformer 연산 직접 구현**: PyTorch의 기본 연산을 활용하여 모델의 단일 레이어 과정을 수동으로 구현하며 수학적 구조를 이해함.
- **데이터 스트리밍**: 대용량 데이터를 메모리에 직접 올리지 않고 `streaming=True` 모드로 효율적으로 읽어오는 방식을 실습함.

## 2. 기술적 핵심 (Technical Insights)

### ① RMSNorm (Root Mean Square Layer Normalization)
- 기존 LayerNorm 보다 계산 효율이 높은 RMSNorm을 직접 함수로 구현함.
- 평균을 빼는 과정 없이 제곱 평균의 제곱근을 활용하여 정규화를 수행함.

### ② Attention 구조 (GQA 기반)
- SmolLM-135M이 채택한 **GQA (Grouped-Query Attention)** 구조를 분석함.
- Query(9 heads)와 KV(3 heads)의 불균형을 해결하기 위해 `repeat_interleave`를 사용하여 헤드 수를 맞추는 과정을 실습함.

### ③ MLP (SwiGLU)
- 최근 LLM의 표준인 SwiGLU 구조(`F.silu(gate) * up`)를 직접 연산함.
- `gate_proj`, `up_proj`, `down_proj`로 이어지는 데이터 확장 및 압축 과정을 파악함.

### ④ Tied Weights
- `embed_tokens`의 가중치와 `lm_head`의 가중치가 동일한 메모리 주소를 공유(`Tied Weight`)하여 파라미터 수를 절약하고 있음을 확인함.

## 3. 실험 코드 가이드
- **[day1_smollm_load.py](file:///c:/Users/user/Desktop/LLM_study/day1/day1_smollm_load.py)**: 모델 로드 및 파라미터 출력, 데이터셋 스트리밍 실습
- **[day1_transformer_math.py](file:///c:/Users/user/Desktop/LLM_study/day1/day1_transformer_math.py)**: 트랜스포머 레이어 0의 연산 과정을 수동으로 한 단계씩 실행

## 4. 학습 회고
- 모델의 전체 파라미터가 약 1.35억 개임을 확인하였으며, 모든 연산이 결국 거대한 행렬곱과 정규화의 반복임을 수학적으로 증명함.
