# Day 2: GQA vs MHA KV Cache Comparison

본 문서는 1주차 2일차 학습 내용인 Grouped-Query Attention(GQA)의 메모리 효율성을 실험하고 분석한 결과입니다.

## 1. 실험 요약
- **목적**: MHA, GQA, MQA 방식 간의 KV 캐시 메모리 소모량 비교
- **환경**: 
  - Precision: float16 (2 bytes per param)
  - Sequence Length: 4096
  - Head Dim: 128 / 256
  - Batch Size: 1

## 2. 수치 분석 결과 (64 Heads 기준)

| Type | KV Heads | Memory (128 Dim) | Memory (256 Dim) |
| :--- | :--- | :--- | :--- |
| **MHA** | 64 | 128 MB | 256 MB |
| **GQA (Your Config)** | 16 | 32 MB | 64 MB |
| **MQA** | 1 | 2 MB | 4 MB |

## 3. 핵심 결론 (User Insights)
1. **Dimension (`dim`)의 독립성**: 전체 모델의 차원(`dim`)은 모델 자체의 물리적 무게(파라미터 수)에는 영향을 주지만, 국소적인 **KV 캐시 메모리에는 영향을 주지 않음**. (K_proj에 의해 압축되기 때문)
2. **지능과 메모리의 분리**: Query 헤드(`n_heads`)가 늘어나면 모델은 똑똑해지지만, GQA 방식에서는 KV 캐시 메모리가 늘어나지 않고 **MHA만 기하급수적으로 늘어남**.
3. **KV 헤드의 직접적 영향**: `n_kv_heads`를 늘리면 모델의 성능이 MHA에 수렴하는 대신, 메모리 소모량이 정직하게 선형적으로 증가함.
4. **헤드 차원의 영향**: `head_dim`이 늘어나면 모든 방식(MHA, GQA, MQA)의 메모리 소모량이 동일한 비율로 증가함.

## 4. 주석 평가 및 피드백
- **GQA Layer 구현**: Llama-3 구조와 유사하게 `expand` 및 `reshape`를 활용하여 메모리를 점유하지 않는 방식(Broadcasting)으로 공유 로직을 구현함.
- **연산의 본질**: `o_proj`를 통한 헤드 간 "콜라보" 과정과 KV 캐시의 "국소적 압축" 원리를 실험 수치를 통해 증명함.
