# LLM Study Project

본 저장소는 8주간의 LLM (Large Language Model) 심층 학습 과정을 기록하고 실습 코드를 관리하기 위한 프로젝트입니다.

## 📅 학습 로드맵

### 1단계: Core Concepts & Coding (1-2주차)
Transformer 구조부터 양자화, 효율적인 파인튜닝 기법까지 모델의 핵심 구조와 하드웨어 제약 조건 내에서의 최적화를 학습합니다.
- Attention & Transformer 구조 분석
- GQA, RoPE, Quantization 이론 및 실습
- PEFT/LoRA, Gradient Checkpointing 적용

### 2단계: 데이터 큐레이션 (3-4주차)
LLM 성능의 핵심인 고품질 데이터 구축 과정을 학습합니다.
- FineWeb-Edu, Datatrove 활용 데이터 분석
- 중복 제거(Deduplication) 및 데이터 믹스 전략
- Token Packing 및 Scaling Laws 이해

### 3단계: Post-training (5-6주차)
사전 학습된 모델을 특정 목적에 맞게 정렬하는 기법을 학습합니다.
- SFT (Supervised Fine-Tuning) 및 채팅 템플릿 구성
- QLoRA를 활용한 효율적 튜닝
- DPO (Direct Preference Optimization) 및 모델 병합

### 4단계: 평가 & 포트폴리오 (7-8주차)
학습된 모델을 정량/정성적으로 평가하고 결과물을 정리합니다.
- 모델 성능 평가 및 실패 사례 분석
- GGUF 변환 및 배포 준비
- 기술 회고 및 포트폴리오 완성

## 🛠 환경 설정
- **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM)
- **주요 라이브러리**: `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`

## 📝 학습 원칙
1. **직접 구현**: AI가 생성한 기본 코드에 직접 주석을 달며 구조를 파악합니다.
2. **제약 조건 극복**: 4060 환경에서의 최대 효율을 이끌어내는 방법을 탐구합니다.
3. **기록의 습관화**: 매일 진행 상황을 깃허브와 블로그에 기록합니다.
