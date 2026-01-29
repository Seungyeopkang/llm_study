import os
import subprocess

def main():
    # 평가 결과를 저장할 디렉토리 설정
    output_dir = "day19/day19_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 평가 모델 정보
    model_name = "HuggingFaceTB/SmolLM2-135M"
    
    print("--- 🚀 LightEval 평가 시작 ---")
    print(f"모델: {model_name}")
    print("평가 항목: MMLU (Abstract Algebra subset)")

    # 실제 실습에서는 실행 시간을 위해 --max_samples를 제한하는 것이 좋습니다.
    # 아래는 터미널에서 실행할 명령어를 예시로 보여주며, subprocess로 실행할 수 있습니다.
    import subprocess
    
    # 윈도우 환경에 맞는 python 실행 파일 경로 추출
    python_exe = r"C:\Users\user\anaconda3\envs\dart\python.exe"
    
    # LightEval 실행 명령어 구성
    # 모델 인수와 태스크는 위치 인자(Positional Arguments)로 전달합니다.
    command = [
        python_exe, "-m", "lighteval", "accelerate",
        "--output-dir", output_dir,
        "--max-samples", "1",
        f"model_name={model_name}",
        "harness|mmlu:abstract_algebra|5"
    ]

    print(f"실행 명령어: {' '.join(command)}")
    
    # 평가 실행
    try:
        # 실시간 로그 확인을 위해 capture_output=False로 설정
        subprocess.run(command, check=True, capture_output=False)
        print("\n--- ✅ 평가 완료 ---")
    except subprocess.CalledProcessError as e:
        print("\n--- ❌ 평가 실패 ---")

    # 결과 파일 확인 안내
    print(f"\n평가 결과는 {output_dir} 폴더 내의 JSON 파일에서 상세히 확인할 수 있습니다.")

if __name__ == "__main__":
    main()
