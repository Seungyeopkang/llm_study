from datatrove.executor.local import LocalPipelineExecutor #파이프라인 수행
from datatrove.pipeline.readers import HuggingFaceDatasetReader #데이터로더
from datatrove.pipeline.filters import GopherQualityFilter, LanguageFilter #데이터 필터
from datatrove.pipeline.writers.jsonl import JsonlWriter #데이터 저장
import os

def main():
    dataset_name = "HuggingFaceTB/smollm-corpus" 
    dataset_subset = "fineweb-edu-dedup"
    
    if not os.path.exists("day12_output"):
        os.makedirs("day12_output")

    pipeline = [
        #데이터 읽기
        HuggingFaceDatasetReader(
            dataset=dataset_name,
            dataset_options={"name": dataset_subset, "split": "train"},
            streaming=True,
            limit=1000 
        ),
        #데이터 필터링
        GopherQualityFilter(
            min_doc_words=50, #최소 50단어 이상
            max_doc_words=100000, #최대 100000단어 이하
            #특수기호 거르기
            min_avg_word_length=3, #평균 단어 길이 3 이상
            max_avg_word_length=10 #평균 단어 길이 10 이하
        ),
        JsonlWriter(
            output_folder="day12_output/filtered_data"
        )
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        workers=1, #cpu 코어 사용 개수
        tasks=1 #병렬 처리 개수
    )

    print("Starting Datatrove Pipeline...")
    executor.run()
    print("Pipeline Finished. Results in day12_output/filtered_data")

if __name__ == "__main__":
    main()



"""
1. Pretrin 단계 : 최소한의 필터 (약한 필터)
2. Continued Pre-training(전문지식) : 강한 필터링 + 퀄리티 (AI분류기)
3. SFT 단계 : 고품질 데이터 + 포맷팅


요즘 트렌드 = 데이터 퀄리티가 더 중요해지고 있음
"""