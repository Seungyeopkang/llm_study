"""
MinHash Deduplication : 수조 단위의 거대 데이터셋에서 거의 비슷한 문서들을 아주 빠른 속도로 찾아내서 제거하는 퍼지 중복 제거 기술

해시 값만 비교하여 두 문서가 비슷한지 추축 가능

효과 1: 중복 데이터를 제거하여 학습 시간을 단축
효과 2: 모델이 특정 문장을 암기하는 현상을 방지
효과 3: 중복된 저품질 데이터가 가중치에 너무 큰 영향을 주는 것을 막아 지능을 높임

1. Heuristic Filtering (기본적 필터링)
2. Exact Deduplication (단순 중복 제거)
3. Fuzzy Deduplication (아주 조금만 다른것)
4. Model-based Filtering


"""


from datatrove.executor.local import LocalPipelineExecutor #파이프라인 수행
from datatrove.pipeline.readers import HuggingFaceDatasetReader #데이터셋 읽기
from datatrove.pipeline.dedup import MinhashDedupSignature #텍스트를 n-gram으로 쪼갠 뒤, $k$개의 서로 다른 해시 함수를 적용하여 각 해시 함수 결과최솟값고정 길이의 벡터 생성
from datatrove.pipeline.dedup import MinhashDedupBuckets  #LSH(Locality Sensitive Hashing) 기법을 사용. 생성된 $k$개의 시그니처를 $b$개의 밴드로 나눕
from datatrove.pipeline.dedup import MinhashDedupCluster #같은 Bucket ID에 들어있는 문서 쌍들을 연결 그래프로 간주하고, Union-Find(Disjoint Set Union) 알고리즘을 사용하여 연결 요소 탐색
from datatrove.pipeline.dedup import MinhashDedupFilter # 순회하면서 각 문서의 ID가 클러스터 맵에 존재하는지 확인
from datatrove.pipeline.dedup import MinhashConfig #MinHash 및 LSH(Locality Sensitive Hashing) 알고리즘의 하이퍼파라미터를 정의하는 데이터 클래스
from datatrove.pipeline.writers.jsonl import JsonlWriter #json으로 출력
import os

"""
1. n gram을 선택해서 그걸로 나눔
2. MinHashing를 통해서 각 독립 시행으로 가장 작은 해시값을 구함 (160번 각각의 기준으로 뽑음, 가장 낮은 점수의 조각 ID를 기록)
3. 바구니에 담음 (걍 순서대로 넣음)
4. 클러스터링 : 같은 바구니에 하나라도 같은 해시값이 있으면 클러스터링 (묶기)
5. 가장 빠른 id만 남기고 나머지 삭제

"""



def main():
    
    dataset_name = "HuggingFaceTB/smollm-corpus"
    dataset_subset = "fineweb-edu-dedup" 
    
    # OUTPUT_DIR
    output_dir = "day13_output"
    signature_folder = f"{output_dir}/signatures"
    buckets_folder = f"{output_dir}/buckets"
    cluster_folder = f"{output_dir}/clusters"
    filtered_folder = f"{output_dir}/filtered_data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Config(파라미터 정의)
    # num_buckets : 20개
    # hashes_per_bucket : 8개
    minhash_config = MinhashConfig(
        num_buckets=20,
        hashes_per_bucket=8,
        n_grams=5,
    ) 

    # STAGE 1 : Signature
    pipeline_1 = [
        HuggingFaceDatasetReader(
            dataset=dataset_name,
            dataset_options={"name": dataset_subset, "split": "train"},
            streaming=True,
            limit=5000, 
            text_key="text",
            id_key="id"
        ),
        #n gram으로 쪼개서 해시값 생성
        MinhashDedupSignature(
            output_folder=signature_folder,
            config=minhash_config
        )
    ]

    # STAGE 2 : Buckets
    # 해시값을 bucket으로 분할
    pipeline_2 = [
        MinhashDedupBuckets(
            input_folder=signature_folder,
            output_folder=buckets_folder,
            config=minhash_config
        )
    ]

    # STAGE 3 : Cluster
    # bucket 내의 문서들을 클러스터링
    pipeline_3 = [
        MinhashDedupCluster(
            input_folder=buckets_folder,
            output_folder=cluster_folder,
            config=minhash_config
        )
    ]
    
    # STAGE 4 : Filter
    # 클러스터링된 문서들을 필터링
    pipeline_4 = [
        HuggingFaceDatasetReader(
            dataset=dataset_name,
            dataset_options={"name": dataset_subset, "split": "train"},
            streaming=True,
            limit=5000,
            text_key="text",
            id_key="id"
        ),
        MinhashDedupFilter(
            input_folder=cluster_folder,
            exclusion_writer=JsonlWriter(f"{output_dir}/removed_duplicates") 
        ),
        JsonlWriter(
            output_folder=filtered_folder
        )
    ]

    executor_1 = LocalPipelineExecutor(pipeline=pipeline_1, tasks=20, workers=4, logging_dir=f"{output_dir}/logs/step1", start_method="spawn")
    executor_2 = LocalPipelineExecutor(pipeline=pipeline_2, tasks=20, workers=4, logging_dir=f"{output_dir}/logs/step2", start_method="spawn")
    executor_3 = LocalPipelineExecutor(pipeline=pipeline_3, tasks=1, workers=1, logging_dir=f"{output_dir}/logs/step3", start_method="spawn")
    executor_4 = LocalPipelineExecutor(pipeline=pipeline_4, tasks=20, workers=4, logging_dir=f"{output_dir}/logs/step4", start_method="spawn")

    print("Stage 1 execution...")
    executor_1.run()
    
    print("Stage 2 execution...")
    executor_2.run()

    print("Stage 3 execution...")
    executor_3.run()
    
    print("Stage 4 execution...")
    executor_4.run()

    print("Done")

if __name__ == "__main__":
    main()
