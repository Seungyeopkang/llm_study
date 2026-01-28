import re
from datasets import load_dataset

# MMLU (지식 측정) : 대학 수준의 상식 및 전문 지식을 측정
# GSM8K (추론 능력 측정) : 초등학교 수준의 수학 문제를 해결하는 능력을 측정
# Few-shot Prompt : 모델이 학습한 데이터를 기반으로 새로운 데이터를 처리하는 능력을 측정

def format_mmlu(example):
    prompt = f"Question: {example['question']}\n"
    choices = ["A", "B", "C", "D"]
    for i, choice in enumerate(example['choices']):
        prompt += f"{choices[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt, choices[example['answer']]

def format_gsm8k(example):
    prompt = f"Question: {example['question']}\nAnswer:"
    answer = re.search(r"####\s*(.*)", example['answer']).group(1)
    return prompt, answer

def main():
    print("--- MMLU (General Knowledge) ---")
    mmlu = load_dataset("cais/mmlu", "abstract_algebra", split="test")
    
    for i in range(2):
        prompt, label = format_mmlu(mmlu[i])
        print(f"Sample {i+1}:\n{prompt} {label}\n")

    print("--- GSM8K (Math Reasoning) ---")
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    
    for i in range(2):
        prompt, label = format_gsm8k(gsm8k[i])
        print(f"Sample {i+1}:\n{prompt} {label}\n")

    print("--- Few-shot Prompt Example (GSM8K) ---")
    few_shot_prompt = ""
    for i in range(3):
        p, a = format_gsm8k(gsm8k[i])
        few_shot_prompt += f"{p} {a}\n\n"
    
    target_p, target_a = format_gsm8k(gsm8k[3])
    few_shot_prompt += target_p
    
    print(few_shot_prompt)

if __name__ == "__main__":
    main()


"""
--- MMLU (General Knowledge) ---
Sample 1:
Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
A. 0
B. 4
C. 2
D. 6
Answer: B

Sample 2:
Question: Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.
A. 8
B. 2
C. 24
D. 120
Answer: C

--- GSM8K (Math Reasoning) ---
Sample 1:
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: 18

Sample 2:
Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
Answer: 3

--- Few-shot Prompt Example (GSM8K) ---
Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: 18

Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
Answer: 3

Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
Answer: 70000

Question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
Answer:


최근 벤치마크 : MMLU-Pro, GPQA(박사 수준 지식), GSM8K, IFEval 
"지식(MMLU), 추론(GSM8K), 상식(HellaSwag), 코딩(HumanEval), 지시이행(IFEval)"
"""