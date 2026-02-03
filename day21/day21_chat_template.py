import torch
from transformers import AutoTokenizer

def main():
    model_id = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how can I use chat templates?"},
        {"role": "assistant", "content": "You can use the apply_chat_template method!"},
        {"role": "user", "content": "Show me an example."}
    ]

    formatted_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("--- Formatted Chat ---")
    print(formatted_chat)

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True, # 마지막에 assistant 꼬리표 추가
        return_tensors="pt"
    )

    print("\n--- Tokenized Chat (Input IDs) ---")
    print(tokenized_chat)

    decoded_back = tokenizer.decode(tokenized_chat[0])
    print("\n--- Decoded Back ---")
    print(decoded_back)

    custom_template = (
        "{% for message in messages %}"
        "{{ '### ' + message['role']|capitalize + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '### Assistant: ' }}"
        "{% endif %}"
    )

    formatted_custom = tokenizer.apply_chat_template(
        messages,
        chat_template=custom_template,
        tokenize=False,
        add_generation_prompt=True
    )

    print("\n--- Custom Template Result ---")
    print(formatted_custom)

if __name__ == "__main__":
    main()


# 사람이 읽기 편한 리스트(JSON)를 모델이 공부하기 좋은 특정 문자열(String)로 변환하는 자동화 규칙