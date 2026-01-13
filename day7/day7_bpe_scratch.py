import collections

def get_stats(ids):
    counts = collections.defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def main():
    
    text = "hello hello world"
    tokens = list(text.encode("utf-8"))
    print(f"Initial Bytes: {tokens}")
    ## 사전의 자리를 좀 내주고 문장 길이를 줄이는게 더 효율적
    #나중에 계산복잡도 시에도 효율적
    vocab_size = 256 + 3 # 원래 8비트(256) + 3회의 merge (짝꿍 만들어서 추가하겠다는 뜻)
    num_merges = 3
    
    ids = list(tokens)
    print(ids)
    merges = {}
    
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        # 가장 많이 등장하는 쌍(pair) 찾기
        top_pair = max(stats, key=stats.get)
        new_idx = 256 + i
        print(f"Merging {top_pair} into new token {new_idx}")
        ids = merge(ids, top_pair, new_idx)
        merges[top_pair] = new_idx
        
    print(f"Final Token IDs: {ids}")
    print(f"Vocabulary Mappings: {merges}")

if __name__ == "__main__":
    main()


"""
Initial Bytes: [104, 101, 108, 108, 111, 32, 104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]

Merging (104, 101) into new token 256
Merging (256, 108) into new token 257
Merging (257, 108) into new token 258

Final Token IDs: [258, 111, 32, 258, 111, 32, 119, 111, 114, 108, 100]

Vocabulary Mappings: {(104, 101): 256, (256, 108): 257, (257, 108): 258}
"""