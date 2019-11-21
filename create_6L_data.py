import os
import json
from tqdm import tqdm

def file_iterator(path):
    with open(path, 'r') as f:
        for line in f:
            example = json.loads(line)
            yield example

for t in ['train', 'test', 'valid']:

    os.makedirs(f'data/6L/final/jsonl/{t}', exist_ok = True)

    for language in ['go', 'java', 'javascript', 'php', 'python', 'ruby']:

        iterator = file_iterator(f'data/{language}/final/jsonl/{t}/{language}_{t}.jsonl')

        with open(f'data/6L/final/jsonl/{t}/6L_{t}.jsonl', 'w+') as f:

            for example in tqdm(iterator, desc=f'{language} {t}'):

                code_tokens = example['code_tokens']
                docstring_tokens = example['docstring_tokens']

                example = {'code_tokens': code_tokens, 'docstring_tokens': docstring_tokens, 'language': language}

                json.dump(example, f)
                f.write('\n')