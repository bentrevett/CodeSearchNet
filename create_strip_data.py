import json
import os
from tqdm import tqdm

def file_iterator(path):
    with open(path, 'r') as f:
        for line in f:
            example = json.loads(line)
            yield example

for language in ['go', 'java', 'javascript', 'php', 'python', 'ruby']:

    for t in ['train', 'test', 'valid']:

        os.makedirs(f'data/{language}-strip/final/jsonl/{t}', exist_ok = True)

        iterator = file_iterator(f'data/{language}/final/jsonl/{t}/{language}_{t}.jsonl')

        with open(f'data/{language}-strip/final/jsonl/{t}/{language}-strip_{t}.jsonl', 'w+') as f:

            for example in tqdm(iterator):

                code_tokens = example['code_tokens']
                docstring_tokens = example['docstring_tokens']
                strip_code_tokens = [t for t in code_tokens if t.isalnum()]

                example = {'code_tokens': strip_code_tokens, 'docstring_tokens': docstring_tokens}

                json.dump(example, f)
                f.write('\n')