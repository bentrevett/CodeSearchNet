import os
import json
from tqdm import tqdm

def file_iterator(path):
    with open(path, 'r') as f:
        for line in f:
            example = json.loads(line)
            yield example

for t in ['train', 'test', 'valid']:

    os.makedirs(f'data/5L/final/jsonl/{t}', exist_ok = True)

    with open(f'data/5L/final/jsonl/{t}/5L_{t}.jsonl', 'w+') as f:

        for language in ['go', 'javascript', 'php', 'python', 'ruby']:

            iterator = file_iterator(f'data/{language}/final/jsonl/{t}/{language}_{t}.jsonl')

            for example in tqdm(iterator, desc=f'{language} {t}'):

                code_tokens = example['code_tokens']
                docstring_tokens = example['docstring_tokens']
                func_name = example['func_name']

                example = {'code_tokens': code_tokens, 'docstring_tokens': docstring_tokens, 'language': language, 'func_name': func_name}

                json.dump(example, f)
                f.write('\n')