import collections
import json
import os
from tqdm import tqdm

for root, dirs, files in os.walk('data', topdown = False):
   
   for name in files:
      if name.endswith('.jsonl') and 'train' in name:

        code_vocab_counter = collections.Counter()
        desc_vocab_counter = collections.Counter()

        with open(os.path.join(root, name), 'r') as f:
            for line in tqdm(f):
                example = json.loads(line)
                code = example['code_tokens']
                desc = example['docstring_tokens']
                code_vocab_counter.update(code)
                desc_vocab_counter.update(desc)

        code_vocab_name = ''.join(name.split('.jsonl')[:-1] + ['.code_vocab'])
        desc_vocab_name = ''.join(name.split('.jsonl')[:-1] + ['.desc_vocab'])

        print(name, code_vocab_name, desc_vocab_name)

        with open(os.path.join(root, code_vocab_name), 'w+') as f:
            for tok, count in code_vocab_counter.most_common(100_000):
                f.write(f'{tok}\t{count}\n')

        with open(os.path.join(root, desc_vocab_name), 'w+') as f:
            for tok, count in desc_vocab_counter.most_common(100_000):
                f.write(f'{tok}\t{count}\n')
