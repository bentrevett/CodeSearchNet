# CodeSearchNet

### Setup

1. `download_data.py` to get data for all 6 languages
1. `create_6L_data.py` to create 6L dataset
1. `create_5L_data.py` to create 5L dataset
1. `python bpe.py --lang java --vocab_max_size 10000 --bpe_pct 0.5` to create bpe dataset for java
1. `python bpe.py --lang 6L --vocab_max_size 10000 --bpe_pct 0.5` to create bpe dataset for 6L
1. `python bpe.py --lang 5L --vocab_max_size 10000 --bpe_pct 0.5` to create bpe dataset for 5L
1. `python create_vocab.py` to create vocabularies

### Initial Experiments

1. `python main_code_retrieval.py --lang java --model transformer --seed 1` model on code retrieval task, in java, w/ random initial weights
1. `python main_method_prediction.py --lang java --model transformer --seed 1` model on method prediction task, in java, w/ random initial weights

### Experiments w/ Extra Data

1. `python main_code_retrieval.py --lang 6L-java --model transformer --seed 1`
1. `python main_method_prediction.py --lang 6L-java --model transformer --seed 1`

### Experiments w/ Extra Data (but no Java!)

1. `python main_code_retrieval.py --lang 5L-java --model transformer --seed 1`
1. `pythom main_method_prediction.py --lang 5L-java --model transformer --seed 1`

### Pre-Training

1. `python main_sequence_lm.py --lang java --model --transformer --data code --seed 1`
1. `python main_sequence_lm.py --lang java --model --transformer --data code --seed 1`
1. `python main_code_retrieval.py --lang java --model transformer --seed 1 --load` model on code retrieval task, in java, fine tuned
1. `python main_method_prediction.py --lang java --model transformer --seed 1 --load` model on method prediction task, in java, fine tuned