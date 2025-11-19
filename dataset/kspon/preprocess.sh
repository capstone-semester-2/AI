# Author
# Soohwan Kim, Seyoung Bae, Cheolhwang Won, Soyoung Cho, Jeongwon Kwak
# python main.py --dataset_path "D:\code\train wav" --vocab_dest "D:\kospeech-latest" --output_unit "character" --preprocess_mode 'phonetic'

DATASET_PATH="SET_YOUR_DATASET_PATH"
VOCAB_DEST='SET_LABELS_DESTINATION'
OUTPUT_UNIT='character'                                          # you can set character / subword / grapheme
PREPROCESS_MODE='phonetic'                                       # phonetic : 칠 십 퍼센트,  spelling : 70%
VOCAB_SIZE=5000                                                  # if you use subword output unit, set vocab size

echo "Pre-process KsponSpeech Dataset.."

python main.py \
--dataset_path $DATASET_PATH \
--vocab_dest $VOCAB_DEST \
--output_unit $OUTPUT_UNIT \
--preprocess_mode $PREPROCESS_MODE \
--vocab_size $VOCAB_SIZE \
