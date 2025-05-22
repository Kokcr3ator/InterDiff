export DATASET="zinc"

# tokenize
python smiles-GPT/data/$DATASET/prepare.py

python smiles-GPT/train.py --batch_size=64
