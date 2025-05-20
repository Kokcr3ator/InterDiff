import os
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

file_path = os.path.join(os.path.dirname(__file__))
def load_smiles(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]
    
zinc_path = os.path.join(file_path, 'zinc250k.txt')   
print(f"Loading SMILES from {zinc_path}") 
smiles_list = load_smiles(zinc_path)

# initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=2048,
    special_tokens=["[PAD]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"],
)

tokenizer.train_from_iterator(smiles_list, trainer=trainer)
tokenizer.save(os.path.join(file_path, 'zinc_tokenizer.json'))

train = smiles_list[:int(len(smiles_list) * 0.9)]
val = smiles_list[int(len(smiles_list) * 0.9):]

print("Tokenizing train and val sets...")
train_ids = tokenizer.encode_batch(train)
val_ids = tokenizer.encode_batch(val)

print(f"Train size: {len(train_ids)}")
print(f"Validation size: {len(val_ids)}")

print("Saving train and val ids to binary files...")
train_flat = np.concatenate([np.array(seq.ids, dtype=np.uint16) for seq in train_ids])
train_flat.tofile(os.path.join(file_path, 'train.bin'))

val_flat = np.concatenate([np.array(seq.ids, dtype=np.uint16) for seq in val_ids])
val_flat.tofile(os.path.join(file_path, 'val.bin'))

print("Saving train and val offsets to binary files...")
offsets = []
start = 0
for seq in train_ids:
    length = len(seq)
    offsets.append((start, length))
    start += length
np.save(os.path.join(file_path, "train_offsets.npy"), np.array(offsets, dtype=np.uint32)) 

offsets = []
start = 0
for seq in val_ids:
    length = len(seq)
    offsets.append((start, length))
    start += length

np.save(os.path.join(file_path, "val_offsets.npy"), np.array(offsets, dtype=np.uint32))

