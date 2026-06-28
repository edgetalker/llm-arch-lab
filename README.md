## Data
```zsh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
## Generate
```python
python -m archlab.script.generate \
  --config archlab/configs/tinystory.yaml \
  --ckpt runs/tinystory/latest.pt \
  --prompt "Once upon a time, there was" \
  --temperature 0.8 \
  --top-p 0.9
```
+ case study
+ lr = 3e-4 && val_loss = 1.47
```
Once upon a time, a little cat wanted to find his mom. Then,
he saw a big dog. The dog wanted to know what was behind the door. The cat was scared of the big dog.
The cat said, "Do not be scared. I will help you find your mom." The dog was happy. They walked together and found the big dog. The big dog was not scared anymore. He said, "Thank you, little cat. You are a good friend."
The moral of the story is to be brave and help others when they need it.
```
+ lr = 3e-3 && val_loss = 1.36
```
Once upon a time, a little cat wanted to find his mom. Then,
he saw a big dog. The dog had a collar. The cat said, "I can't find your mom!" The dog looked at the cat and said, "I can help you!"
The cat and the dog looked for the dog's mom. They walked and walked. They asked other animals if they saw her. No one said it was his mom. But then, they saw something unexpected. A big bird was in the tree! The cat and the dog were scared.
The cat said, "I'm sorry, I was just playing. I don't know where my mom is." The dog said, "It's okay, cat. I'm just playing." The cat, the dog, and the bird became friends. They found the cat's mom, and she was happy. The cat's mom said, "I missed you, cat. Thank you for helping me." The cat and the dog were not sad anymore. They were friends forever.
```

## Tokenizer
### TinyStory
+ train_bpe
```
input:     llm-arch-lab/data/TinyStoriesV2-GPT4-train.txt
input size: 2227.8 MB
vocab_size: 10000
special:    ['<|endoftext|>']


trained in 121.5s
final vocab size: 10000
merges learned:   9743

saved:
  trainer/tokenizer/vocab.pkl  (114.9 KB)
  trainer/tokenizer/merges.pkl (129.0 KB)

first 5 merges learned:
  0: b' ' + b't' → b' t'
  1: b'h' + b'e' → b'he'
  2: b' ' + b'a' → b' a'
  3: b' ' + b's' → b' s'
  4: b' ' + b'w' → b' w'

last 5 merges learned:
  9738: b'S' + b'urrender' → b'Surrender'
  9739: b'Rock' + b'y' → b'Rocky'
  9740: b' meadow' + b's' → b' meadows'
  9741: b' imag' + b'inary' → b' imaginary'
  9742: b' bo' + b'ld' → b' bold'
```
+ encode(540M~ Token)
```
...
[1556.1s]  540,498,265 tokens  (347.3k tok/s)

  done in 1567.1s (encode 1556.9s + write 10.1s)
  tokens:      540,796,778
  output size: 1081.6 MB
  range:       [9, 9999]
  compression: 2.06x

==================================================
all done in 1568.2s
```

## Ablation
+ learning rate: 
+ learning rate warmup:
+ AdamW hyperparameters
+ weight decay


## Status
- [x] BPE tokenizer (32k vocab, OWT)
- [x] Transformer body: RMSNorm, RoPE, SwiGLU, GQA
- [ ] Training loop (in progress)
- [ ] MoE variant (8 experts, top-2)
- [ ] Linear attention ablation