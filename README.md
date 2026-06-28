> 📚 Course version: CS336 Spring 2026  
> 📁 This repo covers: Lab 1 Basics 

## Results

### LR Sweep (KR1)
Fixed: 22.7M params, batch=64, context=256, 20k steps, ~328M tokens.

| LR | val_loss| PPL | Notes |
| --- | --- | ---| --- |
| 3e-4 | 1.478 | 4.38 |> 1.45 |
| 1e-3 | 1.379 | 3.97 |✅ |
| 3e-3 | 1.365 | 3.91 |🏅 |
| 1e-2 | 1.424 | 4.15 |✅ |

![LR Sweep](assets/lr_sweep.png)

**Search strategy**: half-decade grid scan around the 3e-4 baseline. 
The U-shape across {3e-4, 1e-3, 3e-3, 1e-2} establishes both bounds 
of the search interval.  
The 1e-3 vs 3e-3 gap (0.014) is within batch noise, so no second-stage 
refinement was performed.



## Generation samples

Same prompt: `"Once upon a time, there was"`, temperature=0.8, top-p=0.9.

**lr=3e-4, val_loss=1.478** (under-trained baseline):
> Once upon a time, a little cat wanted to find his mom. Then,
he saw a big dog. The dog wanted to know what was behind the door. The cat was scared of the big dog.
The cat said, "Do not be scared. I will help you find your mom." The dog was happy. They walked together and found the big dog. The big dog was not scared anymore. He said, "Thank you, little cat. You are a good friend."
The moral of the story is to be brave and help others when they need it.


Coherent at the sentence level, but the dog's role is contradictory 
(scared the cat then needed help) and the story ends with a moralizing 
template common in under-trained TinyStories models.

**lr=3e-3, val_loss=1.366** (best LR):
> Once upon a time, a little cat wanted to find his mom. Then,
he saw a big dog. The dog had a collar. The cat said, "I can't find your mom!" The dog looked at the cat and said, "I can help you!"
The cat and the dog looked for the dog's mom. They walked and walked. They asked other animals if they saw her. No one said it was his mom. But then, they saw something unexpected. A big bird was in the tree! The cat and the dog were scared.
The cat said, "I'm sorry, I was just playing. I don't know where my mom is." The dog said, "It's okay, cat. I'm just playing." The cat, the dog, and the bird became friends. They found the cat's mom, and she was happy. The cat's mom said, "I missed you, cat. Thank you for helping me." The cat and the dog were not sad anymore. They were friends forever.

Longer narrative arc, multi-character interaction (cat + dog + bird), 
consistent goal tracking (finding mom), and natural resolution. 
A 0.11 val_loss difference translates to clearly improved coherence.

## Quickstart

```bash
# 1. Setup
uv sync

# 2. Download data (TinyStories ~2GB)
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..

# 3. Train tokenizer + tokenize (vocab=10000, compresses 2.06x → 540M tokens)
uv run archlab/trainer/train_tokenizer # ~2min
uv run prepare_data # ~26min

# 4. Train (uses best LR from sweep)
python -m archlab.script.pretrain --config archlab/configs/tinystory_lr_3e-3.yaml

# 5. Generate
python -m archlab.script.generate \
  --config archlab/configs/tinystory_lr_3e-3.yaml \
  --ckpt runs/tinystory_lr_3e-3/latest.pt \
  --prompt "Once upon a time, there was" \
  --temperature 0.8 --top-p 0.9
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



## Status
- [x] BPE tokenizer (32k vocab, OWT)
- [x] Transformer body: RMSNorm, RoPE, SwiGLU, GQA
- [ ] Training loop (in progress)
- [ ] MoE variant (8 experts, top-2)
- [ ] Linear attention ablation