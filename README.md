> 📚 Course version: CS336 Spring 2026  
> 📁 This repo covers: Lab 1 Basics 

## Results

### LR Sweep (KR1)
**Setup**: 22.7M params, batch=64, context=256, 20k steps, ~328M tokens.

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

### Batch Size Experiment (KR2)

**Setup**: All runs use the optimal LR from KR1 (lr=3e-3), with total iterations 
scaled to maintain a fixed ~328M token budget. AdamW, cosine schedule with warmup.

| batch_size | total_iters| warmup | val_loss | grad norm | tok/s |
| --- | --- | ---| --- | --- | --- |
| 32 | 40000 | 1000 | 1.415 | 0.25| 100k 
| 64 | 20000 | 500 | 1.365 | 0.18 | 100k
| 128 | 10000 | 250 |🏅 1.344| 0.13 | 100k


![BSZ Sweep](assets/bsz_sweep.png)

**Findings**:

1. **Gradient noise scales as 1/√B, as predicted by theory.** Mid-training 
   grad_norm at bsz=32 is ~1.9× that of bsz=128, closely matching the theoretical 
   √(128/32)=2.0× ratio. This confirms that larger batches produce more stable 
   gradient estimates.

2. **Throughput is saturated even at bsz=32.** All three batch sizes reach 
   ~100k tok/s on RTX 4090 — increasing batch size beyond 32 yields no throughput 
   gain for this 22.7M-parameter model. For small models on modern GPUs, GPU 
   memory is no longer the practical constraint on batch size; compute saturation 
   is reached well before memory limits.

3. **Larger batches achieve marginally lower final val_loss.** Under fixed token 
   budget, val_loss improves monotonically with batch size (bsz=32: 1.41, 
   bsz=128: 1.35), but with diminishing returns (0.02 gap between bsz=64 and 128 
   vs 0.04 gap between bsz=32 and 64). This is consistent with the gradient noise 
   reduction making optimization more stable, while the LR remains tuned for the 
   bsz=64 reference.

**Practical implication**: For this setup, bsz=128 strictly dominates — same 
throughput, lower final loss. The "default bsz=64" choice is not optimal here.
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

## Status
- [x] BPE tokenizer (32k vocab, OWT)
- [x] Transformer body: RMSNorm, RoPE, SwiGLU, GQA
- [ ] Training loop (in progress)
- [ ] MoE variant (8 experts, top-2)
- [ ] Linear attention ablation