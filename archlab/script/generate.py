# archlab/script/generate.py
import argparse
import torch
from dataclasses import asdict
from archlab.model.baseline import TransformerLM, generate
from archlab.tokenizer.bpe_tokenizer import Tokenizer
from archlab.trainer.train_model import load_checkpoint
from archlab.trainer.config_loader import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载 tokenizer
    tokenizer = Tokenizer.from_files(
        "archlab/tokenizer/tinystory/vocab.pkl",
        "archlab/tokenizer/tinystory/merges.pkl",
        ["<|endoftext|>"],
    )
    
    # 2. 加载 model
    model = TransformerLM(**asdict(cfg.model)).to(device)
    
    # 3. 加载 ckpt(注意 generate 时只需要 model weight,optimizer 不用)
    obj = load_checkpoint(args.ckpt, model)
    model.eval()
    
    # 4. 生成
    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    
    print("=" * 50)
    print(f"Prompt:    {args.prompt}")
    print(f"Output:    {output}")


if __name__ == "__main__":
    main()