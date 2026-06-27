from archlab.tokenizer.bpe_tokenizer import Tokenizer

# 注意把路径改成实际存在的位置
VOCAB  = 'archlab/tokenizer/tinystory/vocab.pkl'
MERGES = 'archlab/tokenizer/tinystory/merges.pkl'

tok = Tokenizer.from_files(VOCAB, MERGES, ['<|endoftext|>'])
print(f'vocab size: {len(tok.vocab)}')
print(f'merges:     {len(tok.merges)}')

# Test 1: TinyStories 风格文本
text = 'Once upon a time, there was a little girl named Lily. She liked to play with her dog.'
ids = tok.encode(text)
decoded = tok.decode(ids)
print(f'\nTest 1: TinyStories sentence')
print(f'  text:    {text!r}')
print(f'  ids[:10]: {ids[:10]}')
print(f'  ids len:  {len(ids)} (compression: {len(text)/len(ids):.2f} chars/token)')
print(f'  decoded:  {decoded!r}')
assert decoded == text, 'FAILED'
print(f'  OK')

# Test 2: special token
text2 = 'First story.<|endoftext|>Second story.'
ids2 = tok.encode(text2)
decoded2 = tok.decode(ids2)
print(f'\nTest 2: special token')
print(f'  text:     {text2!r}')
print(f'  ids:      {ids2}')
print(f'  decoded:  {decoded2!r}')
assert decoded2 == text2, 'FAILED'
print(f'  OK')

# Test 3: unicode 边界(TinyStories 偶尔有花引号)
text3 = 'She said \"Hello\" — and smiled.'
ids3 = tok.encode(text3)
decoded3 = tok.decode(ids3)
print(f'\nTest 3: unicode punctuation')
print(f'  text:     {text3!r}')
print(f'  decoded:  {decoded3!r}')
assert decoded3 == text3, 'FAILED'
print(f'  OK')

# Test 4: 空字符串和单字符
assert tok.decode(tok.encode('')) == '', 'empty string failed'
assert tok.decode(tok.encode('a')) == 'a', 'single char failed'
print(f'\nTest 4: edge cases (empty / single char) OK')

print('\nall tests passed')
