"""练习 01：看清楚 nanoGPT 到底在喂什么数据

跑：
    cd ~/Code/ai/nanoGPT
    .venv/bin/python practice/01_look_at_data.py

这个目录被 .git/info/exclude 关掉了，不在版本控制里。
随便改，改坏了删掉重来就行，不会影响原版代码。
"""

import pickle
import numpy as np

# ---------- 1. 读回字符表：prepare.py 当时存下来的 ----------
with open('data/shakespeare_char/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

itos = meta['itos']          # {0: '\n', 1: ' ', 2: '!', ...}   编号 -> 字符
stoi = meta['stoi']          # 反过来，字符 -> 编号
vocab_size = meta['vocab_size']

print(f"词表大小 vocab_size = {vocab_size}")
print(f"训练集里一共就这 {vocab_size} 种符号，模型只能吐这些\n")

# ---------- 2. 整个训练集在磁盘上就是一串数字 ----------
data = np.memmap('data/shakespeare_char/train.bin', dtype=np.uint16, mode='r')
print(f"训练集长度 = {len(data):,} 个 token")
print(f"最前面 10 个 token 的编号：{data[:10]}\n")

# ---------- 3. 切一小段出来 ----------
block_size = 256
i = 100_000
x = data[i : i + block_size]              # 模型看到的
y = data[i + 1 : i + 1 + block_size]      # 模型要它预测的

# TODO 1  把 x 和 y 解码成能读的文字，打印出来
#         itos 是 {编号: 字符}，''.join(...) 就能拼成字符串
#         注意 data 里取出来的是 numpy 的整数，拿去当字典 key 之前先 int() 一下
x_text = None
y_text = None

print("---------- x（模型看到的） ----------")
print(x_text)
print()
print("---------- y（模型要预测的） ----------")
print(y_text)
print()


# TODO 2  先别往下走，把这两个问题想清楚，答案写在注释里
#
#   (a) y 和 x 一位一位对起来看，差别在哪？
#       答：
#
#   (b) 为什么不干脆让 y = x，非要错开一格？
#       答：


# 跑通之后，脑子里过一遍这个（不用写下来）：
#   如果模型完全不认识字、每次都在 65 个字符里瞎猜，
#   那它猜中的概率是多少？把这个概率换算成 loss 大概是几？
#   —— 下次训练第一行打印出来的 loss，跟这个数应该对得上。
