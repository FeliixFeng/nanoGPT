"""练习 01：看清楚 nanoGPT 到底在喂什么数据

跑：
    cd ~/Code/ai/nanoGPT
    .venv/bin/python practice/01_look_at_data.py

分五段写，写完一段跑一次，能跑再写下一段：
    1) 读 meta.pkl   —— 拿到字符表
    2) 读 train.bin  —— 拿到那一百多万个编号
    3) 切一段出来    —— x 和 y
    4) 解码成文字    —— 打印出来看
    5) 回答两个问题  —— 写在文件末尾

卡住就问，别空着想超过五分钟。
"""

import pickle
import numpy as np



# ---- 1) 读 meta.pkl ----

with open("data/shakespeare_char/meta.pkl", 'rb') as f:
    meta = pickle.load(f)
    
print(meta['vocab_size'])


# ---- 2) 读 train.bin ----

# memmap：把文件当成内存里的数组，用到哪读哪，不整体加载
train = np.memmap('data/shakespeare_char/train.bin', dtype=np.uint16, mode='r')

print('编号总数:', len(train))          # 有多少个编号
print('每个编号占:', train.dtype)       # uint16 = 2 字节
print('前 20 个:  ', train[:20])        # 取前 20 个看看

# ---- 3) 切一段 ----
block_size = 4
i = 100

x = train[i : i + block_size]
y = train[i + 1 : i + block_size + 1]

print('起点', i)
print('x =', x)
print('y =', y)



# ---- 4) 解码打印 ----

x_text = ''.join(meta['itos'][int(c)] for c in x)
y_text = ''.join(meta['itos'][int(c)] for c in y)

print('x ->', repr(x_text))
print('y ->', repr(y_text))



# ---- 5) 两个问题 ----
# (a) y 和 x 一位一位对起来，差别在哪？
#
# (b) 为什么不干脆让 y = x？
