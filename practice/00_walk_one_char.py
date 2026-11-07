"""显微镜：一个字符怎么变成 loss。

不是练习，是给你看数字怎么流动的。跑：
    cd ~/Code/ai/nanoGPT
    uv run python practice/00_walk_one_char.py

小世界设定：词表 4 个字符，向量宽度 4。
nanoGPT 是词表 65、宽度 384，形状的规矩一模一样，只是数字多。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

vocab = ['a', 'b', 'c', 'd']
V = len(vocab)   # 词表大小，nanoGPT 里是 65
D = 4            # 向量宽度，nanoGPT 里是 384

# 两张表：查表用 emb，打分用 head。都是随机初始化的，没训练过。
emb = nn.Embedding(V, D)
head = nn.Linear(D, V, bias=False)


def show(name, t):
    print(f'{name}  形状={tuple(t.shape)}')
    for row in t.detach().tolist():
        print('   ', '  '.join(f'{x:+.4f}' for x in row) if isinstance(row, list)
              else f'{row:+.4f}')
    print()


print('=' * 56)
print('0. 两张表（就是模型里全部的“知识”，现在是随机的）')
print('=' * 56)
print(f'emb.weight   形状={tuple(emb.weight.shape)}   行号=字符编号，每行 4 个数')
print()
for i, ch in enumerate(vocab):
    print(f"  第 {i} 行 ('{ch}')  " + '  '.join(f'{x:+.4f}' for x in emb.weight[i].tolist()))
print()
print(f'head.weight  形状={tuple(head.weight.shape)}   4 个数 -> 4 个分数')
print()
for i, ch in enumerate(vocab):
    print(f"  第 {i} 行 ('{ch}')  " + '  '.join(f'{x:+.4f}' for x in head.weight[i].tolist()))
print()

print('=' * 56)
print('1. 输入：一句话切成 X 和 Y，Y 是 X 错一格')
print('=' * 56)
text = 'abcd'
X = torch.tensor([[0, 1, 2, 3]])
Y = torch.tensor([[1, 2, 3, 0]])
print('原文      ', text)
print('X 的编号  ', X[0].tolist(), '-> ', ''.join(vocab[i] for i in X[0].tolist()))
print('Y 的编号  ', Y[0].tolist(), '-> ', ''.join(vocab[i] for i in Y[0].tolist()))
print('X 形状    ', tuple(X.shape), ' = (句子数, 长度)')
print()

print('=' * 56)
print('2. 查表：每个编号换成一个 4 维向量')
print('=' * 56)
h = emb(X)
print(f'形状 {tuple(X.shape)} -> {tuple(h.shape)}，每个编号变成 4 个数')
print()
for pos in range(4):
    print(f"  位置 {pos}: 编号 {X[0, pos].item()} ('{vocab[X[0, pos].item()]}')"
          f"  ->  [{', '.join(f'{x:+.4f}' for x in h[0, pos].tolist())}]")
print()

print('=' * 56)
print('3. 打分：4 个数 × 4x4 矩阵 -> 4 个分数（logits）')
print('=' * 56)
z = head(h)
print(f'形状 {tuple(h.shape)} -> {tuple(z.shape)}')
print()
for pos in range(4):
    print(f"  位置 {pos} 的 4 个分数: " +
          '  '.join(f'{x:+.4f}' for x in z[0, pos].tolist()))
print()

# 手算验证第一个位置
pos = 0
print('  挑位置 0 手算一遍，核对上面的第一行：')
for k in range(V):
    terms = [h[0, pos, j].item() * head.weight[k, j].item() for j in range(D)]
    s = sum(terms)
    print(f"    分数[{k}] = " + ' + '.join(f'({t:+.4f})' for t in terms)
          + f' = {s:+.4f}')
print()

print('=' * 56)
print('4. softmax：每行 4 个分数 -> 4 个概率')
print('=' * 56)
p = torch.softmax(z, dim=-1)
print(f'形状 {tuple(z.shape)} -> {tuple(p.shape)}，每行加起来 = 1')
print()
for pos in range(4):
    row = z[0, pos]
    ex = torch.exp(row)
    pr = p[0, pos]
    print(f"  位置 {pos}: 分数 " + '  '.join(f'{x:+.4f}' for x in row.tolist()))
    print(f"            exp  " + '  '.join(f'{x:.4f}' for x in ex.tolist())
          + f"   合计 {ex.sum():.4f}")
    print(f"           概率 " + '  '.join(f'{x:.4f}' for x in pr.tolist())
          + f"   合计 {pr.sum():.4f}")
    print()

print('=' * 56)
print('5. 挑出正确答案那一格，取 -ln，四个位置平均')
print('=' * 56)
print(f'{"位置":<5}{"正确答案":<9}{"模型给它的概率":<15}{"罚分 -ln(p)":<12}')
total = 0.0
for pos in range(4):
    ans = Y[0, pos].item()
    prob = p[0, pos, ans].item()
    pen = -torch.log(p[0, pos, ans]).item()
    total += pen
    print(f'{pos:<7}{vocab[ans]:<11}{prob:<17.4f}{pen:<12.4f}')
print()
print(f'  平均罚分 = loss = {total / 4:.4f}')
print()

z_flat = z.view(-1, V)
loss_lib = F.cross_entropy(z_flat, Y.view(-1))
print(f'  库函数 F.cross_entropy 算出来 = {loss_lib.item():.4f}')
print()
print(f'  参考：4 个字符完全瞎猜的 loss = ln(4) = {torch.log(torch.tensor(4.0)).item():.4f}')
