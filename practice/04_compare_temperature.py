"""
04 — 改一个旋钮做对照：temperature 对输出的影响

做的事：
  拿同一个 checkpoint（1000 步、val 1.547），把四段文本喂回去算 loss：

    T=0.1  复读机        "the country of the countryman the world"
    T=0.5  套话          "the world of this daughter"
    T=1.5  乱码          "Uncsnoranquion, for nehing but beat in the Roma"
    原文   莎士比亚真迹   "Before we proceed any further, hear me speak."

  输出是四行 loss。跑完你会看到一个反直觉的结果：
  模型给复读机打的 loss 比给真的莎士比亚还低。

  为什么重要：
  loss 衡量的是"文本有多符合模型的预期"，不是"文本有多好"。
  模型最确定的东西恰恰是它最容易产出的、也最平庸的东西。
  所以"用 loss 挑生成结果"是错的，"loss 低所以模型好"也是错的。

运行：
  cd ~/Code/ai/nanoGPT
  uv run python practice/04_compare_temperature.py
"""

import pathlib
import pickle
import sys

import torch

# 脚本在 practice/ 下，但 model.py 在上一层，得手动加进搜索路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from model import GPT, GPTConfig  # noqa: E402

CKPT = 'out-shakespeare-char-1k/ckpt.pt'
META = 'data/shakespeare_char/meta.pkl'

# 四段文本，都是前面 sample.py 真实跑出来的，不是编的
TEXTS = {
    'T=0.1': """First Citizen:
And the country of the countryman the world.
CORIOLANUS:
The prince of the countryman the countryman was the world?
CORIOLANUS:
The prince of the countryman the countryman the country's good
That w""",
    'T=0.5': """First Citizen:
And not well so for the fault of the court of this death
A good son of blood well this cheek of the bosom,
And the trumpet of the hour of the charm of the country.
COMINIUS:
Sir, for you will say the world of this daughter.
CORIOLANUS:
The country of come the country's brother,""",
    'T=1.5': """First Citizen:
Uncsnoranquion, for nehing but beat in the Roma.
Widow to to'glous Biltol, dowell this,
Deg imaDe waxinacatcle
Heir which unfurnr head and tWoShmen!
Trunpetness'd Tarlis!
Lap Six Of nales; let's a be""",
    '原文': """First Citizen:
Before we proceed any further, hear me speak.
All:
Speak, speak.
First Citizen:
You are all resolved rather to die than to famish?""",
}


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # checkpoint 里 model_args 是个 dict，要先用它构造 GPTConfig 才能建模型
    ckpt = torch.load(CKPT, map_location='cpu')
    model = GPT(GPTConfig(**ckpt['model_args']))
    model.load_state_dict(ckpt['model'])
    model.eval()                      # 关掉 dropout，结果才可比
    model.to(device)

    with open(META, 'rb') as f:
        stoi = pickle.load(f)['stoi']

    # 一次前向最多吃 block_size（256）个字符，超了要截断
    block = model.config.block_size

    print()
    print(f'{"":6s}  {"字符数":>5s}  {"loss":>6s}   不同词/总词')
    print('-' * 46)
    for name, text in TEXTS.items():
        ids = [stoi[c] for c in text if c in stoi][:block]

        # 跟 01 里手写的一样：x 是原串，y 往后挪一格
        x = torch.tensor(ids[:-1])[None].to(device)
        y = torch.tensor(ids[1:])[None].to(device)

        with torch.no_grad():
            _, loss = model(x, y)

        words = text.split()
        r = len(set(words)) / len(words)     # 重复度：1.0 = 一个词都没重复
        print(f'{name:6s}  {len(ids):5d}  {loss.item():6.3f}   {r:.2f}')

    print()
    print('参考：词表 65 个字符，瞎猜的 loss = ln(65) = 4.17')


if __name__ == '__main__':
    main()
