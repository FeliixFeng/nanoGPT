# practice

在 nanoGPT 上做练习的地方。原版代码尽量不动，所有改动落在这里。

运行环境：项目根目录的 `.venv`（torch 2.15 / MPS）。
数据：`data/shakespeare_char/`，已经生成过一次，重来就 `python data/shakespeare_char/prepare.py`。

---

## 进度

| # | 练什么 | 状态 | 产出 |
|:--|:-----|:----:|:-----|
| 01 | 数据是怎么喂进去的 | 完成 | 看懂 x/y 为什么错开一格 |
| 02 | 跑通第一次训练 | 完成 | 一个 checkpoint + 一段能读的文本 |
| 03 | 一次迭代到底干了什么 | 待做 | 不看代码说出那五个动作 |
| 04 | 改一个旋钮做对照 | 完成 | 四段文本的 loss 对照 + 一句结论 |
| 05 | 换成自己的语料 | 待做 | 一个只会写我这个领域的模型 |

---

## 记录

每个练习做完，在下面加一节。别写过程流水，写这四行就够：

```
### 0X 标题
- 日期：
- 做了什么：
- 卡在哪 / 猜错了什么：
- 明白了什么（一句话）：
```

---

### 01 数据是怎么喂进去的
- 日期：2026-11-07
- 做了什么：拆开 meta.pkl（65 个字符的编号表）+ train.bin（100 万个 uint16）；从第 100 个位置切出 x=[1 39 56 43]、y=[39 56 43 1]，解码是 `" are"` / `"are "`
- 卡在哪 / 猜错了什么：`open` 模式写成 `'b'` 报 ValueError（要 `'rb'`）；以为概率和字符有名字对应，其实是**靠下标**——答案存的是编号 1，取 `p[1]`，模型最看好谁都无所谓
- 明白了什么（一句话）：y 错开一格是为了**把答案从输入里拿掉**，否则模型照抄一遍就能得 0 分

---

### 02 跑通第一次训练
- 日期：2026-11-07
- 做了什么：从零训 1000 步，M3 上约 20 分钟。train 4.2872 → 1.2734，val 4.2810 → 1.5471，产出 `ckpt.pt`（123MB，大头是优化器动量）
- 卡在哪 / 猜错了什么：以为起点是 4.17（实测 4.29，随机初始化会略高于 ln(65)）；以为要跑满 5000 步（val 在 900 步后基本不动，最后 100 步只降 0.002）；`sample.py` 报 `No module named tiktoken`
- 明白了什么（一句话）：曲线前快后慢——前 20 步掉 1.5（学字符频率，不需要上下文），后面靠上下文只能一点点挪；**过拟合看 val 自己有没有回头，不看 train/val 差值**

复现命令（`out-*/` 不进 git，clone 下来要自己先训一次）：

```bash
uv run python train.py config/train_shakespeare_char.py \
  --device=mps --compile=False --dtype=float32 \
  --eval_iters=10 --eval_interval=100 --log_interval=20 \
  --max_iters=1000 --out_dir=out-shakespeare-char-1k
```

---

### 04 改一个旋钮做对照（temperature）
- 日期：2026-11-07
- 做了什么：同一个 checkpoint，四个温度各生成一段，再把四段文本喂回模型算 loss（`04_compare_temperature.py`）

  | T | 输出 | loss |
  |:--|:--|:--|
  | 0.1 | 复读机，"the country of the countryman the world" 反复 | **0.752** |
  | 0.5 | 语法通、模板套接，"the world of this daughter" | 0.914 |
  | 0.8 | 半通，有造词（`castless`、`slainment`） | — |
  | 1.5 | 拼写全崩（`Uncsnoranquion`），标点位置却没错 | 2.208 |
  | — | 真实莎士比亚原文 | **1.333** |

- 卡在哪 / 猜错了什么：**猜错了 loss 和"好"的关系**——以为 loss 低就是学得好，实测复读机 0.752 比真莎士比亚 1.333 还低。另以为高温会全乱，实际标点、句首大写、词长分布全保住了（温度只冲垮"选哪个字母"这一层）
- 明白了什么（一句话）：**loss 是"文本有多符合模型预期"，不是"文本有多好"**；模型最确定的东西恰好是最平庸的东西，所以不能用 loss 挑生成结果

采样的完整命令（四档都跑过，随手能重来）：

```bash
uv run python sample.py --out_dir=out-shakespeare-char-1k --device=mps --dtype=float32 \
  --num_samples=1 --start="First Citizen:" --temperature=0.5
```

---
