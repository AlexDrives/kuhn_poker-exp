# Kuhn Poker 博弈求解实验

`kuhn_poker_exp` 围绕 `Kuhn Poker` 展开，包含两部分工作：

- 手写 Kuhn Poker 环境，并用 PyTorch 实现基于 REINFORCE 的策略训练与评估
- 使用 OpenSpiel 复现 `CFR` 和 `CFR+`，作为对照实验验证博弈求解过程

项目的核心问题是：在这个小型不完全信息博弈中，神经网络策略梯度是否能够稳定逼近 Nash equilibrium；如果不能，问题出现在哪里。

## 第一部分：手写环境与策略梯度

### Kuhn Poker 环境

`nn/kuhn_card.py` 实现了一个两人 Kuhn Poker 环境：

- 牌集为 `{1, 2, 3}`，可对应 `J / Q / K`
- 动作集合只有两个：
  - `CHECK = 0`
  - `BET = 1`

动作语义依赖上下文：

- 在无人下注时，`CHECK` 表示过牌，`BET` 表示下注
- 在面对下注时，`CHECK` 表示弃牌，`BET` 表示跟注

收益规则：

- `Check, Check`：摊牌，赢家净收益 `+1`
- `Bet, Check`：下注方获胜，净收益 `+1`
- `Bet, Bet`：摊牌，赢家净收益 `+2`

### 状态表示与策略

`nn/strategies.py` 中定义了：

- `PolicyNet`
  - 两层隐藏层的全连接网络
  - 输入维度为 `12`
  - 输出两个动作的 logits
- `encode_state(card, history, player_id)`
  - 将当前状态编码为 `12` 维向量
  - 结构为：手牌 one-hot、动作历史编码和玩家身份标记
- `RandomPolicy`
- `HumanPolicy`
- `NashPolicy`
- `NeuralPolicy`
- `StrategyPlayer`

### 训练方式

`nn/kuhn_train.py` 使用最基本的 REINFORCE：

1. 网络根据状态输出动作分布
2. 从分布中采样动作并记录 `log_prob`
3. 对局结束后，用该局 reward 更新整局的动作概率
4. 使用 `Adam` 优化器更新参数

默认训练入口为：

```python
train(rounds=50000, lr=5e-4, save_path="kuhn_policy.pt", use_nash_opponent=False)
```

默认对手是 `RandomPolicy`。

### 评估方式

第一部分提供两类评估：

- `nn/main.py`
  - 加载 `nn/kuhn_policy.pt`
  - 让神经网络策略与 `NashPolicy` 连续对战
- `nn/evaluate_policy.py`
  - 枚举多个信息集
  - 比较 AI 与 Nash 在这些状态下的 `CHECK / BET` 概率

## 第一部分的实验结果

当前仓库中的模型 `nn/kuhn_policy.pt` 在与 `NashPolicy` 的对战中表现不理想。

本地运行 `10` 次 `nn/main.py`，每次对战 `10000` 局，AI 的总收益分别为：

```text
-346, -324, -420, -412, -270, -427, -258, -412, -354, -506
```

统计结果：

- 平均总收益：`-372.9 / 10000` 局
- 单局平均收益：`-0.03729`
- 最好一次：`-258`
- 最差一次：`-506`

这说明当前神经网络策略并没有逼近 Nash，而是稳定落后于 `NashPolicy`。

## 为什么策略梯度没有学到 Nash

基于当前实现，可以看到几个主要原因：

- `nn/main.py` 中使用了 `greedy=True`
  - Kuhn Poker 的 Nash 本质上是混合策略
  - 用 `argmax` 做评估会把混合策略压成确定性策略
- 训练目标是“对固定对手优化收益”，不是“求解零和博弈均衡”
  - 默认对手还是 `RandomPolicy`
  - 即使切换到 `NashPolicy`，本质上也更接近 best response 训练，而不是 equilibrium solving
- 当前 REINFORCE 实现较为朴素
  - 使用整局统一 reward
  - 没有 baseline
  - 没有 advantage
  - 没有熵正则
- 状态表示与环境逻辑存在不一致
  - `encode_state(...)` 使用 `player_id` 作为位置特征
  - 但 `KuhnGame._play_one_hand()` 每局会随机打乱实际行动顺序
  - 注册编号与本局先后手并不稳定对应

这一部分记录的是在当前问题设定下使用策略梯度的实验过程，以及为什么它没有收敛到 Nash。

## 第二部分：OpenSpiel 上的 CFR / CFR+

在手写环境里做完策略梯度实验之后，后续工作转向了 OpenSpiel。目的不是重复实现一套新的训练流程，而是换到一个标准化博弈框架里，直接考察 `CFR` 和 `CFR+` 在 Kuhn Poker 上的收敛行为。

这一部分对应 `cfr/` 目录中的三个脚本：

- `cfr/kuhn_cfr.py`
- `cfr/kuhn_cfr_plus.py`
- `cfr/policy_printer.py`

其中 `cfr/kuhn_cfr.py` 使用 `CFRSolver`，`cfr/kuhn_cfr_plus.py` 使用 `CFRPlusSolver`，`cfr/policy_printer.py` 用来遍历信息集并打印策略分布。

### 实验设置

这一部分直接调用 OpenSpiel 提供的标准 `kuhn_poker` 游戏：

- 通过 `pyspiel.load_game("kuhn_poker")` 加载环境
- 分别运行 `CFR` 和 `CFR+`
- 训练过程中使用 average policy 作为评估对象
- 每隔固定步数记录一次 `exploitability`
- 训练结束后打印初始策略、平均策略和理论 Nash 策略

当前脚本里默认训练 `2000` 轮，并按固定间隔记录收敛曲线。

### 为什么使用 exploitability

这一部分的评估重点不再是“和某个固定对手打了多少分”，而是 `exploitability`。

在零和不完全信息博弈里，`exploitability` 衡量的是一个策略距离均衡还有多远：如果一个策略仍然能被 best response 稳定利用，那么它的 `exploitability` 就不会接近 `0`。对 Kuhn Poker 这样的标准小博弈来说，这个指标比单纯看对战胜负更适合分析求解过程。

### 结果

OpenSpiel 部分的结果主要有三点。

第一，`CFR` 和 `CFR+` 的 `exploitability` 都会随着迭代下降并逐渐接近 `0`，说明这两种方法都能在 Kuhn Poker 上收敛到均衡附近。

第二，`CFR+` 的下降速度更快。和标准 `CFR` 相比，`CFR+` 在前期迭代中更快进入低 exploitability 区间，这和文献里对 `CFR+` 收敛效率的结论一致。

第三，策略分布层面的结果和 Kuhn Poker 的理论结构基本一致。例如：

- 先手拿到 `Q` 时基本总是 `check`
- 先手拿到 `K` 时接近总是 `bet`
- 后手面对下注时，拿到 `J` 倾向于 `fold`，拿到 `K` 倾向于 `call`
- 一些信息集上的混合概率会在理论范围附近波动

这里需要注意，Kuhn Poker 的 Nash equilibrium 不是唯一的单点解，而是一族由参数控制的混合策略。因此，`exploitability` 接近 `0` 并不要求每个信息集上的动作概率都和某一张参考表完全相同；更准确的说法是，策略已经落在 Kuhn Poker 的均衡族附近。

### 这一部分和第一部分的关系

这两部分不是彼此独立的。

第一部分关心的是：用手写环境和策略梯度做 Kuhn Poker，会得到什么样的结果。第二部分关心的是：在标准化工具和经典博弈算法下，这个问题的基准表现是什么。

把两部分放在一起看，结论会更清楚：

- Kuhn Poker 这样的不完全信息博弈，天然要求混合策略
- 仅用当前这套 REINFORCE 实现，很难稳定逼近 Nash
- 换成以 regret minimization 为核心的 `CFR / CFR+` 后，收敛行为会清楚得多

## 如何运行

### 第一部分

至少需要安装 PyTorch：

```bash
pip install torch
```

训练模型：

```bash
cd nn
python kuhn_train.py
```

运行神经网络与 Nash 对战：

```bash
cd nn
python main.py
```

比较 AI 与 Nash 的信息集策略：

```bash
cd nn
python evaluate_policy.py
```

### 第二部分

第二部分依赖 OpenSpiel 的 Python 绑定 `pyspiel`。官方文档目前支持两种常见安装方式。

如果是支持官方二进制包的环境，可以直接安装。根据 OpenSpiel 官方安装文档，这条路径适用于 MacOS 或 Linux 上的 `x86_64` 架构：

```bash
python3 -m pip install open_spiel
python3 -m pip install numpy matplotlib
```

如果二进制包不可用，或者需要从源码安装，则需要本地构建工具链。按照 OpenSpiel 官方文档，至少需要准备：

- `CMake`
- `Clang`，或 `g++ >= 9.2`
- Python 3 开发头文件

在 Ubuntu / Debian 上，常见的安装方式是：

```bash
sudo apt-get install cmake clang python3-dev
python3 -m pip install --no-binary=:open_spiel: open_spiel
python3 -m pip install numpy matplotlib
```

这也是第二部分更适合在 Linux 环境中运行的主要原因：如果需要从源码构建 `pyspiel`，Linux 下准备编译环境通常更直接。

运行 CFR：

```bash
cd cfr
python kuhn_cfr.py
```

运行 CFR+：

```bash
cd cfr
python kuhn_cfr_plus.py
```
