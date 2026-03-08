# strategies.py
# -------------------------
# 包含：
#   - PolicyNet：神经网络策略
#   - encode_state：将状态编码为固定长度向量
#   - NeuralPolicy / Trainable 接口
#   - 一个“接近 Kuhn Nash”的手工策略 NashPolicy（用于对战评估）
#   - RandomPolicy / HumanPolicy
#   - StrategyPlayer：将策略包装为游戏玩家（PlayerBase）

import random
from typing import Dict, List

import torch
import torch.nn as nn

from kuhn_card import PlayerBase, Action, ActionHistory


# =========================
# 一、神经网络策略
# =========================

class PolicyNet(nn.Module):
    """
    简单三层全连接网络：
      输入：12 维状态向量
      输出：2 维动作 logits（对应 [CHECK, BET]）
    """

    def __init__(self, input_dim: int = 12, hidden: int = 64, action_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (12,) 或 (batch, 12)
        返回：未归一化 logits
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def encode_state(card: int, history_raw: List[int], player_id: int) -> torch.Tensor:
    """
    将当前状态编码为一个 12 维向量，用于神经网络输入。
    结构：
      [hand_onehot(3)] + [4 steps x 2bits = 8] + [is_player0(1)] = 12

    - hand_onehot: 手牌 1/2/3 对应 [1,0,0] / [0,1,0] / [0,0,1]
    - 动作历史：
        最多取最近 4 个动作，从旧到新排列
        对每个动作用两位编码：[is_check, is_bet]
        CHECK -> [1, 0], BET -> [0, 1]，不足位置补 [0,0]
    - is_player0: 如果该状态对应玩家0（先手），则为 1，否则为 0
    """
    # 1) 手牌 one-hot
    hand = [0.0, 0.0, 0.0]
    assert card in (1, 2, 3), f"非法手牌: {card}"
    hand[card - 1] = 1.0

    # 2) 动作历史编码（最多 4 步，旧 → 新）
    last4 = history_raw[-4:]
    padded = [None] * (4 - len(last4)) + list(last4)  # [None, None, ..., a_k, ..., a_n]

    hist_vec: List[float] = []
    for v in padded:
        if v is None:
            hist_vec.extend([0.0, 0.0])
        else:
            a = Action(v)
            if a == Action.CHECK:
                hist_vec.extend([1.0, 0.0])
            elif a == Action.BET:
                hist_vec.extend([0.0, 1.0])
            else:
                hist_vec.extend([0.0, 0.0])

    # 3) 玩家身份标志：先手玩家=1，后手玩家=0
    player_flag = [1.0 if player_id == 0 else 0.0]

    vec = hand + hist_vec + player_flag
    assert len(vec) == 12, f"状态向量长度错误: {len(vec)}"

    return torch.tensor(vec, dtype=torch.float32)


class NeuralPolicy:
    """
    神经网络策略封装：
      - greedy=False 时使用采样（训练时）
      - greedy=True 时选取最大概率动作（评估时）
    """

    def __init__(self, model: PolicyNet, greedy: bool = False):
        self.model = model
        self.greedy = greedy

    def compute_action(self, state: Dict) -> int:
        """
        state: {
          "card": int,
          "history": List[int],
          "player_id": int,
        }
        返回：0=CHECK, 1=BET
        """
        x = encode_state(state["card"], state["history"], state["player_id"])
        logits = self.model(x)

        if self.greedy:
            return int(torch.argmax(logits).item())
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())


# =========================
# 二、手工策略（接近 Kuhn Nash）
# =========================

class NashPolicy:
    """
    严格正确的 Kuhn Poker 纳什平衡策略。
    使用文献中的 exact Nash Mixing Rates。
    """
    def compute_action(self, state):
        c = state["card"]          # 1=J, 2=Q, 3=K
        h = state["history"]       # 动作历史（整数序列）
        pid = state["player_id"]   # 0/1

        def bet(): return int(Action.BET)
        def chk(): return int(Action.CHECK)
        def mix(p): return bet() if random.random() < p else chk()

        # --- 情况 A：先手开局 ---
        if len(h) == 0:
            if c == 3: return bet()
            if c == 2: return mix(1/3)
            return chk()

        # --- 情况 B：后手面对先手 Check → 后手行动 (history = [CHECK]) ---
        if len(h) == 1 and h[0] == Action.CHECK and pid == 1:
            if c == 3: return bet()
            if c == 2: return mix(1/3)
            return chk()

        # --- 情况 C：面对下注的玩家 (history ends with BET) ---
        if h[-1] == Action.BET:
            # 面对 Bet → Call(Follow) = BET, Fold = CHECK
            if c == 3: return bet()       # K always call
            if c == 2: return mix(1/3)    # Q call 1/3
            return chk()                  # J fold

        # --- 情况 D：先手 Check → 后手 Check → 先手？？ ---
        # 其实标准 Kuhn 在 Check-Check 后直接摊牌，不会轮回到先手
        # 为安全保守策略写一个兜底：
        if len(h) == 1 and h[0] == Action.CHECK and pid == 0:
            if c == 3: return bet()
            if c == 2: return mix(1/3)
            return chk()

        # --- 情况 E：先手 Check → 后手 Bet → 先手行动 (history=[CHECK, BET]) ---
        if len(h) == 2 and h[0] == Action.CHECK and h[1] == Action.BET and pid == 0:
            if c == 3: return bet()
            if c == 2: return mix(1/3)
            return chk()

        return chk()



# =========================
# 三、随机策略 & 人类策略
# =========================

class RandomPolicy:
    """纯随机策略，动作等概率选择"""

    def compute_action(self, state: Dict) -> int:
        return random.randint(0, 1)


class HumanPolicy:
    """人类交互策略，用于手动对战调试"""

    def compute_action(self, state: Dict) -> int:
        card = state["card"]
        history = [Action(a).name for a in state["history"]]

        print(f"你的手牌: {card}")
        print(f"历史动作: {history}  (CHECK=0, BET=1)")
        while True:
            try:
                a = int(input("请输入动作 (0=CHECK 1=BET)："))
                if a in (0, 1):
                    return a
            except ValueError:
                pass
            print("无效输入，请重新输入。")


# =========================
# 四、Player 封装器
# =========================

class StrategyPlayer(PlayerBase):
    """
    封装一个“策略对象”为 PlayerBase，
    使其可以在 KuhnGame 中直接使用。
    策略对象需要实现：compute_action(state: dict) -> int
    """

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def decide_action(self, history: ActionHistory) -> Action:
        state = {
            "card": self.card,
            "history": list(history.raw),
            "player_id": self.player_id,
        }
        a = self.policy.compute_action(state)
        return Action(a)
