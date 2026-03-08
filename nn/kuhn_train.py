# train_kuhn.py
# -------------------------
# 使用 REINFORCE 算法训练一个打 Kuhn 扑克的策略网络。
#
# 设计思路：
#   - 使用 PolicyNet + TrainablePolicy，封装为 StrategyPlayer 加入环境
#   - 对手默认使用 RandomPolicy（你可以切换为 NashPolicy 做更强对手训练）
#   - 每局结束后，先手玩家得到一个总 reward（payoff）
#   - 对该局中所有 log_prob 乘以同一个 reward 进行更新
#
# 未来扩展：
#   - 可以把 KuhnGame 抽象成更通用的“牌局环境”，以支持更多牌型 / 公共牌 / 多轮下注
#   - 状态编码 encode_state 可以随游戏复杂度增加而扩展

import torch
import torch.optim as optim

from kuhn_card import KuhnGame
from strategies import (
    PolicyNet,
    StrategyPlayer,
    RandomPolicy,
    NashPolicy,
    NeuralPolicy,
    encode_state,
)


class TrainablePolicy:
    """
    可训练策略：封装 PolicyNet + REINFORCE 所需的 log_prob 缓存。
    只关心“如何根据状态选动作，并把该动作的 log_prob 存起来”。
    """

    def __init__(self, model: PolicyNet, device: torch.device | None = None) -> None:
        self.model = model
        self.saved_log_probs = []
        self.device = device if device is not None else torch.device("cpu")
        self.model.to(self.device)

    def compute_action(self, state: dict) -> int:
        """
        用当前策略对状态采样一个动作，并缓存其 log_prob。
        返回：0=CHECK, 1=BET
        """
        x = encode_state(state["card"], state["history"], state["player_id"]).to(self.device)
        logits = self.model(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        self.saved_log_probs.append(dist.log_prob(a))
        return int(a.item())

    def clear_buffer(self) -> None:
        self.saved_log_probs.clear()


def train(
    rounds: int = 20000,
    lr: float = 5e-4,
    save_path: str = "kuhn_policy.pt",
    use_nash_opponent: bool = True,
) -> PolicyNet:
    """
    训练入口：
      - rounds: 训练局数
      - lr: 学习率
      - save_path: 模型保存路径
      - use_nash_opponent: 若为 True，对手采用 NashPolicy；否则为 RandomPolicy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 构建模型与可训练策略
    model = PolicyNet(input_dim=12, hidden=64, action_dim=2)
    policy = TrainablePolicy(model, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    opponent = NeuralPolicy(model, greedy=False)

    # 2. 构建对手策略
    if use_nash_opponent:
        opponent_policy = NashPolicy()
    else:
        opponent_policy = RandomPolicy()

    # 3. 构建环境并注册玩家
    env = KuhnGame()
    # player0: 训练中 agent
    env.add_player(StrategyPlayer(policy))
    # player1: 对手
    env.add_player(StrategyPlayer(opponent_policy))

    # 4. 循环训练
    for ep in range(rounds):
        # 玩一局牌，env.start(1) 返回 [先手收益, 后手收益]
        score = env.start(1)
        reward = score[0]  # 先手玩家（训练对象）的总奖励

        # REINFORCE: 对整局的所有动作使用同一个 reward
        if policy.saved_log_probs:
            loss = 0.0
            for lp in policy.saved_log_probs:
                loss = loss - lp * reward  # maximize E[reward] -> minimize -log_prob * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 清空缓存，为下一局做准备
        policy.clear_buffer()

        # 适当打印一下训练进度
        if (ep + 1) % 500 == 0:
            print(f"[Episode {ep + 1}/{rounds}] last_reward = {reward}")

    # 5. 保存模型参数
    torch.save(model.state_dict(), save_path)
    print("训练完成，策略已保存为", save_path)

    return model


if __name__ == "__main__":
    # 示例：用随机对手训练 20000 局
    train(rounds=50000, lr=5e-4, save_path="kuhn_policy.pt", use_nash_opponent=False)
