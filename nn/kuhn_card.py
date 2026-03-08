# kuhn_card.py
# -------------------------
# 核心游戏逻辑：Kuhn 扑克环境
# 支持两名玩家，牌为 {1, 2, 3}，1 最小 3 最大
# 动作只有两种：
#   - CHECK: 未面对下注时为“过牌”，面对下注时为“弃牌（fold）”
#   - BET:   未面对下注时为“下注”，面对下注时为“跟注（call）”
#
# 规则（标准 Kuhn 变体，简化版）：
#   - 两人各自自动下前注 1（我们只体现在收益上，不单独记筹码）
#   - 若双方一路 check 到底（Check, Check）：摊牌，胜者净赢 +1
#   - 若一人下注而对手弃牌（… Bet, Check）：下注者净赢 +1
#   - 若有下注且被跟注（… Bet, Bet）：摊牌，胜者净赢 +2
#
# 注意：这里的 get_payoff 正确区分了“谁下注、谁弃牌”，避免了之前的错误。

import random
from enum import IntEnum
from typing import List, Optional


random.seed()


class Action(IntEnum):
    """动作：CHECK=0, BET=1"""
    CHECK = 0  # 没有下注时是“过牌”，面对下注时是“弃牌”
    BET = 1    # 没有下注时是“下注”，面对下注时是“跟注”

    @staticmethod
    def num() -> int:
        return 2


class ActionHistory:
    """记录一局游戏中发生的动作序列"""

    def __init__(self) -> None:
        # raw 中存的是 int(0/1)，但可用 Action(value) 恢复成枚举
        self.raw: List[int] = []

    def add(self, action: Action) -> None:
        """添加一个动作"""
        self.raw.append(int(action))

    def clone(self) -> "ActionHistory":
        """深拷贝历史（返回新的 ActionHistory 对象）"""
        h = ActionHistory()
        h.raw = list(self.raw)
        return h


class PlayerBase:
    """
    玩家基类，所有策略玩家需要继承此类。

    生命周期：
      - on_register(pid): 注册玩家编号（0/1），同时固定先手/后手身份
      - on_start(card):   一局开始时发牌
      - decide_action(history): 当前轮到该玩家，根据历史决定动作
      - handle_result(history, payoff): 一局结束时接收结果
    """

    def on_register(self, pid: int) -> None:
        self.player_id = pid  # 0 = 先手玩家，1 = 后手玩家

    def on_start(self, card: int) -> None:
        self.card = card      # 手牌 1/2/3

    def decide_action(self, action_history: ActionHistory) -> Action:
        raise NotImplementedError

    def handle_result(self, action_history: ActionHistory, payoff: int) -> None:
        # 默认不做记录，你可以在子类中记录对局、收益等信息
        pass


def get_payoff(action_history: ActionHistory, cards: List[int]) -> Optional[int]:
    """
    根据动作历史和手牌计算先手玩家的收益。
    返回值含义：先手玩家的净收益（后手玩家收益为其相反数）。

    动作模式（只看最后两步即可确定结果）：
      - Check, Check          -> 摊牌，底池=2，胜者 +1，败者 -1
      - ..., Bet, Check(fold) -> 弃牌方输掉 1，下注者 +1
      - ..., Bet, Bet(call)   -> 摊牌，底池=4，胜者 +2，败者 -2
    """
    h = action_history.raw
    n = len(h)

    # 少于 2 个动作一定未结束
    if n < 2:
        return None

    a1 = Action(h[-2])
    a2 = Action(h[-1])

    # 1) 连续 check → 摊牌
    if a1 == Action.CHECK and a2 == Action.CHECK:
        return 1 if cards[0] > cards[1] else -1

    # 2) ... Bet → Check(fold) → 下注者赢 1
    if a1 == Action.BET and a2 == Action.CHECK:
        # 下注者是倒数第二个动作的玩家（索引 n-2）
        bet_player = (n - 2) % 2  # 0=先手, 1=后手
        return 1 if bet_player == 0 else -1

    # 3) ... Bet → Bet(call) → 摊牌，底池=4
    if a1 == Action.BET and a2 == Action.BET:
        return 2 if cards[0] > cards[1] else -2

    # 其它情况（理论上 Kuhn 不会出现），视为未结束
    return None


class KuhnGame:
    """
    Kuhn 扑克环境：
      - 固定 2 名玩家
      - 牌堆为 [1, 2, 3]，每局洗牌，按顺序发给玩家0、玩家1
      - 玩家按顺序行动（先手玩家 id = 0）
    """

    def __init__(self) -> None:
        self.cards = [1, 2, 3]
        self.players: List[PlayerBase] = []
        self.history = ActionHistory()

    def add_player(self, player: PlayerBase) -> None:
        """注册玩家，顺序即为行动顺序（第一个加入的是先手）"""
        assert len(self.players) < 2, "KuhnGame 只支持 2 名玩家"
        pid = len(self.players)
        player.on_register(pid)
        self.players.append(player)

    def _play_one_hand(self) -> List[int]:
        """
        进行一局牌，返回 [先手收益, 后手收益]
        """
        assert len(self.players) == 2, "必须先注册 2 名玩家"

        # 洗牌并发牌
        # --- 随机决定先手 / 后手 ---
        order = [0, 1]
        random.shuffle(order)     # e.g., [1,0] 代表玩家1先手

        self.history = ActionHistory()

        # 按随机顺序分配手牌
        random.shuffle(self.cards)
        p0, p1 = order[0], order[1]

        # 给先行动者发 cards[0]，后行动者发 cards[1]
        self.players[p0].on_start(self.cards[0])
        self.players[p1].on_start(self.cards[1])

        # 当前行动者设为先手
        current_player = p0

        payoff = None
        
        while payoff is None:
            player = self.players[current_player]
            action = player.decide_action(self.history)
            self.history.add(action)

            payoff = get_payoff(self.history, self.cards)
            current_player = p0 if current_player == p1 else p1

        # payoff 是先手玩家的收益
        pay = [0, 0]
        pay[p0] = payoff      # p0 是先手
        pay[p1] = -payoff     # p1 是后手


        for pid, p in enumerate(self.players):
            p.handle_result(self.history, pay[pid])

        return pay

    def start(self, rounds: int = 1) -> List[int]:
        """
        进行指定局数的游戏，返回累计得分 [先手累计, 后手累计]
        """
        score = [0, 0]

        for _ in range(rounds):
            pay = self._play_one_hand()
            score[0] += pay[0]
            score[1] += pay[1]

        return score
