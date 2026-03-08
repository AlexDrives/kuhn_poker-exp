import torch
from strategies import PolicyNet, encode_state, NashPolicy, NeuralPolicy
from kuhn_card import Action

# -------------------------------------------------------------
# 工具：从 AI 模型计算 CHECK/BET 概率（softmax 后）
# -------------------------------------------------------------
def get_ai_probs(model, card, history, player_id):
    x = encode_state(card, history, player_id)
    logits = model(x)
    probs = torch.softmax(logits, dim=0)
    return float(probs[0]), float(probs[1])  # CHECK, BET


# -------------------------------------------------------------
# 工具：从纳什策略计算 CHECK/BET 概率
# -------------------------------------------------------------
def get_nash_probs(nash, card, history, player_id, trials=30000):
    # 因为纳什是混合策略，用采样估计概率（30000 次足够稳定）
    cnt_check = 0
    cnt_bet = 0
    for _ in range(trials):
        a = nash.compute_action({
            "card": card,
            "history": history,
            "player_id": player_id
        })
        if a == Action.CHECK:
            cnt_check += 1
        else:
            cnt_bet += 1
    s = cnt_check + cnt_bet
    return cnt_check/s, cnt_bet/s


# -------------------------------------------------------------
# 枚举所有信息集
# -------------------------------------------------------------
def all_infosets():
    cards = [1, 2, 3]  # J Q K
    histories = [
        [],                # 开局先手
        [Action.CHECK],    # 后手在先手 check 后
        [Action.BET],      # 后手面对先手 bet
        [Action.CHECK, Action.BET],  # 先手 check, 后手 bet
    ]
    # 对应的 player 必须合法
    valid = []

    for h in histories:
        length = len(h)
        # 先手在偶数步行动
        if length == 0:
            valid.append((h, 0))
        elif length == 1:
            valid.append((h, 1))  # 后手行动
        elif length == 2:
            valid.append((h, 0))  # 再轮到先手
    # 返回 (history, acting_player)
    return valid


# -------------------------------------------------------------
# 主函数：对比 AI vs Nash
# -------------------------------------------------------------
def evaluate(model_path="kuhn_policy.pt"):
    model = PolicyNet(input_dim=12, hidden=64, action_dim=2)
    model.load_state_dict(torch.load(model_path))
    nash = NashPolicy()

    print("======== 对比 AI 策略与 纳什策略（CHECK/BET 概率）========\n")

    for card in [1, 2, 3]:
        print(f"\n===== 手牌 {card} (1=J,2=Q,3=K) =====")
        for hist, pid in all_infosets():
            # 得到 AI 概率
            ai_p_check, ai_p_bet = get_ai_probs(model, card, hist, pid)
            # 得到 Nash 概率（采样估计）
            nash_p_check, nash_p_bet = get_nash_probs(nash, card, hist, pid)

            # 格式化历史动作
            hist_str = "[" + ",".join("C" if a == 0 else "B" for a in hist) + "]"

            print(f"信息集: 手牌={card}, 玩家={pid}, 历史={hist_str}")
            print(f"  AI策略:    CHECK={ai_p_check:.3f}, BET={ai_p_bet:.3f}")
            print(f"  Nash策略:  CHECK={nash_p_check:.3f}, BET={nash_p_bet:.3f}")
            print("  ------------------------------------")

    print("\n评估完成！")

if __name__ == "__main__":
    # 模型路径按你实际训练时保存的来
    evaluate("kuhn_policy.pt")
