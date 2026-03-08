from strategies import RandomPolicy, NashPolicy, NeuralPolicy, PolicyNet, StrategyPlayer
import torch
from kuhn_card import KuhnGame

# 随机
ram = StrategyPlayer(RandomPolicy())

# Nash
nash = StrategyPlayer(NashPolicy())

# 已训练好的 NN (greedy eval)
nn_model = PolicyNet(input_dim=12)
nn_model.load_state_dict(torch.load("kuhn_policy.pt"))
ai = StrategyPlayer(NeuralPolicy(nn_model, greedy=True))

env = KuhnGame()
env.add_player(ai)
env.add_player(nash)

print(env.start(10000))
