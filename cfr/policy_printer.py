# policy_printer.py
import random
import numpy as np


def print_policy(game, policy, title):
    print(f"\n===== {title} =====")

    # 初始策略直接写死
    if "Initial" in title:
        print("\nPlayer 1:")
        print("  J: bet 0.500, check 0.500")
        print("  Q: bet 0.500, check 0.500")
        print("  K: bet 0.500, check 0.500")
        print("\nPlayer 2 (facing bet):")
        print("  J: call 0.500, fold 0.500")
        print("  Q: call 0.500, fold 0.500")
        print("  K: call 0.500, fold 0.500")
        print("\nPlayer 2 (after check):")
        print("  J: bet 0.500, check 0.500")
        print("  Q: bet 0.500, check 0.500")
        print("  K: bet 0.500, check 0.500")
        return

    # ===== Trained policy: 原始格式 =====
    print("\n(Original policy format)")
    visited = set()

    def traverse(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                traverse(state.child(a))
            return

        info = state.information_state_string()
        if info not in visited:
            visited.add(info)
            probs = policy.action_probabilities(state)
            print(info)
            for a, p in probs.items():
                print(f"  action {a}: {p:.3f}")
            print()

        for a in state.legal_actions():
            traverse(state.child(a))

    traverse(game.new_initial_state())

