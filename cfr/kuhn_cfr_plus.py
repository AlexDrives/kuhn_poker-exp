import pyspiel
import matplotlib.pyplot as plt

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability

from policy_printer import print_policy

# ======================================================
# Policy utilities
# ======================================================


def print_kuhn_nash():
    """Print theoretical Nash equilibrium of Kuhn Poker."""
    print("\n===== Theoretical Nash Policy (Reference) =====")

    print("Player 1:")
    print("  J: bet 1/3, check 2/3")
    print("  Q: bet 0,   check 1")
    print("  K: bet 1,   check 0\n")

    print("Player 2 (facing bet):")
    print("  J: call 0,   fold 1")
    print("  Q: call 1/3, fold 2/3")
    print("  K: call 1,   fold 0\n")

    print("Player 2 (after check):")
    print("  J: bet 0")
    print("  Q: bet 0")
    print("  K: bet 1")


# ======================================================
# Main experiment (CFR+)
# ======================================================
def main():
    # Load game
    game = pyspiel.load_game("kuhn_poker")

    # ⚠️ 唯一的区别：使用 CFR+
    solver = cfr.CFRPlusSolver(game)

    # 1. Initial policy (before training)
    initial_policy = solver.average_policy()
    print_policy(game, initial_policy, "Initial CFR+ Policy (Before Training)")

    # 2. Train CFR+ and record exploitability
    iters = 2000
    xs, ys = [], []

    for i in range(iters):
        solver.evaluate_and_update_policy()
        if i % 10 == 0:
            avg_policy = solver.average_policy()
            e = exploitability.exploitability(game, avg_policy)
            xs.append(i)
            ys.append(e)

    # 3. Trained policy
    trained_policy = solver.average_policy()
    print_policy(game, trained_policy, "Trained CFR+ Average Policy")

    # 4. Plot exploitability curve
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("Kuhn Poker CFR+ Convergence")
    plt.grid()
    plt.savefig("cfrplus_kuhn_exploitability.png", dpi=200)
    print("Saved figure to cfrplus_kuhn_exploitability.png")

    # 5. Nash reference
    print_kuhn_nash()


if __name__ == "__main__":
    main()
