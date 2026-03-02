import matplotlib.pyplot as plt

def plot_trust(trust_scores):
    plt.figure(figsize=(8,4))
    plt.plot(trust_scores)
    plt.title("Trust Score Over Time")
    plt.xlabel("Sample")
    plt.ylabel("Trust")
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

def plot_drift(p_values):
    plt.figure(figsize=(8,4))
    plt.plot(p_values)
    plt.title("Drift Detection (p-values)")
    plt.xlabel("Step")
    plt.ylabel("p-value")
    plt.axhline(0.01, linestyle="--")
    plt.grid(True)
    plt.show()