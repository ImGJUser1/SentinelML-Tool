import numpy as np
import matplotlib.pyplot as plt


def trust_calibration(trust_scores, correct):

    bins = np.linspace(0, 1, 10)

    bin_acc = []
    bin_centers = []

    for i in range(len(bins)-1):

        mask = (trust_scores >= bins[i]) & (trust_scores < bins[i+1])

        if mask.sum() == 0:
            continue

        acc = correct[mask].mean()

        bin_acc.append(acc)
        bin_centers.append((bins[i] + bins[i+1]) / 2)

    plt.plot(bin_centers, bin_acc, marker="o")

    plt.plot([0,1],[0,1], linestyle="--")

    plt.xlabel("Trust Score")
    plt.ylabel("Actual Accuracy")
    plt.title("Trust Calibration Curve")

    plt.show()