import matplotlib.pyplot as plt
import numpy as np

no_mae = np.array([0.745, 0.697, 0.675, 0.619])
mae1 = np.array([0.753, 0.718, 0.706, 0.668])
mae2 = np.array([0.750, 0.721, 0.707, 0.670])

size = np.array([100, 10, 5, 1])

plt.tight_layout()
plt.xlabel("Wykorzystana część zbioru treningowego (~1000 utworów)")
plt.xscale("log")
plt.ylabel("WCSR")
plt.xticks([100, 10, 5, 1], ["100%", "10%", "5%", "1%"])
plt.plot(size, no_mae, "D-", label="Od zera")
plt.plot(size, mae1, "o-", label="Finetuning 1")
plt.plot(size, mae2, "s-", label="Finetuning 2")
plt.legend()
plt.grid()
plt.savefig("thesis/images/final_results.png")
