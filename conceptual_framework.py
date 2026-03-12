import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=[[-3, 9], [5, 2], [-7, -7]], cluster_std=1.0, random_state=42)
ood_point = np.array([[0, 10]]) 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5, s=20)
ax1.scatter(ood_point[:, 0], ood_point[:, 1], c='red', marker='x', s=100, label='OOD Sample')
ax1.axhline(5, color='black', linestyle='--', alpha=0.3)
ax1.axvline(0, color='black', linestyle='--', alpha=0.3)
ax1.set_title("Softmax Decision Boundaries\n(Unbounded Hyperplanes)", fontsize=18)
ax1.set_xlim(-11, 8); ax1.set_ylim(-11, 13)
ax1.legend()

ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5, s=20)
ax2.scatter(ood_point[:, 0], ood_point[:, 1], c='red', marker='x', s=100, label='OOD (In Boundary)')

centers_fixed = [[-3, 9], [5, 2], [-7, -7]]
for i, center in enumerate(centers_fixed):
    circle = plt.Circle(center, 2.5, color='blue', fill=False, linestyle='-', linewidth=2, alpha=0.6)
    ax2.add_artist(circle)

ax2.set_title("Topological Rough Set\n(Local Knowledge Granules)", fontsize=18)
ax2.set_xlim(-11, 8); ax2.set_ylim(-11, 13)

plt.tight_layout()
plt.savefig('conceptual_framework.pdf')
plt.show()