import matplotlib.pyplot as plt

k_values = [5, 10, 20, 30, 50, 80, 100]
auroc_scores = [0.9612, 0.9745, 0.9821, 0.9868, 0.9810, 0.9740, 0.9705]

plt.figure(figsize=(8, 5), dpi=150)
plt.plot(k_values, auroc_scores, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

plt.annotate(f'Optimal K=30\n(AUROC={auroc_scores[3]:.4f})', 
             xy=(30, 0.9868), xytext=(57, 0.98),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

plt.title('Impact of Neighborhood Size $K$ on OOD Detection (AUROC)', fontsize=14, fontweight='bold')
plt.xlabel('Neighborhood Size ($K$)', fontsize=12)
plt.ylabel('AUROC Score', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(0.95, 1.0)

plt.savefig('sensitivity_k.pdf', bbox_inches='tight')
plt.show()