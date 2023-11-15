import numpy as np
import matplotlib.pyplot as plt

from settings import settings

kohonen_predict_accuracies = [48.4, 47.44, 50, 47.6, 45.99]

kmeans_predict_accuracies = [38.3, 40.38, 40.54, 38.3, 40.22]

hierarchical_predict_accuracies = [40.98, 40.98, 40.98, 40.98, 40.98] 

kohonen_mean = np.mean(kohonen_predict_accuracies)
kohonen_std = np.std(kohonen_predict_accuracies)

kmeans_mean = np.mean(kmeans_predict_accuracies)
kmeans_std = np.std(kmeans_predict_accuracies)

hierarchical_mean = np.mean(hierarchical_predict_accuracies)
hierarchical_std = np.std(hierarchical_predict_accuracies)

accuracies = [kohonen_mean, kmeans_mean, hierarchical_mean]
stds = [kohonen_std, kmeans_std, hierarchical_std]
labels = ['Kohonen', 'K-means', 'Hierarchical']

x_pos = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(x_pos, accuracies, yerr=stds, capsize=5, color='skyblue', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.tick_params(axis='x', labelsize=14)

plt.tight_layout()
plt.savefig(f"{settings.Config.out_dir}/predict_accuracies.png")
plt.close()
