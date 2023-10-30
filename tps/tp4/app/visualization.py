import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from settings import settings

def plot_hist_kde(df, col, xlabel):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Frecuencia", fontsize=14)
    plt.savefig(f"{settings.Config.out_dir}/{col}_hist.png")
    plt.clf()


def boxplot(variables_data, standardized=None):
	plt.figure(figsize=(18, 6))
	sns.boxplot(data=variables_data, palette='pastel')
	standardized = "no" if not standardized else ""
	plt.title(f"Atributos {standardized} estandarizados")
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12) 
	plt.tight_layout()
	plt.savefig(f"{settings.Config.out_dir}/{standardized}StandardizedBoxplot.png")
	plt.clf()


def neuron_heatmap(winner_neurons, k):
	# Create an empty k x k matrix to store the counts
	matrix = np.zeros((k, k), dtype=int)

	# Count the number of entries per cell
	for idx in winner_neurons:
		row, col = divmod(idx, k)
		matrix[row, col] += 1

	# Plot the heatmap
	plt.figure(figsize=(8, 8))
	plt.imshow(matrix, cmap='OrRd')

	# Set tick positions and labels
	plt.xticks(np.arange(k))
	plt.yticks(np.arange(k))
	plt.gca().set_xticklabels(np.arange(k) + 1)
	plt.gca().set_yticklabels(np.arange(k) + 1)

	# Add color bar
	plt.colorbar()

	# Show the plot
	plt.tight_layout()
	plt.savefig(f"{settings.Config.out_dir}/kohonen_heatmap.png")
	plt.clf()


def u_matrix(umatrix):
	plt.figure(figsize=(8, 8))
	cmap = plt.cm.get_cmap('Greys')

	plt.imshow(umatrix, cmap=cmap.reversed())

	plt.colorbar()
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
	plt.savefig(f"{settings.Config.out_dir}/u_matrix.png")
	plt.clf()
