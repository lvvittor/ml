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


def plot_pca(reduced_data, labels):
	# Assuming 'labels' are the cluster labels obtained from k-means
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=labels, cmap='viridis')
	ax.set_xlabel('Principal Component 1')
	ax.set_ylabel('Principal Component 2')
	ax.set_zlabel('Principal Component 3')
	fig.tight_layout()
	plt.savefig(f"{settings.Config.out_dir}/kmeans_pca.png")
	plt.clf()


def plot_pca_weights(df):
	fig, ax = plt.subplots(figsize=(14, 6))

	bar_width = 0.2
	bar_positions = np.arange(len(df.columns))

	colors = ['#FFB2B2', '#9EE09E', '#B2CCFF']

	variances = ["35", "16", "14"]

	for i, (row, color) in enumerate(zip(df.values, colors)):
		ax.bar(
			bar_positions + i * bar_width,
			row,
			width=bar_width,
			label=f'PC {i+1} ({variances[i]}%)',
			color=color
		)

	ax.set_xticks(bar_positions + (len(df.columns) / 2 - 4) * bar_width)
	ax.set_xticklabels(df.columns)
	ax.tick_params(axis='x', labelsize=9)
	ax.set_ylabel('Weights', fontsize=14)

	ax.legend()
	fig.tight_layout()

	plt.savefig(f"{settings.Config.out_dir}/kmeans_pca_weights.png")
	plt.clf()


def elbow_plot(k_values, variations):
	plt.plot(k_values, variations, marker='o')
	plt.xlabel('Number of Clusters (k)', fontsize=14)
	plt.ylabel('Inertia', fontsize=14)
	plt.tight_layout()
	plt.savefig(f"{settings.Config.out_dir}/kmeans_elbow.png")
	plt.clf()


def plot_cluster_observations(cluster_observations):
	clusters = np.arange(len(cluster_observations)) + 1
	count = np.zeros(len(cluster_observations))
	errors = np.zeros(len(cluster_observations))

	for cluster, observations in enumerate(cluster_observations):
		count[cluster] = np.mean(observations)
		errors[cluster] = np.std(observations)
	
	plt.bar(clusters, count, yerr=errors, capsize=5, color='skyblue', edgecolor='black')

	plt.xlabel('Cluster', fontsize=14)
	plt.ylabel('Observaciones', fontsize=14)
	plt.tight_layout()
	plt.savefig(f"{settings.Config.out_dir}/kmeans_obs_per_cluster.png")
	plt.clf()


def plot_centroid_distances(centroid_distances):
	clusters = np.arange(len(centroid_distances)) + 1
	plt.bar(clusters, centroid_distances, capsize=5, color='skyblue', edgecolor='black')
	plt.xlabel('Cluster', fontsize=14)
	plt.ylabel('Distancia entre centroides', fontsize=14)
	plt.tight_layout()
	plt.savefig(f"{settings.Config.out_dir}/kmeans_centroid_distances.png")
	plt.clf()
