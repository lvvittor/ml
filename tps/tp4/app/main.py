import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from settings import settings
import visualization as vis
from kohonen import Kohonen
from kmeans import KMeans

def main():
	df = pd.read_csv(settings.Config.data_dir+"/movie_data.csv", delimiter=";")

	# Analyze data and clean it
	var_df, std_df = data_preprocessing(df)

	match settings.exercise:
		case 1:
			kohonen(std_df)
		case 2:
			kmeans(std_df)
		case _:
			raise ValueError("Invalid exercise number")


def data_preprocessing(df):
	print(df.info())

	# Keep only `float64` columns and drop rows with missing values
	var_df = df.select_dtypes(include='float64').dropna()

	print(var_df.info())

	# Histograms + Kernel Density Estimation (KDE) of numerical variables

	# Note: a KDE plot is produced by drawing a small continuous curve (also called kernel) for every individual
	# data point along an axis, all of these curves are then added together to obtain a single smooth density
	# estimation. Unlike a histogram, KDE produces a smooth estimate.

	vis.plot_hist_kde(var_df, "budget", "Presupuesto en dólares")
	vis.plot_hist_kde(var_df, "popularity", "Popularidad")
	vis.plot_hist_kde(var_df, "production_companies", "Compañías productoras")
	vis.plot_hist_kde(var_df, "production_countries", "Países productores")
	vis.plot_hist_kde(var_df, "revenue", "Ingresos en dólares")
	vis.plot_hist_kde(var_df, "runtime", "Duración en minutos")
	vis.plot_hist_kde(var_df, "spoken_languages", "Idiomas")
	vis.plot_hist_kde(var_df, "vote_average", "Voto Promedio")
	vis.plot_hist_kde(var_df, "vote_count", "Cantidad de votos")

	# Standardize variables
	scaler = StandardScaler()
	std_df = scaler.fit_transform(var_df)
	std_df = pd.DataFrame(data=std_df, columns=var_df.columns.values)

	# Boxplot
	vis.boxplot(std_df, standardized=True)

	return var_df, std_df


def kohonen(df):
	std_vars = df.to_numpy()

	k = 5
	epochs = 200_000

	# TODO: try different values for `k`, `epochs`, initial `eta` and initial `R`
	kohonen = Kohonen(k, std_vars)
	kohonen.train(epochs)

	# Get the winner neuron for each input
	winner_neurons = kohonen.map_inputs(std_vars)
	umatrix = kohonen.get_umatrix()

	# Plot results
	vis.neuron_heatmap(winner_neurons, k)
	vis.u_matrix(umatrix)


def kmeans(df):
	k = 3
	max_epochs = 100

	kmeans = KMeans(k, df)
	labels, _, epochs = kmeans.train(max_epochs)

	print(f"Finished in {epochs} epochs")

	values, counts = np.unique(labels, return_counts=True)
	occurrences = dict(zip(values, counts))

	print("Number of observations in each cluster:")
	print(occurrences)

	vis.plot_pca(df, labels)

	# TODO: show principal components

	# --------- Elbow method (get best `k`) ---------

	# Create an elbow plot for k from 1 to 10
	k_values = range(1, 11)
	variations = []

	for _k in k_values:
		kmeans = KMeans(_k, df)
		labels, inertia, epochs = kmeans.train()
		variations.append(inertia)

		if settings.verbose:
			print(f"Ended KMeans with k={k} in {epochs} epochs.")

	vis.elbow_plot(k_values, variations) # k = 6 looks best

	# --------- Try different centroids ---------

	# TODO: try different initial centroids

	print("Centroids")
	print(kmeans.centroids)

	# 1. Plot average observations per cluster + error

	# 2. Plot average difference between centroids


if __name__ == "__main__":
    main()
