import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from settings import settings
from kmeans import KMeans

# Load the data
movies_df = pd.read_csv(settings.Config.data_dir+"/movie_data.csv", delimiter=";")

# Step 1: Shuffle the data
movies_df = shuffle(movies_df, random_state=42)

# Step 2: Filter the dataset
selected_genres = ['Action', 'Comedy', 'Drama']
subset = movies_df[movies_df['genres'].isin(selected_genres)].dropna()

# Step 3: Split the subset into train and test
train_size = int(0.8 * len(subset))
train_data = subset[:train_size]
test_data = subset[train_size:]

numerical_columns = ["budget","popularity","production_companies","production_countries","revenue","runtime","spoken_languages","vote_average","vote_count"]

scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

# Extract features for clustering (you may need to adjust this based on your dataset)
features = train_data[numerical_columns]

# Step 4: Apply k-means on the training set
k = 3  # match the number of genres

kmeans = KMeans(k, features)

labels, _, _ = kmeans.train()
train_data['Cluster'] = labels

# Step 5: Assign clusters to the test set
test_features = test_data[numerical_columns]

labels = kmeans.predict(test_features)
test_data['Cluster'] = labels

# Step 6: Determine the predominant genre for each cluster based on the training set
cluster_genre_mapping = {}
for cluster in range(k):
    cluster_samples = train_data[train_data['Cluster'] == cluster]
    predominant_genre = cluster_samples['genres'].mode().iloc[0]
    cluster_genre_mapping[cluster] = predominant_genre

# Step 7: Map the cluster labels to genres for the test set
test_data['PredominantGenre'] = test_data['Cluster'].map(cluster_genre_mapping)

# Step 8: Compute accuracy
correct_predictions = (test_data['genres'] == test_data['PredominantGenre']).sum()
total_samples = len(test_data)
accuracy = correct_predictions / total_samples

# Display the test set with assigned clusters, actual genres, and predominant genres
print(test_data[['original_title', 'genres', 'Cluster', 'PredominantGenre']])

# Display the accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')

# Create a confusion matrix
confusion_matrix = pd.crosstab(test_data['genres'], test_data['PredominantGenre'], rownames=['Actual'], colnames=['Predicted'])

print(confusion_matrix)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.savefig(f"{settings.Config.out_dir}/kmeans_conf_matrix.png")
plt.close()
