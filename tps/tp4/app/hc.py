import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from settings import settings

class HC:
    def __init__(self, linkage='complete'):
        self.linkage = linkage
        
    def argmin(self, D):
        """
        Given a 2D array, returns the minimum value that is not in the main diagonal, i.e x==y
        and (x,y) index of that value.
        """
        min_x, min_y = (0,0)
        min_val = float('inf')
        for i in range(D.shape[0]):
            for j in range(D.shape[0]):
                if j==i:
                    continue
                else:
                    if D[i,j] < min_val:
                        min_val = D[i,j]
                        min_x = i
                        min_y = j
                        
        return min_val, min_x, min_y
        
    def cluster_distance(self, cluster_members, X):
        """
        Calculates the cluster euclidean distances. 
        
        Params
        ------
        cluster_members: dict.
            stores the cluster members in format: {key: [item1, item2 ..]}. 
            if key is less than X.shape[0] then, it only has itself in the cluster. 
        
        Returns
        -------
        distance: 2D array. 
            Contains distances between each cluster. 
        """
        n_clusters = len(cluster_members)
        keys = list(cluster_members.keys())
        distance = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                d_in_clusters = euclidean_distances(X[cluster_members[keys[i]]], X[cluster_members[keys[j]]])
                if self.linkage == 'complete':
                    dij = np.max(d_in_clusters)
                elif self.linkage == 'single':
                    dij = np.min(d_in_clusters)
                distance[i,j] = dij

        # so distance has minimum distance between clusters 
        # row and col of distance denotes element in `keys`
        return distance
    
    def fit(self, X):
        """
        Generates the dendrogram.
        
        Params
        ------
        X: Dataset, shape (n_samples, nFeatures)
        
        Returns
        -------
        Z: 2D array. shape (n_samples-1, 4). 
            Linkage matrix. Stores the merge information at each iteration.
        """
        self.n_samples = X.shape[0]

        cluster_members = dict([(i,[i]) for i in range(self.n_samples)])
        Z = np.zeros((self.n_samples-1,4)) # c1, c2, d, count
        
        for i in range(0, self.n_samples-1):
            if settings.verbose:
                print(f'\n-------\nDebug Line at, i={i}\n--------')
            
            keys = list(cluster_members.keys())

            # caculate the distance between existing clusters
            D = self.cluster_distance(cluster_members, X)
            _, tmpx, tmpy = self.argmin(D)
            
            if settings.verbose:
                print(f'Z:\n{Z}, \nCluster Members: {cluster_members}, D: \n {D}')
            
            x = keys[tmpx]
            y = keys[tmpy]

            # update Z
            Z[i,0] = x
            Z[i,1] = y
            Z[i,2] = D[tmpx, tmpy] # that's where the min value is
            Z[i,3] = len(cluster_members[x]) + len(cluster_members[y])
            
            # new cluster created
            cluster_members[i+self.n_samples] = cluster_members[x] + cluster_members[y]

            # remove merged from clusters pool, else they'll be recalculated
            del cluster_members[x]
            del cluster_members[y]
            
        self.Z = Z
        return self.Z
    
    def predict(self, n_cluster=3):
        """
        Get cluster label for specific cluster size.
        
        Params
        ------
        n_cluster: int. 
            Number of clusters to keep. Can not be > n_samples
        
        Returns
        -------
        labels: list.
            Cluster labels for each sample.
        """
        labels = np.zeros((self.n_samples))
        cluster_members = dict([(i,[i]) for i in range(self.n_samples)])
        for i in range(self.n_samples - n_cluster):
            x,y = (Z[i,0], Z[i,1])
            cluster_members[self.n_samples + i] = cluster_members[x] + cluster_members[y]
            del cluster_members[x]
            del cluster_members[y]
            
        keys = list(cluster_members.keys())
        
        for i in range(len(keys)):
            samples_in_cluster = cluster_members[keys[i]]
            labels[samples_in_cluster] = i
            
        return labels