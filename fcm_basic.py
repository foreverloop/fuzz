import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class fuzzy_c_means:

    def __init__(self, n_clusters, m_fuzziness = 2, epsilon_stopping = 0.05):
        self.n_clusters = n_clusters
        self.m_fuzziness = m_fuzziness # 1 ~ k-means
        self.epsilon_stopping = epsilon_stopping
        self.c_means = None

    #compute weight for a point x_i relative to cluster c_j
    def get_point_weight():
        return 0

    #compute cluster c_j
    def get_cluster():
        return 0

    #use fuzzy hyper parameter to calculate cluster weight
    def get_cluster_weights(self, data_points):
        cluster_weights = []
        for i in range(self.n_clusters):
            #clusters vs columns? do we need for n clusters or n columns?
            cluster_weights.append(np.sum(np.power(data_points[:,i], self.m_fuzziness)))
        return np.asarray(cluster_weights)

    #fit data points using the fuzzy algorithm
    def fit(self, data_points):
        '''
        initialise a matrix populated with a guess for weight for each point for each cluster
        with number of rows equal to number of data points
        and number of columns equal to the number of clusters requested
        and each row's elements sum to 1
        '''
        random_initial_matrix = np.random.rand(len(data_points),self.n_clusters)
        random_initial_matrix = np.apply_along_axis(lambda x: x - (np.sum(x) - 1)/len(x),
        1, random_initial_matrix)

        #select an initial guess at centroids of n_clusters, number of clusters
        #start by selecting indexes to choose the random data points from
        idx = np.random.choice(data_points.shape[0], self.n_clusters, replace=False)
        self.c_means = data_points[idx]
        cluster_weights = self.get_cluster_weights(data_points)
        print(cluster_weights)
        #self.c_means = np.random.choice(data_points,1)[0]
        #self.c_means = random.sample(data_points.tolist(),self.n_clusters)
        print(self.c_means)
        print(random_initial_matrix[:10])
        #self.c_means = random.sample(points.tolist(),self.n_clusters)
        return 0

if __name__ == "__main__":
    df_wine = pd.read_csv('wine_clean.csv')
    scaler = MinMaxScaler()
    df_wine[df_wine.columns] = scaler.fit_transform(df_wine[df_wine.columns])
    my_fuzzy = fuzzy_c_means(n_clusters = 3, epsilon_stopping = 0.05)
    my_fuzzy.fit(df_wine[['Malic acid','Color intensity']].values)
