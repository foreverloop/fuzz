import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

class fuzzy_c_means:

    def __init__(self, n_clusters, m_fuzziness = 2, epsilon_stopping = 0.05):
        self.n_clusters = n_clusters
        self.m_fuzziness = m_fuzziness # 0 ~ k-means
        self.epsilon_stopping = epsilon_stopping
        self.cluster_classes = None

    #use fuzzy hyper parameter to calculate weights for the clusters using the data
    def get_data_weights(self, data_weights):
        calculated_weights = []
        for j in range(self.n_clusters):
            calculated_weights.append(np.sum(np.power(data_weights[:,j], self.m_fuzziness)))
        return np.asarray(calculated_weights)

    def get_cluster_centroids(self, data_points, data_weights):

        #get weights as a sum of exponents (like sum square error) as a single scalar values
        #for each data feature, for each cluster
        result_col_weights = self.get_data_weights(data_weights)
        cluster_centroids = []

        #for each cluster, calculate weights as single scalar values (differs since the data points themselves are required in this calculation)
        for j in range(self.n_clusters):
            weight_results = []
            #calculate the centroid using the datapoint and results of the simpler calculation easier
            for point, weight in zip(data_points,data_weights[:,j]):
                weight_results.append(((weight ** self.m_fuzziness) * point) / result_col_weights[j])
            #for the cluster, add the sum of the weights to for centroids
            cluster_centroids.append(sum((weight_results)))

        return np.asarray(cluster_centroids)

    '''get weights for each cluster for each column
        intuitively, this makes sense, since column weights hold information
        about how __likely__ data in this column belongs to any particular cluster'''
    def update_membership_weights(self, data_points, cluster_centroids):

        #exponent to raise distance calculation to for later finding weights
        fuzzy_power = int(1 / (self.m_fuzziness - 1))

        #empty list, later will hold results format [w_1,w_...,w_n] for all data points
        recalculated_weights = []
        #for each data point
        for z in data_points:
            #each data point will have it's own class weights, calculate them as distance from centroids
            data_point_weights = [1 / (np.linalg.norm(x-z) ** fuzzy_power) for x in cluster_centroids]
            #finish the weight calculations by using each distance over the sum of all other distances
            finished_weights = [x / sum(data_point_weights) for x in data_point_weights]
            #needs to be an (number_of_rows, cluster_size) matrix,
            #contains weights for each cluster for each data point
            recalculated_weights.append(finished_weights)

        #return the recalculated weights as a numpy array
        return np.asarray(recalculated_weights)

    #sort into clusters based on weights
    def assign_clusters(self, weights):
        return [np.argmax(x, axis=0) for x in weights]

    #fit data points using the fuzzy algorithm
    def fit(self, data_points):
        '''
            initialise a matrix populated with a guess for weight for each point for each cluster
            with number of rows equal to number of data points
            and number of columns equal to the number of clusters requested
            and each row's elements sum to 1
        '''
        #initialise a matrix with shape described above
        cluster_weights = np.random.rand(len(data_points),self.n_clusters)
        #ensure values in the row always sum up to 1
        cluster_weights = np.apply_along_axis(lambda x: x - (np.sum(x) - 1) / len(x), 1, cluster_weights)

        #assign the points to clusters using the random initial weights (just a starting point)
        classified_points_initial = self.assign_clusters(cluster_weights)
        #declare a holder for points outside the iteration loop
        classified_points_new = []

        #add in early stopping and better convergence checking later
        #for now, this just iterates 100 times and ends
        for _ in range(101):
            cluster_centroids = self.get_cluster_centroids(data_points, cluster_weights)
            cluster_weights = self.update_membership_weights(data_points, cluster_centroids)
            classified_points_new = self.assign_clusters(cluster_weights)

        self.cluster_classes = classified_points_new

if __name__ == "__main__":
    df_wine = pd.read_csv('wine_clean.csv')
    wine_labels = df_wine['wine type'].tolist()
    scaler = MinMaxScaler()
    df_wine[df_wine.columns] = scaler.fit_transform(df_wine[df_wine.columns])
    my_fuzzy = fuzzy_c_means(n_clusters = 3, epsilon_stopping = 0.05)
    my_fuzzy.fit(df_wine[['Malic acid','Color intensity']].values)

    #plot the original cluster labels
    plt.scatter(df_wine['Malic acid'],df_wine['Color intensity'],c=wine_labels)
    plt.show()
    #plot the fuzzy labels
    plt.scatter(df_wine['Malic acid'],df_wine['Color intensity'],c=my_fuzzy.cluster_classes)
    plt.show()
