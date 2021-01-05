import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("error")

class fuzzy_c_means:

    def __init__(self, n_clusters = 3, m_fuzziness = 2, epsilon_stopping = 0.001, max_iter = 500):
        self.n_clusters = n_clusters
        self.m_fuzziness = m_fuzziness
        self.epsilon_stopping = epsilon_stopping
        self.max_iter = max_iter
        self.cluster_weights = None
        self.cluster_classes = None

    def get_cluster_centroids(self, data_points, data_weights):

        cluster_centroids = []

        for j in range(self.n_clusters):
            weight_results = []
            for point, weight in zip(data_points,data_weights[:,j]):
                weight_results.append(((weight ** self.m_fuzziness) * point))
            cluster_centroids.append(sum(weight_results) / np.sum(np.power(data_weights[:,j], self.m_fuzziness)))

        return np.asarray(cluster_centroids)

    def update_membership_weights(self, data_points, cluster_centroids):
        fuzzy_power = 1 / (self.m_fuzziness - 1)
        #empty list, later will hold results format [w_1,w_...,w_n] for all data points
        recalculated_weights = []
        #for each data point
        for z in data_points:
            #each data point will have it's own cluster weights, calculate them as distance from centroids
            try:
                data_point_weights = [(1 / (np.linalg.norm(x-z))) ** fuzzy_power for x in cluster_centroids]
            except RuntimeWarning:
                #print("runtime warning occured: np.linalg.norm returned 0 (same data point means distance == 0)")
                #set the weight to the average so it doesn't mess up the result
                data_point_weights = [1/self.n_clusters for _ in range(self.n_clusters)]

            #finish the weight calculations by using each distance over the sum of all other distances
            finished_weights = [x / (sum(data_point_weights)) for x in data_point_weights]
            #needs to be an (number_of_rows, cluster_size) shape matrix
            recalculated_weights.append(finished_weights)
        print(recalculated_weights)
        #return the recalculated weights as a numpy array
        return np.asarray(recalculated_weights)

    #sort into clusters based on weights
    def assign_clusters(self, weights):
        return [np.argmax(x, axis=0) for x in weights]

    #fit data points using the fuzzy algorithm
    def fit(self, data_points):
        #initialise a matrix same length as input data, and width of number of clusters
        cluster_weights = np.random.rand(len(data_points),self.n_clusters)

        #ensure values in the row always sum up to 1
        cluster_weights = np.apply_along_axis(lambda x: x - (np.sum(x) - 1) / len(x), 1, cluster_weights)

        #assign the points to clusters using the random initial weights (just a starting point)
        classified_points_initial = self.assign_clusters(cluster_weights)

        #declare a holder for points outside the iteration loop
        classified_points_new = []

        #declare an empty matrix for comparing old centroids with new when iterating
        n_rows, n_cols = np.shape(data_points)
        old_centroids = np.zeros((self.n_clusters, n_cols))

        #iterate through the data up to maximum specified
        for i in range(self.max_iter):
            #get the centroids for the weights and data points
            cluster_centroids = self.get_cluster_centroids(data_points, cluster_weights)
            #if the centroids did not change much, end iteration early
            if np.allclose(cluster_centroids, old_centroids, atol=self.epsilon_stopping, rtol=self.epsilon_stopping):
                print("stopped after {0} iterations".format(i))
                break

            #store the centroids calculated so they can be compared with new ones calculated next iteration
            old_centroids = cluster_centroids
            #update cluster membership weights
            cluster_weights = self.update_membership_weights(data_points, cluster_centroids)
            #assign the data points to clusters using the weights
            classified_points_new = self.assign_clusters(cluster_weights)

        self.cluster_classes = classified_points_new
        self.cluster_weights = cluster_weights
        self.cluster_centroids = cluster_centroids

if __name__ == "__main__":
    #load in and prepare data and set output options
    df_wine = pd.read_csv('wine_clean.csv')
    wine_labels = df_wine['wine type'].tolist()
    df_wine[df_wine.columns] = MinMaxScaler().fit_transform(df_wine[df_wine.columns].values)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    #fuzziness is inverse, with higher number making crisper clusters (can easily change calc to make it work in standard)
    my_fuzzy = fuzzy_c_means(n_clusters = 3, m_fuzziness = 2, epsilon_stopping = 0.001)
    #my_fuzzy.fit(df_wine[['Malic acid','Color intensity','Alcalinity of ash','Flavanoids']].values)
    my_fuzzy.fit(df_wine[['Malic acid','Color intensity']].values)

    print("centroids: ",my_fuzzy.cluster_centroids)
    centroids_np = np.asarray(my_fuzzy.cluster_centroids)

    #print(my_fuzzy.cluster_weights)
    #plot the original cluster labels
    plt.scatter(df_wine['Malic acid'], df_wine['Color intensity'], c = wine_labels)
    plt.show()
    #plot the fuzzy labels
    plt.scatter(centroids_np[:,0], centroids_np[:,1], s = 50, color = "orangered",marker='x')
    plt.scatter(df_wine['Malic acid'], df_wine['Color intensity'], c = my_fuzzy.cluster_classes)
    plt.show()
