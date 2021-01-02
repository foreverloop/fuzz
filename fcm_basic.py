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

    #use fuzzy hyper parameter to calculate weights for the clusters using the data
    def get_data_weights(self, data_weights, n_cols):
        calculated_weights = []
        for j in range(self.n_clusters):
            calculated_weights.append(np.sum(np.power(data_weights[:,j], self.m_fuzziness)))
        return np.asarray(calculated_weights)

    #uses get_data_weights to get column weights, use these alongside the data points to classify points into clusters
    def get_cluster_centroids(self, data_points, data_weights, n_cols):
        '''
        get weights for each cluster for each column of weights
        intuitively, this makes sense, since column weights hold information
        about how __likely__ data in this column belongs to any particular cluster
        '''
        result_col_weights = self.get_data_weights(data_weights, n_cols)
        cluster_weights = []

        for j in range(self.n_clusters):
            weight_results = []
            for point, weight in zip(data_points,data_weights[:,j]):
                weight_results.append(((weight ** self.m_fuzziness) * point) / result_col_weights[j])

            cluster_weights.append(sum((weight_results)))

        return np.asarray(cluster_weights)

    def update_membership_weights(self, data_points, cluster_centroids):

        #exponent to raise distance calculation to for later finding weights
        fuzzy_power = int(1 / (self.m_fuzziness - 1))
        recalculated_weights = []

        #for each data point
        for z in data_points:
            #for each cluster cenroid
            data_point_centroids_to_sum = []
            data_point_weights = []
            for idx,x in enumerate(cluster_centroids):
                if idx == 0:
                    alpha = 1 / (np.linalg.norm(x-z) ** fuzzy_power)
                data_point_centroids_to_sum.append(np.linalg.norm(x-z) ** fuzzy_power)
                cluster_x_data_weight = alpha / (1 / (sum(data_point_centroids_to_sum) ** fuzzy_power))
                data_point_weights.append(cluster_x_data_weight)
            recalculated_weights.append(data_point_weights)
        print(recalculated_weights)
        return np.asarray(recalculated_weights)

    #sort into clusters based on weights
    def assign_clusters(self, weights):
        classified_points = []
        for i in weights:
            classified_points.append(np.argmax(i,axis=0))
        return classified_points

    #fit data points using the fuzzy algorithm
    def fit(self, data_points):
        '''
        initialise a matrix populated with a guess for weight for each point for each cluster
        with number of rows equal to number of data points
        and number of columns equal to the number of clusters requested
        and each row's elements sum to 1
        '''
        #get number of rows and columns
        n_rows, n_cols = np.shape(data_points)

        random_initial_weights = np.random.rand(len(data_points),self.n_clusters)
        random_initial_weights = np.apply_along_axis(lambda x: x - (np.sum(x) - 1) / len(x), 1, random_initial_weights)

        #select an initial guess at centroids of n_clusters, number of clusters
        #start by selecting indexes to choose the random data points from
        idx = np.random.choice(data_points.shape[0], self.n_clusters, replace = False)
        self.c_means = data_points[idx]

        classified_points_initial = self.assign_clusters(random_initial_weights)
        classified_points_new = []

        max_iter = 200
        count = 0
        while classified_points_initial != classified_points_new:
            if count == max_iter:
                break
            else:
                count += 1
                cluster_centroids = self.get_cluster_centroids(data_points, random_initial_weights, n_cols)
                new_weights = self.update_membership_weights(data_points, cluster_centroids)
                classified_points_new = self.assign_clusters(new_weights)
        #print(classified_points_initial)
        #print('-----------------------')
        #print(classified_points_new)

if __name__ == "__main__":
    df_wine = pd.read_csv('wine_clean.csv')
    scaler = MinMaxScaler()
    df_wine[df_wine.columns] = scaler.fit_transform(df_wine[df_wine.columns])
    my_fuzzy = fuzzy_c_means(n_clusters = 3, epsilon_stopping = 0.05)
    my_fuzzy.fit(df_wine[['Malic acid','Color intensity']].values)
