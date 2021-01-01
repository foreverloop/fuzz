import random
import numpy as np

class fuzzy_c_means:

    def __init__(self, n_clusters, epsilon_error = 0.05):
        self.n_clusters = n_clusters
        self.epsilon_error = epsilon_error
        self.c_means = None

    def fit(self, data_points):
        #initialise an empty matrix, with rows equal to number of data points
        #and columns equal to the number of clusters requested
        random_initial_matrix = np.zeros((len(data_points),self.n_clusters))

        #select an initial guess at centroid

        #start by selecting an index to choose the random data point from
        n = 1 #number of indicies to choose (may need choose multiple, one for each cluster?)
        idx = np.random.choice(data_points.shape[0], n, replace=False)
        self.c_means = data_points[idx]
        #self.c_means = np.random.choice(data_points,1)[0]
        #print(self.c_means[0])
        #self.c_means = random.sample(data_points.tolist(),self.n_clusters)
        print(self.c_means[0])
        print(np.shape(random_initial_matrix))
        #self.c_means = random.sample(points.tolist(),self.n_clusters)
        return 0
if __name__ == "__main__":
    my_fuzzy = fuzzy_c_means(n_clusters = 3, epsilon_error = 0.05)
    my_fuzzy.fit(np.asarray([[1,2],[5,10],[30,10],[5,2]]))
