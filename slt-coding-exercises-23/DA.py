import sklearn as skl
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, metric="euclidian", T_min=2, T_0=0.02, eta=0.925, sigma=0.1, min_err=10e-6):

        self.n_clusters = n_clusters #this corresponds to K_amx
        self.random_state = random_state
        self.metric = metric
        self.sigma=sigma #noise used when splitting cluster to separate centroids
        self.T = None
        self.T_min = T_min
        self.T_0=T_0 #temperature that we will use for last iteration (we cannot use T=0 so we choose a very small value)
        self.eta=eta #cooling parameter
        self.min_err=min_err

        self.cluster_centers = None
        self.cluster_probs = None
        self.p_y=None

        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()
        self.bifurcation_tree = Tree()


        # Not necessary, depends on your implementation
        self.bifurcation_tree_cut_idx = None

        # Add more parameters, if necessary. You can also modify any other
        # attributes defined above

    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """
        # TODO:

        n_samples, n_features=samples.shape

        K_max=self.n_clusters #this corresponds to K_max in [1]
    
        #covariance=np.cov(samples.T) #C_x (covariance matrix of the data)
        covariance=1/n_samples*np.matmul(samples, np.transpose(samples))

        eigenvalues, eigenvectors=np.linalg.eigh(covariance)
        
        self.T=2*max(eigenvalues)+100 #we initialize the temperature such that it is bigger than twice the maximum eigenvalue, as suggested
        self.K=1 #initialize number of clusters to 1

        self.cluster_centers=np.zeros((self.K, n_features))
        sample_mean=1/n_samples*samples.sum(axis=0)
        self.cluster_centers[0, :]=sample_mean #initialize cluster centers as the mean of the data
        self.cluster_probs=np.ones((n_samples, self.K)) #since we have only one cluster at the beginning, we initialize all the probabilities to 1

        self.p_y=np.ones(self.K)

        self.bifurcation_tree.create_node(identifier=0, data={'cluster_id': 0, 'parent_centroid_position': self.cluster_centers[0, :], 'parent_last_distance': 0})
        
        count=0

        self.directions = [1]
        self.list_of_lists_distances = [[] for x in range(self.n_clusters)]
        self.list_of_lists_distances[0].append(0.0)
        

        while self.T>self.T_0:

            # update probabilities/centroids until convergence -> step 3 and 4

            diff=1

            while diff>self.min_err:

                old_estimate=self.cluster_centers.copy()

                dist_mat = self.get_distance(samples, self.cluster_centers)
                self.cluster_probs = self._calculate_cluster_probs(dist_mat, self.T)
                #self.p_y=1/n_samples*self.cluster_probs.sum(axis=0)

                for i in range(self.K):

                    self.p_y[i] = 1/n_samples*((self.cluster_probs[:, i]).sum())

                    self.cluster_centers[i]=1/self.p_y[i]*1/n_samples*(np.dot(np.transpose(samples), self.cluster_probs[:, i]))
                
                diff=np.sqrt(np.sum((old_estimate-self.cluster_centers)**2))

            
            #check temperature below T_min -> step 5

            if self.T<self.T_min:
                self.T=self.T_0
            

            
            # update temperature (cooling step) -> step 6

            self.T=self.eta*self.T
            count+=1

            self.temperatures.append(self.T)
            self.n_eff_clusters.append(self.K)
            self.distortions.append(1/n_samples*((self.cluster_probs*dist_mat).sum()))


            # checking temperature thresholds -> step 7

            if self.K<K_max:

                for i in range(self.K):

                    #calculate covariance of the cluster (C_x|y)

                    cluster_posterior=1/n_samples*1/self.p_y[i]*self.cluster_probs[:, i]

                    cluster_covariance=np.zeros((n_features, n_features))

                    for j in range(n_samples):
                        cluster_covariance+=cluster_posterior[j]*np.outer((samples[j, :] - self.cluster_centers[i]), (samples[j, :] - self.cluster_centers[i]))  
                    eigenvalues, eigenvectors=np.linalg.eigh(cluster_covariance)
                    critical_temperature=2*max(np.abs(eigenvalues))

                    leaves=self.bifurcation_tree.leaves()

                    for node in leaves:

                        if node.data['cluster_id']==i:
                            current_node=node
                            break


                    if self.T<critical_temperature:

                        self.K+=1

                        #create new_centroid

                        new_centroid=self.cluster_centers[i, :]+np.random.normal(0, self.sigma, n_features)
                        self.cluster_centers=np.vstack((self.cluster_centers, new_centroid))

                        new_p_y=self.p_y[i]/2
                        self.p_y[i]=self.p_y[i]/2
                        self.p_y=self.p_y.tolist()
                        self.p_y.append(new_p_y)
                        self.p_y=np.array(self.p_y)
        
                        #create new nodes in the bifurcation tree

                        self.directions[i] = -1
                        self.directions.append(1)

                        self.list_of_lists_distances[i].append(self.list_of_lists_distances[i][-1])
                        self.list_of_lists_distances[self.K-1].append(self.list_of_lists_distances[i][-1])
                    
                        self.bifurcation_tree.create_node(identifier=self.bifurcation_tree.size(), data={'cluster_id': i, 'parent_centroid_position': self.cluster_centers[i].copy(), 'parent_last_distance':self.list_of_lists_distances[i][-1]}, parent=current_node)
                        self.bifurcation_tree.create_node(identifier=self.bifurcation_tree.size(), data={'cluster_id': self.K-1, 'parent_centroid_position': self.cluster_centers[i].copy(), 'parent_last_distance': self.list_of_lists_distances[i][-1]}, parent=current_node)

                        self.bifurcation_tree_cut_idx=count

                        if self.K==K_max:
                            break
                    
                    else:

                        #if we haven't reached critical temperature, we update

                        new_distance = np.linalg.norm(current_node.data['parent_centroid_position'] - self.cluster_centers[i])
                        self.list_of_lists_distances[i].append(current_node.data['parent_last_distance'] + self.directions[i] * new_distance)
                            
            else:

                #update bifurcation_tree

                leaves=self.bifurcation_tree.leaves()

                for node in leaves:

                    node_cluster_id=node.data['cluster_id']
                    new_distance=np.linalg.norm(node.data['parent_centroid_position']-self.cluster_centers[node_cluster_id])
                    self.list_of_lists_distances[node_cluster_id].append(node.data['parent_last_distance']+self.directions[node_cluster_id]*new_distance)
                    
            
          
                        
        
    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        # TODO:

        tol=np.min(dist_mat, axis=1, keepdims=True)
        un_normalized_probs = np.exp(-(tol+np.square(dist_mat))/temperature) * self.p_y
        return normalize(un_normalized_probs, axis=1, norm='l1')


    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """
        # TODO:

        distance_matrix=np.zeros((samples.shape[0], clusters.shape[0]))

        for i in range(clusters.shape[0]):
            distance_matrix[:, i]=((samples-clusters[i, :])**2).sum(axis=1)
        
        return np.sqrt(distance_matrix)

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting

        This is a pseudo-code showing how you may be using the tree
        information to make a bifurcation plot. Your implementation may be
        entire different or based on this code.
        """
        check_is_fitted(self, ["bifurcation_tree"])

        clusters = [[] for _ in range(len(np.unique(self.n_eff_clusters)))]

        beta = [1/t for t in self.temperatures]


        for i in range(self.n_clusters):

            diff = len(beta) - len(self.list_of_lists_distances[i])
            
            padding = []
            if diff != 0:
                padding = [np.nan for _ in range(diff)]

            clusters[i]= padding + self.list_of_lists_distances[i]
        # Cut the last iterations, usually it takes too long
        cut_idx = self.bifurcation_tree_cut_idx + 20

        plt.figure(figsize=(10, 5))
        for c_id, s in enumerate(clusters):

            plt.plot(s[:cut_idx], beta[:cut_idx], '-k',
                     alpha=1, c='C%d' % int(c_id),
                     label='Cluster %d' % int(c_id))
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()

    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()
