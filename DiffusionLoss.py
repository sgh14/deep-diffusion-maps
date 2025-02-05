import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances


class DiffusionLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        X,
        sigma=None,
        steps=1,
        alpha=1,
        name="diffusion_loss"
    ):
        super().__init__(name=name)
        self.sigma = sigma
        self.gamma = 1/(2*sigma**2)
        self.steps = steps
        self.alpha = alpha

        X_flat = X.reshape((X.shape[0], -1))
        # Compute the kernel matrix and degree vector
        K = self._rbf_kernel(X_flat, X_flat, self.gamma)
        d_K = self._degree_vector(K)
        # Compute the normalized kernel and degree vector
        W = self._normalize_by_degree(K, d_K, d_K, self.alpha)
        d_W = self._degree_vector(W)
        # Compute the stationary distribution
        pi = self._stationary_dist(d_W)
        # Compute the matrix A
        A = self._normalize_by_degree(W, d_W, d_W, 0.5)
        # Compute A^{2t}
        A_t_squared = np.linalg.matrix_power(A, 2*steps)
        # Substract first eigenvalue component from A_t_squared
        A_t_squared_reduced = A_t_squared - np.sqrt(np.outer(pi, pi))
        # Store the A and pi for the training data
        self.A_t_squared_reduced = tf.constant(A_t_squared_reduced, dtype=tf.float32)
        self.pi = tf.constant(pi, dtype=tf.float32)


    @staticmethod
    def _rbf_kernel(X, Y=None, gamma=None):
        gamma = gamma if gamma else 1.0 / X.shape[1]
        distances = pairwise_distances(X, Y, metric='sqeuclidean')
        K = np.exp(-gamma * distances)

        return K


    @staticmethod
    def _normalize_by_degree(M, d_i, d_j=[], alpha=0):
        d_i_alpha = d_i**alpha
        d_j_alpha = d_j**alpha if len(d_j) > 0 else d_i_alpha
        M_alpha = M/np.outer(d_i_alpha, d_j_alpha)

        return M_alpha


    @staticmethod
    def _degree_vector(K):
        d = np.sum(K, axis=1)

        return d


    @staticmethod
    def _stationary_dist(d):
        pi = d / np.sum(d)

        return pi
    

    def call(self, y_true, y_pred):
        """
        Args:
            y_true (tf.Tensor): True labels or indices of the samples.
            y_pred (tf.Tensor): Predicted embeddings of the samples.

        Returns:
            tf.Tensor: Computed loss value.
        """
        # Rename y_true and y_pred
        ids = tf.cast(y_true, tf.int32) # Cast to integer type for indexing
        embeddings = y_pred
        # Gather the corresponding pi and A_t_squared_reduced
        pi = tf.gather(self.pi, ids)
        A_t_squared_reduced = tf.gather(tf.gather(self.A_t_squared_reduced, ids), ids, axis=1)
        # Compute the scaled embeddings and the scaled embeddings gram matrix
        scaled_embeddings = tf.expand_dims(tf.sqrt(pi), axis=1) * embeddings 
        scaled_embeddings_gram_matrix = tf.matmul(scaled_embeddings, scaled_embeddings, transpose_b=True)
        # Compute the squared differences
        loss = tf.reduce_mean(tf.square(scaled_embeddings_gram_matrix - A_t_squared_reduced))

        return loss