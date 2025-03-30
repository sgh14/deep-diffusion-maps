import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances


class DiffusionLoss(tf.keras.losses.Loss):
    """
    Diffusion Maps loss function.

    Attributes
    ----------
    sigma : float
        Width parameter for the RBF kernel.
    gamma : float
        Parameter derived from sigma (1 / (2 * sigma^2)) used in the RBF kernel.
    steps : int
        Number of diffusion steps 't'.
    alpha : float
        Alpha parameter for kernel normalization.
    A_t_squared_reduced : tf.Tensor
        Precomputed matrix A^(2t) minus the component related to the first
        eigenvalue/eigenvector (stationary distribution), stored as a TensorFlow constant.
        Shape (n_samples, n_samples).
    pi : tf.Tensor
        Precomputed stationary distribution of the diffusion process, stored as
        a TensorFlow constant. Shape (n_samples,).
    """
    def __init__(
        self,
        X: np.ndarray,
        sigma: float,
        steps: int = 1,
        alpha: float = 0.0,
        name: str = "diffusion_loss"
    ):
        # --- Input Validation ---
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data 'X' must be a NumPy array.")
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("sigma must be a positive number.")
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer.")
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a number.")
        
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
        pi = self._stationary_distribution(d_W)
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
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray = None, gamma: float = 1.0) -> np.ndarray:
        """
        Computes the RBF (Gaussian) kernel between two sets of points.

        K(x, y) = exp(-gamma * ||x - y||^2)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
            First set of data points.
        Y : np.ndarray of shape (n_samples_Y, n_features), optional
            Second set of data points. If None, computes kernel between X and itself.
        gamma : float
            Parameter for the RBF kernel (1 / (2 * sigma^2)).

        Returns
        -------
        np.ndarray
            The computed kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        # pairwise_distances computes squared Euclidean distances efficiently
        distances_sq = pairwise_distances(X, Y, metric='sqeuclidean', n_jobs=-1)
        K = np.exp(-gamma * distances_sq)

        return K
    

    @staticmethod
    def _degree_vector(K: np.ndarray) -> np.ndarray:
        """
        Computes the degree vector (row sums) of a kernel/affinity matrix.

        d_i = sum_j K_ij

        Parameters
        ----------
        K : np.ndarray of shape (n_samples, n_samples) or (n_samples_X, n_samples_Y)
            The kernel or affinity matrix.

        Returns
        -------
        np.ndarray
            The degree vector (sum along axis 1).
        """
        # Sum along rows (axis=1)
        return np.sum(K, axis=1)


    @staticmethod
    def _normalize_by_degree(M: np.ndarray, d_i: np.ndarray, d_j: np.ndarray = None, alpha: float = 0.0) -> np.ndarray:
        """
        Performs alpha-normalization on a matrix M using degree vectors.

        M_norm[i, j] = M[i, j] / (d_i[i]**alpha * d_j[j]**alpha)

        Parameters
        ----------
        M : np.ndarray of shape (n_samples_i, n_samples_j)
            The matrix to normalize.
        d_i : np.ndarray of shape (n_samples_i,)
            Degree vector corresponding to the rows of M.
        d_j : np.ndarray of shape (n_samples_j,), optional
            Degree vector corresponding to the columns of M. If None, uses d_i.
        alpha : float
            The exponent for the degree normalization.

        Returns
        -------
        np.ndarray
            The normalized matrix M_norm.
        """
        d_i_alpha = np.power(d_i, alpha)
        d_j_alpha = d_i_alpha if d_j is None else np.power(d_j, alpha)
        M_alpha = M / np.outer(d_i_alpha, d_j_alpha)

        return M_alpha


    @staticmethod
    def _stationary_distribution(d_W: np.ndarray) -> np.ndarray:
        """
        Computes the stationary distribution pi of the Markov chain defined by W.

        For the diffusion process associated with the alpha-normalized kernel W,
        the stationary distribution pi is proportional to the degree vector d_W.

        pi_i = d_W_i / sum_k d_W_k

        Parameters
        ----------
        d_W : np.ndarray of shape (n_samples,)
            Degree vector derived from the normalized kernel W (sum of rows of W).

        Returns
        -------
        np.ndarray
            The stationary distribution pi, shape (n_samples,). Sums to 1.
        """
        pi = d_W / np.sum(d_W)

        return pi
    

    def call(self, y_true, y_pred):
        """
        Calculates the diffusion loss for a batch of predictions.

        This method computes the mean squared error between the Gram matrix of
        pi-scaled embeddings and the precomputed reduced diffusion matrix
        (A^{2t}_reduced) corresponding to the input indices.

        Parameters
        ----------
        y_true : tf.Tensor
            Tensor containing the integer indices of the data points in the
            original dataset `X` (used during initialization) for the current
            batch. Expected dtype convertible to int32. Shape (batch_size,).
        y_pred : tf.Tensor
            Tensor containing the predicted embeddings for the data points
            specified by `y_true`. Expected shape (batch_size, embedding_dim).

        Returns
        -------
        tf.Tensor
            A scalar tensor representing the computed diffusion loss for the batch.
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