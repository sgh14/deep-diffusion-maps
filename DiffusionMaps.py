import numpy as np
from sklearn.metrics import pairwise_distances



class DiffusionMaps:
    """
    Diffusion Maps.
    """
    def __init__(self, sigma, n_components, steps=1, alpha=0):
        self.sigma = sigma
        self.gamma = 1/(2*sigma**2)
        self.n_components = n_components
        self.steps = steps
        self.alpha = alpha


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
    def transition_probabilities(W):
        d = np.sum(W, axis=1)
        P = W / d[:, np.newaxis]

        return P
    

    @staticmethod
    def diffusion_distances(P, pi):
        D = pairwise_distances(
            P, metric=lambda P_i, P_j: np.sqrt(np.sum(((P_i - P_j)**2) / pi))
        )

        return D


    @staticmethod
    def _fix_vector_orientation(vectors):
        # Fix the first non-zero component of every vector to be positive
        for i in range(vectors.shape[1]):
            first_nonzero = np.nonzero(vectors[:, i])[0][0]
            if vectors[first_nonzero, i] < 0:
                vectors[:, i] *= -1

        return vectors


    @staticmethod
    def _spectral_decomposition(A):
        # Compute the eigenvalues and right eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # Find the order of the eigenvalues (decreasing order)
        order = np.argsort(eigenvalues)[::-1]
        # Sort eigenvalues and eigenvectors
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        # Fix eigenvectors orientation
        eigenvectors = DiffusionMaps._fix_vector_orientation(eigenvectors)
        
        return eigenvalues, eigenvectors
    

    @staticmethod
    def _degree_vector(K):
        d = np.sum(K, axis=1)

        return d


    @staticmethod
    def _stationary_dist(d):
        pi = d / np.sum(d)

        return pi
    

    def _get_embedding(self, phis, lambdas, pi):
        # Compute P right eigenvectors
        psis = phis / np.sqrt(pi[:, np.newaxis])
        # Compute the new coordinates
        Psis = psis * (lambdas[np.newaxis, :] ** self.steps)

        return Psis


    def fit_transform(self, X, y=None):
        self.X = X.reshape((X.shape[0], -1))
        # Compute the kernel
        K = self._rbf_kernel(self.X, self.X, self.gamma)
        # Compute degree vector        
        self.d_K = self._degree_vector(K)
        # Compute the normalized kernel
        self.W = self._normalize_by_degree(K, self.d_K, self.d_K, self.alpha)
        # Compute degree vector
        self.d_W = self._degree_vector(self.W)
        # Compute the stationary distribution
        self.pi = self._stationary_dist(self.d_W)
        # Compute the matrix A
        A = self._normalize_by_degree(self.W, self.d_W, self.d_W, 0.5)
        # Get the eigenvectors and eigenvalues
        self.lambdas, self.phis = self._spectral_decomposition(A)
        # Reduce dimension
        lambdas_red = self.lambdas[1:self.n_components + 1]
        phis_red = self.phis[:, 1:self.n_components + 1]            
        # Compute the new coordinates
        X_red = self._get_embedding(phis_red, lambdas_red, self.pi)

        return X_red

    
    @staticmethod
    def _nystrom_extension(K_mix, old_eigenvectors, old_eigenvalues):
        new_eigenvectors = (K_mix @ old_eigenvectors) / old_eigenvalues[np.newaxis, :]
        new_eigenvalues = old_eigenvalues

        return new_eigenvectors, new_eigenvalues


    def _get_K_alpha_approx(self, X_new):
        K_mix = self._rbf_kernel(X_new, self.X, gamma=self.gamma)
        d_old = self.d_K
        d_mix = np.sum(K_mix, axis=1)
        K_alpha_mix = self._normalize_by_degree(K_mix, d_mix, d_old, self.alpha)

        return K_alpha_mix
    

    def _get_A_approx(self, X_new):
        if self.alpha !=0:
            K_alpha_mix = self._get_K_alpha_approx(X_new)
        else:
            K_alpha_mix = self._rbf_kernel(X_new, self.X, gamma=self.gamma)

        d_alpha_old = self.d_W
        d_alpha_mix = np.sum(K_alpha_mix, axis=1)
        A_mix = self._normalize_by_degree(K_alpha_mix, d_alpha_mix, d_alpha_old, 0.5)

        return A_mix


    def transform(self, X_new):
        X_new_flat = X_new.reshape((X_new.shape[0], -1))
        lambdas_red = self.lambdas[:self.n_components + 1]
        phis_red = self.phis[:, :self.n_components + 1]
        A_mix = self._get_A_approx(X_new_flat)
        new_phis, new_lambdas = self._nystrom_extension(A_mix, phis_red, lambdas_red)
        new_pi = new_phis[:, 0]**2
        X_new_red = self._get_embedding(new_phis[:, 1:], new_lambdas[1:], new_pi)

        return X_new_red
