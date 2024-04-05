import numpy as np
import networkx as nx
from scipy.special import rel_entr
import kss

def process_text(text):
  sentences=[i for i in kss.split_sentences(text) if len(i)>3]
  sentences=[sentence.replace("앵커", "") if sentence.startswith("앵커") else sentence for sentence in sentences]

  return sentences

class WeightedNetworkDissimilarity:
    def __init__(self, G1, G2,weight=[0.45, 0.45, 0.1]):
        """
        Initializes the class with two weighted networks G1 and G2.
        G1 and G2 are expected to be networkx Graph objects with weight attributes.
        """
        self.G1 = G1
        self.G2 = G2

        # Compute adjacency matrices and normalize
        self.W1 = nx.to_numpy_array(G1)
        self.W2 = nx.to_numpy_array(G2)
        self.normalize_weights()

        # Node strengths and average weights for alpha centrality
        self.S1 = np.sum(self.W1, axis=1)
        self.S2 = np.sum(self.W2, axis=1)
        self.average_weight1 = np.mean(self.W1[self.W1 > 0])
        self.average_weight2 = np.mean(self.W2[self.W2 > 0])

        # Parameters for WD calculation
        self.w1, self.w2, self.w3 = weight

    def normalize_weights(self):
        """
        Normalizes the weight matrices of both networks by dividing by the maximum weight.
        """
        max_weight = max(self.W1.max(), self.W2.max())
        self.W1 /= max_weight
        self.W2 /= max_weight

    def JS_divergence(self, P1, P2):
        """
        Calculate the Jensen-Shannon Divergence between two probability distributions,
        padding them to the same length if necessary.
        """
        # Determine the longer length
        max_length = max(len(P1), len(P2))

        # Extend both arrays to this length by padding with zeros
        P1_extended = np.pad(P1, (0, max_length - len(P1)), 'constant')
        P2_extended = np.pad(P2, (0, max_length - len(P2)), 'constant')

        # Calculate the midpoint distribution
        M = 0.5 * (P1_extended + P2_extended)

        # Calculate the JS divergence
        JS_div = 0.5 * (np.sum(rel_entr(P1_extended, M)) + np.sum(rel_entr(P2_extended, M)))
        return JS_div


    def calculate_mu(self, P_omega):
        """
        Calculate the average proportion of each order neighbors for a given distance probability matrix P_omega.
        """
        return np.sum(P_omega, axis=0) / P_omega.shape[0]

    def calculate_J(self, P_omega):
        """
        Calculate the J divergence value for the given network probability distributions.
        """
        mu = self.calculate_mu(P_omega)
        N, m = P_omega.shape
        J_value = sum(p_ij * np.log(p_ij / mu[j]) for i in range(N) for j, p_ij in enumerate(P_omega[i]) if p_ij > 0)
        return J_value / N

    def WNND(self, P_omega):
        """
        Calculate the Weighted Network Node Dispersion (WNND) for the network.
        """
        J_value = self.calculate_J(P_omega)
        m = P_omega.shape[1]  # Number of columns in P_omega
        return J_value / np.log(m + 1)

    def alpha_centrality(self, W, S, average_weight):
        """
        Computes the alpha centrality for the network represented by adjacency matrix W.
        """
        N = len(W)
        alpha = 1 / N
        beta = S / ((N - 1) * average_weight)
        x = np.linalg.solve(np.eye(N) - alpha * W, beta)
        return x

    def calculate_P_omega(self, L_omega_prime):
        """
        Converts adjusted path length matrix to distance probability matrix P_omega.
        """
        N = L_omega_prime.shape[0]
        max_distance = int(np.max(L_omega_prime))
        P_omega = np.zeros((N, max_distance + 1))  # Including column for disconnected nodes

        for i in range(N):
            for j in range(N):
                if L_omega_prime[i, j] < np.inf:
                    P_omega[i, int(L_omega_prime[i, j])] += 1
        P_omega /= (N - 1)
        return P_omega

    def compute_WD_metric(self):
        """
        Computes the WD metric between the two weighted networks.
        """
        # Calculate P_omega for both networks
        L1 = nx.floyd_warshall_numpy(self.G1, weight='weight')
        L2 = nx.floyd_warshall_numpy(self.G2, weight='weight')
        P_omega1 = self.calculate_P_omega(L1)
        P_omega2 = self.calculate_P_omega(L2)

        # Calculate mu for WD
        mu_G1 = self.calculate_mu(P_omega1)
        mu_G2 = self.calculate_mu(P_omega2)
        J_global = self.JS_divergence(mu_G1, mu_G2)

        # Calculate WNND for both networks
        WNND_G1 = self.WNND(P_omega1)
        WNND_G2 = self.WNND(P_omega2)

        # Calculate alpha centrality for both networks and convert to P_alpha
        alpha_centrality1 = self.alpha_centrality(self.W1, self.S1, self.average_weight1)
        alpha_centrality2 = self.alpha_centrality(self.W2, self.S2, self.average_weight2)
        P_alpha_G1 = alpha_centrality1 / np.sum(alpha_centrality1)
        P_alpha_G2 = alpha_centrality2 / np.sum(alpha_centrality2)

        # Compute WD metric components
        term1 = self.w1 * np.sqrt(J_global / np.log(2))
        term2 = self.w2 * abs(np.sqrt(WNND_G1) - np.sqrt(WNND_G2))
        term3 = self.w3 / 2 * (np.sqrt(self.JS_divergence(P_alpha_G1, P_alpha_G2) / np.log(2)))

        # Summing up WD metric components
        WD_metric = term1 + term2 + term3
        return WD_metric
