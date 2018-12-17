from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import yaml


class MonteCarloEstimator():
    def __init__(self):
        pass

    def _soft_threshold(self, x, tau):
        if x > tau:
            return x - tau
        else:
            return 0

    def _generate_random_orthogonal_matrix(self, shape):
        assert len(shape) == 2
        assert shape[0] >= shape[1]
        X = np.random.normal(0, 1, shape)
        U, _, _ = np.linalg.svd(X, full_matrices = False)
        return U


    def get_param_file(self, param_path, filename):
        param_file = os.path.join(param_path, filename)
        with open(param_file, "rb") as f:
            cfg = yaml.load(f)
        self.param_path = param_path
        self.alpha = float(cfg["alpha"])
        self.estimated_sigma = float(cfg["sigma"])
        self.lambda_filename = cfg["lambda"]
        with open(os.path.join(param_path, self.lambda_filename), 'rb') as f:
            self.empirical_signal = pickle.load(f)

    def estimate_signal(self):
        self.estimated_signal = list(map(lambda x: self._soft_threshold(x, 2 * self.estimated_sigma * np.sqrt(len(self.empirical_signal))), self.empirical_signal))
        rank = len(self.estimated_signal)
        for i in range(len(self.estimated_signal)):
            if self.estimated_signal[i] == 0:
                rank = i
                break
        self.rank = rank

    def estimate_pip_loss(self):
        D = self.estimated_signal
        rank = self.rank
        n = len(self.estimated_signal)
        sigma = self.estimated_sigma
        shape = (n, n)
        alpha = self.alpha
        print("n={}, rank={}, sigma={}".format(n, rank, sigma))
        
        D_gen = D[:rank]
        U_gen = self._generate_random_orthogonal_matrix((n, rank)) 
        V_gen = self._generate_random_orthogonal_matrix((n, rank))
        true_dims = range(rank)

        X = (U_gen * D_gen).dot(V_gen.T)

        E = np.random.normal(0, sigma, size = shape)
        estimation_noise_E = E 

        Y = X + estimation_noise_E

        U, D, V = np.linalg.svd(X)
        U1, D1, V1 = np.linalg.svd(Y)

        embed_gt = U[:,true_dims] * (D[true_dims] ** alpha)
        sim_gt = embed_gt.dot(embed_gt.T) 

        spectrum = D ** alpha
        spectrum_est = D1 ** alpha
        embed = U * spectrum
        embed_est = U1 * spectrum_est

        sim_est = None

        """ the "dumb method" does every step as is:
            a) loop through every dimensionality k
            b) for every k, calculate the dim-k estimated embedding
            c) compare it with the oracle ((g)round (t)ruth) embedding
            d) record the PIP loss
            f) select the dimensionality k that minimizes the PIP loss

            Now, the "smart method" essentially does the same, but only an order of magnitude
            faster. We used some simple linear algebra trick here. Readers can verify that the
            two methods give the same results.
        """
        dumb_method = False
        if dumb_method:
            time_add = 0.0
            time_norm = 0.0
            frobenius_list_est_to_gt = []
            for keep_dim in range(1, rank + 1):
                t0 = time.time()
                if sim_est is None:
                    sim_est = embed_est[:,:keep_dim].dot(embed_est[:,:keep_dim].T)
                else:
                    sim_est += np.outer(embed_est[:,keep_dim-1], embed_est[:,keep_dim-1])
                time_add += time.time() - t0
                t0 = time.time()
                sim_diff_est_to_gt = np.linalg.norm(sim_est - sim_gt, 'fro')
                time_norm += time.time() - t0
                frobenius_list_est_to_gt.append(sim_diff_est_to_gt)
            self.estimated_pip_loss = frobenius_list_est_to_gt
        else:
            time_norm = 0.0
            frobenius_list_est_to_gt = [np.linalg.norm(spectrum ** 2) ** 2]
            for keep_dim in range(1, rank + 1):
                t0 = time.time()
                diff = frobenius_list_est_to_gt[keep_dim-1] + spectrum_est[keep_dim-1] ** 4 - 2 * (
                        np.linalg.norm(embed_est[:, keep_dim-1].T.dot(embed_gt)) ** 2)
                time_norm += time.time() - t0
                frobenius_list_est_to_gt.append(diff)
            self.estimated_pip_loss = list(map(np.sqrt, frobenius_list_est_to_gt[1:]))
        with open(os.path.join(self.param_path, "pip_loss_{}.pkl".format(self.alpha)), 'wb') as f:
            pickle.dump(self.estimated_pip_loss, f)


    def plot_pip_loss(self):
        with open(os.path.join(self.param_path, "pip_loss_{}.pkl".format(self.alpha)), 'rb') as f:
            frobenius_list_est_to_gt = pickle.load(f)
        print("optimal dimensionality is {}".format(np.argmin(frobenius_list_est_to_gt)))
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(frobenius_list_est_to_gt, 'aqua', label = r'PIP loss')
        lgd = ax.legend(loc='upper right')
        plt.title(r'PIP Loss')
        fig_path = '{}/pip_{}.pdf'.format(self.param_path, self.alpha)
        fig.savefig(fig_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        print("a plot of the loss is saved at {}".format(fig_path))
        plt.close()


