# Author: Sining Sun , Zhanheng Yang

import numpy as np
from utils import *
import scipy.cluster.vq as vq
import time
import multiprocessing

num_gaussian = 5
num_iterations = 5
targets = ['Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')


class GMM:
    def __init__(self, D, K=5):
        assert (D > 0)
        self.dim = D
        self.K = K
        # Kmeans Initial
        self.mu, self.sigma, self.pi = self.kmeans_initial()

    def kmeans_initial(self):
        mu = []
        sigma = []
        data = read_all_data('train/feats.scp')
        (centroids, labels) = vq.kmeans2(data, self.K, minit="points", iter=100)
        clusters = [[] for i in range(self.K)]
        for (l, d) in zip(labels, data):
            clusters[l].append(d)

        for cluster in clusters:
            mu.append(np.mean(cluster, axis=0))
            sigma.append(np.cov(cluster, rowvar=0))
        pi = np.array([len(c) * 1.0 / len(data) for c in clusters])
        return mu, sigma, pi

    def gaussian(self, x, mu, sigma):
        """Calculate gaussion probability.

            :param x: The observed data, dim*1.
            :param mu: The mean vector of gaussian, dim*1
            :param sigma: The covariance matrix, dim*dim
            :return: the gaussion probability, scalor
        """
        D = x.shape[0]
        det_sigma = np.linalg.det(sigma) + 0.00001
        inv_sigma = np.linalg.inv(sigma + 0.00001 * np.identity(self.dim))
        mahalanobis = np.dot(np.transpose(x - mu), inv_sigma)
        mahalanobis = np.dot(mahalanobis, (x - mu))
        const = 1 / ((2 * np.pi) ** (D / 2))
        return const * (det_sigma) ** (-0.5) * np.exp(-0.5 * mahalanobis)

    def calc_log_likelihood(self, X: np.ndarray):
        """Calculate log likelihood of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of current model
        """

        log_llh = 0.0
        N = X.shape[0]  # 帧数，观测样本个数
        for n in range(N):
            tmp = 0
            for k in range(self.K):
                xn = X[n, :].transpose()  # dim*1
                tmp += self.pi[k] * self.gaussian(xn, self.mu[k], self.sigma[k])
            log_llh += np.log10(tmp)

        """
            FINISH by YOUSELF
        """
        return log_llh

    def em_estimator(self, X: np.ndarray):
        """Update paramters of GMM

            param: X: A matrix including data samples, num_samples * D
            return: log likelihood of updated model
        """

        log_llh = 0.0
        N = X.shape[0]  # 帧数，观测样本个数
        K = self.K  # 高斯个数
        # E-step
        gamma_nk = np.zeros((N, K))
        for n in range(N):
            for k in range(K):
                den = 0
                xn = X[n, :].transpose()  # dim*1
                for j in range(self.K):
                    den += self.pi[j] * self.gaussian(xn, self.mu[j], self.sigma[j])
                gamma_nk[n, k] = self.pi[k] * self.gaussian(xn, self.mu[k], self.sigma[k]) / den

        # M-step
        for k in range(K):
            N_k = np.sum(gamma_nk[:, k])
            # 更新均值向量
            tmp_mu = np.zeros((self.dim,))
            tmp_sigma = np.zeros((self.dim, self.dim))
            for n in range(N):
                xn = X[n, :].transpose()  # dim*1
                tmp_mu += gamma_nk[n, k] * xn

                tmp_sigma += gamma_nk[n, k] * (xn - self.mu[k])[:, np.newaxis] @ (xn - self.mu[k])[:,
                                                                                 np.newaxis].transpose()
            # 更新均值向量
            self.mu[k] = tmp_mu / N_k

            # 更新协方差矩阵
            self.sigma[k] = tmp_sigma / N_k

            # 更新混合系数
            self.pi[k] = N_k / N

        """
            FINISH by YOUSELF
        """
        log_llh = self.calc_log_likelihood(X)

        return log_llh


def process(target_gmm):
    target = target_gmm[0]
    gmm = target_gmm[1]
    print("train GMM for target {}".format(target))
    feats = get_feats(target, dict_utt2feat, dict_target2utt)
    for i in range(num_iterations):
        log_llh = gmm.em_estimator(feats)
        print("log_llh = {}".format(log_llh))
    return target, gmm


def train(gmms, num_iterations=num_iterations):
    # dict_utt2feat, dict_target2utt = read_feats_and_targets('train/feats.scp', 'train/text')
    # gmms_values = [i for i in gmms.values()]
    #
    # target = [target for target in targets]
    # gmm = [gmm for gmm in gmms_values]
    # target_gmm = list(zip(target, gmm))
    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=1)
    # target_gmm_update = pool.map(process, target_gmm)
    # gmms_update = {}
    # for i in range(len(target_gmm_update)):
    #     gmms_update[target_gmm[i][0]] = target_gmm[i][1]

    # gmms = gmms_update
    for target in targets:
        feats = get_feats(target, dict_utt2feat, dict_target2utt)  #
        print("train GMM for {}".format(target))
        for i in range(num_iterations):
            log_llh = gmms[target].em_estimator(feats)
            print("log_llh = {}".format(log_llh))
    return gmms


def test(gmms):
    correction_num = 0
    error_num = 0
    acc = 0.0
    dict_utt2feat, dict_target2utt = read_feats_and_targets('test/feats.scp', 'test/text')
    dict_utt2target = {}
    for target in targets:
        utts = dict_target2utt[target]
        for utt in utts:
            dict_utt2target[utt] = target
    for utt in dict_utt2feat.keys():
        feats = kaldi_io.read_mat(dict_utt2feat[utt])
        scores = []
        for target in targets:
            scores.append(gmms[target].calc_log_likelihood(feats))
        predict_target = targets[scores.index(max(scores))]
        if predict_target == dict_utt2target[utt]:
            correction_num += 1
        else:
            error_num += 1
    acc = correction_num * 1.0 / (correction_num + error_num)
    return acc


def main():
    gmms = {}
    for target in targets:
        gmms[target] = GMM(39, K=num_gaussian)  # Initial model
    start = time.process_time()
    gmms = train(gmms)
    end = time.process_time()
    print('Running time: %s Seconds' % (end - start))
    acc = test(gmms)
    print('Recognition accuracy: %f' % acc)
    fid = open('acc.txt', 'w')
    fid.write(str(acc))
    fid.close()


def f(x):
    return x * x


if __name__ == '__main__':
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    xs = range(100)
    y = pool.map(f, xs)
    print(y)

    main()
