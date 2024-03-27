import numpy as np
import math

class Estimate_TC:
    def __init__(self, args, train_init):
        self.train_init = train_init
        self.N = args.N
        self.d = args.d
        self.M = args.M
        self.K = args.K
        self.gamma = args.gamma
        self.seed = args.seed
        self.mean = np.zeros(self.d)
        self.cov = args.cov

    def MC_approx(self):
        approx_TC = 0
        for m in range(self.M):
            Y_i_list = np.random.multivariate_normal(mean=self.mean, cov=self.cov**2 * np.eye(self.d),
                                                     size=self.N)
            approx_KL_m = 0
            for Y_i in Y_i_list:
                approx_integral_i = self.estimate_integral(Y_i_list=Y_i_list, Y_i=Y_i)
                approx_KL_m += approx_integral_i
            approx_KL_m /= self.N
            approx_TC += approx_KL_m

        return (approx_TC / self.M)

    def gaussian_KDE_pdf(self, x, mean, cov):
        constant = (2 * math.pi * cov**2) ** (self.d/2)
        abs_diff = x[np.newaxis, :, :] - mean[:, np.newaxis, :]
        abs_square = -np.linalg.norm(abs_diff[:, :, :, np.newaxis],axis=2) ** 2 / (2*cov**2)
        # print('Shape of abs_square', abs_square.shape)
        return np.exp(abs_square) / constant

    def estimate_integral(self, Y_i_list, Y_i):
        x_k_list = np.random.multivariate_normal(mean=Y_i, cov=self.gamma**2 * np.eye(self.d),
                                                 size=self.K)

        denominator = np.mean(self.gaussian_KDE_pdf(x=x_k_list, mean=np.zeros_like(Y_i_list), cov=math.sqrt(self.cov**2 + self.gamma**2)), axis=0)
        numerator = np.mean(self.gaussian_KDE_pdf(x=x_k_list, mean=Y_i_list, cov=self.gamma), axis=0)

        return np.mean(np.log(np.divide(numerator, denominator)))
