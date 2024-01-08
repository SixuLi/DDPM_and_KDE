import copy

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("darkgrid", {'grid.linestyle': '--'})
sns.set_context('poster')

class Generative_model:
    def __init__(self, args, train_init):
        self.args = args
        self.train_init = train_init
        self.num_training_data = args.num_training_data
        self.num_generated_samples = args.num_generated_samples
        self.h = args.h  # Step size
        self.T = args.T  # Time interval [0,T]
        self.K = int(args.T / args.h)  # Total number of steps
        self.mu_p = np.array([-5,5])
        self.sigma_p = math.sqrt(10)

        # Initialize the training data and generated samples
        self.initialization()

    def initialization(self):
        # Training data
        self.training_data = np.random.multivariate_normal(mean=self.mu_p, cov=self.sigma_p**2 * np.eye(2), size=self.num_training_data)

        # Initialize the samples
        self.initialized_samples = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=self.num_generated_samples)
        self.generated_samples = np.zeros_like(self.initialized_samples)

    def sigma(self, t):
        return math.sqrt(1 - math.exp(-2*t))

    def mu(self, t):
        return math.exp(-t)

    def condi_pdf(self,t,x,y):
        constant = 2 * math.pi * self.sigma(t)**2
        abs_diff = self.mu(t) * y[np.newaxis, :, :] - x[:, np.newaxis, :]
        abs_square = - np.linalg.norm(abs_diff[:, :, :, np.newaxis], axis=2) ** 2 / (2 * self.sigma(t)**2)
        return np.exp(abs_square) / constant

    def condi_vec(self,t,x,y):
        vec = (self.mu(t) * y[np.newaxis, :, :] - x[:, np.newaxis, :]) / (self.sigma(t)**2)
        return vec

    def p_t(self,t,x):
        constant = 2 * math.pi * (self.sigma(t)**2 + self.mu(t)**2 * self.sigma_p**2)
        abs_square = np.linalg.norm(x - self.mu(t)*self.mu_p, axis=1) ** 2 / (2 * (self.sigma(t)**2 + self.mu(t)**2 * self.sigma_p**2))
        return np.exp(-abs_square) / constant

    def v_t(self,t,x):
        numerator = self.mu(t) * self.mu_p - x
        denominator = self.sigma(t) ** 2 + self.mu(t) ** 2 * self.sigma_p ** 2
        return np.multiply((numerator / denominator), self.p_t(t,x)[:, np.newaxis])

    def v_t_N(self,t,x,y):
        condi_vec = self.condi_vec(t,x,y)
        condi_pdf = self.condi_pdf(t,x,y)
        return np.squeeze(self.vectorized_matrix_multiplication(condi_vec, condi_pdf) / y.shape[0], axis=2)

    def p_t_N(self,t,x,y):
        condi_pdf = self.condi_pdf(t,x,y)
        return condi_pdf.mean(axis=1)

    def normalize_rows(self, a):
        norms = np.linalg.norm(a, axis=1)
        # Normalize each row
        return a / norms[:, np.newaxis]

    def vectorized_matrix_multiplication(self, a, b):
        result = np.array([np.matmul(ai.T, bi) for ai, bi in zip(a,b)])
        return result

    def u_t(self, t,x):
        vt = self.v_t(t,x)
        pt = self.p_t(t,x)
        return np.divide(vt, pt[:, np.newaxis])

    def u_t_N(self,t,x,y):
        vtN = self.v_t_N(t,x,y)
        ptN = self.p_t_N(t,x,y)
        return np.divide(vtN, ptN)

    def score_approximation_error(self, num_x, num_y, rand_sample_time):
        approximation_error = np.zeros(rand_sample_time)
        for i in range(rand_sample_time):
            np.random.seed(i)
            y = np.random.multivariate_normal(mean=self.mu_p, cov=self.sigma_p ** 2 * np.eye(2), size=num_y)
            for n in range(1, self.K):
                t = n * self.h
                x_t = np.random.multivariate_normal(mean=self.mu(t) * self.mu_p, cov=(self.sigma(t) ** 2 + self.mu(t) ** 2 * self.sigma_p ** 2) * np.eye(2), size=num_x)
                approximation_error[i] += np.linalg.norm(self.u_t_N(t=t,x=x_t,y=y) - self.u_t(t=t, x=x_t)) ** 2 / num_x
            approximation_error[i] /= self.K
        return approximation_error.mean()

    def diffusion_generation(self, x_0, y):
        x_t = copy.deepcopy(x_0)
        for n in range(0, self.K):
            if self.args.is_early_stop and n >= self.K - 20:
                break
            z = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=x_t.shape[0])
            x_t += self.h * (x_t + 2 * self.u_t_N(t=(self.K - n) * self.h, x=x_t, y=y)) + math.sqrt(self.h) * math.sqrt(2) * z
        return x_t

    def true_backward_process(self, x_0):
        x_t = copy.deepcopy(x_0)
        for n in range(0, self.K):
            if self.args.is_early_stop and n >= self.K - 20:
                break
            z = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(2), size=x_t.shape[0])
            x_t += self.h * (x_t + 2 * self.u_t(t=(self.K-n)*self.h, x=x_t)) + math.sqrt(self.h) * math.sqrt(2) * z
        return x_t

    def visualization_sample_generation(self, tag='ddpm'):
        fig, ax = plt.subplots(figsize=(12, 8))
        if tag == 'ddpm':
            ax.scatter(self.training_data[:, 0], self.training_data[:, 1], marker="x", s=120, label='training samples')
            ax.scatter(self.generated_samples_ddpm[:, 0], self.generated_samples_ddpm[:, 1], marker='o', s=30, label='generated samples')
            ax.scatter(self.initialized_samples[:, 0], self.initialized_samples[:, 1], marker='o', s=30, label='initialization')
            if self.args.is_early_stop:
                figsave_path = os.path.join(self.train_init.output_path, '_early_stop_' + 'generated_samples_ddpm.png')
            else:
                figsave_path = os.path.join(self.train_init.output_path, 'generated_samples_ddpm.png')
        elif tag == 'true_process':
            ax.scatter(self.training_data[:, 0], self.training_data[:, 1], marker="x", s=120, label='training samples')
            ax.scatter(self.generated_samples_true_process[:, 0], self.generated_samples_true_process[:, 1], marker='o', s=30,
                       label='generated samples')
            ax.scatter(self.initialized_samples[:, 0], self.initialized_samples[:, 1], marker='o', s=30,
                       label='initialization')
            if self.args.is_early_stop:
                figsave_path = os.path.join(self.train_init.output_path, '_early_stop_' + 'generated_samples_true_process.png')
            else:
                figsave_path = os.path.join(self.train_init.output_path, 'generated_samples_true_process.png')
        plt.legend(loc='best', fontsize=15)
        plt.tick_params(labelleft=False, labelbottom=False)
        plt.savefig(figsave_path, bbox_inches='tight')
        plt.show()

    def visualization_score_approximation_error(self, num_training_data_list, err_list):
        reference_1_start_point = (5.5, 0.85)
        reference_1_end_point = (7.0, -0.65)
        reference_line_1_x = [reference_1_start_point[0], reference_1_end_point[0]]
        reference_line_1_y = [reference_1_start_point[1], reference_1_end_point[1]]

        reference_2_start_point = (5.5, -0.25)
        reference_2_end_point = (7.0, -1.75)
        reference_line_2_x = [reference_2_start_point[0], reference_2_end_point[0]]
        reference_line_2_y = [reference_2_start_point[1], reference_2_end_point[1]]

        plt.figure(figsize=(12,8))
        plt.plot(np.log(num_training_data_list), np.log(err_list), marker='X', mec='darkorange', mfc='darkorange')
        plt.plot(reference_line_1_x, reference_line_1_y, linestyle='dashed', c='darkgreen')
        plt.plot(reference_line_2_x, reference_line_2_y, linestyle='dashed', c='darkgreen')
        # plt.axline((np.log(num_training_data_list)[0], np.log(err_list)[0]), slope=-1)
        plt.xlabel(r'$\log$ (Number of sample size $N$)')
        plt.ylabel(r'$\log (\mathbb{E}_{\{y_i\} \sim p_{*}^{\otimes N}} \left|E_{\{y_i\}} \right|^2)$')
        figsave_path = os.path.join(self.train_init.output_path, 'score_approximation_error.png')
        plt.savefig(figsave_path, bbox_inches='tight')
        plt.show()

    def run_generative_algorithm(self):
        self.generated_samples_true_process = self.true_backward_process(x_0=self.initialized_samples)
        self.generated_samples_ddpm = self.diffusion_generation(x_0=self.initialized_samples, y=self.training_data)





