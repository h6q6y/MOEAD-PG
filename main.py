from Moead import *
from globalVar import *
from select_and_reproduct import *
from get_neighbors import *
from draw import *
import torch
import torch.nn as nn
import problem.LIRCMOP1 as LIR1
import problem.LIRCMOP2 as LIR2

import matplotlib.pyplot as pl
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if torch.cuda.is_available(): # 检查cuda可用
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.lstm = nn.LSTM(POP_SIZE, CELL_SIZE, LAYERS_NUM)
        self.mu = nn.Linear(CELL_SIZE, 30)
        self.sigma = nn.Linear(CELL_SIZE, 30)
        self.distribution = torch.distributions.Normal

    def forward(self, x, h, c):
        cell_out, (h_, c_) = self.lstm(x, (h, c))
        mu = self.mu(cell_out)
        sigma = torch.sigmoid(self.sigma(cell_out))
        return mu, sigma, h_, c_

    def sampler(self, inputs, ht, ct):
        mu, sigma, ht_, ct_ = self.forward(inputs, ht, ct)
        normal = self.distribution(mu, sigma)
        sample_w = np.clip(normal.sample().numpy(), 0, 1)
        return sample_w, ht_, ct_

def discounted_norm_rewards(r):
    for ep in range(TRAJECTORY_NUM*PROBLEM_NUM):
        single_rs = r[ep*TRAJECTORY_LENGTH:ep*TRAJECTORY_LENGTH+TRAJECTORY_LENGTH]
        discounted_rs = np.zeros_like(single_rs)
        running_add = 0.
        for t in reversed(range(0, TRAJECTORY_LENGTH)):
            running_add = running_add * GAMMA + single_rs[t]
            discounted_rs[t] = running_add
        if ep == 0:
            all_disc_norm_rs = discounted_rs
        else:
            all_disc_norm_rs = np.hstack((all_disc_norm_rs, discounted_rs))
    return all_disc_norm_rs

moead = MOEAD()
MN = PolicyNet()

optimizer = torch.optim.Adam(MN.parameters(), lr=LEARNING_RATE)
for iter_data in range(num_train_data):
    inputs, sf_crs, hs, cs, rewards = [], [], [], [], []
    fits = []
    pop_ini = np.array(moead.Pop)
    for p in range(PROBLEM_NUM):
        [fit_ini, cv] = moead.Test_fun.Func(pop_ini)
        cv = overall_cv(cv)
        Z = np.array([min(fit_ini[0, :]), min(fit_ini[1, :])])
        population = np.c_[moead.Pop, fit_ini.T, cv.T]
        lamda = get_lambda()
        B = get_neighbors(moead.Pop_size, lamda, moead.T)
        n_dim = moead.Test_fun.Dimention
        lu = np.ones((2, 30))
        lu[0, :] = 0
        pcrossover = 0.9    # crossover probability
        de_factor = 0.5     # crossover distribution index
        mdi = 20            # mutation distribution index
        pmutation = 1 / n_dim  # mutation probability
        M = moead.M  # 目标函数个数

        for l in range(TRAJECTORY_NUM):
            pop = pop_ini.copy()
            fit = fit_ini.copy()
            h0 = torch.zeros(LAYERS_NUM, 2, CELL_SIZE)
            c0 = torch.zeros(LAYERS_NUM, 2, CELL_SIZE)
            for t in range(TRAJECTORY_LENGTH):
                sf_cr, h_, c_ = MN.sampler(torch.FloatTensor(fit[None, :]), h0, c0)
                sf_cr = np.squeeze(sf_cr, axis=0)
                cr = sf_cr[0]

                subproblem_elected = torch.randperm(moead.Pop_size)
                se = subproblem_elected[t]
                [Y, P] = select_and_reproduct(population, B, se, moead.delta, n_dim, lu, pcrossover, cr, mdi,pmutation)
                Y = Y.T.reshape((1, moead.Test_fun.Dimention))
                [obj_Y, con_Y] = moead.Test_fun.Func(Y)
                con_Y = overall_cv(con_Y)
                offspring = np.c_[Y, obj_Y.T, con_Y.T]
                Z = np.array([min(Z[0], obj_Y[0]), min(Z[1], obj_Y[1])], dtype=object)
                population = update_neighbors(moead, 1e-6, population, offspring, P, moead.n_r, lamda, Z, n_dim, M)
                bsf = np.sum(fit)
                pop_next = population[:,0:n_dim]
                fit_next = population[:,n_dim:n_dim + M]
                bsf_next = np.sum(fit_next)
                reward = (bsf - bsf_next) / bsf
                inputs.append(fit)
                sf_crs.append(sf_cr)
                fits.append(fit_next)

                hs.append(np.squeeze(h0.data.numpy(), axis=0))
                cs.append(np.squeeze(c0.data.numpy(), axis=0))
                rewards.append(reward)
                fit = fit_next.copy().T
                pop = pop_next.copy()
                h0 = h_
                c0 = c_
    #             #print("a")
    all_eps_mean, all_eps_std, all_eps_h, all_eps_c = MN.forward(torch.FloatTensor(np.vstack(inputs)[None, :]),
                                                                 torch.Tensor(np.vstack(hs)[None, :]),
                                                                 torch.Tensor(np.vstack(cs)[None,
                                                                              :]))  # all_esp_mean,all_esp_std: tensor(20000,100);all_eps_h,all_esp_c:tensor(1,20000,100)
    sf_crs = torch.FloatTensor(np.vstack(sf_crs))
    # if iter_data == 20:
    #     print("????")
    all_eps_mean = torch.squeeze(all_eps_mean, 0)
    all_eps_std = torch.squeeze(all_eps_std, 0)
    normal_dis = torch.distributions.Normal(all_eps_mean, all_eps_std)
    log_prob = torch.sum(normal_dis.log_prob(sf_crs + 1e-8), 1)
    log_prob = log_prob[0:4000]
    all_eps_dis_reward = discounted_norm_rewards(rewards)
    loss = - torch.mean(log_prob * torch.FloatTensor(all_eps_dis_reward))  # 一个常数
    loss.backward()
    optimizer.step()
print("PG done")
draw_Pareto(moead,fits[-1])
# path = os.path.abspath('.') + "/model/pg_net"
# torch.save(MN.state_dict(), path)  # save model
# print("test starts")
# all_fs_bsf = []
# test_moead = MOEAD()
# test_moead.Test_fun = LIR2
# for repeat in range(NUM_RUNS):
#     test_pop = np.array(moead.Pop)
#     for test_p in range(TEST_PROBLEM):
#         test_fit,_ = moead.Test_fun.Func(test_pop)
#         test_nfes = POP_SIZE
#         test_h0 = torch.zeros(LAYERS_NUM, 2, CELL_SIZE)
#         test_c0 = torch.zeros(LAYERS_NUM, 2, CELL_SIZE)
#         for t in range(POP_SIZE):
#             if t == 0:
#                 pop_ = test_pop.copy()
#                 fit_ = test_fit.copy()
#             sf_cr_, test_h, test_c = MN.sampler(torch.FloatTensor(fit_[None, :]), test_h0, test_c0)
#             sf_cr = np.squeeze(sf_cr)
#             sf = sf_cr[:, 0:POP_SIZE]
#             cr = sf_cr[:, POP_SIZE:2 * POP_SIZE]
#
#             subproblem_elected = torch.randperm(moead.Pop_size)
#             se = subproblem_elected[t]
#             [Y, P] = select_and_reproduct(population, B, se, moead.delta, n_dim, lu, pcrossover, de_factor, mdi,
#                                           pmutation)
#             Y = Y.T.reshape((1, moead.Test_fun.Dimention))
#             [obj_Y, con_Y] = moead.Test_fun.Func(Y)
#             con_Y = overall_cv(con_Y)
#             offspring = np.c_[Y, obj_Y.T, con_Y.T]
#             Z = np.array([min(Z[0], obj_Y[0]), min(Z[1], obj_Y[1])], dtype=object)
#             population = update_neighbors(moead, 1e-6, population, offspring, P, moead.n_r, lamda, Z, n_dim, M)
#             bsf = np.min(fit)
#             pop_next = population[:, 0:n_dim]
#             fit_next = population[:, n_dim:n_dim + M]
#             test_h0 = test_h
#             test_c0 = test_c
#             fit_ = fit_next.copy().T
#             pop_ = pop_next.copy()
#             bst_test = np.min(fit_,axis=1)
#         all_fs_bsf.append(bst_test.tolist())
# all_fs_bsf = np.array(all_fs_bsf).reshape(NUM_RUNS, -1)
# np.savetxt('CEC17_all_51run_'+str(PROBLEM_NUM)+'DN'+str(POP_SIZE)+'size'+str(CELL_SIZE)+'L'+str(TRAJECTORY_NUM)
#            +'_PG_MAXFE.txt', np.transpose(all_fs_bsf))
# print("saved")
# #draw_Pareto(moead,all_fs_bsf)


