import numpy as np
def select_and_reproduct(pop,B,i,delta,n_dim,lu,pcrossover,de_factor,miu,pmutation):
    if np.random.rand() < delta:
        P = B[i,:]
    else:
        P = np.arange(1,pop.shape[0])
    p_len = len(P)
    parent = np.zeros((2,pop.shape[1]))
    rand_index = np.random.randint(p_len, size=(1,2))
    while rand_index[0][0] == rand_index[0][1]:
        rand_index = np.random.randint(p_len, size=(1,2))
    m1 = rand_index[0][0]
    m2 = rand_index[0][1]
    n1 = P[m1]
    n2 = P[m2]
    parent[0,:] = pop[int(n1)-1,:]
    parent[1,:] = pop[int(n2)-1,:]
    y = de_crossover(pop[i,:], parent[0,:], parent[1,:], n_dim, lu, pcrossover, de_factor)

    y1 = y
    rand_value = np.random.rand();
    if rand_value < 0.5:
        theta = (2 * rand_value) ** (1 / (1 + miu)) - 1
    else:
        theta = 1 - (2 - 2 * rand_value) ** (1 / (1 + miu))
    rand_mut = np.random.rand(n_dim, 1)
    muted = rand_mut < pmutation
    y = y1 + theta * (lu[1,:] - lu[0,:])
    y[muted[:,0] == False] = y1[muted[:,0] == False]
    y[y < 0] = 0
    y[y > 1] = 1
    return np.array([y,P],dtype=object)


def de_crossover(parent1, parent2,parent3, n, lu, pcrossover, de_factor):
    parent_1 = parent1[0: n];
    parent_2 = parent2[0: n];
    parent_3 = parent3[0: n];

    cross_idx1 = np.random.rand(1, n) < pcrossover;
    cross_idx1[0][np.random.randint(n)] = True

    child = parent_1 + de_factor * (parent_3 - parent_2);

    child[cross_idx1[0] == False] = parent_1[cross_idx1[0] == False];
    child[child > 1] = 1
    child[child < 0] = 0

    return child

