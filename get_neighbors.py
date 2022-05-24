import numpy as np

def get_neighbors(n,lamda,t):
    distance_matrix = np.zeros((n, n))
    B = np.zeros((n,t))
    for i in range(n):
        for j in range(i+1, n):
            a = lamda[i,:]-lamda[j,:]
            distance_matrix[i, j] = np.linalg.norm(lamda[i,:]-lamda[j,:],ord=2)
            distance_matrix[j, i] = distance_matrix[i, j]
        near_index = np.argsort(distance_matrix[i,:])
        B[i,:] = near_index[0: t]
    return B+1


def get_lambda():
    path = 'vector_csv_file/W2D_300.csv'
    W = np.loadtxt(fname=path)
    return W

def update_neighbors(moead,epsilon,chromosome,offspring,p,nr,lamda,z,v,m):

    def teFit(fit, i,z):
        i = int(i)
        z = z.tolist()
        maxx = z[0]
        for j in range(moead.Test_fun.Func_num):
            maxx = max(lamda[i][j] * abs(fit[j] - z[j]), maxx)
        return maxx
    c = 0
    while (c < nr) and p.size != 0:
        p_len = len(p)
        rand_index = np.random.randint(0,p_len)
        m1 = p[rand_index]
        m1 = int(m1) - 1
        m2 = chromosome[m1,-1]
        if offspring[0,-1] <= epsilon and m2 <= epsilon:
            if teFit(offspring[0,v:v+m],m1,z) <= teFit(chromosome[m1,v:v+m],m1,z):
                chromosome[m1,:] = offspring
                c = c + 1

        elif offspring[0,-1] < m2:
            chromosome[m1, :] = offspring
            c = c + 1
        p = np.delete(p, rand_index)

    return chromosome






