import numpy as np
import matplotlib.pyplot as plt


def draw_Pareto(moead,fit):

    file = moead.name2 + '.pf'
    path = 'vector_csv_file/' + file
    pf_data = np.loadtxt(fname=path)
    for pi, pp in enumerate(pf_data):
        plt.scatter(pp[0], pp[1], c='black', s=5)
    for pi, pp in enumerate(fit):
        plt.scatter(fit[pi][0],fit[pi][1], c='red', s=5)
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)

    plt.show()





