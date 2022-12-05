import argparse
import os
import numpy as np
from mpi4py import MPI
from utils.problem import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# code for DMO (Decentralized Minimax OPtimizer): a simple and effient framework for decentralized minimax optimization

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Decentralized Minimax Problem')
parser.add_argument('--data_path', default='./Data', type=str, help='path of data')
parser.add_argument('--data_file', default='a9a', type=str, help='file of data')
parser.add_argument('--kappa', default=0.999, type=float, help='inner loop paramter of gossip matrix')
parser.add_argument('--lambda2', default=1e-5, type=float, help='coefficient of regularization')
parser.add_argument('--alpha', default=10.0, type=float, help='coefficient in the regularization')
parser.add_argument('--lr_x', default=0.1, type=float, help='learning rate for x')
parser.add_argument('--lr_y', default=1, type=float, help='learning rate for y')
parser.add_argument('--beta', default=0.01, type=float, help='momentum weight of SGD')
parser.add_argument('--b', default=1, type=int, help='mini batch size')
parser.add_argument('--multi_step', default=20, type=int, help='steps for solving max problem')
parser.add_argument('--num_mix', default=2, type=int, help='numer of FastMix/Mix rounds')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs (one pass of data)')
parser.add_argument('--print_freq', default=1, type=int, help='frequency to print train stats')
parser.add_argument('--out_fname', default='./result_DMO.csv', type=str, help='path of output file')
parser.add_argument('--out_fig', default= './img/', type=str, help = 'path of output picture')
# --------------------------------------------------------------------------- #

def main():
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    path = args.data_path
    datafile = args.data_file + '.txt'
    data, label = load_data(path, datafile)
    n, d = data.shape
    
    # ditribute the data on different machines 
    p = n // size
    n = p * size

    # cast the tail data
    data = data[:n, :]
    label = label[:n]

    # e.g. for dataset a9a, n = 48840, m =8, p =6105
    if rank == 0:
        if not os.path.exists(args.out_fname):
            with open(args.out_fname, 'w') as f:
                print('===begin experiment==', file=f)

    x0 = np.ones(d) * 1
    y0 = np.ones(n) * 1.0 / n
    
    # stochastic counterpart GT-DA, namely GT-SGDA
    num_epochs = 2 * args.epochs * n  // ((args.multi_step +1 ) * size * args.b) 
    grad_lst,comm_lst, res = GTDA(data, label, x0, y0, args, comm, num_epochs, consensus_method=None)
    if rank == 0:
        res_sgd = res
        grad_sgd = grad_lst
        comm_sgd = comm_lst

    # compare with the basline DMHSGD (NeurIPS21)
    # DMHSGD can be implemented in our framwork DMO with: "gradient_estimator='MR', consensus_method=None"
    num_epochs = args.epochs * n  //  (size * args.b)
    grad_lst,comm_lst, res = DMO(data, label, x0, y0, args, comm, num_epochs, gradient_estimator='MR', consensus_method=None)
    if rank == 0:
        res_dmhsgd = res 
        grad_dmhsgd = grad_lst
        comm_dmhsgd = comm_lst
    # our methods

    # DREAM with the same number of epochs as DMHSGD
    grad_lst, comm_lst, res = DMO(data, label, x0, y0, args, comm, num_epochs // 2, gradient_estimator='VR', consensus_method='FM')
    if rank == 0:
        res_dream = res 
        grad_dream = grad_lst
        comm_dream = comm_lst

    if rank == 0:
        
        plt.rc('font', size=15)
        plt.figure()
        plt.plot(grad_sgd, res_sgd, ':r', label = 'GT-SGDA', linewidth = 3)
        plt.plot(grad_dmhsgd, res_dmhsgd, '-.b', label = 'DM-HSGD', linewidth = 3)
        plt.plot(grad_dream, res_dream, '-k', label = 'DREAM', linewidth = 3)
        plt.xlim((0, grad_dream[-1]))
        plt.legend(fontsize=20,frameon=False,loc='lower left')
        plt.tick_params('x',labelsize=15)
        plt.tick_params('y',labelsize=15)
        plt.ticklabel_format(style='sci',scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(args.out_fig + args.data_file + '.sfo.png')
        plt.savefig(args.out_fig + args.data_file + '.sfo.eps', format = 'eps')

        plt.rc('font', size=15)
        plt.figure()
        plt.plot(comm_sgd, res_sgd, ':r', label = 'GT-SGDA', linewidth = 3)
        plt.plot(comm_dmhsgd, res_dmhsgd, '-.b', label = 'DM-HSGD', linewidth = 3)
        plt.plot(comm_dream, res_dream, '-k', label = 'DREAM', linewidth = 3)
        plt.xlim((0, comm_dream[-1]))
        plt.legend(fontsize=20,frameon=False,loc='lower left')
        plt.tick_params('x',labelsize=15)
        plt.tick_params('y',labelsize=15)
        plt.ticklabel_format(style='sci',scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(args.out_fig + args.data_file + '.comm.png')
        plt.savefig(args.out_fig + args.data_file + '.comm.eps', format = 'eps')

# run the program with "mpiexec -n 8 python xxx"
if __name__ == '__main__':
    main()
