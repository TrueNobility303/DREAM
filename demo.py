import argparse
import os
import numpy as np
from mpi4py import MPI
from utils.problem import *
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Decentralized Minimax Problem')
parser.add_argument('--data_path', default='./Data', type=str, help='path of data')
parser.add_argument('--data_file', default='ijcnn1', type=str, help='file of data')
parser.add_argument('--kappa', default=0.999, type=float, help='inner loop paramter of gossip matrix')
parser.add_argument('--lambda2', default=1e-5, type=float, help='coefficient1 of regularization')
parser.add_argument('--alpha', default=10.0, type=float, help='coefficient2 of regularization')
parser.add_argument('--lr_x', default=0.1, type=float, help='learning rate for x')
parser.add_argument('--lr_y', default=1, type=float, help='learning rate for y')
parser.add_argument('--beta', default=0.01, type=float, help='momentum weight of SGD, for DM-HSGD only')
parser.add_argument('--b', default=64, type=int, help='mini batch size')
parser.add_argument('--b0', default=128, type=int, help='mega batch size')
parser.add_argument('--prob', default=0.5, type=int, help='probability for calucate mege batch')
parser.add_argument('--q', default=0.5, type=int, help='probability for update gradient in mini batch')
parser.add_argument('--multi_step', default=4, type=int, help='steps for solving max problem, for GT-DA only')
parser.add_argument('--num_mix', default=2, type=int, help='numer of FastMix rounds')
parser.add_argument('--epochs', default=40, type=int, help='number of epochs (one pass of data)')
parser.add_argument('--print_freq', default=1, type=int, help='frequency to print train stats')
parser.add_argument('--out_fname', default='./result_ijcnn1.csv', type=str, help='path of output file')
parser.add_argument('--out_fig', default= './img/', type=str, help = 'path of output picture')
parser.add_argument('--out_res', default= './res/', type=str, help = 'path of output data')
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
    

    num_iters = args.epochs * n  //  (size * args.b)

    # DMHSGD 

    grad_lst,comm_lst, res = DMHSGD(data, label, x0, y0, args, comm, num_iters)
    if rank == 0:
        res_dmhsgd = res 
        grad_dmhsgd = grad_lst
        comm_dmhsgd = comm_lst
    

    # DREAM 
    grad_lst, comm_lst, res = DREAM(data, label, x0, y0, args, comm, num_iters  // 2)
    if rank == 0:
        res_dream = res 
        grad_dream = grad_lst
        comm_dream = comm_lst

    # GT-DA
    grad_lst, comm_lst, res = DA(data, label, x0, y0, args, comm, num_iters)
    if rank == 0:
        res_gtda = res 
        grad_gtda = grad_lst
        comm_gtda = comm_lst

    #GT-GDA    
    grad_lst, comm_lst, res = GDA(data, label, x0, y0, args, comm, num_iters)
    if rank == 0:
        res_gtgda = res 
        grad_gtgda = grad_lst
        comm_gtgda = comm_lst

    #GT-SRVR    
    grad_lst, comm_lst, res = SRVR(data, label, x0, y0, args, comm, num_iters)
    if rank == 0:
        res_gtsrvr = res 
        grad_gtsrvr = grad_lst
        comm_gtsrvr = comm_lst
    
    if rank == 0:
        
        with open(args.out_res + args.data_file + '.dream.res', 'w') as f:
            print(res_dream, file=f)
        with open(args.out_res + args.data_file + '.dream.comm', 'w') as f:
            print(comm_dream, file=f)
        with open(args.out_res + args.data_file + '.dream.grad', 'w') as f:
            print(grad_dream, file=f)
        
        with open(args.out_res + args.data_file + '.dmhsgd.res', 'w') as f:
            print(res_dmhsgd, file=f)
        with open(args.out_res + args.data_file + '.dmhsgd.comm', 'w') as f:
            print(comm_dmhsgd, file=f)
        with open(args.out_res + args.data_file + '.dmhsgd.grad', 'w') as f:
            print(grad_dmhsgd, file=f)
        
        with open(args.out_res + args.data_file + '.gtda.res', 'w') as f:
            print(res_gtda, file=f)
        with open(args.out_res + args.data_file + '.gtda.comm', 'w') as f:
            print(comm_gtda, file=f)
        with open(args.out_res + args.data_file + '.gtda.grad', 'w') as f:
            print(grad_gtda, file=f)
        
        with open(args.out_res + args.data_file + '.gtgda.res', 'w') as f:
            print(res_gtgda, file=f)
        with open(args.out_res + args.data_file + '.gtgda.comm', 'w') as f:
            print(comm_gtgda, file=f)
        with open(args.out_res + args.data_file + '.gtgda.grad', 'w') as f:
            print(grad_gtgda, file=f)
        
        with open(args.out_res + args.data_file + '.gtsrvr.res', 'w') as f:
            print(res_gtsrvr, file=f)
        with open(args.out_res + args.data_file + '.gtsrvr.comm', 'w') as f:
            print(comm_gtsrvr, file=f)
        with open(args.out_res + args.data_file + '.gtsrvr.grad', 'w') as f:
            print(grad_gtsrvr, file=f)


        plt.rc('font', size=18)
        plt.figure()
        plt.plot(grad_gtda, res_gtda, '-g', label = 'GT-DA',  marker = 'd', markersize = 12, markerfacecolor = 'none', markevery=9)
        plt.plot(grad_gtgda, res_gtgda, '-b', label = 'GT-GDA', marker = 'o', markersize = 12, markerfacecolor = 'none', markevery=11)
        plt.plot(grad_dmhsgd, res_dmhsgd, '-m', label = 'DM-HSGD',  marker = '.', markersize = 12,  markevery=300)
        plt.plot(grad_gtsrvr, res_gtsrvr, ':r', label = 'GT-SRVR', linewidth = 3)
        plt.plot(grad_dream, res_dream, '-k', label = 'DREAM', linewidth = 3)
        plt.xlim((0, grad_dream[-1]))
        plt.legend(fontsize=18,frameon=False,loc='lower left')
        plt.tick_params('x',labelsize=18)
        plt.tick_params('y',labelsize=18)
        plt.ticklabel_format(style='sci',scilimits=(0,0))
        plt.xlabel('#SFO')
        plt.ylabel(r'$P(\bar x)$')
        plt.tight_layout()
        plt.savefig(args.out_fig + args.data_file + '.sfo.png')
        plt.savefig(args.out_fig + args.data_file + '.sfo.eps', format = 'eps')

        plt.rc('font', size=18)
        plt.figure()
        plt.plot(comm_gtda, res_gtda, '-g', label = 'GT-DA',  marker = 'd', markersize = 12, markerfacecolor = 'none', markevery=700)
        plt.plot(comm_gtgda, res_gtgda, '-b', label = 'GT-GDA', marker = 'o', markersize = 12, markerfacecolor = 'none', markevery=900)
        plt.plot(comm_dmhsgd, res_dmhsgd, '-m', label = 'DM-HSGD', marker = '.', markersize = 12,  markevery=1100)
        plt.plot(comm_gtsrvr, res_gtsrvr, ':r', label = 'GT-SRVR', linewidth = 3)
        plt.plot(comm_dream, res_dream, '-k', label = 'DREAM', linewidth = 3)
        plt.xlim((0, comm_dream[-1]))
        plt.legend(fontsize=18,frameon=False,loc='lower left')
        plt.tick_params('x',labelsize=18)
        plt.tick_params('y',labelsize=18)
        plt.ticklabel_format(style='sci',scilimits=(0,0))
        plt.xlabel('#Communication')
        plt.ylabel(r'$P(\bar x)$')
        plt.tight_layout()
        plt.savefig(args.out_fig + args.data_file + '.comm.png')
        plt.savefig(args.out_fig + args.data_file + '.comm.eps', format = 'eps')

# run the program with "mpiexec -n 8 python xxx"
if __name__ == '__main__':
    main()