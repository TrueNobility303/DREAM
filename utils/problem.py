from ast import arg
from sklearn.datasets import load_svmlight_file
import numpy as np
from mpi4py import MPI
from tqdm import tqdm
import time
import math

def load_data(path, file):
    source = path + '/' + file
    data = load_svmlight_file(source)
    x_raw = data[0]
    y = np.array(data[1])
    x = np.ones([x_raw.shape[0], x_raw.shape[1] + 1])
    x[:, :-1] = x_raw.todense()
    return x, y

def cal_gradient_x(data, label, x, y, idx, lambda2, alpha):
    # calculate the the logit term  
    term1 = np.exp(- np.sum(data[idx,:] * x, axis=1) * label[idx]) 
    term2 = - y[idx] * label[idx] * term1 / (1 + term1)
    grad1 = np.matmul(term2, data[idx,:]) / len(idx)
    denominator = (1 + alpha * x * x) * (1 + alpha * x * x)
    numerator = 2 * lambda2 * alpha * x
    grad2 = numerator / denominator
    return grad1 + grad2

def cal_gradient_y(data, label, x, y, idx, n):
    logistic_loss = np.log(1 + np.exp(- np.sum(data[idx,:] * x, axis=1) * label[idx])) / len(idx)
    grad = np.ones(n) * 1.0 / n
    grad[idx] += logistic_loss
    return grad - y 

# consensus step in a ring graph
def consensus(variable, comm, rank, size, kappa):
    left = (rank + size - 1) % size
    right = (rank + 1) % size
    send_buffer = np.copy(variable)
    recv_left = np.zeros_like(send_buffer, dtype=float)
    recv_right = np.zeros_like(send_buffer, dtype=float)
    req_left = comm.Isend(send_buffer, dest=left, tag=1)
    req_right = comm.Isend(send_buffer, dest=right, tag=2)
    comm.Recv(recv_left, source=left, tag=2)
    comm.Recv(recv_right, source=right, tag=1)
    req_left.wait()
    req_right.wait()
    return (recv_left * (1-kappa)/2  + recv_right * (1-kappa)/2  + variable * kappa) 

def multi_consensus(variable, comm, rank, size, kappa, K):
    for i in range(K):
        variable = consensus(variable, comm, rank, size, kappa)
    return variable

# project y onto the n-dimentinal simplex
def projection(y, n):
    y_sort = np.sort(y)
    y_sort = y_sort[::-1]
    sum_y = 0
    t = 0
    for i in range(n):
        sum_y = sum_y + y_sort[i]
        t = (sum_y - 1.0) / (i + 1)
        if i < n - 1 and y_sort[i + 1] <= t < y_sort[i]:
            break
    return np.maximum(y - t, 0)

def cal_phi(data, label, x, n, lambda2, alpha):
    logistic_loss = np.log(1 + np.exp(- np.sum(data * x, axis=1) * label))
    un = np.ones(n) * 1.0 / n

    y_star = projection(logistic_loss + un, n)
    phi = np.inner(y_star, logistic_loss) - 0.5 * np.sum((y_star - un) * (y_star - un)) + lambda2 * \
          np.sum(alpha * x * x / (alpha * x * x + 1))
    gx = cal_gradient_x(data, label, x, y_star, np.arange(n), lambda2, alpha)
    return phi , np.linalg.norm(gx)

def fast_mix(variable, comm, rank, size, kappa, K):
    variable_old = np.copy(variable)
    # calculate the second largest singular value
    sig = kappa + (1-kappa) * np.cos(2 * (size -1) / size * np.pi)
    q = np.sqrt(1-sig*sig)
    eta = (1-q) / (1+q)  
    for i in range(K):
        variable = (1+eta) * consensus(variable,comm,rank,size,kappa) - eta * variable_old
        variable_old = np.copy(variable)
    return variable

## gradient_estimator: 'MR': momentum-based recursive approach; 'VR': variance reduction approach, 
## consensus_methid: 'None' ; 'MC': multi-consensus; 'FM': FastMix

# DMO (Decentralized Minimax Optimizer: for single loop decentralized NC-SC minimax)
def DMO(data, label, x0, y0, args, comm, epochs, gradient_estimator='VR', consensus_method='FM'):
    alpha = args.alpha
    lambda2 = args.lambda2
    kappa = args.kappa

    rank = comm.Get_rank()
    size = comm.Get_size()

    # use different random seeds for each agent
    np.random.seed(42 + rank)

    n, d = data.shape
    p = n // size
    
    # initialize 
    x = np.copy(x0)
    y = np.copy(y0)
    x_old = np.copy(x)
    gx_old = np.zeros_like(x)
    vx = np.zeros_like(x)
    x_bar = np.copy(x)
    y_old = np.copy(y)
    gy_old = np.zeros_like(y)
    vy = np.zeros_like(y)

    comm_comp_lst = []
    grad_comp_lst = []
    phi_lst = []
    grad_cnt = 0
    comm_cnt = 0
    
    min_phi = np.inf
    for epoch in tqdm(range(epochs)):
        # decentralized minimax optimizer 
        if gradient_estimator == 'MR':
            # we find that the initial large batch size may be ununcessary in practice
            # if epoch == 0:
            #     idx = np.random.randint(0, p, b0) * rank 
            #     gx = cal_gradient_x(data, label, x, y, idx, lambda2, alpha)
            #     gy = cal_gradient_y(data, label, x, y, idx, n)
            #     grad_cnt = grad_cnt + b0
            idx = np.random.randint(0, p, args.b) * rank
            gx = cal_gradient_x(data, label, x, y, idx, lambda2, alpha) + (1 - args.beta) * \
                (gx_old - cal_gradient_x(data, label, x_old, y_old, idx, lambda2, alpha))
            gy = cal_gradient_y(data, label, x, y, idx, n) + (1 - args.beta) * \
                (gy_old - cal_gradient_y(data, label, x_old, y_old, idx, n))
            grad_cnt = grad_cnt + 4 * size * args.b
        elif gradient_estimator == 'VR':
            # set both the probability, mini-batch size and mega-batch size according to theoretical choice
            b0 = args.b * args.b 
            prob = args.b / (args.b + b0)
            rad = np.random.random()
            if rad < prob:
                id0 = np.random.randint(0, p, b0) * rank
                # id0 = range(n)
                gx = cal_gradient_x(data, label, x, y, id0, lambda2, alpha)
                gy = cal_gradient_y(data, label, x, y, id0, n)
                grad_cnt = grad_cnt + 2 * size * b0
            else:
                idx = np.random.randint(0, p, args.b) * rank
                gx = cal_gradient_x(data, label, x, y, idx, lambda2, alpha) + \
                    (gx_old - cal_gradient_x(data, label, x_old, y_old, idx, lambda2, alpha))
                gy = cal_gradient_y(data, label, x, y, idx, n) + \
                    (gy_old - cal_gradient_y(data, label, x_old, y_old, idx, n))
                grad_cnt = grad_cnt + 4 * size * args.b
        else:
            idx = np.random.randint(0, p, args.b) * rank 
            gx = cal_gradient_x(data, label, x, y, idx, lambda2, alpha)
            gy = cal_gradient_y(data, label, x, y, idx, n)
            grad_cnt = grad_cnt + 2 * size * args.b
        # gradient tracking
        vx = vx + gx - gx_old
        vy = vy + gy - gy_old
        if consensus_method is None:
            vx = consensus(vx, comm, rank, size, kappa)
            vy = consensus(vy, comm, rank, size, kappa)
        elif consensus_method == 'MC':
            vx = multi_consensus(vx, comm, rank, size, kappa, args.num_mix)
            vy = multi_consensus(vy, comm, rank, size, kappa, args.num_mix)
        elif consensus_method == 'FM':
            vx = fast_mix(vx, comm, rank, size, kappa, args.num_mix)
            vy = fast_mix(vy, comm, rank, size, kappa, args.num_mix)
            
        gx_old = np.copy(gx)
        gy_old = np.copy(gy)
        # update x and y
        x_old = np.copy(x)
        x = x - args.lr_x * vx
        y_old = np.copy(y)
        y = y + args.lr_y * vy
        if consensus_method is None:
            x = consensus(x, comm, rank, size, kappa)
            y = consensus(y, comm, rank, size, kappa)
        elif consensus_method == 'MC':
            x = multi_consensus(x, comm, rank, size, kappa, args.num_mix)
            y = multi_consensus(y, comm, rank, size, kappa, args.num_mix)
        elif consensus_method == 'FM':
            x = fast_mix(x, comm, rank, size, kappa, args.num_mix)
            y = fast_mix(y, comm, rank, size, kappa, args.num_mix)

        y = projection(y, n)

        if consensus_method is not None:
            comm_cnt = comm_cnt + 4 * args.num_mix
        else:
            comm_cnt = comm_cnt + 4

        if (epoch + 1) % args.print_freq == 0:
            # compute x_bar on agent rank 0
            comm.Reduce(x, x_bar, op=MPI.SUM)
            if rank == 0:
                x_bar = x_bar / size
                phi,grad_phi = cal_phi(data, label, x_bar, n, lambda2, alpha)
                with open(args.out_fname, '+a') as f:
                    print('DMO with {cm}:{ep:d},{phi:.8f},{grad_phi:.8f}'.format(cm=gradient_estimator, ep=epoch + 1, phi=float(phi),grad_phi=float(grad_phi)), file=f)
                # store the minimal loss
                if phi < min_phi:
                    min_phi = phi
                phi_lst.append(phi / n)
                grad_comp_lst.append(grad_cnt)
                comm_comp_lst.append(comm_cnt)
        comm.barrier()
    return grad_comp_lst, comm_comp_lst, phi_lst

# GT-GDA
def GDA(data, label, x0, y0, args, comm, epochs):
    alpha = args.alpha
    lambda2 = args.lambda2
    kappa = args.kappa

    rank = comm.Get_rank()
    size = comm.Get_size()

    # use different random seeds for each agent
    np.random.seed(42 + rank)

    n, d = data.shape
    p = n // size
    
    # initialize 
    x = np.copy(x0)
    y = np.copy(y0)
    x_old = np.copy(x)
    gx_old = np.zeros_like(x)
    vx = np.zeros_like(x)
    x_bar = np.copy(x)
    y_old = np.copy(y)
    gy_old = np.zeros_like(y)
    vy = np.zeros_like(y)

    comm_comp_lst = []
    grad_comp_lst = []
    phi_lst = []
    grad_cnt = 0
    comm_cnt = 0
    
    min_phi = np.inf
    for epoch in tqdm(range(epochs)):
        idx = np.arange(p) * rank 
        gx = cal_gradient_x(data, label, x, y, idx, lambda2, alpha)
        gy = cal_gradient_y(data, label, x, y, idx, n)
        grad_cnt = grad_cnt + 2 * size * p
        # gradient tracking
        vx = vx + gx - gx_old
        vy = vy + gy - gy_old
        vx = consensus(vx, comm, rank, size, kappa)
        vy = consensus(vy, comm, rank, size, kappa)
            
        gx_old = np.copy(gx)
        gy_old = np.copy(gy)
        # update x and y
        x_old = np.copy(x)
        x = x - args.lr_x * vx
        y_old = np.copy(y)
        y = y + args.lr_y * vy
        x = consensus(x, comm, rank, size, kappa)
        y = consensus(y, comm, rank, size, kappa)
        y = projection(y, n)

        comm_cnt = comm_cnt + 4

        if (epoch + 1) % args.print_freq == 0:
            # compute x_bar on agent rank 0
            comm.Reduce(x, x_bar, op=MPI.SUM)
            if rank == 0:
                x_bar = x_bar / size
                phi,grad_phi = cal_phi(data, label, x_bar, n, lambda2, alpha)
                with open(args.out_fname, '+a') as f:
                    print('GT-GDA:{ep:d},{phi:.8f},{grad_phi:.8f}'.format(ep=epoch + 1, phi=float(phi),grad_phi=float(grad_phi)), file=f)
                # store the minimal loss
                if phi < min_phi:
                    min_phi = phi
                phi_lst.append(phi / n)
                grad_comp_lst.append(grad_cnt)
                comm_comp_lst.append(comm_cnt)
        comm.barrier()
    return grad_comp_lst, comm_comp_lst, phi_lst

# GT-SRVR
def SRVR(data, label, x0, y0, args, comm, epochs):
    alpha = args.alpha
    lambda2 = args.lambda2
    kappa = args.kappa

    rank = comm.Get_rank()
    size = comm.Get_size()

    # use different random seeds for each agent
    np.random.seed(42 + rank)

    n, d = data.shape
    p = n // size
    
    # initialize 
    x = np.copy(x0)
    y = np.copy(y0)
    x_old = np.copy(x)
    gx_old = np.zeros_like(x)
    vx = np.zeros_like(x)
    x_bar = np.copy(x)
    y_old = np.copy(y)
    gy_old = np.zeros_like(y)
    vy = np.zeros_like(y)

    comm_comp_lst = []
    grad_comp_lst = []
    phi_lst = []
    grad_cnt = 0
    comm_cnt = 0
    
    min_phi = np.inf
    q = math.ceil(math.sqrt(p))
    for epoch in tqdm(range(epochs)):
        if epoch % q ==0:
            idx = np.arange(p) * rank 
            gx = cal_gradient_x(data, label, x, y, idx, lambda2, alpha)
            gy = cal_gradient_y(data, label, x, y, idx, n)
            grad_cnt = grad_cnt + 2 * size * p
        else:
            idx = np.random.randint(0, p, q) * rank
            gx = cal_gradient_x(data, label, x, y, idx, lambda2, alpha) + \
                (gx_old - cal_gradient_x(data, label, x_old, y_old, idx, lambda2, alpha))
            gy = cal_gradient_y(data, label, x, y, idx, n) + \
                (gy_old - cal_gradient_y(data, label, x_old, y_old, idx, n))
            grad_cnt = grad_cnt + 4 * size * q
        # gradient tracking
        vx = vx + gx - gx_old
        vy = vy + gy - gy_old
        vx = consensus(vx, comm, rank, size, kappa)
        vy = consensus(vy, comm, rank, size, kappa)
            
        gx_old = np.copy(gx)
        gy_old = np.copy(gy)
        # update x and y
        x_old = np.copy(x)
        x = x - args.lr_x * vx
        y_old = np.copy(y)
        y = y + args.lr_y * vy
        x = consensus(x, comm, rank, size, kappa)
        y = consensus(y, comm, rank, size, kappa)
        y = projection(y, n)

        comm_cnt = comm_cnt + 4

        if (epoch + 1) % args.print_freq == 0:
            # compute x_bar on agent rank 0
            comm.Reduce(x, x_bar, op=MPI.SUM)
            if rank == 0:
                x_bar = x_bar / size
                phi,grad_phi = cal_phi(data, label, x_bar, n, lambda2, alpha)
                with open(args.out_fname, '+a') as f:
                    print('GT-SRVR:{ep:d},{phi:.8f},{grad_phi:.8f}'.format(ep=epoch + 1, phi=float(phi),grad_phi=float(grad_phi)), file=f)
                # store the minimal loss
                if phi < min_phi:
                    min_phi = phi
                phi_lst.append(phi / n)
                grad_comp_lst.append(grad_cnt)
                comm_comp_lst.append(comm_cnt)
        comm.barrier()
    return grad_comp_lst, comm_comp_lst, phi_lst

