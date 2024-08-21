import numpy as np
from numpy import matlib as mb
from scipy.optimize import curve_fit
import os
from numba import njit
import matplotlib as mpl
import matplotlib.pyplot as plt
import platform

current_folder = __file__[:-len(os.path.basename(__file__))-1]
my_colors = ["#2f4b7c","#83B5D1","#26C485", "#AF90A9" ,"#a05195","#d45087", "#ff7c43", "#FF5400", "#F1D302"]

@njit
def my_choice_adj(sizes, l, alpha, network_p):
    matrix = np.zeros(sizes)
    while np.count_nonzero(matrix) == 0:
        for i in range(sizes[0]):
            for j in range(sizes[1]):
                matrix[i,j] = 1 if np.random.uniform(0,1) < alpha*network_p**l else 0
    return matrix

@njit
def asymetrical_finite_adjacency_maker(M_0,alpha,network_p,b,s, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = M_0 * (b**s)
    A = my_choice_adj((N,N), s, alpha, network_p)

    for l in range(s-1,0,-1):
        for m in range(0,b**(s-l)):
            fator = N//(b**(s-l))
            aux_matrix = my_choice_adj((fator, fator), l, alpha, network_p)
            A[fator*m:fator*(m+1),fator*m:fator*(m+1)] = aux_matrix
            # print(np.count_nonzero(aux_matrix))
        # plt.imshow(A)
        # plt.show()
    for i in range(N):
        A[i,i] = 0
    return A

@njit
def symmetrical_finite_adjacency_maker(M_0,alpha,network_p,b,s, seed=None):
    if seed is not None:
        np.random.seed(seed)
    N = M_0 * (b**s)
    A = my_choice_adj((N,N), s, alpha, network_p)

    for l in range(s-1,0,-1):
        for m in range(0,b**(s-l)):
            fator = N//(b**(s-l))
            aux_matrix = my_choice_adj((fator, fator), l, alpha, network_p)
            A[fator*m:fator*(m+1),fator*m:fator*(m+1)] = aux_matrix
            # print(np.count_nonzero(aux_matrix))
        # plt.imshow(A)
        # plt.show()
    for i in range(A.shape[0]):
        for j in range(i, A.shape[1]):
            A[i, j] = A[j, i]
    for i in range(N):
        A[i,i] = 0
    return A

@njit
def kh_adjacency_maker(N,E,m,h):
    def sort_ci(N_i, p_i):
        c_i = np.zeros((N_i,N_i))
        for i in range(N_i):
            for j in range(N_i):
                if i != j:
                    c_i[i,j] = 1 if np.random.uniform(0,1) < p_i else 0
        return c_i
    
    E_i = E/(h+1)
    CIJ = np.zeros((N,N))

    p_is_list = []
    for i in range(h):
        A_i = (m-1)/(m**(i+1))
        p_i = E_i/(A_i*(N**2))
        p_is_list.append(p_i)
        N_i = int(N*(1/m)**i)
        N_c = int(N/N_i)
        
        for j in range(N_c):
            c_i = sort_ci(N_i, p_i)
            r_0 = int(1 + j*N_i)
            r_1 = int(r_0 + N_i - 1)
            CIJ[(r_0-1):(r_1) , (r_0-1):(r_1)] = c_i

    return CIJ, p_is_list

@njit
def CIJ_to_P(CIJ):
    N = len(CIJ)
    max_K = 0
    for i in range(N):
        K = np.count_nonzero(CIJ[i,:])
        max_K = K if K > max_K else max_K
    P = np.zeros((N,max_K))
    for i in range(N):
        k=0
        for j in range(N):
            if CIJ[i,j] != 0:
                p = CIJ[i,j]
                P[i,k] = j
                k+=1
        P[i,k:N].fill(-1)
    return P

@njit
def P_to_CIJ(P):
    N, max_k = P.shape
    CIJ = np.zeros((N,N))
    for i in range(N):
        for j in range(max_k):
            if P[i,j] != -1:
                CIJ[i,int(P[i,j])] = 1
            else:
                break
    return CIJ

@njit
def degree_calc(P):
    N, max_k = P.shape
    K = 0
    for i in range(N):
        K += list(P[i,:]).count(-1)
        K -= max_k
    return -K/N

@njit    
def autoval_JG(matrix):
    """Calculates the largest eigenvalue of a matrix by the iterative JG method.

    Args:
        matrix (float64): Desired square matrix

    Returns:
        float: The eigenvalue.
    """
    iterations=80
    b = np.random.rand(len(matrix))
    for i in range(iterations):
        prox_b = (matrix @ b)/np.linalg.norm(matrix @ b)
        mi = (prox_b.reshape(1,-1)) @ (matrix @ b)
        b = prox_b
    # Converter para float por causa do numba
    for mi_temp in mi:
        final_mi = mi_temp
    return final_mi

def Adjac_file_man(M_0, alpha, network_p, b, s, mode, maker):
    desired = int(mode[1:])
    folder_adress = current_folder+f'/Data/Networks/{maker.__name__}/{M_0=}/{b=}_{s=}/{alpha=}_{network_p=}'
    
    os.makedirs(folder_adress,exist_ok=True)
    file_adress = folder_adress+f'/n={mode[1:]}.txt'

    if mode[0] == 'r':
        try:
            P = np.loadtxt(file_adress,dtype='int',delimiter='\t', comments='#')
            oi = open(file_adress, 'r')
            N = M_0 *b**s
            for l in range(N):
                oi.readline()
            oi.seek(oi.tell() + 1)
            p_crit = float(oi.readline())
            oi.close()
            net = mode[1:]
        except FileNotFoundError:
            A = maker(M_0, alpha, network_p, b, s)
            P = CIJ_to_P(A)
            p_crit = 1/autoval_JG(A)
            np.savetxt(folder_adress+f'/n={mode[1:]}.txt', P, '%i', delimiter='\t', newline='\n', footer=str(p_crit), comments='#')
            net = int(mode[1:])

    elif mode[0] == 's':
        seed_number = int(mode[1:])
        A = maker(M_0, alpha, network_p, b, s, seed=seed_number)
        P = CIJ_to_P(A)
        try:
            seed_file = open(folder_adress+'/seeds_table.txt', 'r+')
            seed_file.readline()
        except FileNotFoundError:
            seed_file = open(folder_adress+'/seeds_table.txt', 'w+')
            seed_file.write("Seed \t p_crit \n")
        for l in range(1,seed_number-1):
            seed_file.readline()
        this_line = seed_file.readline()

        if int(this_line[:len(mode[1:])]) == seed_number:
            net, p_crit = np.float64(leitura_ate_tab(seed_file.readline()))
        else:
            p_crit = 1/autoval_JG(A)
            seed_file.write(str(seed_number)+" \t"+str(p_crit)+"\n")
            net = seed_number
    
    return P, p_crit, int(net)

def mle(x, tauRange, nIterations, dataType, xmin=None, xmax=None):
    """Performs a maximum likelihood estimation for a data set ```x``` that has to be previously known to follow a power law distribution. The program is sensitive to integer (i.e., discrete) or decimal (i.e., continuous) distributions, that need to be informed through the ```dataType``` arg. 

    Args:
        x (array): The dataset. This is not the histogram, it is just the data itself.
        tauRange (list): Interval that the exponent ```tau``` may be located in.
        nIterations (int): Number of iterations to be performed. Small integers suffice. In a similar manner, the precision may be used, that is, 10**(-nIterations).
        dataType (str): Informs the dataType, 'INTS' for integers and 'CONT' for continuous data. Note: If the user prefers, there is a commented snippet that does this decision automatically. Haven't tested it, but it's easy to build a conditional to automate this value.
        xmin (float, optional): Applies a mask to remove values less than it. Defaults to None.
        xmax (float, optional): Applies a mask to remove values bigger than it. Defaults to None.

    Returns:
        tau (float): Exponent of the distribution.
        Ln (float): Maximum of the Log-likelihood function for the final exponent.
    """
    if (xmin==None):
        xmin = np.min(x)
    if (xmax==None):
        xmax = np.max(x)

    x = np.reshape(x, len(x))
    tauRange = sorted(-np.array(tauRange))

    #Determine data type
    # if np.count_nonzero(np.absolute(x - np.round(x)) > 3*(np.finfo(float).eps)) > 0:
    #     dataType = 'CONT'
    # else:
    #     dataType = 'INTS'
    #     x = np.round(x)

    # print(dataType)
    #Truncate
    z = x[(x>=xmin) & (x<=xmax)]
    unqZ = np.unique(z)
    nZ = len(z)
    nUnqZ = len(unqZ)
    allZ = np.arange(xmin,xmax+1)
    nallZ = len(allZ)

    #MLE calculation

    r = xmin / xmax

    for iIteration in range(1, nIterations+1):

        spacing = 10**(-iIteration)

        if iIteration == 1:
            taus = np.arange(min(tauRange), max(tauRange)+spacing, spacing)

        else:
            if tauIdx == 0:
                taus = np.arange(taus[0], taus[1]+spacing, spacing)
                #return (taus,0,0,0)
            elif tauIdx == len(taus):
                taus = np.arange(taus[-2], taus[-1]+spacing, spacing)#####
            else:
                taus = np.arange(taus[tauIdx-1], taus[tauIdx]+spacing, spacing)

        #return(dataType)
        nTaus = len(taus)

        if dataType=='INTS':
            #replicate arrays to equal size
            allZMat = mb.repmat(np.reshape(allZ,(nallZ,1)),1,nTaus)
            tauMat = mb.repmat(taus,nallZ,1)

            #compute the log-likelihood function
            #L = - np.log(np.sum(np.power(allZMat,-tauMat),axis=0)) - (taus/nZ) * np.sum(np.log(z))
            L = - nZ*np.log(np.sum(np.power(allZMat,-tauMat),axis=0)) - (taus) * np.sum(np.log(z))


        elif dataType=='CONT':
            #return (taus,r, nZ,z)
            L = np.log( (taus - 1) / (1 - r**(taus - 1)) )- taus * (1/nZ) * np.sum(np.log(z)) - (1 - taus) * np.log(xmin)

            if np.in1d(1,taus):
                L[taus == 1] = -np.log(np.log(1/r)) - (1/nZ) * np.sum(np.log(z))
        tauIdx=np.argmax(L)

    tau = taus[tauIdx]
    Ln = L[tauIdx]
    #return (taus,L,tau
    return (-tau, Ln)

def leitura_ate_tab(linha):
    el = []
    antigo_barrat = 0
    for i in range(1,len(linha)):
        if linha[i] == '\t':
            barrat = i
            if antigo_barrat != 0:
                el.append(linha[(antigo_barrat+1):barrat])
            else:
                el.append(linha[(antigo_barrat):barrat])
            antigo_barrat = barrat
    el.append(linha[(barrat+1):-1])
    return el



@njit
def initial_condition(N):
    actives = sorted(np.random.choice(N,N,True))
    x = np.zeros(N)
    for n in actives:
        x[n] = 1
    return x, actives

@njit
def iterator_model_A(P, x, lamb, mu, r, seed):
    # All active nodes are selected each step (could be just one)
    delta_t = 1
    activ_prob = mu/(lamb+mu)
    deactiv_prob = lamb/(lamb+mu)
    ext_stim_prob = 1 - np.exp(-r*delta_t)
    x_new = np.zeros_like(x)
    # Synchronous update
    for i in range(len(x)):
        if i==seed and x[seed]==0:
            x_new[i] = 1 if np.random.uniform(0,1) < ext_stim_prob else 0

        if x[i] == 1:
            x_new[i] = 0 if np.random.uniform(0,1) < deactiv_prob else 1
            neighb = np.array(neighbours_counter(P, i))
            for j in neighb:
                if x[j] == 0 and np.random.uniform(0,1) < activ_prob:
                    x_new[j] = 1
                    break
    return x_new

@njit
def iterator_model_B(P, x, lamb, mu, r, seed):
    ...

@njit
def copelli_iterator(P, x, states, iteration_p, r):
    """Iterates one timestep of a network, according to the K&C 2006 model.
    Updated and recommended iteration step. See ```iterator``` for a legacy model, that uses ```CIJ``` instead of ```P```.

    Args:
        P ((N,k) array): The neighborhood matrix, with the value on the i-th row indicating the index of a connection from the element to i.
        x ((N,) array): Current state of the network.
        states (int): Number of possible states. 0 and 1 are reserved for the inactive and polarized state, so ```states > 2```.
        p (float): Real probability of a connection between the nodes.

    Returns:
        (N,) array: Next state of the system.
        int: The total number of excitations.
    """
    N, k = P.shape
    
    x_new = np.copy(x)
    activated_xs = []
    for i in range(N):
        if x[i] != 0:
            x_new[i] = (x[i]+1)%states # Caminha para frente. 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 5 -> 0.
        else: # Caso o neurônio não esteja polarizado
            for j in range(k): # Começa a caminhar pela matriz de vizinhanças
                con = int(P[i,j])
                if con != -1:
                    u = np.random.uniform(0,1)
                    if x[con] == 1 and u<iteration_p: # Se o neurônio tiver uma conexão polarizado
                        x_new[i] = 1
                        activated_xs.append(i)
                        break
                else: break
            stim = np.random.uniform(0,1)
            if stim <= 1-np.exp(-r*1):
                x_new[i] = 1
                activated_xs.append(i)
    return x_new, activated_xs

@njit
def Munoz_time_evaluation(P, lamb, mu, r, seed, T, model):
    N, k = P.shape
    rho = np.zeros(T)
    
    for i in range(T):
        if i == 0:
            x = initial_condition(N)
            
        else:
            x = model(P, x, lamb, mu, r, seed)
        if np.sum(x) == 0:
            return x, rho
        rho[i] = list(x).count(1)/N
        
    return x, rho

@njit
def P_time_evaluation(P, iteration_p, states, T, r):
    N, k = P.shape
    rho = np.zeros(T)
    
    for i in range(T):
        if i == 0:
            x, activated_xs = initial_condition(N)
        else:
            x, activated_xs = copelli_iterator(P, x, states, iteration_p, r)
        if np.sum(x) == 0 and r == 0:
            return x, rho
        rho[i] = len(activated_xs)/N

    return x, rho


@njit
def one_time_cumulative_neighbours_counter(P, ind_pre, tspre):
    tspo = [] # Initializes a list for postsynaptic neurons in 'this' step
    for presynaptic in tspre: # For each presynaptic neuron in the list
        nc = neighbours_counter(P, presynaptic)
        tspo.extend(nc) # Appends the neighbours of the specific presynaptic
    for postsynaptic in tspo: # Loop for excluding the neurons that have been accounted for
        if ind_pre.count(postsynaptic) != 0: # If the postsynaptic is on the indistinct list
            tspo.remove(postsynaptic) # Removes it, because it has been accounted for
        else: # If it isn't, thet leave it there but appends it to the indistinct list
            ind_pre.append(postsynaptic)
    return tspo, ind_pre

@njit
def cumulative_neighbours_counter(P: np.ndarray, steps: int, N_max: int=1e+4):
    initial_neuron = np.random.randint(0,P.shape[0])
    ind_pre = [initial_neuron] # Indistinct (whether accounted for or not) neurons list
    tspre = [initial_neuron]
    len_list = [] # The amount of postsynaptic neurons in each step

    for i in range(steps): # Loop for each step
        print(f"step {i}...")
        prev_len = len(ind_pre)
        tspre, ind_pre = one_time_cumulative_neighbours_counter(P, ind_pre, tspre)
        new_len = len(ind_pre)
        len_list.append(new_len - prev_len)
        if i>5 and np.cumsum(np.array(len_list[:i]))[-1] >= N_max:
            return np.array(len_list)

    return np.array(len_list)

def dimension_calculator_brain(P, steps):
    len_list = cumulative_neighbours_counter(P, steps)
    cumsum = np.cumsum(len_list)
    r_vec = np.arange(len(len_list))

    r_art = r_vec[len(len_list)//10:]
    cumsum_art = cumsum[len(len_list)//10:]
    def poly_fit(x, a, b):
        return a*(x**b)

    param,param_cov = curve_fit(poly_fit, r_art, cumsum_art, p0=[2,1],maxfev=2000)
    D = param[1]
    return D, r_vec, cumsum

@njit
def neighbours_counter(P, presynaptic):
    N, max_K = P.shape
    all_postsynaptics = []
    for postsynaptic in range(N):
        for k in range(max_K):
            if P[postsynaptic,k] == presynaptic:
                all_postsynaptics.append(postsynaptic)
    return all_postsynaptics



@njit
def find_dyn_range(lista_rs, lista_Fs, F_tax_inf, F_tax_sup):
    rmin_adjust = np.zeros(2)
    rmax_adjust = np.zeros(2)
    F_rmin_adjust = np.zeros(2)
    F_rmax_adjust = np.zeros(2)
    
    for i in range(len(lista_rs)):
        if lista_Fs[i] > F_tax_inf:
            rmin_adjust[0] = lista_rs[i-1]
            rmin_adjust[1] = lista_rs[i]
            F_rmin_adjust[0] = lista_Fs[i-1]
            F_rmin_adjust[1] = lista_Fs[i]
            break
    list(lista_Fs).reverse()
    for i in range(len(lista_Fs)):
        if lista_Fs[i] > F_tax_sup:
            rmax_adjust[0] = lista_rs[i-1]
            rmax_adjust[1] = lista_rs[i]
            F_rmax_adjust[0] = lista_Fs[i-1]
            F_rmax_adjust[1] = lista_Fs[i]
            break
    
    rmin, F_rmin = find_intersection(rmin_adjust, F_rmin_adjust, F_tax_inf)
    rmax, F_rmax = find_intersection(rmax_adjust, F_rmax_adjust, F_tax_sup)
    sig_rmin = - abs((rmin_adjust[1] - rmin_adjust[0])/np.sqrt(2)) + np.sqrt((rmin-rmin_adjust[1])**2 + (rmin-rmin_adjust[0])**2)
    sig_rmax = - abs((rmax_adjust[1] - rmax_adjust[0])/np.sqrt(2)) + np.sqrt((rmax-rmax_adjust[1])**2 + (rmax-rmax_adjust[0])**2)

    Delta = 10*np.log10(rmax/rmin)
    sig_Delta = (10/np.log(10)) * np.sqrt((sig_rmax**2 / rmax**2 + sig_rmin**2 / rmin**2))
    return rmin, F_rmin, rmax, F_rmax, Delta, sig_Delta

@njit
def mediador_F_auto(P, iteration_p, systems, states, T, r):
    F_list = np.zeros(systems)
    rho_std, F_std = 0, 0
    for j in range(systems):
        x_fin, rho = P_time_evaluation(P, iteration_p, states, T, r)
        if np.all(x_fin != 0):
            rho_not_trans = rho[int(0.3*len(rho)):]
            F_list[j] = np.mean(rho_not_trans)/T
    F = np.mean(F_list)
    F_std = np.std(F_list)
    return F, F_std

@njit
def find_intersection(r_adjust, F_adjust, y_des):
    xa, xb = r_adjust[0], r_adjust[1]
    ya, yb = F_adjust[0], F_adjust[1]
    b = (yb*np.log(xa) - ya*np.log(xb))/np.log(xa/xb)
    a = (ya - yb)/np.log(xa/xb)
    x_des = np.exp((y_des-b)/a)
    return x_des, y_des

def new_connectivity_file_man(M_0, alpha, network_p, b, s, mode):
    desired = int(mode[1:])
    folder_adress = current_folder+f'/Data/Networks/{M_0=}/{b=}_{s=}/{alpha=}_{network_p=}'
    os.makedirs(folder_adress, exist_ok=True)
    how_many_networks_present = len(os.listdir(folder_adress))

    while how_many_networks_present < desired:
        A = asymetrical_finite_adjacency_maker(M_0, alpha, network_p, b, s)
        P = CIJ_to_P(A)
        p_crit = 1/autoval_JG(A)
        file_loc = folder_adress + f"/n={how_many_networks_present+1}.txt"
        np.savetxt(file_loc, P, '%i', delimiter='\t', newline='\n', footer=str(p_crit), comments='#')
        how_many_networks_present += 1
    else:
        file_loc = folder_adress + f"/n={desired}.txt"
    
    P = np.loadtxt(file_loc, dtype='int', delimiter='\t', comments='#')
    file = open(file_loc, 'r')
    for line in file:
        pass
    p_crit = float(line[1:])
    return P, p_crit, desired

def custom_plot():
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['text.color'] = '000000'

    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.linestyle'] = 'dashed'
    mpl.rcParams['grid.alpha'] = 0.3

    mpl.rcParams['axes.facecolor'] = 'white'
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    mpl.rcParams['xtick.labelcolor'] = 'black'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.labelcolor'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True

    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.shadow'] = True
    mpl.rcParams['legend.facecolor'] = 'cfcfcf'
    # mpl.rcParams['legend.fontsize'] = 11
    # mpl.rcParams['legend.labelcolor'] = '000000'

    if "Linux" in platform.platform():
        mpl.rcParams['font.family'] = 'manjari'
    elif "Windows" in platform.platform():
        mpl.rcParams['font.family'] = 'century gothic'
     
    # mpl.rcParams['font.size'] = 12
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=my_colors)

if __name__=="__main__":
    M_0, alpha, network_p, b, s = 1, 0.08, 2, 5, 4
    # P, p_crit, net = new_connectivity_file_man(M_0, alpha, network_p, b, s, 'r1')
    # A = P_to_CIJ(P)
    A = np.zeros((int(M_0*b**s),int(M_0*b**s)))
    for i in range(20):
        A += asymetrical_finite_adjacency_maker(M_0, alpha, network_p, b, s)/20
    plt.pcolor(A)
    plt.show()
