import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit
from tqdm import tqdm
import matplotlib as mpl
import itertools
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import time
# from scipy.optimize import curve_fit

from imports import *

markerstyles = ['.', '^', '1', 's', 'p', '*', 'x', 'd', '+', 'v']

import platform
if "Windows" in platform.platform():
    data_current_folder = "E:/Codes/IC/Griffiths"
else:
    data_current_folder = "/media/filipe/Novo volume/Codes/IC/Griffiths"
try:
    os.listdir(data_current_folder)
except:
    print("Storage not found. Check the external data storage.")
    exit()
else:
    print("Storage accessed succesfully.")

def no_execution_rho_brain(N, folder_loc, lamb, T, T_threshold, desired_qtde, use_all_presents=True, unique=0) -> np.ndarray:
    how_many_already_present_systems = len(os.listdir(folder_loc))
    remaining = desired_qtde - how_many_already_present_systems

    if remaining <= 0:
        if use_all_presents:
            final_systems_qtde = how_many_already_present_systems
        else:
            final_systems_qtde = desired_qtde

    if remaining > 0:
        print("Ainda faltam sistemas. Recorra às versões locais, processe os dados e armazene-os aqui.")
        print(f"Parâmetros: \n N={N} \t {T=} \t {lamb=}\n")
        exit()

    if desired_qtde == 1:
        file_loc = folder_loc + f"/i{unique}.txt"
        nonzero_rho = np.loadtxt(file_loc)
        actually_active = len(nonzero_rho)
        one_sys_rho = np.pad(nonzero_rho, (0,T-actually_active), mode='constant')
        return one_sys_rho

    rho = np.zeros(T)
    effective_systems = 0
    for j in range(final_systems_qtde):
        file_loc = folder_loc + f"/i{j}.txt"
        nonzero_rho = np.loadtxt(file_loc)
        actually_active = len(nonzero_rho)
        if actually_active > T_threshold:
            effective_systems += 1
            one_sys_rho = np.pad(nonzero_rho, (0,T-actually_active), mode='constant')
            rho += one_sys_rho
    # print(f"\nOut of {desired_qtde}, {effective_systems} lasted at least {T_threshold} timesteps.")
    if effective_systems == 0:
        print("You need to lower the threshold for survivability.")
        exit()
    rho = rho/final_systems_qtde
    return rho
        
def rho_plotter(M_0, alpha, network_p, b, s, mode, lambs_list, T, T_threshold, systems, show_dimension=False, plotstyle="Plotly"):
    if plotstyle == "Matplotlib":
        fig, ax = plt.subplots(figsize=(4.2,4.2))
        ax.set_ylabel(r"$\rho$")
        ax.set_xlabel("t")
    elif plotstyle == "Plotly":
        fig = go.Figure()
        fig.update_yaxes(title_text=r"$\rho$")
        fig.update_xaxes(title_text="t")
    N = M_0*b**s

    if show_dimension:
        ...
    
    for l, lamb in enumerate(tqdm(lambs_list, colour='green')):
        data_folder_loc = data_current_folder + f'/Data/rho/{M_0=}/{alpha=}_{network_p=}/{b=}_{s=}/{T=}/net={mode[1:]}/lamb={lamb}/'
        os.makedirs(data_folder_loc,exist_ok=True)
        rho = no_execution_rho_brain(N, data_folder_loc, lamb, T, T_threshold, systems)
        ax.loglog(rho, '-') if plotstyle == "Matplotlib" else fig.add_trace(go.Scatter(x=np.arange(T), y=rho))

    return fig

def no_execution_new_mean_activity_brain(N, folder_loc, lamb, T, T_threshold, sys_perp, use_all_presents=True):
    how_many_already_present_systems = len(os.listdir(folder_loc))
    remaining = sys_perp - how_many_already_present_systems

    if remaining <= 0:
        if use_all_presents:
            final_systems_qtde = how_many_already_present_systems
        else:
            final_systems_qtde = sys_perp

    if remaining > 0:
        print("Ainda faltam sistemas. Recorra às versões locais, processe os dados e armazene-os aqui.")
        print(f"Parâmetros: \n N={N} \t {T=} \t {lamb=}\n")
        exit()

    one_sys_F = []
    one_sys_std_F = []
    effective_systems = 0
    for j in range(final_systems_qtde):
        file_loc = folder_loc + f"/i{j}.txt"
        nonzero_rho = np.loadtxt(file_loc)
        actually_active = len(nonzero_rho)
        if actually_active > T_threshold:
            effective_systems += 1
            one_sys_rho = np.pad(nonzero_rho, (0,T-actually_active), mode='constant')
            one_sys_F.append(np.mean(one_sys_rho))
            one_sys_std_F.append(np.std(one_sys_rho))
    # print(f"\nOut of {sys_perp}, {effective_systems} lasted at least {T_threshold} timesteps.")
    if effective_systems == 0:
        print("You need to lower the threshold for survivability.")
        exit()
    
    # print(final_systems_qtde - effective_systems)
    F = np.mean(one_sys_F)
    std_F = np.std(one_sys_F)
    # std_F = np.std(np.trim_zeros(one_sys_F))
    return F, std_F

def streamlit_new_mean_activity_plotter(M_0s_list, alpha, network_p, bs_list, ss_list, mode, lambs_list, T, T_threshold, sys_perp, use_all_presents=True, plotstyle="Plotly"):
    if plotstyle=="Matplotlib":
        fig, ax = plt.subplots()
        ax.set_xlabel(r"Maior autovalor $\lambda$")
        ax.set_ylabel(r"Atividade Média $F (ms^{-1})$")
    elif plotstyle=="Plotly":
        fig = go.Figure()
        fig.update_xaxes(title_text=r"Maior autovalor $\lambda$")
        fig.update_yaxes(title_text=r"Atividade Média $F (ms^{-1})$")

    iterable = itertools.product(M_0s_list, bs_list, ss_list)
    for M_0, b, s in tqdm(iterable, total=max(len(M_0s_list), len(bs_list), len(ss_list)), colour='yellow'):
        N = M_0*b**s
        F = np.zeros((len(lambs_list), len(T_threshold)))
        std_F = np.zeros_like(F)
        for l, lamb in enumerate(tqdm(lambs_list, colour='green', leave=False, desc=f"Network with {M_0=}, {b=}, {s=}")):
            data_folder_loc = data_current_folder + f'/Data/rho/{M_0=}/{alpha=}_{network_p=}/{b=}_{s=}/{T=}/net={mode[1:]}/lamb={lamb}/'
            os.makedirs(data_folder_loc,exist_ok=True)
            for t, thr in enumerate(T_threshold):
                F[l,t], std_F[l,t] = no_execution_new_mean_activity_brain(N, data_folder_loc, lamb, T, thr, sys_perp, use_all_presents)
        for t, thr in enumerate(T_threshold): 
            if plotstyle == "Matplotlib":
                ax.errorbar(lambs_list, F[:,t], std_F[:,t], marker=str(t), linestyle='none', label=f"N={M_0*b**s}"+r" $T_{thr} = $"+ f"{thr}")
            elif plotstyle=="Plotly":
                fig.add_trace(go.Scatter(mode='markers',x=lambs_list, y=F[:,t], error_y=dict(type='data', array=std_F[:,t],visible=True),name=f"N={M_0*b**s}"+r" $T_{thr} = $"+ f"{thr}"))
        if plotstyle == "Matplotlib": ax.legend()

@njit
def count_each_state(P, x):
    # count_0, count_1, count_01, count_all_pairs = 0,0,0,0
    # for presynaptic in range(len(x)):
    #     all_postsynaptics = neighbours_counter(P, presynaptic)
    #     if x[presynaptic] == 1:
    #         for post in all_postsynaptics:
    #             if x[post] == 0:
    #                 count_0 += 1
    #                 count_all_pairs += 1
    #                 count_01 += 1
    #             elif x[post] == 1:
    #                 count_1 += 1
    #                 count_all_pairs += 1
                
    #     elif x[presynaptic] == 0:
    #         for post in all_postsynaptics:
    #             if x[post] == 0:
    #                 count_0 += 1
    #                 count_all_pairs += 1
    #             elif x[post] == 1:
    #                 count_1 += 1
    #                 count_all_pairs += 1

    count_01, count_all_pairs = 0,0
    for presynaptic in range(len(x)):
        all_postsynaptics = neighbours_counter(P, presynaptic)
        if x[presynaptic] == 1:
            for post in all_postsynaptics:
                if x[post] == 0:
                    count_all_pairs += 1
                    count_01 += 1
                elif x[post] == 1:
                    count_all_pairs += 1
                
        elif x[presynaptic] == 0:
            for post in all_postsynaptics:
                if x[post] == 0:
                    count_all_pairs += 1
                elif x[post] == 1:
                    count_all_pairs += 1
    
    count_1 = list(x).count(1)
    count_0 = list(x).count(0)
    # print(count_all_pairs)
    return count_0, count_1, count_01, count_all_pairs

@njit
def verify_independence(P, iteration_p, T):
    N, max_k = P.shape
    Prob_0 = np.zeros(T)
    Prob_1 = np.zeros(T)
    Prob_01 = np.zeros(T)
    ratio = np.zeros(T)

    for i in range(T):
        if i==0:
            x, activated_xs = initial_condition(N)
        else:
            x, activated_xs = copelli_iterator(P, x, states, iteration_p, 0)

        count_0, count_1, count_01, count_all_pairs = count_each_state(P, x)
        
        if count_1 == 0:
            Prob_01[i] = count_01/count_all_pairs
            Prob_1[i] = count_1/(count_0+count_1)
            Prob_0 = 1 - Prob_1
            ratio = Prob_01/Prob_1
            return ratio[:i], Prob_1[:i]
        
        Prob_01[i] = count_01/count_all_pairs
        Prob_1[i] = count_1/(count_0+count_1)
    Prob_0 = 1 - Prob_1
    ratio = Prob_01/Prob_1
    return ratio, Prob_0

states = 2

def rho_site_maker():
    bib = st.selectbox("Selecione a biblioteca de exibição:",["Plotly", "Matplotlib"])
    st.write("Selecione os parâmetros desejados da rede")
    network_choosing_cols = st.columns(4)

    

    folder = current_folder+"/Data/avalanches/"
    possible_M_0s = sorted(os.listdir(folder))
    with network_choosing_cols[0]:
        selected_M_0 = st.selectbox("Número de neurônios no módulo mais interno:", possible_M_0s)
        M_0 = int(selected_M_0[selected_M_0.index("=")+1:])
    folder += f"{selected_M_0}/"
    possible_alphap = sorted(os.listdir(folder))
    with network_choosing_cols[1]:
        selected_alphap = st.selectbox(r"$\alpha$ e $p$ da rede", possible_alphap)
        alpha = float(selected_alphap[selected_alphap.index("=")+1:selected_alphap.index("_")])
        if alpha%1 == 0: alpha = int(alpha)
        network_p = float(selected_alphap[selected_alphap.index("k")+4:])
    folder += f"{selected_alphap}/"
    possible_bs = sorted(os.listdir(folder))
    with network_choosing_cols[2]:
        selected_bs = st.selectbox("modularização da rede:", possible_bs)
        b = int(selected_bs[selected_bs.index("b")+2:selected_bs.index("_")])
        s = int(selected_bs[selected_bs.index("s")+2:])
    folder += f"{selected_bs}/"
    possible_T = sorted(os.listdir(folder))
    with network_choosing_cols[3]:
        selected_T = st.selectbox("Passos de tempo iterados:", possible_T)
        T = int(selected_T[selected_T.index("=")+1:])
    folder += f"{selected_T}/{n=1}"

    T_threshold = st.number_input("Informe um valor limite para sobrevivência de amostras", min_value=0, max_value=T, value=0, step=1, format="%i")
    processed_lambs = os.listdir(folder)
    for i, pl in processed_lambs:
        processed_lambs[i] = pl[:pl.index(t)-2]

    selected_lambs = st.multiselect("Selecione valores desejados para lambda. Eles são só para otimização, é possivel deselecioná-los posteriormente.", processed_lambs, on_change=lambda: time.sleep(4))
    
    selected_lamb_numbers = []
    for i, lamb in selected_lambs:
        selected_lamb_numbers = lamb[lamb.index("=")+1:]

    fig = rho_plotter(M_0, alpha, network_p, b, s, 'r1', selected_lamb_numbers, T, T_threshold, 0, show_dimension=False, plotstyle=bib)
    if bib == "Plotly":
        st.plotly_chart(fig)
    elif bib == "Matplotlib":
        st.pyplot(fig)


class exec_independence_verifier:
    ...
    # custom_plot()
    # M_0, alpha, network_p, b, s = 2, 2, 1/4, 2, 12
    # gp_P, p_crit, net = Adjac_file_man(M_0, alpha, network_p, b, s, 'r1', asymetrical_finite_adjacency_maker)
    # T = 100
    # N = M_0 * b**s
    # k = degree_calc(gp_P)
    # grif_iteration_ps = np.array([1.4,1.675,2.0])*p_crit
    # random_iteration_ps = np.array([0.9,1.0,1.1])/int(k)

    # rand_P = np.random.choice(N, (N, int(k)))
    # fig2, ax2 = plt.subplots(ncols=3,sharey=True,figsize=(18,4.2))
    # i=0
    # for iteration_p, sig in zip(grif_iteration_ps, random_iteration_ps):
    #     ratio_gp, Prob_0_gp = verify_independence(gp_P, iteration_p, T)
    #     ratio_rand, Prob_0_rand = verify_independence(rand_P, sig, T)
        
    #     if i==0:
    #         finite_pts, = ax2[i].plot(ratio_gp - Prob_0_gp, 'x', label='Finite dimensional Topology on GP', color=my_colors[3])
    #         random_pts, = ax2[i].plot(ratio_rand - Prob_0_rand, 'p', label='Random Network',color=my_colors[-1])
    #     else:
    #         finite_pts, = ax2[i].plot(ratio_gp - Prob_0_gp, 'x', color=my_colors[3])
    #         random_pts, = ax2[i].plot(ratio_rand - Prob_0_rand, 'p',color=my_colors[-1])
    #     ax2[i].set_xlabel(r"$t$")
    #     i+=1
    # finite_pts.set_label('Finite dimensional Topology on GP')
    # random_pts.set_label('Random Network')
    # ax2[0].set_ylabel(r"$\frac{P(0,1)}{P(1)} - P(0)$")
    # fig2.legend(loc='upper center',ncols=2,bbox_to_anchor=(0.5,1.01),fancybox=True,shadow=True)
    # fig2.tight_layout()
    # fig_folder_loc = current_folder + f"/Figs/independence/{M_0=}/{alpha=}_{network_p=}/{b=}_{s=}/{T=}/"
    # os.makedirs(fig_folder_loc, exist_ok=True)
    # fig2.savefig(fig_folder_loc+f'over_time.png')
    # plt.show()

class rho_several_rums:
    ...
    # M_0, alpha, network_p, b, s = 2, 2, 1/4, 2, 12
    # N = M_0*b**s
    # lambs_list = np.round(np.arange(1.5,1.76,0.01),2)
    # # lambs_list = np.array([1.6,1.64,1.7,1.76])
    # T, T_threshold = int(1e+6), int(0)
    # systems = int(30)
    # custom_plot()
    # rho_plotter(M_0, alpha, network_p, b, s, 'r1', lambs_list, T, T_threshold, systems)
    # plt.show()

class binder_cumulant_visualizer:
    ...
    # binder_cumulant_plotter([2], [2], [1/4], [2], [12], 'r1', [1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75], T=int(1e+6), T_threshold=int(30), sys_perp=int(30))

class mean_activity_visualizer:
    ...
    # M_0s_list, alpha, network_p, bs_list, ss_list = [2], 2, 1/4, [2], [12]
    # lambs_list = np.round(np.arange(1.6,1.84,0.04),3)
    # T, T_threshold = int(1e+6), [2000,5000]
    # systems = int(50)
    # custom_plot()
    # new_mean_activity_plotter(M_0s_list, alpha, network_p, bs_list, ss_list, 'r1', lambs_list, T, T_threshold, systems)
    # plt.show()

class susc_visualizer:
    ...
    # M_0, alpha, network_p, b, s = [2], 2, 1/4, 2, 12
    # lambs_list = [1.5,1.56,1.6,1.66,1.7,1.75,1.76,1.8,1.84,1.88,1.92,1.96,2.0,2.04,2.08,2.12,2.16,2.2,2.24,2.32,2.36]
    # plot_susc(M_0, alpha, network_p, b, s, 'r1', lambs_list, int(1e+6), 0)

