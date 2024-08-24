import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from numba import njit
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import time

from imports import mle, current_folder, my_colors, custom_plot

def power_fit(x, a, b):
    return a*np.float_power(x,b)
def crackling_noise_fit(x, a):
    return a*(x-1) + 1

@njit
def acumulator(counts):
    soma = np.zeros(len(counts))
    for i in range(len(counts)):
        soma[i] = np.sum(counts[i:])
    return soma

def acumulator_power_law_fitting(bins, acum, factors):
    zeta_matrix = pd.DataFrame(np.matrix.transpose(np.array([bins[:-1],acum])),columns=["Bins","Acumulada"])

    mask = zeta_matrix["Bins"] < factors[1]
    zeta_matrix_adjust = zeta_matrix[mask]
    mask2 = zeta_matrix_adjust["Bins"] > factors[0]
    zeta_matrix_adjust = zeta_matrix_adjust[mask2]
    adjust = zeta_matrix_adjust["Bins"]
    zeta_adjust = zeta_matrix_adjust["Acumulada"]
    param, param_cov = curve_fit(power_fit, adjust, zeta_adjust, p0=[3e+4, -1/2])
    
    return param, param_cov

def probability_power_law_fitting(bins, probs, factors):
    probs_matrix = pd.DataFrame(np.matrix.transpose(np.array([bins[:-1],probs])),columns=["Bins","Probability"])

    mask = probs_matrix["Probability"] > factors[0]
    probs_matrix_adjust = probs_matrix[mask]
    mask2 = probs_matrix_adjust["Probability"] < factors[1]
    probs_matrix_adjust = probs_matrix_adjust[mask2]
    adjust = probs_matrix_adjust["Bins"]
    probs_adjust = probs_matrix_adjust["Probability"]

    param, param_cov = curve_fit(power_fit, adjust, probs_adjust, p0=[3e+4, -3/2])
    return param, param_cov

def power_law_lsq(bins, counts, bins_min=None, bins_max=None, num_points=None, axes=None):
    if (bins_min==None):
        bins_min = np.min(bins)
    if (bins_max==None):
        bins_max = np.max(bins)
    
    bins = np.copy(bins[:-1])
    if num_points == None:
        mask = (bins >= bins_min) & (bins <= bins_max)
        masked_bins = bins[mask]
        masked_counts = counts[mask]
    elif num_points != None:
        masked_bins = bins[1:num_points]
        masked_counts = counts[1:num_points]
    param, param_cov = curve_fit(power_fit, masked_bins, masked_counts, p0=[3e+4, -3/2])
    if axes != None:
        axes.loglog(masked_bins, masked_counts, 'o', c='k', zorder=1)
    return param, param_cov

def prob_updated_mle_powers(data, bins, counts, tauRange, datamin=None, datamax=None, axes=None):
    if (datamin==None):
        datamin = np.min(data)
    if (datamax==None):
        datamax = np.max(data)
    mask = (data >= datamin) & (data <= datamax)
    masked_data = data[mask]
    tau, Ln = mle(data, tauRange, 9, 'INTS', xmin=datamin, xmax=datamax)
    bins = np.copy(bins[:-1])
    mask_for_bins = (bins >= datamin) & (bins <= datamax)
    def lin_power_law(x, a):
        return a*x**tau
    
    param, _ = curve_fit(lin_power_law, bins[mask_for_bins], counts[mask_for_bins])
    if axes != None:
        axes.loglog(bins[mask_for_bins], counts[mask_for_bins], 'o', c='k', zorder=1)
    return param[0], tau

def acum_updated_mle_powers(data, bins, tauRange, datamin=None, datamax=None, axes=None):
    if (datamin==None):
        datamin = np.min(data)
    if (datamax==None):
        datamax = np.max(data)
    bins_to_determine_a = 10**np.linspace(np.log10(datamin), np.log10(datamax), len(data))
    mask = (data >= datamin) & (data <= datamax)
    masked_data = data[mask]
    tau, Ln = mle(data, tauRange, 9, 'INTS', xmin=datamin, xmax=datamax)
    def lin_power_law(x, a):
        return a*x**tau
    
    param, _ = curve_fit(lin_power_law, bins_to_determine_a[mask], masked_data)
    if axes != None:
        axes.loglog(bins[:-1][mask], masked_data, 'o', c='k', zorder=1)
    return param[0], tau

def no_execution_avalanches_brain(N, T:int, qtde:int, file_loc:str, lamb:float):
    """Functions as an operating center that returns and manages files for ```qtde``` avalanches subject to a connectivity matrix ```P``` and a synaptic strength ```iteration_p```

    Args:
        P (array): Neighbourhood Matrix. All entries on line i are postsynaptic to the neuron i.
        T (int): Maximum discrete time value.
        qtde (int): Number of avalanches to be performed.
        file_loc (str): File adress location. Must be a .txt file.
        iteration_p (float): Synaptic strength between neurons.
        lamb (float): Eigenvalue of the connectivity matrix CIJ. also means iteration_p/mu, where mu is the largest eigenvalue of the adjacency matrix.
        override_qtde (bool, optional): If True, the quantity of avalanches informed will be overwritten by the quantity present on the file if the latter is greater than the former. If False, only the informed quantity will be returned. Defaults to True.

    Returns:
        pandas DataFrame: Total spikes per avalanche and duration of each one.
    """
    # lê o arquivo se ele existir e cria se não. Tb registra quantos sistemas já foram rodados.
    try:
        data_file = open(file_loc, 'r')
        data_file.seek(11)
        previously_ran_systems = int(data_file.readline())
    except FileNotFoundError:
        data_file = open(file_loc, 'w')
        data_file.write(f"Sistemas = 000000000\n")
        data_file.write("Spikes \t Durations \n")
        previously_ran_systems = 0
    remaining = qtde - previously_ran_systems
    data_file.close()
    
    # Se falta algum ainda a ser rodado
    if remaining > 0:
        print("Ainda faltam sistemas. Recorra às versões locais, processe os dados e armazene-os aqui.")
        print(f"Parâmetros: \n N={N} \t {T=} \t {qtde=} \t {lamb=}\n")
        exit()
    
    # finalmente, ler os sistemas do arquivo todo
    avalanche_table = pd.read_csv(file_loc, delimiter='\t', header=1)
    avalanche_table.columns = avalanche_table.columns.str.strip()
    return avalanche_table

def streamlit_plotly_avalanche_plotter(M_0: int, alpha: float, network_p: float, b: int, s: int, mode: str, T: int, fips, qtde: int, Nhist: int, wherefit: bool, override_qtde=True, plot_cumulative=False):
    fig_prob = make_subplots(cols=3,rows=1)
    fig_acum = make_subplots(cols=2,rows=1) if plot_cumulative else None
    fig_crack = make_subplots(cols=2,rows=1) if len(fips) > 2 else None
    N = M_0*b**s

    exponents_matrix = pd.DataFrame(np.matrix.transpose(np.array([np.zeros_like(fips)]*6)), index=[str(fip) for fip in fips], columns=["τ", "τ_std", "τ_t", "τ_t_std", "Gamma", "Gamma_std"])
    # threshold_dur, threshold_spikes = [], []
    data_folder_loc = current_folder + f"/Data/avalanches/{M_0=}/{alpha=}_{network_p=}/{b=}_{s=}/{T=}/"
    os.makedirs(data_folder_loc,exist_ok=True)
    for i, lamb in enumerate(fips, colour='green'):
        data_file_loc = data_folder_loc + f"net={mode[1:]}_lamb={lamb}.txt"
        
        avalanche_table = no_execution_avalanches_brain(N, T, qtde, data_file_loc, lamb)
        grouped = avalanche_table.groupby("Durations")
        med_spik = grouped.mean()
        
        bins_func = lambda sample: (np.logspace(np.log10(1), np.log10(max(avalanche_table[sample])), Nhist))
        x_art_dur = np.linspace(1, max(avalanche_table["Durations"]), 100)
        x_art_spik = np.linspace(1, max(avalanche_table["Spikes"]), 100)

        separation_coeff = 10**(2*i)
                                   
        counts_spik, bins_spik = np.histogram(avalanche_table["Spikes"], bins=bins_func("Spikes"), density=True)
        counts_dur, bins_dur = np.histogram(avalanche_table["Durations"], bins=bins_func("Durations"), density=True)
        fig_prob.add_trace(go.Scatter(mode="markers", x=bins_spik[:-1],y=separation_coeff*counts_spik, name=r"$\lambda = $"+ f"{round(lamb,4)}", marker=dict(color=my_colors[i%len(my_colors)])), 1,1)
        fig_prob.add_trace(go.Scatter(mode="markers", x=bins_dur[:-1],y=separation_coeff*counts_dur, name=r"$\lambda = $"+ f"{round(lamb,4)}", marker=dict(color=my_colors[i%len(my_colors)]), showlegend=False), 1,2)
        fig_prob.add_trace(go.Scatter(mode="markers", x=med_spik.index.to_list(), y=separation_coeff*med_spik.to_numpy(), name=r"$\lambda = $"+ f"{round(lamb,4)}", marker=dict(color=my_colors[i%len(my_colors)]), showlegend=False), 1,3)

        mask_for_med_fit = med_spik.index < 30
        med_spik_param, med_spik_param_cov = curve_fit(power_fit, list(med_spik.index[mask_for_med_fit]), list(med_spik["Spikes"][mask_for_med_fit]))
        exponents_matrix.loc[str(lamb), "Gamma"] = med_spik_param[1]
        exponents_matrix.loc[str(lamb), "Gamma_std"] = np.sqrt(np.diag(med_spik_param_cov))[1]
        # fig_prob.add_trace(go.Scatter(mode="lines", x=x_art_dur, y=separation_coeff*power_fit(x_art_dur, *med_spik_param), marker=dict(color='pink'), showlegend=False),1,3)

        if wherefit == 'probability_fit':
            prob_spik_param, prob_spik_param_cov = power_law_lsq(bins_spik, separation_coeff*counts_spik, bins_min=10, bins_max=10000)
            fig_prob.add_trace(go.Scatter(mode="lines", x=x_art_spik, y=power_fit(x_art_spik, *prob_spik_param), marker=dict(color='pink'), showlegend=False),1,1)

            prob_dur_param, prob_dur_param_cov = power_law_lsq(bins_dur, separation_coeff*counts_dur, bins_min=12, bins_max=700)
            fig_prob.add_trace(go.Scatter(mode="lines", x=x_art_dur, y=power_fit(x_art_dur, *prob_dur_param), marker=dict(color='pink'), showlegend=False),1,2)
            
            exponents_matrix.loc[str(lamb), "τ"] = -prob_spik_param[1]
            exponents_matrix.loc[str(lamb), "τ_std"] = np.sqrt(np.diag(prob_spik_param_cov))[1]

            exponents_matrix.loc[str(lamb), "τ_t"] = -prob_dur_param[1]
            exponents_matrix.loc[str(lamb), "τ_t_std"] = np.sqrt(np.diag(prob_dur_param_cov))[1]
        
        elif wherefit == 'probability_mle':
            spikes_a, spikes_tau = prob_updated_mle_powers(avalanche_table["Spikes"], bins_spik, separation_coeff*counts_spik, [-2,-1], datamin=10, datamax=1e+4)
            fig_prob.add_trace(go.Scatter(mode="lines", x=x_art_spik, y=spikes_a*x_art_spik**(spikes_tau), marker=dict(color='pink'), showlegend=False),1,1)

            durations_a, durations_tau = prob_updated_mle_powers(avalanche_table["Durations"], bins_dur, separation_coeff*counts_dur, [-3,-1], datamin=12, datamax=1e+3)
            fig_prob.add_trace(go.Scatter(mode="lines", x=x_art_dur, y=durations_a*x_art_dur**(durations_tau), marker=dict(color='pink'), showlegend=False),1,2)

            exponents_matrix.loc[str(lamb), "τ"] = -spikes_tau
            exponents_matrix.loc[str(lamb), "τ_std"] = 0

            exponents_matrix.loc[str(lamb), "τ_t"] = -durations_tau
            exponents_matrix.loc[str(lamb), "τ_t_std"] = 0

        if plot_cumulative:
            counts_acum_spikes, bins_acum_spikes = np.histogram(avalanche_table["Spikes"], bins=np.linspace(2, max(avalanche_table["Spikes"]), len(avalanche_table["Spikes"])))
            acum_spikes = separation_coeff*acumulator(counts_acum_spikes)
            fig_acum.add_trace(go.Scatter(mode="markers", x=bins_acum_spikes[1:-1],y=acum_spikes[1:], name=r"$\lambda = $"+ f"{round(lamb,4)}", marker=dict(color=my_colors[i%len(my_colors)])), 1,1)

            counts_acum_dur, bins_acum_dur = np.histogram(avalanche_table["Durations"], bins=np.linspace(2, max(avalanche_table["Durations"]), len(avalanche_table["Durations"])))
            acum_dur = separation_coeff*acumulator(counts_acum_dur)
            fig_acum.add_trace(go.Scatter(mode="markers", x=bins_acum_dur[1:-1],y=acum_dur[1:], name=r"$\lambda = $"+ f"{round(lamb,4)}", marker=dict(color=my_colors[i%len(my_colors)])), 1,2)

            if wherefit == 'cumulative_fit':
                # acum_spikes_param, acum_spikes_param_cov = power_law_lsq(bins_acum_spikes, acum_spikes, bins_max=2e+2, axes=ax2[0])
                acum_spikes_param, acum_spikes_param_cov = power_law_lsq(bins_acum_spikes, acum_spikes, num_points=40)
                fig_acum.add_trace(go.Scatter(mode="lines", x=x_art_spik, y=power_fit(x_art_spik, *acum_spikes_param), marker=dict(color='pink'), showlegend=False),1,1)

                # acum_dur_param, acum_dur_param_cov = power_law_lsq(bins_acum_dur, acum_dur, bins_max=8e+1, axes=ax2[1])
                acum_dur_param, acum_dur_param_cov = power_law_lsq(bins_acum_dur, acum_dur, num_points=40)
                fig_acum.add_trace(go.Scatter(mode="lines", x=x_art_dur, y=power_fit(x_art_dur, *acum_dur_param), marker=dict(color='pink'), showlegend=False),1,2)

                exponents_matrix.loc[str(lamb), "τ"] = -acum_spikes_param[1] + 1
                exponents_matrix.loc[str(lamb), "τ_std"] = np.sqrt(np.diag(acum_spikes_param_cov))[1]

                exponents_matrix.loc[str(lamb), "τ_t"] = -acum_dur_param[1] + 1
                exponents_matrix.loc[str(lamb), "τ_t_std"] = np.sqrt(np.diag(acum_dur_param_cov))[1]
            
            elif wherefit == 'cumulative_mle':
                acum_spikes_a, acum_spikes_tau = acum_updated_mle_powers(acum_spikes, bins_acum_spikes, [-1,-0.1], datamax=1e+4, datamin=1e+3)
                fig_acum.add_trace(go.Scatter(mode="lines", x=x_art_spik, y=acum_spikes_a*x_art_spik**(acum_spikes_tau), marker=dict(color='pink'), showlegend=False),1,1)

                acum_durations_a, acum_durations_tau = acum_updated_mle_powers(acum_dur, bins_acum_dur, [-1.5,-0.75], datamax=1e+4, datamin=3e+2)
                fig_acum.add_trace(go.Scatter(mode="lines", x=x_art_dur, y=acum_durations_a*x_art_dur**(acum_durations_tau), marker=dict(color='pink'), showlegend=False),1,2)

                exponents_matrix.loc[str(lamb), "τ"] = -acum_spikes_param[1]+1
                exponents_matrix.loc[str(lamb), "τ_std"] = 0

                exponents_matrix.loc[str(lamb), "τ_t"] = -acum_dur_param[1]+1
                exponents_matrix.loc[str(lamb), "τ_t_std"] = 0
                
        #  ax2[2].errorbar(exponents_matrix.loc[str(lamb), "τ"],exponents_matrix.loc[str(lamb), "τ_t"], exponents_matrix.loc[str(lamb), "τ_std"],exponents_matrix.loc[str(lamb), "τ_t_std"], '.')
        if len(fips)>2 and wherefit != 'no': fig_crack.add_trace(go.Scatter(mode="markers", x=[exponents_matrix.loc[str(lamb), "τ"]],y=[exponents_matrix.loc[str(lamb), "τ_t"]], name=r"$\lambda = $"+ f"{round(lamb,4)}", marker=dict(color=my_colors[i%len(my_colors)], size=9)),1,1)

    if len(fips) > 2 and wherefit != 'no':
        fig_crack.add_trace(go.Scatter(mode="lines", x=fips, y=(exponents_matrix["τ_t"] - 1)/(exponents_matrix["τ"] - 1), name=r"$\frac{\tau _t - 1}{\tau - 1}$", marker=dict(color='blue')),1,2)
        fig_crack.add_trace(go.Scatter(mode="lines", x=fips, y=exponents_matrix["Gamma"], name=r"$\frac{1}{\sigma \nu z}$", marker=dict(color="grey")),1,2)

        fig_crack.update_xaxes(title_text=r"$\lambda$", row=1,col=2)
        fig_crack.update_yaxes(title_text="Critical exponent", row=1,col=2)
        fig_crack.update_xaxes(title_text=r"$\tau$", row=1,col=1)
        fig_crack.update_yaxes(title_text=r"$\tau _t$", row=1,col=1)
        
    print(exponents_matrix)
    
    if wherefit != 'no' and len(fips) > 2:
        param, param_cov = curve_fit(crackling_noise_fit, exponents_matrix["τ"],exponents_matrix["τ_t"],p0=[1.28])
        x_artificial_exps = np.linspace(0.95*exponents_matrix["τ"].min(),1.05*exponents_matrix["τ"].max(),10)
        fig_crack.add_trace(go.Scatter(mode="lines", x=x_artificial_exps,y=crackling_noise_fit(x_artificial_exps,*param), marker=dict(color='blue'), showlegend=False),1,1)
        fig_crack.add_trace(go.Scatter(mode="markers", x=[1.5],y=[2], marker=dict(color='white',size=10), name="MF-DP exponent"),1,1)
        fig_crack.add_annotation(x=1.5,y=1.5,text=r"$\frac{\tau _t - 1}{\tau - 1} = $"+f"{round(param[0],3)}",row=1,col=1)

    fig_prob.update_xaxes(type='log',title_text="Tamanho S",row=1,col=1)
    fig_prob.update_xaxes(type='log',title_text="Duração D",row=1,col=2)
    fig_prob.update_xaxes(type='log',title_text="Duração D",row=1,col=3)
    fig_prob.update_yaxes(type='log',title_text="P(S)",row=1,col=1)
    fig_prob.update_yaxes(type='log',title_text="P(D)",row=1,col=2)
    fig_prob.update_yaxes(type='log',title_text=r"$\langle S \rangle$",row=1,col=3)

    if plot_cumulative:
        fig_acum.update_xaxes(type='log',title_text="Tamanho S",row=1,col=1)
        fig_acum.update_xaxes(type='log',title_text="Duração D",row=1,col=2)
        fig_acum.update_yaxes(type='log',title_text=r"$\zeta (S)$",row=1,col=1)
        fig_acum.update_yaxes(type='log',title_text=r"$\zeta (D)$",row=1,col=2)
    
    return fig_prob, fig_acum, fig_crack, exponents_matrix

def streamlit_matplotlib_avalanche_plotter(M_0: int, alpha: float, network_p: float, b: int, s: int, mode: str, T: int, fips, qtde: int, Nhist: int, wherefit: bool, override_qtde=True, plot_cumulative=False):
    custom_plot()
    fig_prob, ax_prob = plt.subplots(ncols=3, figsize=(18,4.8))
    if plot_cumulative: 
        fig_cum, ax_cum = plt.subplots(ncols=2, figsize=(12,4.8))
    else:
        fig_cum, ax_cum = None, None
    if len(fips) > 2 and wherefit != 'no':
        fig_crack, ax_crack = plt.subplots(ncols=2,figsize=(12,5))
    else:
        fig_crack, ax_crack = None, None

    N= M_0*b**s
    exponents_matrix = pd.DataFrame(np.matrix.transpose(np.array([np.zeros_like(fips)]*6)), index=[str(fip) for fip in fips], columns=["τ", "τ_std", "τ_t", "τ_t_std", "Gamma", "Gamma_std"])
    # threshold_dur, threshold_spikes = [], []
    data_folder_loc = current_folder + f"/Data/avalanches/{M_0=}/{alpha=}_{network_p=}/{b=}_{s=}/{T=}/"
    os.makedirs(data_folder_loc,exist_ok=True)
    for i, lamb in enumerate(fips, colour='green'):
        lamb = fips[i]
        data_file_loc = data_folder_loc + f"net={mode[1:]}_lamb={lamb}.txt"
        
        avalanche_table = no_execution_avalanches_brain(N, T, qtde, data_file_loc, lamb)
        grouped = avalanche_table.groupby("Durations")
        med_spik = grouped.mean()
        
        bins_func = lambda sample: (np.logspace(np.log10(1), np.log10(max(avalanche_table[sample])), Nhist))
        x_art_dur = np.linspace(1, max(avalanche_table["Durations"]), 100)
        x_art_spik = np.linspace(1, max(avalanche_table["Spikes"]), 100)

        separation_coeff = 10**(2*i)
                                   
        counts_spik, bins_spik = np.histogram(avalanche_table["Spikes"], bins=bins_func("Spikes"), density=True)
        counts_dur, bins_dur = np.histogram(avalanche_table["Durations"], bins=bins_func("Durations"), density=True)
        spik_plot, = ax_prob[0].loglog(bins_spik[:-1],separation_coeff*counts_spik, '.', label=r"$\lambda = $"+ f"{round(lamb,4)}",linestyle="None")
        dur_plot, = ax_prob[1].loglog(bins_dur[:-1],separation_coeff*counts_dur, '.', linestyle="None")
        mean_plot, = ax_prob[2].loglog(med_spik.index.to_list(),separation_coeff*med_spik, '.', linestyle="None")

        mask_for_med_fit = med_spik.index < 30
        med_spik_param, med_spik_param_cov = curve_fit(power_fit, list(med_spik.index[mask_for_med_fit]), list(med_spik["Spikes"][mask_for_med_fit]))
        exponents_matrix.loc[str(lamb), "Gamma"] = med_spik_param[1]
        exponents_matrix.loc[str(lamb), "Gamma_std"] = np.sqrt(np.diag(med_spik_param_cov))[1]
        ax_prob[2].loglog(x_art_dur, separation_coeff*power_fit(x_art_dur, *med_spik_param),color='pink')

        if wherefit == 'probability_fit':
            prob_spik_param, prob_spik_param_cov = power_law_lsq(bins_spik, separation_coeff*counts_spik, bins_min=10, bins_max=20000, axes=ax_prob[0])
            ax_prob[0].loglog(x_art_spik, power_fit(x_art_spik, *prob_spik_param), '-', color='pink')

            prob_dur_param, prob_dur_param_cov = power_law_lsq(bins_dur, separation_coeff*counts_dur, bins_min=12, bins_max=2000, axes=ax_prob[1])
            ax_prob[1].loglog(x_art_dur, power_fit(x_art_dur, *prob_dur_param), '-', color='pink')
            
            exponents_matrix.loc[str(lamb), "τ"] = -prob_spik_param[1]
            exponents_matrix.loc[str(lamb), "τ_std"] = np.sqrt(np.diag(prob_spik_param_cov))[1]

            exponents_matrix.loc[str(lamb), "τ_t"] = -prob_dur_param[1]
            exponents_matrix.loc[str(lamb), "τ_t_std"] = np.sqrt(np.diag(prob_dur_param_cov))[1]
        
        elif wherefit == 'probability_mle':
            spikes_a, spikes_tau = prob_updated_mle_powers(avalanche_table["Spikes"], bins_spik, separation_coeff*counts_spik, [-2,-1], datamin=10, datamax=1e+4, axes=ax_prob[0])
            ax_prob[0].loglog(x_art_spik, spikes_a*x_art_spik**(spikes_tau),color='pink')

            durations_a, durations_tau = prob_updated_mle_powers(avalanche_table["Durations"], bins_dur, separation_coeff*counts_dur, [-3,-1], datamin=12, datamax=1e+3,axes=ax_prob[1])
            ax_prob[1].loglog(x_art_dur, durations_a*x_art_dur**(durations_tau),color='pink')

            exponents_matrix.loc[str(lamb), "τ"] = -spikes_tau
            exponents_matrix.loc[str(lamb), "τ_std"] = 0

            exponents_matrix.loc[str(lamb), "τ_t"] = -durations_tau
            exponents_matrix.loc[str(lamb), "τ_t_std"] = 0

        if plot_cumulative:
            counts_acum_spikes, bins_acum_spikes = np.histogram(avalanche_table["Spikes"], bins=np.linspace(2, max(avalanche_table["Spikes"]), len(avalanche_table["Spikes"])))
            acum_spikes = separation_coeff*acumulator(counts_acum_spikes)
            ax_cum[0].loglog(bins_acum_spikes[1:-1], acum_spikes[1:], '.', label=r"$\lambda = $"+ f"{lamb}",linestyle="None", color=spik_plot.get_color())

            counts_acum_dur, bins_acum_dur = np.histogram(avalanche_table["Durations"], bins=np.linspace(2, max(avalanche_table["Durations"]), len(avalanche_table["Durations"])))
            acum_dur = separation_coeff*acumulator(counts_acum_dur)
            ax_cum[1].loglog(bins_acum_dur[1:-1], acum_dur[1:], '.', linestyle="None", color=spik_plot.get_color())

            if wherefit == 'cumulative_fit':
                acum_spikes_param, acum_spikes_param_cov = power_law_lsq(bins_acum_spikes, acum_spikes, num_points=500, axes=ax_cum[0])
                ax_cum[0].loglog(x_art_spik, power_fit(x_art_spik, *acum_spikes_param), '-',color='pink')

                acum_dur_param, acum_dur_param_cov = power_law_lsq(bins_acum_dur, acum_dur, num_points=500, axes=ax_cum[1])
                ax_cum[1].loglog(x_art_dur, power_fit(x_art_dur, *acum_dur_param), '-',color='pink')

                exponents_matrix.loc[str(lamb), "τ"] = -acum_spikes_param[1] + 1
                exponents_matrix.loc[str(lamb), "τ_std"] = np.sqrt(np.diag(acum_spikes_param_cov))[1]

                exponents_matrix.loc[str(lamb), "τ_t"] = -acum_dur_param[1] + 1
                exponents_matrix.loc[str(lamb), "τ_t_std"] = np.sqrt(np.diag(acum_dur_param_cov))[1]
            
            elif wherefit == 'cumulative_mle':
                acum_spikes_a, acum_spikes_tau = acum_updated_mle_powers(acum_spikes, bins_acum_spikes, [-1,-0.1], datamax=1e+4, datamin=1e+3, axes=ax_cum[0])
                ax_cum[0].loglog(x_art_spik, acum_spikes_a*x_art_spik**(acum_spikes_tau),color='pink')

                acum_durations_a, acum_durations_tau = acum_updated_mle_powers(acum_dur, bins_acum_dur, [-1.5,-0.75], datamax=1e+4, datamin=3e+2, axes=ax_cum[1])
                ax_cum[1].loglog(x_art_dur, acum_durations_a*x_art_dur**(acum_durations_tau),color='pink')

                exponents_matrix.loc[str(lamb), "τ"] = -acum_spikes_tau+1
                exponents_matrix.loc[str(lamb), "τ_std"] = 0

                exponents_matrix.loc[str(lamb), "τ_t"] = -acum_durations_tau+1
                exponents_matrix.loc[str(lamb), "τ_t_std"] = 0

            # threshold_dur.append(max(bins_acum_dur[np.where(acum_dur==min(acum_dur))]))
            # threshold_spikes.append(max(bins_acum_spikes[np.where(acum_spikes==min(acum_spikes))]))

        if i==0 and len(fips) > 1:
            ax_prob[0].text(min(avalanche_table["Spikes"]), max(separation_coeff*counts_spik[counts_spik != 0]/2e+5),
                            r"$\lambda = $"+f"{fips[0]}",
                            color=('black',0.5),
                            fontsize=12,
                            bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                            )
            ax_prob[1].text(min(avalanche_table["Durations"]), max(separation_coeff*counts_dur[counts_dur != 0]/1e+5),
                            r"$\lambda = $"+f"{fips[0]}",
                            color=('black',0.5),
                            fontsize=12,
                            bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                            )
            ax_prob[2].text(max(avalanche_table["Durations"]), min(separation_coeff*med_spik["Spikes"][med_spik["Spikes"] != 0]*1e+3),
                            r"$\lambda = $"+f"{fips[0]}",
                            color=('black',0.5),
                            fontsize=12,
                            bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                            )
            
            if plot_cumulative:
                ax_cum[0].text(min(avalanche_table["Spikes"])*10, max(separation_coeff*counts_spik[counts_spik != 0]/1e+3),
                                r"$\lambda = $"+f"{fips[0]}",
                                color=('black',0.5),
                                fontsize=12,
                                bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                                )
                ax_cum[1].text(min(avalanche_table["Durations"])*3, max(separation_coeff*counts_dur[counts_dur != 0]/5e+3),
                                r"$\lambda = $"+f"{fips[0]}",
                                color=('black',0.5),
                                fontsize=12,
                                bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                                )
        
        if i==len(fips)-1 and len(fips) > 1:
            ax_prob[0].text(min(avalanche_table["Spikes"]), max(separation_coeff*counts_spik[counts_spik != 0]*10),
                            r"$\lambda = $"+f"{fips[-1]}",
                            color=('black',0.5),
                            fontsize=12,
                            bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                            )
            ax_prob[1].text(min(avalanche_table["Durations"]), max(separation_coeff*counts_dur[counts_dur != 0]*10),
                            r"$\lambda = $"+f"{fips[-1]}",
                            color=('black',0.5),
                            fontsize=12,
                            bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                            )
            ax_prob[2].text(max(avalanche_table["Durations"]), min(separation_coeff*med_spik["Spikes"][med_spik["Spikes"] != 0]*1e+10),
                            r"$\lambda = $"+f"{fips[-1]}",
                            color=('black',0.5),
                            fontsize=12,
                            bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                            )

            if plot_cumulative:
                ax_cum[0].text(min(avalanche_table["Spikes"])*100, max(separation_coeff*counts_spik[counts_spik != 0]*5),
                               r"$\lambda = $"+f"{fips[-1]}",
                               color=('black',0.5),
                               fontsize=12,
                               bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                               )
                ax_cum[1].text(min(avalanche_table["Durations"])*100, max(separation_coeff*counts_dur[counts_dur != 0]*5),
                               r"$\lambda = $"+f"{fips[-1]}",
                               color=('black',0.5),
                               fontsize=12,
                               bbox=dict(facecolor=(spik_plot.get_color(),0.3), edgecolor=spik_plot.get_color(),boxstyle="Round4, pad=0.2")
                               )
                
        if len(fips)>2 and wherefit != 'no': ax_crack[0].scatter(exponents_matrix.loc[str(lamb), "τ"],exponents_matrix.loc[str(lamb), "τ_t"],linestyle="None",label=r"$\lambda = $"+ f"{round(lamb,4)}", color=spik_plot.get_color())

        if len(fips)>2 and wherefit != 'no': ax_crack[0].errorbar(exponents_matrix.loc[str(lamb), "τ"],exponents_matrix.loc[str(lamb), "τ_t"], exponents_matrix.loc[str(lamb), "τ_t_std"],exponents_matrix.loc[str(lamb), "τ_std"], linestyle="None",label=r"$\lambda = $"+ f"{round(lamb,4)}", color=spik_plot.get_color())

    if len(fips) > 2 and wherefit != 'no':
        ax_crack[1].plot(fips, (exponents_matrix["τ_t"] - 1)/(exponents_matrix["τ"] - 1), label=r"$\frac{\tau _t - 1}{\tau - 1}$")
        ax_crack[1].plot(fips, exponents_matrix["Gamma"], label=r"$\frac{1}{\sigma \nu z}$")

        ax_crack[1].set_xlabel(r"$\lambda$")
        ax_crack[1].set_ylabel("Critical exponent")
        ax_crack[0].set_xlabel(r"$\tau$")
        ax_crack[0].set_ylabel(r"$\tau _t$")

        ax_crack[1].legend()
    print(exponents_matrix)
    
    if wherefit != 'no' and len(fips) > 2:
        param, param_cov = curve_fit(crackling_noise_fit, exponents_matrix["τ"],exponents_matrix["τ_t"],p0=[1.28])
        x_artificial_exps = np.linspace(0.95*exponents_matrix["τ"].min(),1.05*exponents_matrix["τ"].max(),10)
        ax_crack[0].plot(x_artificial_exps,crackling_noise_fit(x_artificial_exps,*param),color='blue')
        # ax2[2].plot(x_artificial_exps,crackling_noise_fit(x_artificial_exps,1.28),color='brown')
        mfdp, = ax_crack[0].plot([1.5], [2], 'o',c='black')
        ax_crack[0].text(1.5,1.5,r"$\frac{\tau _t - 1}{\tau - 1} = $"+f"{round(param[0],3)}",color='blue')
        ax_crack[0].legend()

    ax_prob[0].set_xlabel("Tamanho S")
    ax_prob[1].set_xlabel("Duração D")
    ax_prob[2].set_xlabel("Duração D")
    ax_prob[0].set_ylabel("P(S)")
    ax_prob[1].set_ylabel("P(D)")
    ax_prob[2].set_ylabel(r"$\langle S \rangle$")

    if plot_cumulative:
        ax_cum[0].set_xlabel("S")
        ax_cum[0].set_ylabel(r"$\zeta (S)$")
        ax_cum[1].set_xlabel("D")
        ax_cum[1].set_ylabel(r"$\zeta (D)$")

    fig_prob.tight_layout()
    if plot_cumulative: fig_cum.tight_layout()

    return fig_prob, fig_cum, fig_crack, exponents_matrix

def colorize_multiselect_options(colors: list[str]) -> None:
    rules = ""
    n_colors = len(colors)

    for i, color in enumerate(colors):
        rules += f""".stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"]:nth-child({n_colors}n+{i+1}){{background-color: {color};}}"""

    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)

def avalanche_site_maker():
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
    folder += f"{selected_T}/"

    possible_netslambs = sorted(os.listdir(folder))
    lamb_texts = []
    lamb_values = []
    for netslambs in possible_netslambs:
        lamb_texts.append(netslambs[netslambs.index("_")+1:netslambs.index("x")-2])

    fit_selecting_cols = st.columns(spec=[0.2,0.8])
    with fit_selecting_cols[0]:
        plot_cumulative = st.checkbox("Plot cumulative data",value=False,help="It makes the program take additional time to render, and it may crash your browser.")
        ignore_fits = st.checkbox("Ignorar Ajuste de dados")
    with fit_selecting_cols[1]:
        adjust_options = ["MLE with probability", "MLE with acumulated", "Fit with probability", "Fit with acumulated"]
        adjust = st.radio("Selecione o tipo de ajuste desejado:", adjust_options,horizontal=True,disabled=ignore_fits)
        if ignore_fits:
            adj= 'no'
        elif adjust == "MLE with probability":
            adj = "probability_mle"
        elif adjust == "MLE with acumulated" and plot_cumulative:
            adj = "cumulative_mle"
        elif adjust == "Fit with probability":
            adj = "probability_fit"
        elif adjust == "Fit with acumulated" and plot_cumulative:
            adj = "cumulative_fit"


    selected_lamb_texts = st.multiselect("Selecione valores desejados para lambda.", lamb_texts,[], on_change=lambda: time.sleep(4))
    colorize_multiselect_options(my_colors)
    for i, lamb in enumerate(selected_lamb_texts):
        j = lamb_texts.index(lamb)
        # selected_files.append(files[j])
        lamb_values.append(lamb[lamb.index("=")+1:])
            
    lamb_values = np.sort(np.array(lamb_values,dtype=np.float64))

    if bib == "Plotly":
        fig_prob, fig_acum, fig_crack, exponents_matrix = streamlit_plotly_avalanche_plotter(M_0, alpha, network_p, b, s, 'r1', T, lamb_values, 0, 100, adj, plot_cumulative=plot_cumulative)
    elif bib == "Matplotlib":
        fig_prob, fig_acum, fig_crack, exponents_matrix = streamlit_matplotlib_avalanche_plotter(M_0, alpha, network_p, b, s, 'r1', T, lamb_values, 0, 100, adj, plot_cumulative=plot_cumulative)

    if bib == "Plotly":
        st.plotly_chart(fig_prob)
        if plot_cumulative: st.plotly_chart(fig_acum)
        if len(lamb_values)>2: st.plotly_chart(fig_crack)
    elif bib == "Matplotlib":
        st.pyplot(fig_prob)
        if plot_cumulative: st.pyplot(fig_acum)
        if len(lamb_values)>2: st.pyplot(fig_crack)
    
    st.write(exponents_matrix)

st.set_page_config(layout='wide')
bib = st.selectbox("Selecione a biblioteca de exibição:",["Plotly", "Matplotlib"])
avalanche_site_maker()
