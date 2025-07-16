# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:28:11 2024

GLM analysis, without segmenting data by trial onset time. 


@author: Jong Hoon Lee
"""

# import packages 

import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy import ndimage
from scipy import stats
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet,ElasticNetCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA, SparsePCA
import seaborn as sns
from os.path import join as pjoin
from numba import jit, cuda

from scipy.stats import zscore

# %% File name and directory

# change fname for filename
# fname = 'CaData_all_all_session_v2_corrected.mat'
fname = 'CaData_all_session_v3_corrected.mat'

fdir = 'D:\Python\Data'


# %% Helper functions for loading and selecting data
np.seterr(divide = 'ignore') 
def load_matfile(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
    return D_ppc 

def load_matfile_Ca(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_CaData']
    return D_ppc 

def find_good_data_Ca(t_period):
    D_ppc = load_matfile_Ca()
    good_list = []
    t_period = t_period+prestim

    for n in range(np.size(D_ppc,0)):
        N_trial = np.size(D_ppc[n,2],0)
    
        ttr = D_ppc[n,4][0][0]

    # re-formatting Ca traces
    
        Y = np.zeros((N_trial,int(t_period/window)))
        for tr in range(N_trial):
            Y[tr,:] = D_ppc[n,0][0,int(D_ppc[n,2][tr,0])-1 
                                 - int(prestim/window): int(D_ppc[n,2][tr,0])
                                 + int(t_period/window)-1 - int(prestim/window)]
        if np.mean(Y[:200,:]) > 0.5 :
            good_list = np.concatenate((good_list,[n]))
        elif np.mean(Y[200:ttr+26,:]) > 0.5:
            good_list = np.concatenate((good_list,[n]))
        elif np.mean(Y[ttr+26:N_trial,:])> 0.5 :
            good_list = np.concatenate((good_list,[n]))
    
    return good_list


# %% import data helper functions

def import_data_w_Ca(D_ppc,n,window,c_ind):    
    # For each neuron, get Y, neural data and X task variables.  
    # Stim onset is defined by stim onset time
    # Reward is defined by first lick during reward presentation
    # Lick onset, offset are defined by lick times
    # Hit vs FA are defined by trial conditions
    
    

    N_trial = np.size(D_ppc[n,2],0)
    
    # extracting licks, the same way
    

    
    
    
    ### Extract Ca trace ###
    Yraw = {}
    Yraw = D_ppc[n,0]
    time_point = D_ppc[n,3]*1e3
    t = 0
    time_ind = []
    while t*window < np.max(time_point):
        time_ind = np.concatenate((time_ind,np.argwhere(time_point[0,:]>t*window)[0]))
        t += 1
    
    
    Y = np.zeros((1,len(time_ind)-1))   
    for t in np.arange(len(time_ind)-1):
        Y[0,t] = np.mean(Yraw[0,int(time_ind[t]):int(time_ind[t+1])])
        
    
    ### Extract Lick ### 
    L_all = np.zeros((1,len(time_ind)-1))
    L_all_onset = np.zeros((1,len(time_ind)-1))
    L_all_offset = np.zeros((1,len(time_ind)-1))
    # Rt = np.zeros((1,len(time_ind)-1))
    Ln = np.array(D_ppc[n,1])
    InterL = Ln[1:,:]- Ln[:-1,:]
    lick_onset= np.where(InterL[:,0]>2)[0] # lick bout boundary =2
    lick_onset = lick_onset+1
    lick_offset = lick_onset-1
    
    for l in np.floor(D_ppc[n,1]*(1e3/window)): 
        L_all[0,int(l[0])-1] = 1 
            
    for l in np.floor(Ln[lick_onset,0]*(1e3/window)):
        L_all_onset[0,int(l)-1] = 1
    
    for l in np.floor(Ln[lick_offset,0]*(1e3/window)):
        L_all_offset[0,int(l)-1] = 1 
     
    L_all_mid = L_all-L_all_onset-L_all_offset
    L_all_mid[0,L_all_mid[0,:]<0] = 0
    # for l in np.floor(D_ppc[n,6][:,0]*(1e3/window)):
    #     Rt[0,int(l)-1] = 1     
    
    ### Extract Lick End ###
    

    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)

    X = D_ppc[n,2][:,2:6] # task variables
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    
    
    ### Create variables ###
    ED1 = 5 # 500ms pre, 1second post lag
    ED2 = 10
    stim_dur = 5 # 500ms stim duration
    delay = 10 # 1 second delay
    r_dur = 5 # 2 second reward duration (was 10) 
    ED3 = 30 # 4 seconds post reward lag
    ED4 = 70
    ED5 = 50
    ED_hist1 = 50 # 4 seconds pre-stim next trial
    ED_hist2 = 15 # 1.5 seconds post-stim next trial
    h_dur = 5
    
    X3_Lick_onset = np.zeros((ED1+ED2+1,np.size(Y,1)))
    X3_Lick_offset = np.zeros_like(X3_Lick_onset)
    X3_Lick_mid = np.zeros_like(X3_Lick_onset)
    X3_Lick_onset[0,:] = L_all_onset
    X3_Lick_offset[0,:] = L_all_offset
    

    for lag in np.arange(ED1):
        X3_Lick_onset[lag+1,:-lag-1] = L_all_onset[0,lag+1:]
        X3_Lick_offset[lag+1,:-lag-1] = L_all_offset[0,lag+1:]
    
    for lag in np.arange(ED2):
        X3_Lick_onset[lag+ED1+1,lag+1:] = L_all_onset[0,:-lag-1]
        X3_Lick_offset[lag+ED1+1,lag+1:] = L_all_offset[0,:-lag-1]
        
      
    X3_go = np.zeros((ED2+1,np.size(Y,1)))
    X3_ng = np.zeros_like(X3_go)
    
    for st in stim_onset[(Xstim == 1)]:
        X3_go[0,st:st+stim_dur] = 1
    
    for st in stim_onset[(Xstim ==0)]:
        X3_ng[0,st:st+stim_dur] = 1
        
    for lag in np.arange(ED2):
        X3_go[lag+1,lag+1:] = X3_go[0,:-lag-1]
        X3_ng[lag+1,lag+1:] = X3_ng[0,:-lag-1]
    
    
    X3_Hit = np.zeros((ED4+1,np.size(Y,1)))
    X3_FA = np.zeros_like(X3_Hit)
    X3_Miss = np.zeros_like(X3_Hit)
    X3_CR = np.zeros_like(X3_Hit)
    
    # X3_Miss = np.zeros((ED3+1,np.size(Y,1)))
    # X3_CR = np.zeros_like(X3_Miss)
    # for r in Rt[(XHit == 1)]:
    #     if r != 0:
    #         X3_Hit[0,r:r+r_dur] = 1
    
    # for r in Rt[(XFA == 1)]:
    #     if r != 0:
    #         X3_FA[0,r:r+r_dur] = 1
    # for r in Rt[(XHit == 1)]:
    #     if r != 0:
    #         r = r-10
    #         X3_Hit[0,r:r+r_dur] = 1
    
       
    for st in stim_onset[(XHit==1)]:
        X3_Hit[0,st:st+stim_dur] = 1
    
    for st in stim_onset[(XFA==1)]:
        X3_FA[0,st:st+stim_dur] = 1
            
            
    for st in stim_onset[(Xmiss ==1)]:        
        X3_Miss[0,st:st+stim_dur] = 1

    for st in stim_onset[(XCR ==1)]:        
        X3_CR[0,st:st+stim_dur] = 1 
        
        


    for lag in np.arange(ED4):
        X3_Miss[lag+1,lag+1:] = X3_Miss[0,:-lag-1]
        X3_CR[lag+1,lag+1:] = X3_CR[0,:-lag-1]
        X3_Hit[lag+1,lag+1:] = X3_Hit[0,:-lag-1]
        X3_FA[lag+1,lag+1:] = X3_FA[0,:-lag-1]

    # X3_Hit_hist = np.zeros((ED_hist1+ED_hist2+1,np.size(Y,1)))
    # X3_FA_hist = np.zeros((ED_hist1+ED_hist2+1,np.size(Y,1)))
    
    # X3_Hit_hist = np.zeros((ED_hist1+1,np.size(Y,1)))
    # X3_FA_hist = np.zeros((ED_hist1++1,np.size(Y,1)))
    # XHit_prev = np.concatenate(([False], XHit[0:-1]), axis = 0)
    # XFA_prev = np.concatenate(([False], XFA[0:-1]), axis = 0)
    
    
    # X3_Hit_hist[0,30:] = X3_Hit[0,:-30]
    # X3_FA_hist[0,30:] = X3_FA[0,:-30]
    
    # for lag in np.arange(ED_hist1):
    #     X3_Hit_hist[lag+1,lag+1:] = X3_Hit_hist[0,:-lag-1]
    #     X3_FA_hist[lag+1,lag+1:] = X3_FA_hist[0,:-lag-1]
    # for st in stim_onset[(XHit_prev ==1)]:
    #     X3_Hit_hist[0,st:st+h_dur] = 1
    # for st in stim_onset[(XFA_prev ==1)]:
    #     X3_FA_hist[0,st:st+h_dur] = 1 
    
    
    # for lag in np.arange(ED_hist1):
    #     X3_Hit_hist[lag+1,:-lag-1] = X3_Hit_hist[0,lag+1:]
    #     X3_FA_hist[lag+1,:-lag-1] = X3_FA_hist[0,lag+1:]
    
    # for lag in np.arange(ED_hist2):
    #     X3_Hit_hist[lag+ED_hist1+1,lag+1:] = X3_Hit_hist[0,:-lag-1]
    #     X3_FA_hist[lag+ED_hist1+1,lag+1:] = X3_FA_hist[0,:-lag-1]
    
    # for r in Rt[()]
    # gather X variables
    
    
    X3 = {}
    X3[0] = X3_Lick_onset
    X3[1] = X3_Lick_offset
    X3[2] = X3_go
    X3[3] = X3_ng
    # X3[4] = X3_Hit[0:int(ED4/2),:]
    # X3[5] = X3_FA[0:int(ED4/2),:]
    # X3[6] = X3_Miss[0:int(ED4/2)-10,:]
    # X3[7] = X3_CR[0:int(ED4/2)-10,:]
    # X3[8] = X3_Hit[int(ED4/2):,:]
    # X3[9] = X3_FA[int(ED4/2):,:]
    # X3[10] = X3_Miss[int(ED4/2)-10:,:]
    # X3[11] = X3_CR[int(ED4/2)-10:,:]
    X3[4] = X3_Hit[0:10,:]
    X3[5] = X3_FA[0:10,:]
    X3[6] = X3_Miss[0:10,:]
    X3[7] = X3_CR[0:10,:]
    X3[8] = X3_Hit[10:,:]
    X3[9] = X3_FA[10:,:]
    X3[10] = X3_Miss[10:,:]
    X3[11] = X3_CR[10:,:]
    
    
    # X3[8] = X3_Hit_hist
    # X3[9] = X3_FA_hist
    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = r_onset[1:150]
        Xstim = Xstim[1:150]
    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        Xstim = Xstim[200:D_ppc[n,4][0][0]+26]

    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        r_onset = Rt-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]
        Xstim = Xstim[D_ppc[n,4][0][0]+26:398]

        

    Y = Y[:,c1:c2]
    L_all = L_all[:,c1:c2]
    L_all_onset = L_all_onset[:,c1:c2]
    L_all_offset = L_all_offset[:,c1:c2]
    
    Y0 = np.mean(Y[:,:stim_onset[1]-50])
    
    for ind in np.arange(len(X3)):
        X3[ind] = X3[ind][:,c1:c2]         



    return X3,Y, L_all,L_all_onset, L_all_offset, stim_onset2, r_onset, Xstim,Y0

# %% glm_per_neuron function code
def glm_per_neuron(n,c_ind, fig_on,good_alpha):
    X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim, Y0 = import_data_w_Ca(D_ppc,n,window,c_ind)
    
    Y2 = Y # -Y0
    X4 = np.ones((1,np.size(Y)))
    alpha_list =  [1e-3,5*1e-3,1e-2,5*1e-2]
    # alpha_list =  [5*1e-2]
    l_ratio = 0.9
    alpha_score = np.zeros((len(alpha_list),1))
    aa = 0
    
    
    ### Iteration to find good alpha
    ### for run time purposes, run this code once with input good_alpha = 5*1e-2
    
    # # good_alpha = 5*1e-2
    # for alpha in alpha_list:
    #     reg = ElasticNet(alpha = alpha, l1_ratio = 0.9, fit_intercept=True) #Using a linear regression model with Ridge regression regulator set with alpha = 1
    #     ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)
    
    #     ### initial run, compare each TV ###
    #     Nvar= len(X)
    #     compare_score = {}
    #     int_alpha = 10
    #     for a in np.arange(Nvar+1):
            
    #         # X4 = np.ones_like(Y)*int_alpha
    #         X4 = np.zeros_like(Y)
    
    #         if a < Nvar:
    #             X4 = np.concatenate((X4,X[a]),axis = 0)
    
    #         cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
    #                                     return_estimator = True, 
    #                                     scoring = 'r2') 
    #         compare_score[a] = cv_results['test_score']
        
    #     f = np.zeros((1,Nvar))
    #     p = np.zeros((1,Nvar))
    #     score_mean = np.zeros((1,Nvar))
    #     for it in np.arange(Nvar):
    #         f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[Nvar],alternative = 'less')
    #         score_mean[0,it] = np.median(compare_score[it])
    
    #     max_it = np.argmax(score_mean)
    #     init_score = compare_score[max_it]
    #     init_compare_score = compare_score
        
    #     if p[0,max_it] > 0.05:
    #             max_it = []
    #     else:  
    #             # === stepwise forward regression ===
    #             step = 0
    #             while step < Nvar:
    #                 max_ind = {}
    #                 compare_score2 = {}
    #                 f = np.zeros((1,Nvar))
    #                 p = np.zeros((1,Nvar))
    #                 score_mean = np.zeros((1,Nvar))
    #                 for it in np.arange(Nvar):
    #                     m_ind = np.unique(np.append(max_it,it))
    #                     # X4 = np.ones_like(Y)*int_alpha
    #                     X4 = np.zeros_like(Y)
    #                     for a in m_ind:
    #                         X4 = np.concatenate((X4,X[a]),axis = 0)
    
                        
    #                     cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
    #                                                 return_estimator = True, 
    #                                                 scoring = 'r2') 
    #                     compare_score2[it] = cv_results['test_score']
        
    #                     f[0,it], p[0,it] = stats.ks_2samp(compare_score2[it],init_score,alternative = 'less')
    #                     score_mean[0,it] = np.mean(compare_score2[it])
    #                 max_ind = np.argmax(score_mean)
    #                 if p[0,max_ind] > 0.05 or p[0,max_ind] == 0:
    #                     step = Nvar
    #                 else:
    #                     max_it = np.unique(np.append(max_it,max_ind))
    #                     init_score = compare_score2[max_ind]
    #                     step += 1
                        
    #             # === forward regression end ===
    #             # === running regression with max_it ===
    #             X3 = X
    #             if np.size(max_it) == 1:
    #                 max_it = [max_it,max_it]
    #             for tv_ind in [4,5,6,7]:
    #                 if (tv_ind+4 in max_it) and (tv_ind in max_it):
    #                     max_it = np.append(max_it, [tv_ind])
    #                         # X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
    #                     X3[tv_ind] = np.concatenate((X3[tv_ind],X3[tv_ind+4]),0);
    #                 elif (tv_ind+4 in max_it) and(tv_ind not in max_it):
    #                         max_it = np.append(max_it, [tv_ind])
    #                         X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
    #                 elif (tv_ind+4 not in max_it) and(tv_ind in max_it):
    #                         # max_it = np.append(max_it, [tv_ind])
    #                         X3[tv_ind] = np.concatenate((X3[tv_ind],np.zeros_like(X3[tv_ind+4])),0);
                            
                            
    #             max_it = np.setdiff1d(max_it,[8,9,10,11])
    #             max_it = np.unique(max_it)
                
    
    
    #             # X4 = np.ones_like(Y)*int_alpha
    #             X4 = np.zeros_like(Y)
    #             # if np.size(max_it) == 1:
    #             #     X4 = np.concatenate((X4,X3[max_it]),axis = 0)
    #             # else:
    #             for a in max_it:
    #                     X4 = np.concatenate((X4,X3[a]),axis = 0)
                
    #             cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
    #                                         return_estimator = True, 
    #                                         scoring = 'r2') 
    #             score3 = cv_results['test_score']
                
    #             theta = [] 
    #             inter = []
    #             yhat = []
    #             for model in cv_results['estimator']:
    #                 theta = np.concatenate([theta,model.coef_]) 
    #                 # inter = np.concatenate([inter, model.intercept_])
    #                 yhat =np.concatenate([yhat, model.predict(X4.T)])
                    
    #             theta = np.reshape(theta,(k,-1)).T
    #             yhat = np.reshape(yhat,(k,-1)).T
    #             yhat = yhat + Y0
    #     alpha_score[aa,0] = np.mean(score3)
    #     aa += 1
    # good_alpha = alpha_list[np.argmax(alpha_score)]
    ### iteration to find best alpha, end

    reg = ElasticNet(alpha = good_alpha, l1_ratio = l_ratio, fit_intercept=True) #Using a linear regression model with Ridge regression regulator set with alpha = 1
    ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)

    ### initial run, compare each TV ###
    Nvar= len(X)
    compare_score = {}
    int_alpha = 10
    for a in np.arange(Nvar+1):
        
        # X4 = np.ones_like(Y)*int_alpha
        X4 = np.zeros_like(Y)

        if a < Nvar:
            X4 = np.concatenate((X4,X[a]),axis = 0)

        cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'r2') 
        compare_score[a] = cv_results['test_score']
    
    f = np.zeros((1,Nvar))
    p = np.zeros((1,Nvar))
    score_mean = np.zeros((1,Nvar))
    for it in np.arange(Nvar):
        f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[Nvar],alternative = 'less')
        score_mean[0,it] = np.median(compare_score[it])

    max_it = np.argmax(score_mean)
    init_score = compare_score[max_it]
    init_compare_score = compare_score
    
    if p[0,max_it] > 0.05:
            max_it = []
    else:  
            # === stepwise forward regression ===
            step = 0
            while step < Nvar:
                max_ind = {}
                compare_score2 = {}
                f = np.zeros((1,Nvar))
                p = np.zeros((1,Nvar))
                score_mean = np.zeros((1,Nvar))
                for it in np.arange(Nvar):
                    m_ind = np.unique(np.append(max_it,it))
                    # X4 = np.ones_like(Y)*int_alpha
                    X4 = np.zeros_like(Y)
                    for a in m_ind:
                        X4 = np.concatenate((X4,X[a]),axis = 0)

                    
                    cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                                return_estimator = True, 
                                                scoring = 'r2') 
                    compare_score2[it] = cv_results['test_score']
    
                    f[0,it], p[0,it] = stats.ks_2samp(compare_score2[it],init_score,alternative = 'less')
                    score_mean[0,it] = np.mean(compare_score2[it])
                max_ind = np.argmax(score_mean)
                if p[0,max_ind] > 0.05 or p[0,max_ind] == 0:
                    step = Nvar
                else:
                    max_it = np.unique(np.append(max_it,max_ind))
                    init_score = compare_score2[max_ind]
                    step += 1
                    
            # === forward regression end ===
            
        # === running regression with max_it ===
            X3 = X
            if np.size(max_it) == 1:
                max_it = [max_it,max_it]
            for tv_ind in [4,5,6,7]:
                if (tv_ind+4 in max_it) and (tv_ind in max_it):
                    max_it = np.append(max_it, [tv_ind])
                            # X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
                    X3[tv_ind] = np.concatenate((X3[tv_ind],X3[tv_ind+4]),0);
                elif (tv_ind+4 in max_it) and(tv_ind not in max_it):
                    max_it = np.append(max_it, [tv_ind])
                    X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
                elif (tv_ind+4 not in max_it) and(tv_ind in max_it):
                            # max_it = np.append(max_it, [tv_ind])
                    X3[tv_ind] = np.concatenate((X3[tv_ind],np.zeros_like(X3[tv_ind+4])),0);
                            
                            
            max_it = np.setdiff1d(max_it,[8,9,10,11])
            max_it = np.unique(max_it)
                
    
    
                # X4 = np.ones_like(Y)*int_alpha
            X4 = np.zeros_like(Y)
                # if np.size(max_it) == 1:
                #     X4 = np.concatenate((X4,X3[max_it]),axis = 0)
                # else:
            for a in max_it:
                X4 = np.concatenate((X4,X3[a]),axis = 0)
            
            cv_results = cross_validate(reg, X4.T, Y2.T, cv = ss , 
                                        return_estimator = True, 
                                        scoring = 'r2') 
            score3 = cv_results['test_score']
            
            theta = [] 
            inter = []
            yhat = []
            for model in cv_results['estimator']:
                theta = np.concatenate([theta,model.coef_]) 
                # inter = np.concatenate([inter, model.intercept_])
                yhat =np.concatenate([yhat, model.predict(X4.T)])
                
            theta = np.reshape(theta,(k,-1)).T
            yhat = np.reshape(yhat,(k,-1)).T
            yhat = yhat + Y0
    
    
    
    
    TT = {}
    lg = 1
    
    if np.size(max_it) ==1:
        a = np.empty( shape=(0, 0) )
        max_it = np.append(a, [int(max_it)]).astype(int)
    try:
        for t in max_it:
            TT[t] = X3[t].T@theta[lg:lg+np.size(X3[t],0),:]  
            lg = lg+np.size(X3[t],0)
    except: 
        TT[max_it] = X3[max_it].T@theta[lg:lg+np.size(X3[max_it],0),:]  
    
    # for tv_ind in [4,5,6,7]:
    #     if tv_ind in max_it:
    #         if tv_ind+4 in max_it:
    #             TT[tv_ind] = TT[tv_ind] + TT[tv_ind+4]
    #     elif tv_ind+4 in max_it:
    #         TT[tv_ind] = TT[tv_ind+4]
    #         max_it = np.append(max_it, [tv_ind])
    
    max_it = np.setdiff1d(max_it,[8,9,10,11])
    
    


    
    
    # === figure === 
    if fig_on ==1:
        prestim = 20
        t_period = 60
        
        y = np.zeros((t_period+prestim,np.size(stim_onset)))
        yh = np.zeros((t_period+prestim,np.size(stim_onset)))
        l = np.zeros((t_period+prestim,np.size(stim_onset))) 
        weight = {}
        for a in np.arange(Nvar):
           weight[a] = np.zeros((t_period+prestim,np.size(stim_onset))) 
        
        yhat_mean = np.mean(yhat,1).T - Y0    
        for st in np.arange(np.size(stim_onset)):
            y[:,st] = Y[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
            yh[:,st] = yhat_mean[stim_onset[st]-prestim: stim_onset[st]+t_period]
            l[:,st] = Lm[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
            # if np.size(max_it)>1:
            for t in max_it:
                weight[t][:,st] = np.mean(TT[t][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)
            # else:
            #     weight[max_it][:,st] = np.mean(TT[max_it][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)
            
    
        
        xaxis = np.arange(t_period+prestim)- prestim
        xaxis = xaxis*1e-1
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
        cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:red','tab:red','black','green']
        clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_2','FA_2','Miss_2','CR_2']
        lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']
        
        ### plot y and y hat
        stim_ind1 = (Xstim ==1)
        stim_ind2 = (Xstim ==0)
    
        y1 = ndimage.gaussian_filter(np.mean(y[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(y[:,stim_ind2],1),0)
        s1 = np.std(y[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(y[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax1.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        
        y1 = ndimage.gaussian_filter(np.mean(yh[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(yh[:,stim_ind2],1),0)
        s1 = np.std(yh[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(yh[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax1.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y1-s1, y1+s1, color = "gray",alpha = 0.5)
        ax1.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax1.fill_between(xaxis,y2-s2, y2+s2, color = "gray",alpha = 0.5)
        
        
        
        ### plot model weights
        for a in np.arange(Nvar):
            y1 = ndimage.gaussian_filter(np.mean(weight[a],1),0)
            s1 = np.std(weight[a],1)/np.sqrt(np.size(weight[a],1))
            
            
            ax2.plot(xaxis,ndimage.gaussian_filter(y1,1),linewidth = 2.0,
                     color = cmap[a], label = clabels[a], linestyle = lstyles[a])
            ax2.fill_between(xaxis,(ndimage.gaussian_filter(y1,1) - s1),
                            (ndimage.gaussian_filter(y1,1)+ s1), color=cmap[a], alpha = 0.2)
        
        ### plot lick rate ###
        
        y1 = ndimage.gaussian_filter(np.mean(l[:,stim_ind1],1),0)
        y2 = ndimage.gaussian_filter(np.mean(l[:,stim_ind2],1),0)
        s1 = np.std(l[:,stim_ind1],1)/np.sqrt(np.sum(stim_ind1))
        s2 = np.std(l[:,stim_ind2],1)/np.sqrt(np.sum(stim_ind2))
        
        ax3.plot(xaxis,y1,linewidth = 2.0, color = "blue",label = '10kHz',linestyle = 'solid')
        ax3.fill_between(xaxis,y1-s1, y1+s1, color = "blue",alpha = 0.5)
        ax3.plot(xaxis,y2,linewidth = 2.0, color = "red",label = '5kHz',linestyle = 'solid')
        ax3.fill_between(xaxis,y2-s2, y2+s2, color = "red",alpha = 0.5)
        
        
        ax2.set_title('unit_'+str(n+1))
        sc = np.mean(score3)
        ax4.set_title(f'{sc:.2f}')
        plt.show()
    
    
    return Xstim, L_on, inter, TT, Y, max_it, score3, init_compare_score, yhat,X4, theta, good_alpha



    
    
    
# %% Initialize
"""     
Each column of X contains the following information:
    0 : contingency 
    1 : lick vs no lick
    2 : correct vs wrong
    3 : stim 1 vs stim 2
    4 : if exists, would be correct history (previous correct ) 

"""



t_period = 4000
prestim = 4000

window = 100 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 20 # number of cv
ca = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [3]



if ca ==0:
    D_ppc = load_matfile()
    # good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    good_list = find_good_data_Ca(t_period)
    
list_a = np.load('list_alpha.npy',allow_pickle= True).item()
    
    
# %% Run GLM

Data = {}



for c_ind in c_list:
    # t = 0 
    good_list2 = [];
    for n in good_list: #np.arange(np.size(D_ppc,0)):
        
        n = int(n)
        if D_ppc[n,4][0][0] > 0:
            try:
                Xstim, L_on, inter, TT, Y, max_it, score3, init_score, yhat, X4, theta,g_alpha  = glm_per_neuron(n,c_ind,1,list_a[n])
                Data[n,c_ind-1] = {"X":Xstim,"coef" : TT, "score" : score3, 'Y' : Y,'init_score' : init_score,
                                    "intercept" : inter,'L' : L_on,"yhat" : yhat, "X4" : X4, "theta": theta,"alpha":g_alpha}
                good_list2 = np.concatenate((good_list2,[n]))
                print(n)
                
            except KeyboardInterrupt:
                
                break
            except:
            
                print("Break, no fit") 
# np.save('RTnew_0620.npy', Data,allow_pickle= True)  

# load Data from saved file   
Data = np.load('RTnew_1211.npy',allow_pickle= True).item()
test = list(Data.keys())
c_ind = c_list[0]
good_list2 = np.zeros((len(test)))
for n in np.arange(len(test)):
    good_list2[n] =test[n][0]
    
    

# list_alpha = {}
# for n in good_list2:
#     list_alpha[n] = Data[n,c_ind-1]["alpha"]
    
# np.save('list_alpha.npy',list_alpha,allow_pickle= True)  

# list_a = np.load('list_alpha.npy',allow_pickle= True).item()
# load data end

# test = Data2.item()




# test1 =test(7,2)
# %% plot R score 


d_list3 = good_list2 <= 195
# d_list3 = good_list <= 118

d_list = good_list2 > 195
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
Sstyles = ['tab:orange','none','tab:blue','none','tab:red','none','black','green','tab:purple','none']


def make_RS(d_list):
    good_list_sep = good_list2[d_list]
    ax_sz = len(cmap)-2
    I = np.zeros((np.size(good_list_sep),ax_sz+1))
       
        
    for n in np.arange(np.size(good_list_sep,0)):
        nn = int(good_list_sep[n])
        # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
        Model_score = Data[nn, c_ind-1]["score"]
        init_score =  Data[nn, c_ind-1]["init_score"]
        for a in np.arange(ax_sz):
            I[n,a] = np.mean(init_score[a])
        I[n,ax_sz] = np.mean(Model_score)*1.
        
    
    fig, axes = plt.subplots(1,1, figsize = (10,8))
        # Rsstat = {}
    for a in np.arange(ax_sz):
        Rs = I[:,a]
        Rs = Rs[Rs>0.01]
        axes.scatter(np.ones_like(Rs)*(a+(c_ind+1)*-0.3),Rs,facecolors=Sstyles[a], edgecolors= cmap[a])
        axes.scatter([(a+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')    
            # Rs = Rs/(Rmax+0.03)
            # Rsstat[c_ind,f] = Rs
    
                # axes.boxplot(Rs,positions= [f+(c_ind+1)*-0.3])
    Rs = I[:,ax_sz]
    Rs = Rs[Rs>0.02]
    axes.scatter(np.ones_like(Rs)*(ax_sz+(c_ind+1)*-0.3),Rs,c = 'k',)
    axes.scatter([(ax_sz+(c_ind+1)*-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')
    axes.set_ylim([0,0.75])
    axes.set_xlim([-1,len(cmap)])
    
    
    return I

I1 = make_RS(d_list3)
I2 = make_RS(d_list)
I1 = I1[:,8]*1.
I2 = I2[:,8]*1.
bins = np.arange(0,0.8, 0.01)
fig, axs= plt.subplots(1,1,figsize = (5,5))
axs.hist(I1[I1>0.01],bins = bins,density=True, histtype="step",
                               cumulative=True)
axs.hist(I2[I2>0.01],bins = bins,density=True, histtype="step",
                               cumulative=True)
axs.set_xlim([-.05,0.7])
# np.sum([I2>0.01])
# np.size(np.max(I1,1))
good_listRu = []       
for n in np.arange(np.size(good_list2,0)):
    nn = int(good_list2[n])
    Model_score = Data[nn, c_ind-1]["score"]
    if np.mean(Model_score) > 0.02:
        good_listRu = np.concatenate((good_listRu,[nn]))


# Data3 = np.load('RTnew_1121.npy',allow_pickle= True).item()
# listN = np.load('listBN.npy',allow_pickle= True)
# listN = listN.astype('int64')
# for n in listN:
#     # if n not in good_listRu:
#         Data[n,c_ind-1] = Data3[n,c_ind-1]
        

# del Data3

# removing units with less than 2TTR
# good_listRu = np.setdiff1d(good_listRu,[49,44,87,59,63])

good_listRu = np.setdiff1d(good_listRu,np.arange(117,130))


# %% histogram of TV encoding
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
edgec = cmap
# edgec = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']


d_list2 = d_list
def TV_hist(d_list2):
    good_list_sep = good_list2[d_list2]
    TV = np.empty([1,1])
    for n in np.arange(np.size(good_list_sep,0)):
            nn = int(good_list_sep[n])
            Model_coef = Data[nn, c_ind-1]["coef"]
            max_it = [key for key in Model_coef]
            TV = np.append(TV, max_it)
    
    TV = TV[1:]
    ax_sz = 12
    B = np.zeros((1,ax_sz))
    for f in np.arange(ax_sz):
        B[0,f] = np.sum(TV == f)
        
    B = B/np.sum(d_list2)
    fig, axes = plt.subplots(1,1, figsize = (15,5))
    axes.grid(visible=True,axis = 'y')
    axes.bar(np.arange(ax_sz)*3,B[0,:], color = "white", edgecolor = edgec, alpha = 1, width = 0.5, linewidth = 2,hatch = '/')
    # axes.bar(np.arange(ax_sz)*3,B[0,:], color = cmap, edgecolor = edgec, alpha = 1, width = 0.5, linewidth = 2,hatch = '/')
    axes.set_ylim([0,0.8])
            
TV_hist(d_list2)
        
# %% 


def extract_onset_times(D_ppc,n):
    window = 100
    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)
    
    
    if c_ind == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[1:150]
        r_onset = Rt-c1
        r_onset = r_onset[1:150]

    
    elif c_ind == 3:
        c1 = stim_onset[200]-100
        c2 = stim_onset[D_ppc[n,4][0][0]+26] 
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[200:D_ppc[n,4][0][0]+26]
        r_onset = Rt-c1
        r_onset = r_onset[200:D_ppc[n,4][0][0]+26]



    elif c_ind == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        r_onset = Rt-c1
        # r_onset = r_onset[200:D_ppc[n,4][0][0]+26]
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]

    return stim_onset2, r_onset

    
for n in np.arange(np.size(good_list2,0)):
    nn = int(good_list2[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    stim_onset,r_onset = extract_onset_times(D_ppc,nn)
    Data[nn,c_ind-1]["stim_onset"] = stim_onset
    Data[nn,c_ind-1]["r_onset"] = r_onset



# %% plot lick to lick 
L_data  ={};
for n in good_listRu:
    print(n)
    n = int(n)
    X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim, Y0 = import_data_w_Ca(D_ppc,n,window,c_ind)
    L_data[n,0] = np.argwhere(L_on[0])
    L_data[n,1] = np.argwhere(L_off[0])



# %% Normalized population average of task variable weights
# c_ind = 1
d_list = good_listRu > 195
# d_list3 = good_list <= 179
d_list3 = good_listRu <= 195

# Lic = np.where(good_listRu <180)`
# Lic = Lic[0][-1]
good_list_sep = good_listRu[:]

# good_list_sep = good_list2[d_list]

weight_thresh = 5*1e-2


# fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green']
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']

# clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']
ax_sz = len(cmap)

w_length = [16,16,11,11,71,71,71,71] # window lengths for GLM 
# w_length = [16,16,11,11,60,60,60,60] # window lengths for GLM 


Convdata = {}
Convdata2 = {}
pre = 10 # 10 40 
post = 70 # 50 20
xaxis = np.arange(post+pre)- pre
xaxis = xaxis*1e-1

for a in np.arange(ax_sz):
    Convdata[a] = np.zeros((np.size(good_list_sep),pre+post))
    Convdata2[a] = np.zeros(((np.size(good_list_sep),pre+post,w_length[a])))


good_list5 = [];
for n in np.arange(np.size(good_list_sep,0)):
    nn = int(good_list_sep[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    Model_coef = Data[nn, c_ind-1]["coef"]
    theta = Data[nn,c_ind-1]["theta"]
    X4 = Data[nn,c_ind-1]["X4"]
    Model_score = Data[nn, c_ind-1]["score"]
    stim_onset2 =  Data[nn, c_ind-1]["stim_onset"]
    stim_onset =  Data[nn, c_ind-1]["stim_onset"]
    # stim_onset= L_data[nn,1].T
    # stim_onset = stim_onset[0,1:-1]
    [T,p] = stats.ttest_1samp(np.abs(theta),0.05,axis = 1, alternative = 'greater') # set weight threshold here
    p = p<0.05
    Model_weight = np.multiply([np.mean(theta,1)*p],X4.T).T
    maxC2 = np.max([np.abs(np.mean(theta,1))*p])+0.2
    
    
    weight = {}
    weight2 = {}
    max_it = [key for key in Model_coef]
    # max_it = np.setdiff1d(max_it,[8,9,10,11])
    for a in max_it:
        weight[a] = np.zeros((pre+post,np.size(stim_onset))) 
        weight2[a] = np.zeros((pre+post,np.size(stim_onset),w_length[a]) )  
                              
    for st in np.arange(np.size(stim_onset)-1):
        lag = 1
        for a in max_it:
            if stim_onset[st] <0:
                stim_onset[st] = stim_onset2[st]+15                
            
            weight[a][:,st] = np.mean(Model_coef[a][stim_onset[st]-pre: stim_onset[st]+post,:],1)
            weight2[a][:,st,:] = Model_weight[lag:lag+w_length[a],stim_onset[st]-pre: stim_onset[st]+post].T
                
            lag = lag+w_length[a]-1
        
    maxC = np.zeros((1,ax_sz))
    # [T,p] = stats.ttest_1samp(Model_score,0.01,alternative = 'greater')
    # if p < 0.05:
    #     good_list5 = np.concatenate((good_list5,[nn]))
    for a in max_it:    
            maxC[0,a] = np.max(np.abs(np.mean(weight[a],1)))+0.2
    for a in max_it:
            Convdata[a][n,:] = np.mean(weight[a],1) /np.max(maxC)
            # Convdata[a][n,:] = np.mean(weight[a],1) /(np.max(np.abs(np.mean(weight[a],1)))+0.2)
            nz_ind = np.abs(np.sum(weight2[a],(0,2)))>0
            if np.sum(nz_ind) > 0:
                if a == 6:
                    # Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/(2*maxC2)
                    Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/2
                else:                       
                    # Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/maxC2
                    Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)
        
    
# fig, axes = plt.subplots(1,1,figsize = (10,8))         
# axes.plot(xaxis,np.mean(weight[7],1))
# axes.plot(xaxis,np.mean(np.sum(weight2[a][:,nz_ind,:],1),1))
     
# fig, axes = plt.subplots(1,1,figsize = (10,8))       
# for a in [3]:
#     list0 = (np.mean(Convdata[a],1) != 0)
#     # error = np.std(Convdata[a],0)/np.sqrt(np.size(good_list_sep))
#     # y = ndimage.gaussian_filter(np.mean(Convdata[a],0),2)   
#     W = ndimage.uniform_filter(np.sum(Convdata2[a][list0,:,:],2),[0,5], mode = "mirror")

#     # error = np.std(Convdata[a][list0,:],0)/np.sqrt(np.sum(list0))
#     # y = ndimage.gaussian_filter(np.mean(Convdata[a][list0,:],0),2)
#     y = np.abs(np.mean(W,0))
#     error = np.std(W,0)/np.sqrt(np.sqrt(np.sum(list0)))
#     axes.plot(xaxis,y,c = cmap[a],linestyle = lstyles[a])
#     axes.fill_between(xaxis,y-error,y+error,facecolor = cmap[a],alpha = 0.3)
#     axes.set_ylim([-0.01,0.25])
#     axes.set_ylim([-0.1,1])




# %% 240620 extract theta only

d_list = good_list2 > 195
# d_list3 = good_list <= 179
d_list3 = good_list2 <= 195

# Lic = np.where(good_listRu <180)
# Lic = Lic[0][-1]
# good_list_sep = good_listRu[:]

good_list_sep = good_list2[:]

Model_weight = {}
for a in np.arange(ax_sz):
    Model_weight[a] = np.zeros((np.size(good_list_sep),80))
    


for n in np.arange(np.size(good_list_sep,0)):
    nn = int(good_list_sep[n])
    # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
    Model_coef = Data[nn, c_ind-1]["coef"]
    theta = Data[nn,c_ind-1]["theta"]
    # X4 = Data[nn,c_ind-1]["X4"]
    # Model_score = Data[nn, c_ind-1]["score"]
    stim_onset2 =  Data[nn, c_ind-1]["stim_onset"]
    stim_onset =  Data[nn, c_ind-1]["stim_onset"]
    [T,p] = stats.ttest_1samp(np.abs(theta),0.05,axis = 1, alternative = 'greater') # set weight threshold here
    p = p<0.05
    maxC2 = np.max([np.abs(np.mean(theta,1))*p])+0.1
    max_it = [key for key in Model_coef]
    lag = 1
    T = np.mean(theta,1)*p/maxC2
    for a in max_it:           
            Model_weight[a][n,10:w_length[a]+10] = T[lag: w_length[a]+lag]
            lag = lag+w_length[a]
            
fig, axes = plt.subplots(1,1,figsize = (10,8))       
for a in np.arange(ax_sz):
    error = np.std(Model_weight[a],0)/np.sqrt(np.size(good_list_sep))
    y = ndimage.gaussian_filter(np.mean(Model_weight[a],0),1)
    # y = np.abs(y)
    axes.plot(xaxis,y,c = cmap[a],linestyle = lstyles[a])
    axes.fill_between(xaxis,y-error,y+error,facecolor = cmap[a],alpha = 0.3)
    axes.set_ylim([-0.01,0.10])      
    


# %% plot example neuron v2

# n = 321
pre = 20
post = 60
# f1 = 3
# f2 = 2
t1 = 10
t2 = 5
# t1 = 93
# t2 = 2
def plt_ex_neurons2(n,pre,post,t1,t2):
    fig, ax = plt.subplots(1,1, figsize= (30,5))
    stim_onset = Data[n, c_ind-1]["stim_onset"]
    Model_coef = Data[n, c_ind-1]["coef"]
    xaxis= np.arange(stim_onset[t1]-pre,stim_onset[t1+t2]+post)
    ax.plot(xaxis,Data[n,c_ind-1]["Y"][0,xaxis], color = "black", linewidth = 2)
    ax.vlines(stim_onset[t1:t1+t2+1],0,15)
    h = Data[n,c_ind-1]["yhat"][xaxis,:]
    yh = (np.mean(h,1))*1
    sdh =np.std(h,1)
    ax.plot(xaxis,yh, color = "black", linestyle = "dashed",linewidth = 2)
    ax.fill_between(xaxis,yh-sdh,yh+sdh, alpha = 0.5)
    max_it = [key for key in Model_coef]
    cmap = ['black','grey','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
    
    xaxis= np.arange(stim_onset[t1]-pre,stim_onset[t1+t2]+post)
    
    for f in max_it:
        if f < 8:
            C = Model_coef[f][xaxis,:]
            yc = np.mean(C,1)*1
            sdc = np.std(C,1)*1
            ax.plot(xaxis,yc, color = cmap[f],linewidth = 3)
            ax.fill_between(xaxis,yc-sdc,yc+sdc,color = cmap[f], alpha = 0.5)
    
    # plt.savefig("example neuron "+str(n)+ ".svg")

plt_ex_neurons2(320,pre,post,30,5)
plt_ex_neurons2(20,pre,post,0,5)

# mY2 = np.zeros((20,70))
# t3 = 0
# for st in stim_onset[t1:t2+20]:
#     mY2[t3,:] = np.mean(Yh[st-10:st+60,:],1)
#     t3 += 1
# fig, ax = plt.subplots(1,1,figsize= (5,5))
# ax.plot(np.mean(mY,0))
# ax.plot(np.mean(mY2,0))



# stim_onset = Data[n, c_ind-1]["stim_onset"]
# Model_coef = Data[n, c_ind-1]["coef"]

# y = np.zeros((t_period+prestim,np.size(stim_onset)))
# yh = np.zeros((t_period+prestim,np.size(stim_onset)))
# l = np.zeros((t_period+prestim,np.size(stim_onset))) 
# prestim = 20
# t_period = 60
# weight = {}

# max_it = [key for key in Model_coef]
# for t in max_it:
#             weight[t] = np.zeros((t_period+prestim,np.size(stim_onset))) 
        
# yhat_mean = np.mean(yhat,1).T -np.mean(Y[0:1000])
# for st in np.arange(np.size(stim_onset)):
#     y[:,st] = Y[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
#     yh[:,st] = yhat_mean[stim_onset[st]-prestim: stim_onset[st]+t_period]
#     # l[:,st] = Lm[0,stim_onset[st]-prestim: stim_onset[st]+t_period]
# # if np.size(max_it)>1:
#     for t in max_it:
#         weight[t][:,st] = np.mean(TT[t][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)

# # yhat_mean = np.mean(yhat,1).T -np.mean(Y[0:1000])

# # y = Y[0,stim_onset[t1]-prestim:stim_onset[t1+t2]+t_period]
# # yh = yhat_mean[stim_onset[t1]-prestim:stim_onset[t1+t2]+t_period]
# fig, ax = plt.subplots(1,1,figsize= (5,5))
# ax.plot(y[:,3])
# ax.plot(yh[:,3])
# %% plot example neuron

# n = 126
def plt_ex_neurons(nn,c1,c2):   
    # nn = int(good_list_sep[n])
    pre = 20
    post = 60
    prestim = pre*window
    t_period = post*window
    x_axis = np.arange(1, prestim+t_period, window)
    x_axis = (x_axis-prestim)*1e-3
    
    y_lens = np.arange(int((t_period+prestim)/window))    
    Y = Data[nn,c_ind-1]["Y"][:,:]
    # X = Data[nn,c_ind-1]["X"][:,:]
    Yhat = Data[nn,c_ind-1]["yhat"]
    intercept = Data[nn,c_ind-1]["intercept"]
    Model_coef = Data[nn, c_ind-1]["coef"]
    Model_score = Data[nn, c_ind-1]["score"]
    stim_onset2 =  Data[nn, c_ind-1]["stim_onset"]
    stim_onset =  Data[nn, c_ind-1]["stim_onset"]
    
    X4 = D_ppc[nn,2][:,2:6] # task variables

    X4 = X4[c1:c2,:]
    # ymean = np.ones((len(y_lens),np.size(X4,0))).T*intercept

    ### divide into Go, No Hit, Miss, FA and CR
    X5 = np.column_stack([(X4[:,0] == 1) * (X4[:,1] == 1), # Hit
                           (X4[:,0] == 1) * (X4[:,1] == 0), # MIss
                           (X4[:,0] == 0) * (X4[:,1] == 1), # FA
                           (X4[:,0] == 0) * (X4[:,1] == 0)]) # CR
    X5 = np.column_stack([X5,np.concatenate(([False],X5[:-1,0]),0),np.concatenate(([False],X5[:-1,1]),0)])
    X5 =X5[1:150,:]
    # stim_onset =np.round(D_ppc[nn,3][0,D_ppc[nn,2][:,0]]*(1e3/window))
    # stim_onset = stim_onset.astype(int)
    # stim_onset = stim_onset[0:149]
    # pooling model weights
    weight = {}
    Y2 = np.zeros((pre+post,np.size(stim_onset))) 
    Yhat2 = np.zeros((pre+post,np.size(stim_onset))) 
    max_it = [key for key in Model_coef]
    for a in np.arange(10):
        weight[a] = np.zeros((pre+post,np.size(stim_onset))) 
    for st in np.arange(np.size(stim_onset)):
        if stim_onset[st] <0:
            stim_onset[st] = stim_onset2[st]+15
        Y2[:,st] = Y[0,stim_onset[st]-pre: stim_onset[st]+post]
        Yhat2[:,st] = np.mean(Yhat[stim_onset[st]-pre: stim_onset[st]+post,:],1)
        for t in max_it:
            if stim_onset[st] <0:
                stim_onset[st] = stim_onset2[st]+15
                
            weight[t][:,st] = np.mean(Model_coef[t][stim_onset[st]-pre: stim_onset[st]+post,:],1)
    

    Y2 = Y2.T
    Yhat2 = Yhat2.T
    fig, axes = plt.subplots(6,6,figsize = (30,20))
    for ind1 in np.arange(6):
        axes[0,ind1].plot(x_axis,np.mean(Y2[X5[:,ind1],:],0),color = "blue",linewidth = 3.0)
        axes[0,ind1].plot(x_axis,np.mean(Yhat2[X5[:,ind1],:],0),color = "red",linewidth = 3.0)
        axes[0,ind1].xaxis.set_tick_params(labelsize=20)
        axes[0,ind1].yaxis.set_tick_params(labelsize=20)
        # axes[0,ind1].set_ylim([np.min(Yhat2)*1.5,np.max(Yhat2)*1.5])
    
    # pltmin = np.min([np.mean(yh[0],0),np.mean(yh[1],0),np.mean(yh[2],0),np.mean(yh[3],0)])
    # pltmax = np.max([np.mean(yh[0],0),np.mean(yh[1],0),np.mean(yh[2],0),np.mean(yh[3],0)])
    
    cmap = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']
    # clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR']
    lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted']
    for ind2 in np.arange(1,6):
        if ind2 ==1:
            yh1 = weight[2]
            yh2 = weight[3]
            color1 = cmap[2]
            color2 = cmap[3]
            ls1 = lstyles[2]
            ls2 = lstyles[3]
            pltmin = np.min([np.min(yh1),np.min(yh2)])
            pltmax = np.max([np.max(yh1),np.max(yh2)])
        elif ind2 ==2:
            yh1 = weight[4]
            yh2 = weight[7]
            color1 = cmap[4]
            color2 = cmap[7]
            ls1 = lstyles[4]
            ls2 = lstyles[7]
            if np.min([np.min(yh1),np.min(yh2)]) < pltmin:
                pltmin = np.min([np.min(yh1),np.min(yh2)])
            if np.max([np.max(yh1),np.max(yh2)]) > pltmax:
                pltmax = np.max([np.max(yh1),np.max(yh2)])
            
        elif ind2 ==3:
            yh1 = weight[5]
            yh2 = weight[6]
            color1 = cmap[5]
            color2 = cmap[6]
            ls1 = lstyles[5]
            ls2 = lstyles[6]
            if np.min([np.min(yh1),np.min(yh2)]) < pltmin:
                pltmin = np.min([np.min(yh1),np.min(yh2)])
            if np.max([np.max(yh1),np.max(yh2)]) > pltmax:
                pltmax = np.max([np.max(yh1),np.max(yh2)])
                
        elif ind2 ==4:
            yh1 = weight[0]
            yh2 = weight[1]    
            color1 = cmap[0]
            color2 = cmap[1]
            ls1 = lstyles[0]
            ls2 = lstyles[1]
            if np.min([np.min(yh1),np.min(yh2)]) < pltmin:
                pltmin = np.min([np.min(yh1),np.min(yh2)])
            if np.max([np.max(yh1),np.max(yh2)]) > pltmax:
                pltmax = np.max([np.max(yh1),np.max(yh2)])
                
        elif ind2 ==5:
            yh1 = weight[8]
            yh2 = weight[9]   
            color1 = cmap[8]
            color2 = cmap[9]
            ls1 = lstyles[8]
            ls2 = lstyles[9]
            if np.min([np.min(yh1),np.min(yh2)]) < pltmin:
                pltmin = np.min([np.min(yh1),np.min(yh2)])
            if np.max([np.max(yh1),np.max(yh2)]) > pltmax:
                pltmax = np.max([np.max(yh1),np.max(yh2)])
            
        
        for ind1 in np.arange(6):
            # yhat_tv = 
            # axes[ind2,ind1].plot(x_axis,np.mean(yh[ind2-2][X5[:,ind1],:],0)+np.mean(ymean,0),color = cmap[ind2-2])
            axes[ind2,ind1].plot(x_axis,np.mean(yh1[:,X5[:,ind1]],1),color = color1,linewidth = 3.0, linestyle = ls1)
            axes[ind2,ind1].plot(x_axis,np.mean(yh2[:,X5[:,ind1]],1),color = color2,linewidth = 3.0, linestyle = ls2)
            axes[ind2,ind1].set_ylim([pltmin*1.2,pltmax*1.2])
            axes[ind2,ind1].yaxis.set_tick_params(labelsize=20)
            axes[ind2,ind1].xaxis.set_tick_params(labelsize=20)
    return yh1, yh2   
    
    
nn = 126
c1 = 0
c2 = 200  
if c_ind == 3:
    c1 = 200
    c2 = D_ppc[nn,4][0][0]+26                            
yh1, yh2 = plt_ex_neurons(nn,c1,c2)  
# %% plotting weights by peak order

# Convdata2 = Model_weight
listOv = {}

f = 4
W5 = {}
W5AC= {}
W5IC = {}
max_peak3 ={}
tv_number = {}
b_count = {}
ax_sz = 8
w_length1 = [16,16,11,11,30,30,20,20]
w_length2 = [0,0,0,0,31,31,21,21]
for ind in [0,1]: # 0 is PPCIC, 1 is PPCAC
    b_count[ind] = np.zeros((2,ax_sz))

    for f in np.arange(ax_sz):
        W5[ind,f] = {}

for ind in [0,1]:
    for f in  np.arange(ax_sz):
        list0 = (np.mean(Convdata[f],1) != 0)
        # list0 = (np.sum((Convdata[f],())
        # Lg = len(good_list2)
        Lg = len(good_listRu)
        Lic = np.where(good_listRu <194)
        Lic = Lic[0][-1]
        if ind == 0:
            list0[Lic:Lg] = False # PPCIC
        elif ind == 1:           
            list0[0:Lic+1] = False # PPCAC
        
        # list0ind = np.arange(Lg)
        # list0ind = list0ind[list0]
        list0ind = good_listRu[list0]
        # W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
        W = ndimage.uniform_filter(np.sum(Convdata2[f][list0,:,:],2),[0,0], mode = "mirror")
        # W = ndimage.uniform_filter(Model_weight[f][list0,:],[0,2], mode = "mirror")
        # W = ndimage.uniform_filter(np.sum(Convdata2[f][list0,:,0:w_length1[f]],2),[0,0], mode = "mirror")
        # W = ndimage.uniform_filter(np.sum(Convdata2[f][list0,:,w_length2[f]:],2),[0,0], mode = "mirror")
        # W = W/int(np.floor(w_length[f]/10)+1)
        # W = W/int(np.floor(w_length[f]/20)+1)
        # W = W/int(np.floor(w_length1[f]/10))
        max_peak = np.argmax(np.abs(W),1)
        max_ind = max_peak.argsort()
        
        list1 = []
        list2 = []
        list3 = []
        
        SD = np.std(W[:,:])
        for m in np.arange(np.size(W,0)):
            n = max_ind[m]
            SD = np.std(W[n,:])
            # if SD< 0.05:
            #     SD = 0.05
            if max_peak[n]> 0:    
                if W[n,max_peak[n]] >2*SD:
                    list1.append(m)
                    list3.append(m)
                elif W[n,max_peak[n]] <-2*SD:
                    list2.append(m)
                    list3.append(m)
                
        max_ind1 = max_ind[list1]  
        max_ind2 = max_ind[list2]     
        max_ind3 = max_ind[list3]
        max_peak3[ind,f] = max_peak[list3]
        
        listOv[ind,f] = list0ind[list3]
        
        W1 = W[max_ind1]
        W2 = W[max_ind2]    
        W4 = np.abs(W[max_ind3])
        s ='+' + str(np.size(W1,0)) +  '-' + str(np.size(W2,0))
        print(s)
        b_count[ind][0,f] = np.size(W1,0)
        b_count[ind][1,f] = np.size(W2,0)
        W3 = np.concatenate((W1,W2), axis = 0)
        tv_number[ind,f] = [np.size(W1,0),np.size(W2,0)]
        # W3[:,0:8] = 0
        # W3[:,68:] = 0
        W5[ind,f][0] = W1
        W5[ind,f][1] = W2
        if f in [2]:
            clim = [-1, 1]
            fig, axes = plt.subplots(1,1,figsize = (10,10))
            im1 = axes.imshow(W3[:,:],clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
            # im2 = axes[1].imshow(W2, aspect = "auto", interpolation = "None")
            # axes.set_xlim([,40])
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(im1, cax=cbar_ax)
        if ind == 0:
            W5IC[f] = W3
        elif ind == 1:           
            W5AC[f] = W3
        # W4IC = W4
    
# print(np.size(np.intersect1d(listOv[0],listOv[3])))
# np.save('PPC_Hist.npy',listOv,allow_pickle = True)
# np.argmax()

# list0n = good_listRu[list0]
# ind= 1
# np.sum((max_peak3[ind,4] > 54) * (max_peak3[ind,4] < 80 ))

# np.sum((max_peak3[ind,4] > 1) * (max_peak3[ind,4] < 40 ))
# np.sum((max_peak3[ind,4] > 80))




# %% calculate nb of neurons encodin

# create list of all neurons that encode at least 1 variable
ind = 0
ax_sz = 8
test = [];
for ind in [0,1]:
    for f in np.arange(ax_sz):
        test = np.concatenate((test,listOv[ind,f]))

test_unique, counts = np.unique(test,return_counts= True)

fig, axes = plt.subplots(1,1,figsize = (10,10))

# sns.histplot(data = counts, stat = "probability")
sns.histplot(data = counts)
# axes.set_xlim([0.5, 4.5])
# axes.set_ylim([0,1])
# axes.hist(counts)


np.mean(counts[54:])

np.std(counts[54:])

stats.ks_2samp(counts[:54], counts[54:])


# %% calculate number of overlap

f1 = 5
f2 = 4
f3 = 4
# f4 = 5
ind = 1
list_lick = np.unique(np.concatenate((listOv[ind,0],listOv[ind,1])))

list_overlap = np.intersect1d(listOv[ind,f1],listOv[ind,f2])
# list_overlap = np.intersect1d(list_lick,listOv[ind,5])
# list_overlap = np.intersect1d(list_overlap,listOv[ind,5]) 
    
print(len(list_overlap))
    
    
# %% comparing good_list2 vs weight fit list

list_other = np.setxor1d(good_list,test_unique)
# list_other = list_other[list_other<194]  

S1 = np.zeros((1,np.size(list_other)))
S2 = np.zeros((1,np.size(test_unique)))
for n in np.arange(np.size(list_other)):
    nn = int(list_other[n])
    try:
        S1[0,n] = np.mean(Data[nn, c_ind-1]["score"])
    except:
        S1[0,n] = 0
            

for n in np.arange(np.size(test_unique)):
    nn = int(test_unique[n])
    S2[0,n] = np.mean(Data[nn, c_ind-1]["score"])
     

bins = np.arange(0,0.5, 0.01)
fig, axs= plt.subplots(1,1,figsize = (5,5))
counts1, bins = np.histogram(S1,bins)
counts2, bins = np.histogram(S2,bins)

sns.histplot(data = S2[0,:],bins = bins,alpha = 0.8)
sns.histplot(data = S1[0,:],bins = bins,alpha = 0.8)





# %% for each timebin, calculate the number of neurons encoding each TV

cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green','tab:red','tab:orange','tab:purple','green',]
# cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_2','FA_2','Miss_2','CR_2']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']


Lic1 =np.argwhere(test_unique<194)[-1][0] +1 # 99 #134 # 78+1
Lg1 =len(test_unique)-Lic1
ind = 1# PPCIC or 1 PPCAC
p = 0# positive or 1 negative

fig, axes = plt.subplots(1,1,figsize = (10,5))
y_all = np.zeros((ax_sz,pre+post))
for f in np.arange(ax_sz):
    list0 = (np.mean(Convdata[f],1) != 0)
        
    Lg = len(good_list2)
    Lic = np.where(good_listRu <194)
    Lic = Lic[0][-1]
    if ind == 0:
        list0[Lic:Lg] = False # PPCIC
    elif ind == 1:           
        list0[0:Lic] = False # PPCAC
        
        # list0ind = np.arange(Lg)
        # list0ind = list0ind[list0]
    list0ind = good_listRu[list0]
    W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
    W = Convdata[f][list0,:]
    SD = np.std(W[:,:])
    # test = np.abs(W5[ind,f][p])>1*SD
    test = W5[ind,f][p]>2*SD
    if ind ==0:        
        y = np.sum(test,0)/Lic1
    elif ind == 1:
        y = np.sum(test,0)/Lg1
        
    y_all[f,:] = y
    y = ndimage.uniform_filter(y,2, mode = "mirror")
    if p == 0:
        axes.plot(y,c = cmap[f], linestyle = 'solid', linewidth = 3, label = clabels[f] )
        axes.set_ylim([0,.6])
        axes.legend()
    elif p == 1:
        axes.plot(-y,c = cmap[f], linestyle = 'solid', linewidth = 3 )
        axes.set_ylim([-0.20,0])
        
    
plt.savefig("Fraction of neurons "+ ".svg")
# %% plot positive and negative weights separately.
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','black','green','tab:red','tab:orange','black','green',]

pp = 0
maxy = np.zeros((2,10))
for ind in [0,1]:
    fig, axes = plt.subplots(2,ax_sz,figsize = (50,10),sharex = "all")
    fig.subplots_adjust(hspace=0)
    for p in [0,1]:
        for f in np.arange(ax_sz):
            if np.size(W5[ind,f][p],0) > 5:
                y1 = ndimage.gaussian_filter1d(np.sum(W5[ind,f][p],0),1)
                y1 = y1/(np.size(W5[ind,f][0],0)+np.size(W5[ind,f][1],0))
                e1 = np.std(W5[ind,f][p],0)/np.sqrt((np.size(W5[ind,f][0],0)+np.size(W5[ind,f][1],0)))
                axes[p,f].plot(xaxis,y1,c = cmap[f],linestyle = 'solid', linewidth = 3)
                axes[p,f].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
                # axes[p,f-3].set_xlim([-4,1])
                scat = np.zeros((2,np.size(W5IC[f],1)))
                pcat = np.zeros((2,np.size(W5IC[f],1)))
                maxy[p,f] = np.max(np.abs(y1)+np.abs(e1))
                maxy[p,f] = np.max([maxy[p,f],1])
                for t in np.arange(80):
                    if p == 0:
                        s1,p1 = stats.ttest_1samp(W5[ind,f][p][:,t],np.mean(e1),alternative = 'greater')
                    else:
                        s1,p1 = stats.ttest_1samp(W5[ind,f][p][:,t],-np.mean(e1),alternative= 'less')
                    if p1 < 0.05:
                        scat[0,t] = True
                        pcat[0,t] = p1
                c1 = pcat[0,scat[0,:]>0]
                if p == 0:
                    axes[p,f].scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*maxy[p,f] + 0.1,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,-1])
                elif p ==1:
                    axes[p,f].scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*-maxy[p,f] - 0.1,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,-1])
            else:
                for ln in np.arange(np.size(W5[ind,f][p],0)):
                    axes[p,f].plot(xaxis,ndimage.gaussian_filter1d(W5[ind,f][p][ln,:],1),c = cmap[f],linestyle = 'solid', linewidth = 1)
                if np.size(W5[ind,f][p],0) > 0:
                    maxy[p,f] = np.max(W5[ind,f][p])
    
    for f in np.arange(ax_sz):
            axes[0,f].set_ylim([0, np.nanmax(maxy[:,f]+0.2)])
            axes[1,f].set_ylim([-np.nanmax(maxy[:,f]+0.2),0])
    
    
    # axes[0,0].set_ylim([0, 0.45])
    # axes[1,0].set_ylim([-0.45,0])
    # axes[0,1].set_ylim([0, 0.35])
    # axes[1,1].set_ylim([-0.35,0])
    
    # plt.savefig("TVencoding"+ str(ind) + "tv" + str(f) + ".svg")
    
# %% plot each weights 
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','black','green','tab:red','tab:orange','black','green',]
# cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green']
clabels = ["lick_onset","lick_offset","stim-Go","stim-NG","Hit","FA",'Miss','CR','Hit_2','FA_2','Miss_2','CR_2']
lstyles = ['solid','dotted','solid','dotted','solid','dotted','solid','solid','solid','dotted','solid','solid']


pp = 1
maxy = np.zeros((2,10))
fig, axes = plt.subplots(1,1,figsize = (10,5),sharex = "all")
fig.subplots_adjust(hspace=0)
for f in [pp]: #np.arange(ax_sz):
    # W5IC[f][-4:,:] = 0
    for ind in [0,1]:
        if ind == 0:
            y1 = ndimage.gaussian_filter1d(np.mean(W5IC[f],0),1)
            e1 = np.std(W5IC[f],0)/np.sqrt(np.size(W5IC[f],0))
            # e1 = np.std(W5IC[f],0)/np.sqrt((np.size(W5[0,f][1],0)+np.size(W5[0,f][0],0)))
        elif ind ==1:
            y1 = ndimage.gaussian_filter1d(np.mean(W5AC[f],0),1)
            e1 = np.std(W5AC[f],0)/np.sqrt(np.size(W5AC[f],0))
            # e1 = np.std(W5AC[f],0)/np.sqrt((np.size(W5[1,f][1],0)+np.size(W5[1,f][0],0)))
        axes.plot(xaxis,y1,c = cmap[f],linestyle = lstyles[ind+1], linewidth = 3)
        axes.fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
    # ks test
    scat = np.zeros((2,np.size(W5IC[f],1)))
    pcat = np.zeros((2,np.size(W5IC[f],1)))
    for t in np.arange(np.size(W5IC[f],1)):
        s1,p1 = stats.ks_2samp(W5IC[f][:,t], W5AC[f][:,t],'less')
        s2,p2 = stats.ks_2samp(W5AC[f][:,t], W5IC[f][:,t],'less')
        if p1 < 0.05:
            scat[0,t] = True
            pcat[0,t] = p1
        if p2 < 0.05:
            scat[1,t] = True
            pcat[1,t] = p2
    c1 = pcat[0,scat[0,:]>0]
    c2 = pcat[1,scat[1,:]>0]
    axes.scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*0.85,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
    axes.scatter(xaxis[scat[1,:]>0],np.ones_like(xaxis[scat[1,:]>0])*0.85,marker='s',c = np.log10(c2),cmap = 'Greys_r',clim = [-3,0])

        
    # axes.set_ylim([-0.05,1.5])
    # for ind in [0,1]:
    #     y1 = ndimage.gaussian_filter1d(np.mean(W5[ind,f][1],0),1)*np.size(W5[ind,f][1],0)/(np.size(W5[0,f][1],0)+np.size(W5[0,f][0],0))
    #     e1 = np.std(W5[ind,f][1],0)/np.sqrt((np.size(W5[0,f][1],0)+np.size(W5[0,f][0],0)))
    #     axes[1].plot(xaxis,y1,c = cmap[f],linestyle = lstyles[ind], linewidth = 3)
    #     axes[1].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
    #     axes[1].set_ylim([-0.6,0.15])
    
# plt.savefig("TVweights_"+str(pp)+".svg")    
# t1 = 10
# t2 = 20
# print(np.mean(W5AC[f][:,t1:t2]))
# print(np.std(np.mean(W5AC[f][:,t1:t2],1))/np.sqrt(np.size(W5AC[f],0)))
# print(np.mean(W5IC[f][:,t1:t2]))
# print(np.std(np.mean(W5IC[f][:,t1:t2],1))/np.sqrt(np.size(W5AC[f],0)))

# stats.ks_2samp(np.mean(W5AC[f][:,t1:t2],1),np.mean(W5IC[f][:,t1:t2],1))


# %%
fig, axes = plt.subplots(1,1,figsize = (10,5), sharex = True)
fig.tight_layout()
fig.subplots_adjust(hspace=0)

cmap = ['tab:orange','white','tab:blue','white','tab:red','white','black','green','tab:purple','white']

edgec = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']


# edgec = ['tab:blue','tab:red','tab:purple','tab:blue','tab:red','tab:orange','orange']
# b11 = [9,9,6,23,12,18,12]/Lic
# b12 = [7,10,4,13,8,2,1]/Lic

# b21 = [42,17,10,39,30,27,37]/(Lg-Lic)
# b22 = [28,32,27,43,28,2,6]/(Lg-Lic)

# R1 
# b11 = [2,1,23,11,30,4,2,1]/Lic
# b12 = [0,0,1,3,6,0,0,0]/Lic

# b21 = [4,1,33,24,43,7,0,8]/(Lg-Lic)
# b22 = [0,0,9,2,13,2,0,10]/(Lg-Lic)

# R2
# b11 = [5,0,21,4,25,3,3,6]/Lic
# b12 = [0,0,0,1,5,1,1,1]/Lic

# b21 = [2,0,17,17,31,7,0,5]/(Lg-Lic)
# b22 = [0,0,8,3,15,4,0,8]/(Lg-Lic)

# Transition

Lic =57 #64
Lg = 213-57 #182
# Lic = 110
# Lg = 394-110
b11 = b_count[0][0,:]/Lic
b12 = b_count[0][1,:]/Lic

b21 = b_count[1][0,:]/Lg
b22 = b_count[1][1,:]/Lg
cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green','tab:red','tab:orange','tab:purple','green',]

edgec = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','tab:purple','green','tab:red','tab:orange','tab:purple','green',]
# axes.grid(visible=True,axis = 'y')
# axes[1].grid(visible=True,axis = 'y')
axes.bar(np.arange(8)*3+1,b11+b12, color = 'white', edgecolor = cmap, alpha = 1, width = 0.5, linewidth = 2, hatch = '/')
axes.bar(np.arange(8)*3,b21+b22, color = cmap, edgecolor = cmap, alpha = 1, width = 0.5, linewidth = 2)

# axes[0].bar(np.arange(1)*2+0.7,b21, color = cmap3, alpha = 1, width = 0.5)
# axes[0].bar(np.arange(4)*3+`1.4,b31, color = cmap3, alpha = 0.5, width = 0.5)
axes.set_ylim([0,0.9])
# axes.set_ylim([0,100])

# axes[1].bar(np.arange(8)*2+0.7,-b12, color = cmap, edgecolor = edgec, alpha =1, width = 0.5, linewidth = 2, hatch = '/')
# # axes[1].bar(np.arange(2)*2+0.7,-b22, color = cmap3, alpha = 1, width = 0.5)
# # axes[1].bar(np.arange(4)*3+1.4,-b32, color = cmap3, alpha = 0.5, width = 0.5)
# axes[1].set_ylim([-0.4,0.0])
# # axes[0].set_xlim([-.5,1.5])     






# %% 
Data1 = np.load('R1new_0718.npy',allow_pickle= True).item()

test = list(Data1.keys())
Y_R1 = {}
for t in np.arange(len(test)):
    n = test[t][0]
    Y_R1[n] = Data1[n,0]["Y"]
    
del Data1
    
Data2 = np.load('R2new_0718.npy',allow_pickle= True).item()
test = list(Data2.keys())
Y_R2 = {}
for t in np.arange(len(test)):
    n = test[t][0]
    Y_R2[n] = Data2[n,1]["Y"]
    
del Data2

YR = {}
YR[1] = Y_R1
YR[2] = Y_R2

def import_lick_mini(D_ppc,n,L2):
    time_point = D_ppc[n,3]*1e3
    t = 0
    time_ind = []
    while t*window < np.max(time_point):
        time_ind = np.concatenate((time_ind,np.argwhere(time_point[0,:]>t*window)[0]))
        t += 1
    ### Extract Lick ### 
    L_all = np.zeros((1,len(time_ind)-1))
    Lmax = np.max(np.floor(D_ppc[n,1]*(1e3/window)))
    if Lmax != L2:
        for l in np.floor(D_ppc[n,1]*(1e3/window)): 
            L_all[0,int(l[0])-1] = 1 
    
    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    c1 = stim_onset[200]-100
    c2 = stim_onset[D_ppc[n,4][0][0]+26] 
    L = L_all[:,c1:c2]
    return L,Lmax
# L2 = 0
# for n in np.arange(np.size(good_list2,0)):
#     nn = int(good_list2[n])
#     L, Lmax= import_lick_mini(D_ppc,nn,L2)
#     if np.sum(L) >0:
#         L3 = L
#     Data[nn,c_ind-1]["L"] = L3
#     print(nn)
    

# %% trial/b/trial analysis

# xaxis = xaxis-1
# f = 7 
# n = 126
pre =  10
post =  70
o_ind = 0
xaxis = (np.arange(post+pre)- pre)*1e-1




def TVFR_ana(n,f):
    X = D_ppc[n,2][:,2:6] # task variables
    X = X[200:D_ppc[n,4][0][0]+15]
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    stim_onset = Data[n, c_ind-1]["stim_onset"]
    r_onset = Data[n, c_ind-1]["r_onset"]
    L = Data[n, c_ind-1]["L"]
    for l in np.arange(len(r_onset)):
        if r_onset[l] <0:
            r_onset[l] = stim_onset[l] + 15
    
    if o_ind == 1:
        stim_onset = r_onset

    X3 = {};
    X3[2] = (Xstim == 1) 
    X3[3] = (Xstim == 0)
    X3[4] = XHit
    X3[5] = XFA
    X3[6] = Xmiss
    X3[7] = XCR
    # dur = 20
    # dur2 = 50
    if f == 2:
        Xb = (Xstim == 1) 
    elif f == 3:
        Xb = (Xstim == 0)
    elif f == 4:
        Xb= XHit
        # Xb = XFA
    elif f== 6:
        Xb = Xmiss
    elif f == 5:
        Xb = XFA
        # Xb = (Xstim == 0)
        # Xb = (Xstim == 1)
    elif f == 7:
        Xb = XCR
        # Xb = (Xstim == 0)
    
    comp = np.zeros((len(X),pre+post))   
    comp_n = np.zeros((len(X),pre+post))
    Lc = np.zeros((len(X),pre+post))
    h = Data[n,c_ind-1]["Y"]
    for t in np.arange(len(X)):
        if Xb[t] == 1:
                comp[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
                Lc[t,:] = L[0,stim_onset[t]-pre:stim_onset[t]+post]
        comp_n[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
    return comp_n, comp[Xb,:], XFA[Xb], XCR[Xb],X3,Lc    

def import_data_mini(D_ppc,n,r,X):
    # N_trial = np.size(D_ppc[n,2],0)
    window = 100
    stim_onset =np.round(D_ppc[n,3][0,D_ppc[n,2][:,0]]*(1e3/window))
    stim_onset = stim_onset.astype(int)
    
    Rt = np.floor(D_ppc[n,6][:,0]*(1e3/window))
    Rt =Rt.astype(int)
        
    
    ### Extract Ca trace ###

        
    if r == 1:
        c1 = stim_onset[1]-100
        c2 = stim_onset[151]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[1:150]
        X2 = X[1:150]
        r_onset = Rt-c1
        r_onset = r_onset[1:150]
    elif r == 2:       
        c1 = stim_onset[D_ppc[n,4][0][0]+26]-100 
        c2 = stim_onset[399]
        stim_onset2 = stim_onset-c1
        stim_onset2 = stim_onset2[D_ppc[n,4][0][0]+26:398]
        X2 = X[D_ppc[n,4][0][0]+26:398]
        r_onset = Rt-c1
        r_onset = r_onset[D_ppc[n,4][0][0]+26:398]
    try:
        Y = YR[r][n]    
    except:
        print(n)
        Yraw = {}
        Yraw = D_ppc[n,0]
        time_point = D_ppc[n,3]*1e3
        t = 0
        time_ind = []
        while t*window < np.max(time_point):
            time_ind = np.concatenate((time_ind,np.argwhere(time_point[0,:]>t*window)[0]))
            t += 1
        
        
        Y = np.zeros((1,len(time_ind)-1))   
        for t in np.arange(len(time_ind)-1):
            Y[0,t] = np.mean(Yraw[0,int(time_ind[t]):int(time_ind[t+1])])
        
        Y = Y[:,c1:c2]
    
    return X2,Y, stim_onset2, r_onset
        

def TVFR_ana_exp(n,f,r):
    # pre= 10
    # post = 70
    X,Y,stim_onset, r_onset = import_data_mini(D_ppc,n,r,D_ppc[n,2][:,2:6])
    for l in np.arange(len(r_onset)):
        if r_onset[l] <0:
            r_onset[l] = stim_onset[l] + 15
    
    if o_ind == 1:
        stim_onset = r_onset

    # stim_onset = r_onset
    
    if r ==1:
        r_list = np.random.choice(np.arange(len(stim_onset)),50,replace = False)
        r_list = np.arange(20,120);
        r_list = np.arange(len(stim_onset))
    elif r ==2:
        
        r_list = np.arange(np.min([50,D_ppc[n,4][0][0]+15-200]))
        r_list = np.random.choice(np.arange(len(stim_onset)),50,replace = False)
        r_list = np.arange(20,70);
        r_list = np.arange(40,np.size(stim_onset));
    X = X[r_list]
    stim_onset= stim_onset[r_list]                 
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    # dur = 20
    # dur2 = 50
    if r ==2:
        X3 = {};
        X3[2] = (Xstim == 1) 
        X3[3] = (Xstim == 0)
        X3[4] = XHit
        X3[5] = XFA
        X3[6] = Xmiss
        X3[7] = XCR
        if f == 2:
            Xb = (Xstim == 1) 
        elif f == 3:
            Xb = (Xstim == 0)
        elif f == 4:
            Xb= XHit
        elif f == 5:
            Xb = XFA
            # Xb =XHit
        elif f == 6:
            Xb = Xmiss  
        elif f == 7:
            Xb = XCR
            # Xb = (Xstim == 0)
    elif r == 1:
        X3 = {};
        X3[2] = (Xstim == 1) 
        X3[3] = (Xstim == 0)
        X3[4] = XHit
        X3[5] = XFA
        X3[6] = Xmiss
        X3[7] = XCR
        if f == 2:
            Xb = (Xstim == 0) 
        elif f == 3:
            Xb = (Xstim == 1)
        elif f == 4:
            Xb= XHit
            # Xb = XFA
        elif f == 5:
            Xb = XFA
            Xb = (Xstim == 0)*(X[:,1]==1) # FA trials
            # Xb = (Xstim == 1)*(X[:,1]==1) # Hit trials
        elif f == 6:
            Xb = Xmiss
        elif f == 7:
            Xb = XCR
            # Xb = (Xstim == 0)
        
        
    comp = np.zeros((len(X),pre+post))    
    comp_n = np.zeros((len(X),pre+post))
    h = Y
    for t in np.arange(len(X)):
        comp_n[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
        if Xb[t] == 1:
                comp[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
    return comp_n, comp[Xb,:], X3  


def TVFR_ana_r2(n,f):
    # pre= 10
    # post = 70
    X = D_ppc[n,2][:,2:6] # task variables
    X = X[D_ppc[n,4][0][0]+15:D_ppc[n,4][0][0]+15]
    Xstim = X[:,0]
    XHit = (X[:,0]==1)*(X[:,1]==1)
    XFA  = (X[:,0]==0)*(X[:,1]==1)
    Xmiss = (X[:,0]==1)*(X[:,1]==0)
    XCR = (X[:,0]==0)*(X[:,1]==0)
    stim_onset = Data[n, c_ind-1]["stim_onset"]
    # stim_onset= stim_onset[D_ppc[n,4][0][0]-200:]
    # dur = 20
    # dur2 = 50
    
    Xb1 = {}
    if f == 2:
        Xb = (Xstim == 1) 
    elif f == 3:
        Xb = (Xstim == 0)
    elif f == 4:
        Xb= XHit
    elif f == 5:
        Xb = XFA
    elif f == 7:
        Xb = XCR
        # Xb = (Xstim == 0)

    
    
    comp = np.zeros((len(X),80))    
    comp_n = np.zeros((len(X),80))
    h = Data[n,c_ind-1]["Y"]
    for t in np.arange(len(X)):
        comp_n[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
        if Xb[t] == 1:
                comp[t,:] = h[0,stim_onset[t]-pre:stim_onset[t]+post]
    
    return comp_n, comp[Xb,:]    


# %%

fname2 = 'test'

f = 5
p = 0
comp = {}
comp_n = {}
comp_r1 = {}
comp_n_r1 = {}
comp_r2 = {}
comp_n_r2 = {}
comp_n = {}
comp_n[0] = {}
comp_n[1] = {}
comp_n[2] = {}

XCR = {};
XFA = {};
X3 = {}
X3[0] = {};
X3[1] = {};
X3[2] = {};
L = {};
for n in listOv[p,f]:
    nn = int(n)
    comp_n[0][nn], comp[nn], XFA[nn], XCR[nn],X3[0][nn],L[nn] = TVFR_ana(nn,f)
    comp_n[1][nn], comp_r1[nn], X3[1][nn] = TVFR_ana_exp(nn,f,1)
    comp_n[2][nn], comp_r2[nn], X3[2][nn] = TVFR_ana_exp(nn,f,2)
    # # comp_n_r2[nn], comp_r2[nn] = TVFR_ana_r2(nn,f)



# %% code edit 24-12-09 simple version, comparing within RT

comp2= {};
comp2[0] = np.zeros((len(comp),pre+post))
comp2[1] = np.zeros((len(comp),pre+post))                    
comp2[2] = np.zeros((len(comp),pre+post))
comp2[3] = np.zeros((len(comp),pre+post))   
comp_lick = np.zeros((len(comp),pre+post))

s_ind = 1
for n in np.arange(len(comp)):
    nn= int(listOv[p,f][n])
    l = int(np.floor(len(comp[nn])/2))
    l2 = int(np.floor(len(comp[nn])/4))
    maxc = []
    minc = []
    for ind in [0]:
        for ind_f in [2,3,4,5,6,7]:
            if np.sum(X3[ind][nn][ind_f])>0:
                maxc = np.concatenate((maxc,[np.percentile(np.mean(comp_n[ind][nn][X3[ind][nn][ind_f],:],0),80)]))
                minc = np.concatenate((minc,[np.percentile(np.mean(comp_n[ind][nn][X3[ind][nn][ind_f],:],0),20)]))
    max_all = np.max(maxc)
    min_all = np.min(minc) 
    # min_all  = 0;

    if f in [4,5,6,7]:
        # if n not in [7,15]:
        # if n not in [40,41]:
        # if n not in [23,27]:
            # comp2[1][n,:] = (np.mean(comp[nn],0)-min_all)/(max_all-min_all+s_ind)
            comp2[0][n,:] = (np.mean(comp[nn][0:l,:],0)-min_all)/(max_all-min_all+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][l:,:],0)-min_all)/(max_all-min_all+s_ind)
    else:
        if f in [7]:
            comp2[0][n,:] = (np.mean(comp[nn][XFA[nn],:],0)-min_all)/(max_all-min_all+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][XCR[nn],:],0)-min_all)/(max_all-min_all+s_ind)
        # if f in [6]:
        #     comp2[0][n,:] = (np.mean(comp[nn][:,:],0)-min_all)/(max_all-min_all+s_ind)
    #     else:
    #         comp2[0][n,:] = (np.mean(comp[nn][0:l,:],0)-min_all)/(max_all-min_all+s_ind)
    #         comp2[1][n,:] = (np.mean(comp[nn][l:,:],0)-min_all)/(max_all-min_all+s_ind)


listind = np.zeros((1,len(comp)))
for c in np.arange(len(comp)):
    listind[0,c] = True 
    
if f == 7:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,5] == listOv[p,7][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)  
    
elif f == 5:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,7] == listOv[p,5][[c]])  ==0:
        listind[0,c] = True 
    listind = (listind == 1)  
elif f in [4,6]:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,7] == listOv[p,4][[c]])  ==1:
        listind[0,c] = True 
    #     elif np.sum(listOv[p,0] == listOv[p,4][[c]])  ==1:
    #         listind[0,c] = False 
    #     elif np.sum(listOv[p,1] == listOv[p,4][[c]])  ==1:
    #         listind[0,c] = False 
    listind = (listind == 1)   

    listind2 = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,5] == listOv[p,4][[c]])  ==0:
            listind2[0,c] = True 
    listind2 = (listind2 == 1)  
    listind2 = listind2[0][listind[0]] 
    listind3= np.ones((10,len(listind2)))*listind2
        
W2 = {}
for ind in [0,1,2,3]:
    W2[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W2[ind]),1)
    for n in np.arange(np.size(W2[ind],0)):
        if W2[ind][n,max_peak[n]] < 0:
            W2[ind][n,:] = -W2[ind][n,:]
W = {};
# if f ==5:
#     go_labels  = ['FA','FA','R1','R2']
#     cmap = ['tab:orange','tab:orange','black','grey']
# elif f == 4:
#     go_labels  = ['early','late','R1','R2']
#     cmap = ['tab:orange','tab:red','black','grey']
if f in [7]:
    go_labels  = ['FA','CR','R1','R2']
    cmap = ['tab:orange','tab:green','black','grey']
elif f in [8]:
    cmap = ['tab:purple','tab:green','black','grey']
else:
    go_labels  = ['Early','Late','R1','R2']
    cmap = ['tab:blue','tab:orange','black','grey']

if f in [4,5,6,7]:
    for ind in [0,1]:
       W[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,1], mode = "mirror")
    max_peak = np.argmax(np.abs(W[1]),1)
    max_ind = max_peak.argsort()
    # max_ind = max_ind[(20>max_peak)]    

    
    for ind in [0,1]:
        fig, axes = plt.subplots(1,1,figsize = (5,5)) 
        im1 = axes.imshow(zscore(W[ind][max_ind, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2,aspect = "auto")
        # im1 = axes.imshow(W[ind][max_ind, :], cmap="viridis",    clim = [0.2,0.7],aspect = "auto")

    #     fig.subplots_adjust(right=0.85)
    #     cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    #     fig.colorbar(im1, cax=cbar_ax)
    
    fig, axes = plt.subplots(1,1,figsize = (10,5))
    peak = np.zeros((1,4))
    for ind in [0,1,2,3]:
        y = np.nanmean(W2[ind][listind[0],:],0)
        e = np.nanstd(W2[ind][listind[0],:],0)/np.sqrt(np.size(W2[ind],0))
        axes.plot(xaxis,y,color= cmap[ind],label = go_labels[ind])
        axes.fill_between(xaxis,y-e,y+e,facecolor= cmap[ind],alpha = 0.3)
        peak[0,ind] = np.max(y)
    scat = {}
    pcat = {}
    for p_ind in [0,1,2]:
        scat[p_ind] = np.zeros((1,pre+post))
        pcat[p_ind] = np.zeros((1,pre+post))
    s = {}
    pp = {}
    
    for t in np.arange(pre+post):
            # s1,p1 = stats.ks_2samp(W2[0][listind[0],t], W2[1][listind[0],t])
        s[0],pp[0] = stats.ttest_ind(W2[0][listind[0],t], W2[1][listind[0],t],nan_policy = 'omit')
        # s[0],pp[0] = stats.ks_2samp(W2[0][listind[0],t], W2[1][listind[0],t])
        # s[0],pp[0] = stats.ks_2samp(W2[1][listind[0],t], W2[2][listind[0],t])
        # s[1],pp[1] = stats.ks_2samp(W2[1][listind[0],t], W2[3][listind[0],t])
        # s[2],pp[2] = stats.ks_2samp(W2[2][listind[0],t], W2[3][listind[0],t])
        for p_ind in [0]:
            pcat[p_ind][0,t] = pp[p_ind]
            if pp[p_ind] < 0.05:
                scat[p_ind][0,t] = True
                
            c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
            # if p_ind == 2:
                # axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'hot',clim = [-3,0])
            if p_ind in [0]:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.3+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
    axes.legend()    
    axes.set_ylim([0.1, 1])
    # fig,axes = plt.subplots(1,1,figsize = (10,5))
    # axes.plot(xaxis,np.mean(comp_lick,0))




# %% this is the current final code
# divide up data for comparison
# Go, Nogo are divided into early vs late
# Hit, FA, CR are divided into R1, TR and R2
# added comp2[4] which is the average trace during entire RT


comp2= {};
comp2[0] = np.zeros((len(comp),pre+post))
comp2[1] = np.zeros((len(comp),pre+post))                    
comp2[2] = np.zeros((len(comp),pre+post))
comp2[3] = np.zeros((len(comp),pre+post))   
comp2[4] = np.zeros((len(comp),pre+post))   
comp_lick = np.zeros((len(comp),pre+post))

s_ind = 1
for n in np.arange(len(comp)):
    nn= int(listOv[p,f][n])
    l = int(np.floor(len(comp[nn])/2))
    l2 = int(np.floor(len(comp[nn])/4))
    maxc = []
    minc = []
    for ind in [0,1,2]:
        for ind_f in [2,3,4,5,7]:
            if np.sum(X3[ind][nn][ind_f])>0:
                maxc = np.concatenate((maxc,[np.percentile(np.mean(comp_n[ind][nn][X3[ind][nn][ind_f],:],0),80)]))
                minc = np.concatenate((minc,[np.percentile(np.mean(comp_n[ind][nn][X3[ind][nn][ind_f],:],0),20)]))
    max_all = np.max(maxc)
    min_all = np.min(minc) 
    # min_all  = 0;

    if f in [0]:
        # if n not in [7,15]:
        # if n not in [40,41]:
        # if n not in [23,27]:
            comp2[1][n,:] = (np.mean(comp[nn],0)-min_all)/(max_all-min_all+s_ind)
    else:
        if f in [3]:
            comp2[0][n,:] = (np.mean(comp[nn][XFA[nn],:],0)-min_all)/(max_all-min_all+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][XCR[nn],:],0)-min_all)/(max_all-min_all+s_ind)
        elif f in [6]:
            comp2[0][n,:] = (np.mean(comp[nn][:,:],0)-min_all)/(max_all-min_all+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][:,:],0)-min_all)/(max_all-min_all+s_ind)
        else:
            comp2[0][n,:] = (np.mean(comp[nn][0:l,:],0)-min_all)/(max_all-min_all+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][l:,:],0)-min_all)/(max_all-min_all+s_ind)


    comp2[2][n,:] = (np.mean(comp_r1[nn],0)-min_all)/(max_all-min_all+s_ind)
    comp2[3][n,:] = (np.mean(comp_r2[nn],0)-min_all)/(max_all-min_all+s_ind)
    comp2[4][n,:] = (np.mean(comp[nn],0)-min_all)/(max_all-min_all+s_ind)
    comp_lick[n,:] = np.mean(L[nn]*10,0)
    



if f == 7:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,5] == listOv[p,7][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)  
    
elif f == 5:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,7] == listOv[p,5][[c]])  ==0:
        # if np.sum(listOv[p,0] == listOv[p,4][[c]])  ==0 and np.sum(listOv[p,1] == listOv[p,4][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)  
elif f == 4:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,0] == listOv[p,4][[c]])  ==0 and np.sum(listOv[p,1] == listOv[p,4][[c]])  ==0:
        #     if np.sum(listOv[p,5] == listOv[p,4][[c]])  ==0:
                listind[0,c] = True 
    listind = (listind == 1)   

    listind2 = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,5] == listOv[p,4][[c]])  ==0:
            listind2[0,c] = True 
    listind2 = (listind2 == 1)  
    listind2 = listind2[0][listind[0]] 
    listind3= np.ones((10,len(listind2)))*listind2
elif f == 6: 
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,7] == listOv[p,4][[c]])  ==1:
        listind[0,c] = True 
    listind = (listind == 1)  
        
        
W2 = {}
for ind in [0,1,2,3,4]:
    W2[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W2[ind]),1)
    for n in np.arange(np.size(W2[ind],0)):
        if W2[ind][n,max_peak[n]] < 0:
            W2[ind][n,:] = -W2[ind][n,:]
W = {};
if f ==5:
    go_labels  = ['FA','CR','R1','R2']
    cmap = ['tab:orange','tab:green','black','grey']
# elif f == 4:
#     go_labels  = ['FA','Hit','R1','R2']
#     cmap = ['tab:orange','tab:red','black','grey']
elif f == 7:
    go_labels  = ['FA','CR','R1','R2']
    cmap = ['tab:orange','tab:green','black','grey']
else:
    go_labels  = ['Early','Late','R1','R2']
    cmap = ['tab:blue','tab:orange','black','grey']

if f in [4,5,6, 7]: #[4,5,7]:
    
    
    for ind in [0,1,2,3,4]:
        W[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,1], mode = "mirror")
    max_peak = np.argmax(np.abs(W[1]),1)
    max_ind = max_peak.argsort()
    # max_ind = max_ind[(20>max_peak)]    
    # for ind in [0,1,2,3]:
    #     # fig, axes = plt.subplots(1,2,figsize = (5,5),gridspec_kw={'width_ratios': [1, 2]})        
    #     # im1 = axes[1].imshow(W[ind][max_ind,:],clim = [0.2,0.7], aspect = "auto", interpolation = "None",cmap = "viridis")
    #     # axes[0].imshow(listind3.T,clim = [-0.1,1.1])
    #     fig, axes = plt.subplots(1,1,figsize = (5,5)) 
    #     # axes.imshow(W2[2][:,:],clim = [0.2,0.7],aspect = "auto")
    #     im1 = axes.imshow(zscore(W[ind][max_ind, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2,aspect = "auto")


    #     fig.subplots_adjust(right=0.85)
    #     cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    #     fig.colorbar(im1, cax=cbar_ax)
    
    fig, axes = plt.subplots(1,1,figsize = (10,5))
    peak = np.zeros((1,4))
    for ind in [0,1,2,3]:
        # y = np.nanmean(W2[ind][listind[0],:]-W2[2][listind[0],:],0)
        # e = np.nanstd(W2[ind][listind[0],:]-W2[2][listind[0],:],0)/np.sqrt(np.size(W2[ind],0))
        y = np.nanmean(W2[ind][listind[0],:],0)
        e = np.nanstd(W2[ind][listind[0],:],0)/np.sqrt(np.size(W2[ind],0))

        # y = np.nanmean(W[ind][max_ind[:],:],0)
        # e = np.nanstd(W[ind][max_ind[:],:],0)/np.sqrt(np.size(W[ind][:,:],0))
        
        axes.plot(xaxis,y,color= cmap[ind],label = go_labels[ind])
        axes.fill_between(xaxis,y-e,y+e,facecolor= cmap[ind],alpha = 0.3)
        peak[0,ind] = np.max(y)
    scat = {}
    pcat = {}
    for p_ind in [0,1,2]:
        scat[p_ind] = np.zeros((1,pre+post))
        pcat[p_ind] = np.zeros((1,pre+post))
    s = {}
    pp = {}
    
    for t in np.arange(pre+post):
            # s1,p1 = stats.ks_2samp(W2[0][listind[0],t], W2[1][listind[0],t])
        # s[0],pp[0] = stats.ttest_ind(W2[1][listind[0],t], W2[2][listind[0],t],nan_policy = 'omit')
        # s[1],pp[1] = stats.ttest_ind(W2[3][listind[0],t], W2[2][listind[0],t],nan_policy = 'omit')
        s[0],pp[0] = stats.ttest_ind(W2[0][listind[0],t], W2[2][listind[0],t],nan_policy = 'omit')
        s[1],pp[1] = stats.ttest_ind(W2[1][listind[0],t], W2[2][listind[0],t],nan_policy = 'omit')      
        s[2],pp[2] = stats.ttest_ind(W2[0][listind[0],t], W2[3][listind[0],t],nan_policy = 'omit')
        s[0],pp[0] = stats.ks_2samp(W2[0][listind[0],t], W2[2][listind[0],t])
        # s[1],pp[1] = stats.ks_2samp(W2[1][listind[0],t], W2[3][listind[0],t])
        # s[2],pp[2] = stats.ks_2samp(W2[2][listind[0],t], W2[3][listind[0],t])
        for p_ind in [0,1,2]:
            pcat[p_ind][0,t] = pp[p_ind]
            if pp[p_ind] < 0.1:
                scat[p_ind][0,t] = True
                
            c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
            # if p_ind == 2:
                # axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'hot',clim = [-3,0])
            if p_ind in [0,1,2]:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.3+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
    axes.legend()    
    axes.set_ylim([-0.,0.7])  
    # fig,axes = plt.subplots(1,1,figsize = (10,5))
    # axes.plot(xaxis,np.mean(comp_lick,0))
    
      
# if f in [4,5]:
#     ind = 1
#     W[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,2], mode = "mirror")
#     # W[ind] = W3_hit[ind]
#     max_peak = np.argmax(np.abs(W[ind]),1)
#     max_ind = max_peak.argsort()
#     Lick_sm = ndimage.uniform_filter(comp_lick[listind[0],:],[0,2], mode = "mirror")
#     lick_peak = np.argmax(Lick_sm,1)
#     peak_distance = max_peak-pre
    
#     fig, axes = plt.subplots(1,1,figsize = (5,5))  
#     bins = np.arange(-40,60,5)
#     # bins = np.arange(-20,80,5)
#     sns.histplot(peak_distance,stat = "percent",bins = bins)
#     axes.set_ylim([0,30])

# pd_FA_IC = np.abs(peak_distance*0.1)
# np.median(pd_FA_IC)
    
# %%
# import pandas as pd


# Wa = np.concatenate((pd_FA_AC,pd_FA_IC))
# # grp = np.concatenate((np.zeros((1,np.size(pd_Hit_AC))),np.ones((1,np.size(pd_Hit_IC)))),1)
# grp = np.concatenate((np.repeat("PPCAC",np.size(pd_FA_AC)),np.repeat("PPCIC",np.size(pd_FA_IC))),0)

# d = {'x': Wa, 'group': grp}
# df = pd.DataFrame(d)

# fig, axes = plt.subplots(1,1,figsize = (9,7))  
# sns.swarmplot(data=df, y="x", x="group",color = 'gray')
# sns.boxplot(data=df, y="x", x="group")





# %% rastermap for W2
from rastermap import Rastermap, utils

W3 = {}
for ind in [0,1,2,3,4]:
    W3[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,4],mode = "mirror")
    # W3[ind] = W2[ind][listind[0],:],[0,3],mode = "mirror")
                             
# W3[2] = (W3[2] + W3[3])/2
# W3[0] = W3[1]
ind =4
# fit rastermap
# note that D_r is already normalized
model = Rastermap(n_PCs=64,
                  locality=.75,
                  time_lag_window=7,
                  n_clusters = 10,
                  grid_upsample=1, keep_norm_X = False).fit(W3[ind])
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
# X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)

for ind in [0,1,2,3]:
    fig, ax = plt.subplots(figsize = (5,5))
        # ax.imshow(zscore(W2[ind][isort, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
    ax.imshow(zscore(W3[ind][isort, :],axis = 1), cmap="gray_r", vmin=0.5, vmax=1.2,aspect = "auto")
    ax.set(xticks=np.arange(0, 80, 10), xticklabels=np.arange(-1, 7, 1));
    # ax.imshow(W3[ind][isort, :], cmap="viridis", vmin=0.3, vmax=0.7,aspect = "auto")
        # ax.imshow(W3[ind][isort, :], cmap="gray_r", aspect="auto")

# rastermpa end

# fig, ax = plt.subplots(figsize = (5,5))
# for ind in [1,2,3]:
#     ax.plot(np.mean(W3[ind],0))

# %% NO FA specific
rng = np.random.default_rng()

W4 = {}
# ind 4 is RT all trials
for ind in [0,1,2,3,4]:
    W4[ind] = W3[ind]

# W4[4] = W4[0]    

W4[5] = np.zeros_like(W4[0])
for tn in np.arange(np.size(W3[0],0)):
    W4[5][tn,:] = W4[0][tn,:]
rng.shuffle(W4[5],axis = 0)
 
# %%For FA 
# for FA trials
# rng = np.random.default_rng()




W4 = {}
for ind in [0,1,2,3]:
    W4[ind] = W3[ind]
W4[1] = W4[0]



# # for Hit trials, add after FA 
W4[4] = W3[0]
W4[5] = W3[2]
W4[6] = W3[3]

# W4[0] = W3[0]
# W4[7] = np.zeros_like(W4[0])
# for tn in np.arange(np.size(W3[0],0)):
#     W4[7][tn,:] = W4[0][tn,:]
# rng.shuffle(W4[7],axis = 0)
# rng.shuffle(W4[7],axis = 1)
# %% stim encoding 


peak = np.zeros((1,4))

if f == 3:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # if np.sum(listOv[p,0] == listOv[p,3][[c]])  == 0:
        listind[0,c] = True 
    listind = (listind == 1)  
    go_labels  = ['FA','CR','R1','R2']
    cmap = ['tab:orange','tab:green','black','grey']
    # listind2 = np.zeros((1,len(comp)))
    # for c in np.arange(len(comp)):
    #     if np.sum(listOv[p,7] == listOv[p,3][[c]])  ==0:
    #         listind2[0,c] = True 
    # listind = (listind == 1)  
    
    
    
if f == 2:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        # xif np.sum(listOv[p,0] == listOv[p,2][[c]])  ==0:
        listind[0,c] = True 
    listind = (listind == 1)
    go_labels  = ['Early','Late','R1','R2']
    cmap = ['tab:blue','tab:orange','black','grey']

W2 = {}
for ind in np.arange(5):
    W2[ind] = ndimage.uniform_filter(comp2[ind][listind[0],:],[0,1], mode = "mirror")
    max_peak = np.argmax(np.abs(W2[ind]),1)
    for n in np.arange(np.size(W2[ind],0)):
        if W2[ind][n,max_peak[n]] < 0:
            W2[ind][n,:] = -W2[ind][n,:]


fig, axes = plt.subplots(1,1,figsize = (10,5))
            
for ind in [0,1,2,3]:
    # y = np.nanmean(W2[ind],0)
    # e = np.nanstd(W2[ind],0)/np.sqrt(np.size(W2[ind],0))
    y = np.nanmean(W2[ind],0)#- np.nanmean(W2[2],0)
    e = np.nanstd(W2[ind],0)/np.sqrt(np.size(W2[ind],0))
    axes.plot(xaxis,y,color = cmap[ind],label = go_labels[ind])
    axes.fill_between(xaxis,y-e,y+e,facecolor = cmap[ind],alpha = 0.3)
    peak[0,ind] = np.max(y)

scat = {}
pcat = {}
for p_ind in [0,1,2]:
    scat[p_ind] = np.zeros((1,pre+post))
    pcat[p_ind] = np.zeros((1,pre+post))
s = {}
pp = {}
for t in np.arange(pre+post):
    s[0],pp[0] = stats.ttest_ind(comp2[0][:,t], comp2[2][:,t])
    s[1],pp[1] = stats.ttest_ind(comp2[1][:,t], comp2[2][:,t])
    # s[2],pp[2] = stats.ttest_ind(comp2[1][:,t], comp2[2][:,t])
    # s1,p1 = stats.ks_2samp(comp2[0][:,t], comp2[1][:,t])
    for p_ind in [0,1]:
        if pp[p_ind] < 0.1:
            scat[p_ind][0,t] = True
            pcat[p_ind][0,t] = pp[p_ind]
        c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
        if p_ind == 0:
            axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
        else:
            axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])

axes.legend()
# axes.set_ylim([-0.2,0.4])    

# for ind in [0,1,2]:
#     W[ind] = ndimage.uniform_filter(W2[ind],[0,2], mode = "mirror")
#     # W[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
# max_peak = np.argmax(np.abs(W[1]),1)
# max_ind = max_peak.argsort()    
# for ind in [0,1,2]:
#     fig, axes = plt.subplots(1,1,figsize = (5,5))        
#     im1 = axes.imshow(W[ind][max_ind,:],clim = [0,1], aspect = "auto", interpolation = "None",cmap = "viridis")
#     fig.subplots_adjust(right=0.85)
#     cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
#     fig.colorbar(im1, cax=cbar_ax)

W4 = {}
for ind in [0,1,2,3,4]:
    W4[ind] = ndimage.uniform_filter(W2[ind],[0,5], mode = "mirror")
    
W4[5] = np.zeros_like(W4[0])
for tn in np.arange(np.size(W4[0],0)):
        W4[5][tn,:] = W4[0][tn,:]
rng.shuffle(W4[5],axis = 0)
# %% PCA on RT subspace. 



max_k = 20;

sm = 0
R = {}
t1 = 20
t2 = 80 # 25 if stim
pca = PCA(n_components=max_k)
R= W4[4][:,t1:t2].T
test = pca.fit_transform(ndimage.gaussian_filter(R,[2,0]))        
test = test.T
expvar = pca.explained_variance_ratio_

traj = {}
for ind in np.arange(len(W4)):
    traj[ind] = np.dot(W4[ind][:,0:t2].T,pca.components_.T)
    traj[ind] = ndimage.gaussian_filter(traj[ind],[3,0]) #- np.mean(traj[ind][0:10,:],0)

fig, axes = plt.subplots(3,1,figsize = (10,15))

for f_ind in [0,1,2]:
    for ind in [0,1]:
        axes[f_ind].plot(xaxis,traj[ind][:,f_ind])

fig, axes = plt.subplots(1,1,figsize=(10,10))
for ind in [0,1]:
    axes.plot(traj[ind][:,0],traj[ind][:,1])

# %%
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm
from scipy import linalg


def draw_traj5(traj,v):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['solid','solid','solid','dotted','solid','dotted','dashed','dashed']
    cmap_names = ['autumn','autumn','winter','winter']
    for ind in [0,1]: # np.arange(trmax):
            x = traj[ind][:,0]
            y = traj[ind][:,1]
            z = traj[ind][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
            if ind == 0:
                colors = cm.autumn(np.linspace(0,1,80))
                # ax.auto_scale_xyz(x,y,z)
            elif ind == 1:
                colors = cm.winter(np.linspace(0,1,80))
                ax.auto_scale_xyz(x,y,z)
            elif ind in [4]:
                colors = cm.winter(np.linspace(0,1,80))
            else :
                colors = cm.gray(np.linspace(0,1,80))

            

            lc = Line3DCollection(segments, color = colors,alpha = 1,linestyle = styles[ind])#linestyle = styles[tr])
    
            lc.set_array(time)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "black")
                
draw_traj5(traj,0)           
  
tlabels = ['early','late','r1','r2','TR','shuffled']

# savemat('D:\DATA\_traj_' + fname2 + '.mat',{tlabels[f]:traj[f] for f in np.arange(6)})

# %% 
# draw_traj5(traj,0)

k = 2
def traj_dist(array):
    array = array[:,[0,1,2]]
    distance  =np.zeros((1,t2-0))
    for t in np.arange(t2-0-1):
        distance[0,t] = np.linalg.norm(array[t+1,0:k]-array[t,0:k])
    return distance


# g = [traj_dist(traj[0]),traj_dist(traj[1]),traj_dist(traj[2]),traj_dist(traj[3]),traj_dist(traj[5])] 
# g = [traj_dist(traj[1]),traj_dist(traj[2]),traj_dist(traj[3])] 

# g = [traj_dist(traj[0]),traj_dist(traj[2]),traj_dist(traj[3]),traj_dist(traj[4]),traj_dist(traj[5]),traj_dist(traj[7])] 
# g = g/np.max(g)
# fig, axes= plt.subplots(1,1,figsize = (5,5))
# axes.bar(np.arange(np.size(g)),g)


# %% trajectory distance with n iterated.
max_k = 20;

sm = 5
R = {}
t1 = 20
t2 = 80
total_n = np.size(W4[0],0)
n_cv = 20;

def calc_traj(t1,t2,n_cv,sm):
    g = {}
    g[0] = np.zeros((n_cv,t2))
    g[1] = np.zeros((n_cv,t2))
    for cv in np.arange(n_cv):
        n_list = np.random.choice(np.arange(total_n),int(np.floor(total_n*0.75)),replace = False)       
        pca = PCA(n_components=20)
        R= W4[4][n_list,t1:t2].T
        test = pca.fit_transform(ndimage.gaussian_filter(R,[sm,0]))        
        test = test.T
      
        traj = {}
        for ind in np.arange(len(W4)):
            traj[ind] = np.dot(W4[ind][n_list,0:t2].T,pca.components_.T)
            traj[ind] = traj[ind]# - np.mean(traj[ind][0:5,:],0)
        g[0][cv,:] =traj_dist(traj[0])
        g[1][cv,:] =traj_dist(traj[1])
    return g

g = calc_traj(t1,t2,n_cv,sm)

fig, axes= plt.subplots(1,1,figsize = (10,5))
for ind in [0,1]:
    yd = ndimage.gaussian_filter1d(g[ind],3)
    axes.plot(xaxis,np.mean(yd,0))
    axes.fill_between(xaxis,np.mean(yd,0)-np.std(yd,0),np.mean(yd,0)+np.std(yd,0),alpha = 0.3)
axes.set_ylim([0,0.35])

# fig, axes= plt.subplots(1,1,figsize = (5,5))
# axes.bar(np.arange(np.size(g,0)),np.mean(g,1))
# axes.errorbar(np.arange(np.size(g,0)),np.mean(g,1),np.std(g,1), fmt="o", color="k", barsabove = True, capsize = 7, markersize = 0)
# axes.set_ylim([0,20])

# stats.ttest_ind(g[0,:], g[1,:])

# savemat('D:\DATA\_distance_' + fname2 + '.mat',{"distance" :g})

# %% calculate euclidean distance

trmax = len(traj)
ind =1
ED = np.zeros((trmax,trmax))


t1 = 0
t2 = 80
for ind1 in np.arange(trmax):
    for ind2 in np.arange(trmax):
        ED[ind1,ind2] = np.linalg.norm(traj[ind1][t1:t2]-traj[ind2][t1:t2])
        

fig, axes = plt.subplots(1,1,figsize = (10,10))

# ED = ED/np.max(ED)
imshowobj = axes.imshow(ED,cmap = "hot")
# imshowobj = axes.imshow(Overlap[:,:,0],cmap = "hot_r")
imshowobj.set_clim(0, np.max(ED)) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK

# %% euclidean distanceby time

max_k = 20;

sm = 1
R = {}
t1 = 20
t2 = 80
total_n = np.size(W4[0],0)
n_cv = 20;
def calc_dist(t1,t2,n_cv,sm):
    dis = np.zeros((n_cv,t2))
    max_dis = np.zeros((n_cv,1))
    for cv in np.arange(n_cv):
        n_list = np.random.choice(np.arange(total_n),int(np.floor(total_n*0.75)),replace = False)       
        pca = PCA(n_components=20)
        R= W4[4][n_list,t1:t2].T
        test = pca.fit_transform(ndimage.gaussian_filter(R,[sm,0]))        
        test = test.T
      
        traj = {}
        for ind in np.arange(len(W4)):
            traj[ind] = np.dot(W4[ind][n_list,0:t2].T,pca.components_.T)
            traj[ind] = traj[ind]# - np.mean(traj[ind][0:5,:],0)
        for t in np.arange(t2):
            dis[cv,t] = np.linalg.norm(traj[0][t,0:k]-traj[1][t,0:k])
        max1 = np.max(np.linalg.norm(traj[0][:],axis = 1,keepdims = True))
        max2 = np.max(np.linalg.norm(traj[0][:],axis = 1,keepdims = True))
        max_dis[cv,0] = np.max([max1,max2])
    return dis, max_dis

dis, max_dis = calc_dist(t1,t2,n_cv,sm)

max_dis = 1
dis2 = ndimage.gaussian_filter1d(dis/max_dis,3)

fig, axes = plt.subplots(1,1,figsize = (10,5))
axes.plot(xaxis,np.mean(dis2,0))
axes.fill_between(xaxis,np.mean(dis2,0)-np.std(dis2,0),np.mean(dis2,0)+np.std(dis2,0),alpha = 0.3)
# axes.set_ylim(0.3, 0.9)


# %% comparing stim response


go_labels  = ['Early','Late','R1','R2']
cmap = ['tab:blue','tab:orange','black','grey']

W2 = {}
for ind in [0,1,2,3]:
    W2[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W2[ind]),1)
    for n in np.arange(np.size(W2[ind],0)):
        if W2[ind][n,max_peak[n]] < 0:
            W2[ind][n,:] = -W2[ind][n,:]

fig, axes = plt.subplots(1,1,figsize = (10,5))
peak = np.zeros((1,4))
for ind in [0,1,2,3]:
        y = np.nanmean(comp2[ind]-comp2[2],0)
        e = np.nanstd(comp2[ind]-comp2[2],0)/np.sqrt(np.size(comp2[ind],0))
        # y = np.nanmean(comp2[ind],0)
        # e = np.nanstd(comp2[ind],0)/np.sqrt(np.size(comp2[ind],0))
        axes.plot(xaxis,y,color = cmap[ind],label = go_labels[ind])
        axes.fill_between(xaxis,y-e,y+e,facecolor = cmap[ind],alpha = 0.3)
        peak[0,ind] = np.max(y)
scat = {}
pcat = {}
for p_ind in [0,1,2]:
        scat[p_ind] = np.zeros((1,pre+post))
        pcat[p_ind] = np.zeros((1,pre+post))
s = {}
pp = {}
for t in np.arange(pre+post):
         # s1,p1 = stats.ks_2samp(W2[0][listind[0],t], W2[1][listind[0],t])
     s[0],pp[0] = stats.ttest_ind(W2[0][:,t], W2[2][:,t],nan_policy = 'omit',alternative = 'greater')
     s[1],pp[1] = stats.ttest_ind(W2[1][:,t], W2[2][:,t],nan_policy = 'omit',alternative = 'greater')
     
     s[2],pp[2] = stats.ttest_ind(W2[2][:,t], W2[3][:,t],nan_policy = 'omit',alternative ='less')
     # s[0],pp[0] = stats.ks_2samp(W2[1][listind[0],t], W2[2][listind[0],t])
     # s[1],pp[1] = stats.ks_2samp(W2[1][listind[0],t], W2[3][listind[0],t])
     # s[2],pp[2] = stats.ks_2samp(W2[2][listind[0],t], W2[3][listind[0],t])
     for p_ind in [0,1,2]:
          pcat[p_ind][0,t] = pp[p_ind]
          if pp[p_ind] < 0.1:
              scat[p_ind][0,t] = True
             
          c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
          # if p_ind == 2:
              # axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'hot',clim = [-3,0])
          if p_ind in [0,1,2]:
              axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.3+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
         


axes.legend()
axes.set_ylim([-0.3,0.5])    

# %%




import pandas as pd

wdin = 10

group = ['Early','Late','R1','R2']
Wa = []
grp = []
for ind in np.arange(4):
    # W2[ind] = np.abs(W2[ind])
    Wa = np.concatenate((Wa,np.mean(W2[ind][:,pre:pre+wdin],1)))
    grp  = np.concatenate((grp,np.repeat(group[ind],np.size(W2[ind],0))),0)


d = {'x': Wa, 'group': grp}

df = pd.DataFrame(d)
fig, axes = plt.subplots(1,1,figsize = (9,7))  
sns.swarmplot(data=df, y="x", x="group",color = 'r')
sns.violinplot(y = df['x'],x = df['group'], hue = df['group'], palette = cmap,fill=False)

Wa2 = [];
grp2 = [];

for ind in np.arange(2):
    Wa2 = np.concatenate((Wa2,np.mean(W2[ind][:,pre:pre+wdin],1)-np.mean(W2[2][:,pre:pre+wdin],1) ))
    grp2  = np.concatenate((grp2,np.repeat(group[ind],np.size(W2[ind],0))),0)


Wa2 = np.concatenate((Wa2,np.mean(W2[2][:,pre:pre+wdin],1)-np.mean(W2[3][:,pre:pre+wdin],1) ))
grp2  = np.concatenate((grp2,np.repeat(group[3],np.size(W2[3],0))),0)

d = {'x': Wa2, 'group': grp2}
df = pd.DataFrame(d)
fig, axes = plt.subplots(1,1,figsize = (5,5))  
sns.violinplot(y = df['x'],x = df['group'],hue = df['group'], palette = ['tab:blue','tab:orange','grey'],
               fill=False,inner_kws=dict(box_width=15,whis_width=2))
sns.swarmplot(data=df, y="x", x="group",color = 'r')
axes.set_ylim([-1,1])

wc1 = np.mean(W2[0][:,pre:pre+wdin],1) 
wc2 = np.mean(W2[1][:,pre:pre+wdin],1) 
wc3 = np.mean(W2[2][:,pre:pre+wdin],1) 
wc4 = np.mean(W2[3][:,pre:pre+wdin],1) 
[t,proba1] = stats.ttest_1samp(wc1-wc3,0)
[t,proba2] = stats.ttest_1samp(wc2-wc3,0)
[t,proba3] = stats.ttest_1samp(wc4-wc3,0)
print(proba1,proba2,proba3)
# stats.ttest_1samp(wc1-wc3,0)          
# stats.ks_1samp(wc1-wc3, stats.norm.cdf, alternative='lesser')

# %% correlation 
lick_corr = np.zeros((1,len(comp)))
for n in np.arange(len(comp)):
    nn= int(listOv[p,f][n])
    y1 = ndimage.gaussian_filter1d(np.reshape(comp_n[0][nn],(1,-1)),5)
    e1 = {}
    lc = np.zeros((1,10))
    for li in np.arange(10):
        e1[li] = np.reshape(L[nn],(1,-1))
        if li > 0:
            e1[li] = np.concatenate((np.zeros((1,li))[0,:],e1[li][0,0:-li]))    
        lc[0,li] = np.corrcoef(y1,e1[li])[1,0]
    lick_corr[0,n] = np.max(lc)        
# lick_corr_IC = lick_corr

# lick_corr_AC = lick_corr

# grp = np.concatenate((np.zeros_like(lick_corr_AC),np.ones_like(lick_corr_IC)),1)[0,:]
# Wa = np.concatenate((lick_corr_AC,lick_corr_IC),1)[0,:]

# d = {'x': Wa, 'group': grp}
# df = pd.DataFrame(d)
# fig, axes = plt.subplots(1,1,figsize = (5,5))  
# sns.violinplot(y = df['x'],x = df['group'],hue = df['group'], palette = ['tab:blue','tab:orange'],
#                fill=False,inner_kws=dict(box_width=15,whis_width=2))
# sns.swarmplot(data=df, y="x", x="group",color = 'black',size  = 3)
# axes.set_ylim([-.35,.8])
# stats.ttest_ind(lick_corr_AC.T,lick_corr_IC.T)

# %% PCA on individual Subspace overlap calculation 


max_k = 20;
# fig, axs = plt.subplots(4,6,figsize = (5,6))

# W3[2] = W4[2]
# W3[3] = W4[3]
sm = 0
R = {}
t1 = 60
t2 = 80
n_cv = 20   
k = 15
def calc_subspace(t1,t2,sm):
    pca = {} 
    for g in  np.arange(len(W4)):
        pca[g] = PCA(n_components=15)
        R[g] = W4[g][:,t1:t2].T
        test = pca[g].fit_transform(ndimage.gaussian_filter(R[g],[sm,0]))        
        test = test.T
    # for t in range(0,5):
    #     axs[g,t].plot(test[t,:], linewidth = 2)
    # axs[g,5].plot(np.cumsum(pca[g].explained_variance_ratio_), linewidth = 4)
    
    
    Overlap = np.zeros((len(W4),len(W4),n_cv)); # PPC_IC
    # Overlap_across = np.zeros((trmax,trmax,n_cv));
    total_n = np.size(W4[0],0)
    
    k1 = k
    k2 = 19
    
    for cv in np.arange(n_cv):
        n_list = np.random.choice(np.arange(total_n),int(np.floor(total_n*0.80)),replace = False)    
        U = {}
        for g in  np.arange(len(W4)):
            pca[g] = PCA(n_components=15)
            test = pca[g].fit_transform(ndimage.gaussian_filter(R[g][:,n_list],[1,0]))  
            
        
        for g1 in np.arange(len(W4)): #np.arange(trmax):
           for g2 in np.arange(len(W4)): # np.arange(trmax):
               S_value = np.zeros((1,k1))
               for d in np.arange(0,k1):
                   S_value[0,d] = np.abs(np.dot(pca[g1].components_[d,:], pca[g2].components_[d,:].T))
                   S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[g1].components_[d,:])*np.linalg.norm(pca[g2].components_[d,:]))
               Overlap[g1,g2,cv] = np.max(S_value)
    return pca, Overlap


d_overlap = np.zeros((n_cv,7))
p = 0
for t in np.arange(7):
    pca, O = calc_subspace(t*10,t*10+20,5)
    d_overlap[:,p] = O[0,1,:]
    p +=1

d2 = d_overlap*1
    
fig, axes = plt.subplots(1,1,figsize = (10,5))
axes.plot(np.arange(7),np.mean(d2,0))
axes.fill_between(np.arange(7),np.mean(d2,0)-np.std(d2,0),np.mean(d2,0)+np.std(d2,0),alpha = 0.3)
axes.set_ylim([0.1,0.8])
# import pandas as pd
# for k in np.arange(n_cv):
# #     Overlap[2,3,k] = (Overlap[2,3,k] + Overlap[1,3,k])/2
# #     Overlap[3,2,k] = (Overlap[2,3,k])
#     Overlap[0,2,k] = 0.329
#     Overlap[2,0,k] = 0.329

# savemat('D:\DATA\Overlap_' + fname2 + '.mat',{"O" :Overlap})

pca, Overlap = calc_subspace(20, 80, 5)
# fig, axes = plt.subplots(1,1,figsize = (10,10))
np.mean(Overlap[0,1,:])

# imshowobj = axes.imshow(np.mean(Overlap,2),cmap = "hot_r")
# # imshowobj = axes.imshow(Overlap[:,:,0],cmap = "hot_r")
# imshowobj.set_clim(0.15, 1) #correct
# plt.colorbar(imshowobj) #adjusts scale to value range, looks OK


# stats.normaltest(Overlap[0,1,:])
# sh_mean = np.mean(Overlap[5,:-1,:],0)

# # sh_mean = np.reshape(sh_mean,(1,-1))

# stats.ks_2samp(Overlap[1,2,:],sh_mean)

# stats.ttest_ind(Overlap[0,1,:],sh_mean)
# O = np.concatenate((Overlap[0,2,:],Overlap[0,3,:]))
# O = np.concatenate((O,Overlap[2,3,:]))
# O = O
# G = np.concatenate((np.zeros((1,n_cv)),np.ones((1,n_cv))),1)
# G = np.concatenate((G,np.ones((1,n_cv))*2),1)
                   

# d = {'x': O, 'group': G[0]}
# df = pd.DataFrame(d)
# fig, axes = plt.subplots(1,1,figsize = (3,5))
# sns.barplot(df, x="group", y="x")
# axes.set_ylim([0.,1])

# for ind in [0,1,2]:
#     print(np.mean(O[G[0,:]==ind]), np.std(O[G[0,:]==ind]))
# fig, axes = plt.subplots(1,1,figsize = (10,10))        
# imshowobj = axes.imshow(np.mean(Overlap,2),cmap = "hot_r")
# imshowobj.set_clim(0.5, 1) #correct
# plt.colorbar(imshowobj) #adjusts scale to value range, looks OK




# %% PCA and trajectory space with grouped units

sm = 1
# R =  np.empty( shape=(160,0) )
R =  np.empty( shape=(0,np.size(W3[1],0)) )
label = np.empty( shape=(2,0) )
# tr_max = 8
for ind in [1,2,3]:
            R = np.concatenate((R,ndimage.gaussian_filter(W3[ind].T,[sm,0])),0)
            # R = np.concatenate((R,zscore(D[ind,tr][d_list2,:].T,axis = 1)),0)
            lbl = np.concatenate((np.ones((1,np.size(W3[1],0)))*ind,np.ones((1,np.size(W3[1],0)))*ind),0)
            label = np.concatenate((label,lbl),1)



# RT = np.dot(R,raster_model.Usv)

pca = PCA(n_components=64)
test = pca.fit_transform(R)    







# %% draw trajectories


# from mpl_toolkits import mplot3d
# from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm

cmap = ['tab:orange','tab:red','black','grey']
def draw_traj4(traj,v):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    # styles = ['solid','solid']
    # cmap_names = ['autumn','winter','winter']
    for ind in [1,2,3]:
            x = traj[ind][:,0]
            y = traj[ind][:,1]
            z = traj[ind][:,2]
            
            x = ndimage.gaussian_filter(x,1)
            y = ndimage.gaussian_filter(y,1)
            z = ndimage.gaussian_filter(z,1)            
                
            time = np.arange(len(x))
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(segments, color = cmap[ind])#linestyle = styles[tr])

    
            lc.set_array(time)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "black")
            # if tr == 0:
            #     ax.auto_scale_xyz(x,y,z)
            ax.auto_scale_xyz(x,y,z)
# %% 
traj = {}
R = {}

trmax = 7
t_len = np.size(W3[1],1)

m = 0 
sm = 5
for ind in [1,2,3]:
        traj[ind] = ndimage.gaussian_filter(test[m*t_len:(m+1)*t_len,:],[sm,0])
        traj[ind] = traj[ind]- np.mean(traj[ind][10:30,:],0)
        m += 1


draw_traj4(traj,0)


# %%
comp2= {};
comp2[0] = np.zeros((len(comp),80))
comp2[1] = np.zeros((len(comp),80))                    
comp2[2] = np.zeros((len(comp),80))
comp2[3] = np.zeros((len(comp),80))   

s_ind = 1
for n in np.arange(len(comp)):
    nn= int(listOv[p,f][n])
    l = int(np.floor(len(comp[nn])/2))
    l2 = int(np.floor(len(comp[nn])/4))
    maxc = np.percentile(np.mean(comp_n[nn],0),90)
    # maxc = np.max(np.mean(comp[nn],0))
    minc = np.percentile(np.mean(comp_n[nn],0),10)
    
    # comp2[0][n,:] = (np.mean(comp[nn][0:l,:],0)-minc)/(maxc-minc+1)
    # comp2[1][n,:] = (np.mean(comp[nn][l:,:],0)-minc)/(maxc-minc+1)
    if f in [3,5,7]: #[3,5,7]:
        # if n not in [7,15]:
            comp2[0][n,:] = (np.mean(comp[nn][XFA[nn],:],0)-minc)/(maxc-minc+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][XCR[nn],:],0)-minc)/(maxc-minc+s_ind)
    else:
        # if n not in [7,15]:
            comp2[0][n,:] = (np.mean(comp[nn][0:l,:],0)-minc)/(maxc-minc+s_ind)
            comp2[1][n,:] = (np.mean(comp[nn][l:,:],0)-minc)/(maxc-minc+s_ind)


    # 
    maxc = np.percentile(np.mean(comp_n_r1[nn],0),90)
    # maxc = np.max(np.mean(comp_r1[nn],0))
    minc = np.percentile(np.mean(comp_n_r1[nn],0),20)

    comp2[2][n,:] = (np.mean(comp_r1[nn],0)-minc)/(maxc-minc+s_ind)
    
    maxc = np.percentile(np.mean(comp_r2[nn],0),90)
    # maxc = np.max(np.mean(comp_r2[nn],0))
    minc = np.percentile(np.mean(comp_r2[nn],0),10)

    comp2[3][n,:] = (np.mean(comp_r2[nn][:,:],0)-minc)/(maxc-minc+1)
    
    # comp2[0][n,:] = np.mean(comp[int(listOv[p,f][n])][0:l,:],0)/(np.max(np.mean(comp[int(listOv[p,f][n])],0))+1)
    # comp2[1][n,:] = np.mean(comp[int(listOv[p,f][n])][l:,:],0)/(np.max(np.mean(comp[int(listOv[p,f][n])],0))+1)

# for ind in [0,1,2,3]:
#     fig, axes = plt.subplots(1,1,figsize = (10,10))
#     im1 = axes.imshow(comp2[ind],clim = [0,1], aspect = "auto", interpolation = "None",cmap = "viridis")

# if f == 2:
#     listind = np.zeros((1,len(comp)))
#     for c in np.arange(len(comp)):
#         if np.sum(listOv[p,3] == listOv[p,2][[c]])  ==0:
#             listind[0,c] = True 
#     listind = (listind == 1)  

if f == 0:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,2] == listOv[p,3][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1) 


if f == 7:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,5] == listOv[p,7][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)  
    
elif f == 5:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,7] == listOv[p,5][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)  
elif f == 4:
    listind = np.zeros((1,len(comp)))
    for c in np.arange(len(comp)):
        if np.sum(listOv[p,2] == listOv[p,4][[c]])  ==0:
            listind[0,c] = True 
    listind = (listind == 1)        
        

W2 = {}
for ind in [0,1,2,3]:
    W2[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W2[ind]),1)
    for n in np.arange(np.size(W2[ind],0)):
        if W2[ind][n,max_peak[n]] < 0:
            W2[ind][n,:] = -W2[ind][n,:]
                    

# listind[0,15] = False
# listind[0,40] = False
# fig, axes = plt.subplots(1,1,figsize = (5,5))        
# im1 = axes.imshow(W2[2][:,:],clim = [0,1], aspect = "auto", interpolation = "None",cmap = "viridis")

# # rastermpa
# from rastermap import Rastermap, utils
# from scipy.stats import zscore

# W3 = {}
# for ind in [0,1,2]:
#     W3[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,3],mode = "mirror")
                              
    
# ind = 1
# # fit rastermap
# # note that D_r is already normalized
# model = Rastermap(n_PCs=64,
#                   locality=0.75,
#                   time_lag_window=5,
#                   n_clusters = 20,
#                   grid_upsample=5, keep_norm_X = False).fit(W3[ind])
# y = model.embedding # neurons x 1
# isort = model.isort

# # bin over neurons
# # X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)

# for ind in [0,1]:
#     fig, ax = plt.subplots(figsize = (5,5))
#         # ax.imshow(zscore(W2[ind][isort, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2, aspect="auto")
#     ax.imshow(zscore(W3[ind][isort, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2,aspect = "auto")
#         # ax.imshow(W3[ind][isort, :], cmap="gray_r", aspect="auto")

# # rastermpa end

W = {};
if f in [3,5,7]:
    go_labels  = ['FA','CR','R1','R2']
    cmap = ['tab:orange','tab:green','black','grey']
else:
    go_labels  = ['Early','Late','R1','R2']
    cmap = ['tab:blue','tab:orange','black','grey']

if f in [3, 4, 7]: #[4,5,7]:
    
    
    for ind in [0,1,2,3]:
        W[ind] = ndimage.uniform_filter(W2[ind][listind[0],:],[0,2], mode = "mirror")
        # W[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W[0]),1)
    max_ind = max_peak.argsort()    
    for ind in [0,1,2,3]:
        fig, axes = plt.subplots(1,1,figsize = (5,5))        
        im1 = axes.imshow(W[ind][max_ind,:],clim = [0,1], aspect = "auto", interpolation = "None",cmap = "viridis")\
        # im1 = axes.imshow(zscore(W[ind][max_ind, :],axis = 1), cmap="gray_r", vmin=0, vmax=1.2,aspect = "auto")

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
    
    fig, axes = plt.subplots(1,1,figsize = (10,5))
    peak = np.zeros((1,4))
    for ind in [1,2,3]:
        # y = np.mean(comp2[ind][listind[0],:],0)
        # e = np.std(comp2[ind][listind[0],:],0)/np.sqrt(np.size(comp2[ind],0))
        y = np.nanmean(W2[ind][listind[0],:],0)
        e = np.nanstd(W2[ind][listind[0],:],0)/np.sqrt(np.size(W2[ind],0))
        axes.plot(xaxis,y,color= cmap[ind],label = go_labels[ind])
        axes.fill_between(xaxis,y-e,y+e,facecolor= cmap[ind],alpha = 0.3)
        peak[0,ind] = np.max(y)
    scat = {}
    pcat = {}
    for p_ind in [1,2,3]:
        scat[p_ind] = np.zeros((1,80))
        pcat[p_ind] = np.zeros((1,80))
    s = {}
    pp = {}
    
    for t in np.arange(80):
            # s1,p1 = stats.ks_2samp(W2[0][listind[0],t], W2[1][listind[0],t])
        s[0],pp[0] = stats.ttest_ind(W2[0][listind[0],t], W2[1][listind[0],t])
        s[1],pp[1] = stats.ttest_ind(W2[0][listind[0],t], W2[2][listind[0],t])
        s[2],pp[2] = stats.ttest_ind(W2[1][listind[0],t], W2[2][listind[0],t])
        for p_ind in [1,2,3]:
            if pp[p_ind] < 0.05:
                scat[p_ind][0,t] = True
                pcat[p_ind][0,t] = pp[p_ind]
            c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
            if p_ind == 0:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'hot',clim = [-3,0])
            else:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])
    axes.legend()    
    # axes.set_ylim([-0.1,1.0])    
else:
    
    
    fig, axes = plt.subplots(1,1,figsize = (10,5))
    peak = np.zeros((1,3))
    for ind in [0,1,2]:
        y = np.nanmean(comp2[ind],0)
        e = np.nanstd(comp2[ind],0)/np.sqrt(np.size(comp2[ind],0))
        axes.plot(xaxis,y,color = cmap[ind],label = go_labels[ind])
        axes.fill_between(xaxis,y-e,y+e,facecolor = cmap[ind],alpha = 0.3)
        peak[0,ind] = np.max(y)

    scat = {}
    pcat = {}
    for p_ind in [0,1,2]:
        scat[p_ind] = np.zeros((1,80))
        pcat[p_ind] = np.zeros((1,80))
    s = {}
    pp = {}
    for t in np.arange(80):
        s[0],pp[0] = stats.ttest_ind(comp2[0][:,t], comp2[1][:,t])
        s[1],pp[1] = stats.ttest_ind(comp2[0][:,t], comp2[2][:,t])
        s[2],pp[2] = stats.ttest_ind(comp2[1][:,t], comp2[2][:,t])
        # s1,p1 = stats.ks_2samp(comp2[0][:,t], comp2[1][:,t])
        for p_ind in [0,1,2]:
            if pp[p_ind] < 0.05:
                scat[p_ind][0,t] = True
                pcat[p_ind][0,t] = pp[p_ind]
            c1 = pcat[p_ind][0,scat[p_ind][0,:]>0]
            if p_ind == 0:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'hot',clim = [-3,0])
            else:
                axes.scatter(xaxis[scat[p_ind][0,:]>0],np.ones_like(xaxis[scat[p_ind][0,:]>0])*np.max(peak)*1.1+0.05*p_ind,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,0])

    axes.legend()
    axes.set_ylim([-0.1,1.0])    

    for ind in [0,1,2]:
        W[ind] = ndimage.uniform_filter(W2[ind],[0,2], mode = "mirror")
        # W[ind] = ndimage.uniform_filter(comp2[ind],[0,2], mode = "mirror")
    max_peak = np.argmax(np.abs(W[1]),1)
    max_ind = max_peak.argsort()    
    for ind in [0,1,2]:
        fig, axes = plt.subplots(1,1,figsize = (5,5))        
        im1 = axes.imshow(W[ind][max_ind,:],clim = [0,1], aspect = "auto", interpolation = "None",cmap = "viridis")
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im1, cax=cbar_ax)


# comp2 = np.zeros(((2,len(comp))))
# for n in np.arange(len(comp)):
#     l = np.floor(len(comp[int(listOv[p,f][n])])/2)
#     l2 = len(comp[int(listOv[p,f][n])])- l
#     l  =int(l)
#     l2 = int(l2)
#     comp2[0,n] = np.mean(comp[int(listOv[p,f][n])][0:l])/(np.max(comp[int(listOv[p,f][n])])+1)
#     comp2[1,n] = np.mean(comp[int(listOv[p,f][n])][l:])/(np.max(comp[int(listOv[p,f][n])])+1)
# fig, axes = plt.subplots(1,1,figsize = (10,5))
# for n in np.arange(len(comp)):
#     axes.plot(comp2)

# np.mean(comp2[1,:])

 


# def getY():
#     Y = np.zeros((len(good_list2),80))
#     nn = 0
#     for n in good_list2:
#         h = Data[n,c_ind-1]["Y"]
#         stim_onset = Data[n, c_ind-1]["stim_onset"]
#         Yn= np.zeros((len(stim_onset),80))
#         for t in np.arange(len(stim_onset)):
#             Yn[t,:] = h[0,stim_onset[t]-20:stim_onset[t]+60]
#         Y[nn,:] = np.mean(Yn,0)
#         nn  += 1
#     return Y


# fig, axes = plt.subplots(1,1,figsize = (10,5))
# axes.plot(np.mean(Y[:99,:],0))


    
# %% scatter plot weights, reward and ITI
# ind = 1
f = 5
# W5[0,4][0][-4:,:] = 0
B = {}
for ind in [0,1]:
    B[ind] = np.zeros((3,np.size(W5[ind,f][0],0)))
    for n in np.arange(np.size(W5[ind,f][0],0)):
        B[ind][0,n] = np.mean(W5[ind,f][0][n,20:40])
        B[ind][1,n] = np.mean(W5[ind,f][0][n,40:60])
        B[ind][2,n] = np.mean(W5[ind,f][0][n,60:])

    fig, ax = plt.subplots(1,1, figsize = (5,5))
    ax.vlines(0,-1,1,'k')    
    ax.hlines(0,-1,1,'k') 
    ax.scatter(B[ind][0,:],B[ind][1,:])
    ax.set_ylim([-0.9,0.9])
    ax.set_xlim([-0.9,0.9])
    fig, ax = plt.subplots(1,1, figsize = (5,5))
    ax.vlines(0,-1,1,'k')    
    ax.hlines(0,-1,1,'k') 
    ax.scatter(B[ind][1,:],B[ind][2,:])
    ax.set_ylim([-0.9,0.9])
    ax.set_xlim([-0.9,0.9])
    
    # np.sum(B[0,:]>B[1,:])
    

# B = np.abs(B)
cmap = ['tab:blue','tab:orange','tab:green']
hatches = ['/','']
fig, ax = plt.subplots(2,1, figsize = (6,10))
for ind in [0,1]:
    for p in [0,1,2]:
        for b_ind in [0]:
            B1 = B[ind][p]
            # if b_ind == 0:
            #     B1 = B1[B1>0]
            # else:
            #     B1 = B1[B1<0]
            ax[b_ind].bar([3*p+ind],np.mean(B1),width = 0.5,hatch = hatches[ind],color = cmap[p])
            ax[b_ind].errorbar([3*p+ind],np.mean(B1),np.std(B1)/np.sqrt(np.size(B1)),color="black")
        
            ax[b_ind].set_xlim([-1,8])
            if b_ind == 0:
                ax[b_ind].set_ylim([0,0.3])
            else:
                ax[b_ind].set_ylim([-0.3,0.0])
 

for ind in [0,1]:
    max_ind = np.argmax(B[ind],0)
    B2 = np.histogram(max_ind,bins = 3)
    B2 = B2[0]
    print(str(B2))
    for p in [0,1,2]:
        ax[1].bar([3*p+ind],B2[p]/np.sum(B2),width = 0.5,hatch = hatches[ind],color = cmap[p])
    ax[1].set_xlim([-1,8])
    ax[1].set_ylim([-0.0,0.8])

# %% plot weights positive and negative

for f in [4]: #np.arange(ax_sz):
    fig, axes = plt.subplots(2,1,figsize = (10,10),sharex = "all")
    fig.subplots_adjust(hspace=0)
    for ind in [0,1]:
        if ind == 0:
            y1 = ndimage.gaussian_filter1d(np.mean(W5IC[f],0),1)
            e1 = np.std(W5IC[f],0)/np.size(W5IC[f],0)
        elif ind ==1:
            y1 = ndimage.gaussian_filter1d(np.mean(W5AC[f],0),1)
            e1 = np.std(W5AC[f],0)/np.size(W5AC[f],0)
        axes[ind].plot(xaxis,y1,c = cmap[f],linestyle = 'solid', linewidth = 3)
        axes[ind].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)

   
# %% faction of cells for GLM fit

b1 = np.array([98,85,84,79])/(110)
b2 = np.array([204,173,164,153])/(284)
fig, axes = plt.subplots(1,1,figsize = (5,5), sharex = True)
# axes.grid(visible=True, axis = 'y')
axes.bar(np.arange(4)*3,b1,color='white',edgecolor = 'k', width = 0.5, linewidth = 2)
axes.bar(np.arange(4)*3+0.7,b2,color='black',edgecolor = 'k', width = 0.5, linewidth = 2)
axes.set_ylim([0,1])




# %% convert and save Convdata
C_data = {}
for f in np.arange(ax_sz):
    C_data[f] = np.zeros((788,80))
    p = 0
    for n in good_list2:
        nn = int(n)
        C_data[f][nn,:] = Convdata[f][p,:]
        p+=1

# fig, axes = plt.subplots(1,1,figsize = (10,10))
# axes.plot(np.mean(C_data[4],0)) 

# %% save listOv
np.save('C_data_R2.npy', C_data,allow_pickle= True)     

np.save('nlist_R2.npy', listOv,allow_pickle= True)     
# %% analysis with overlapping neurons 

nlist =  {};
nlist[0] = np.load('nlist_R1.npy',allow_pickle = True).item()
nlist[1] = np.load('nlist_RT.npy',allow_pickle = True).item()
nlist[2] = np.load('nlist_R2.npy',allow_pickle = True).item()

CD = {};
CD[0] = np.load('C_data_R1.npy',allow_pickle= True).item()  
CD[1] = np.load('C_data_RT.npy',allow_pickle= True).item()  
CD[2] = np.load('C_data_R2.npy',allow_pickle= True).item()  

# %%
W1 = {};
W2 = {};
W3 = {};
W4 = {};


# %%
ind = 0
f = 4
# int_list = np.intersect1d(nlist[0][ind,f],nlist[1][ind,f])
# int_list2 = np.setxor1d(nlist[2][ind,f], int_list)
# list0 = []
# for n in int_list:
#     nn = int(n)
#     list0 = np.concatenate((list0,np.where(good_list2 == nn)[0]))

# list0 = list0.astype(int)


list0 = np.concatenate((nlist[0][ind,f],nlist[1][ind,f],nlist[2][ind,f]),0)
list0 = np.unique(list0).astype(int)


# W = ndimage.uniform_filter(Convdata[f][list0,:],[0,1], mode = "mirror")
# W4 = {};

good_list = np.arange(np.size(D_ppc,0))
# p = 2
for p in [0]:
    if ind == 0:
        d_list = good_list < 193
    elif ind == 1:
        d_list = good_list > 194
    
    W = CD[p][f][d_list,:]
    # W = W[(np.mean(W,1) !=0),:]    
    # W = W/int(np.floor(w_length1[f]/10)+1)
    
    
    max_peak = np.argmax(np.abs(W),1)
    max_ind = max_peak.argsort()
    
    
    list1 = []
    list2 = []
    list3 = []
    
    SD = np.std(W[:,:])
    for m in np.arange(np.size(W,0)):
        n = max_ind[m]
        # SD = np.std(W[n,:])
        # if SD< 0.05:
        #     SD = 0.05
        if max_peak[n]> 0:    
            if W[n,max_peak[n]] > 2*SD:
                list1.append(m)
                list3.append(m)
            elif W[n,max_peak[n]] <-2*SD:
                list2.append(m)
                list3.append(m)
            
    max_ind1 = max_ind[list1]  
    max_ind2 = max_ind[list2]     
    max_ind3 = max_ind[list3]
    
    
    W1[p] = W[max_ind1]
    W2[p] = W[max_ind2]    
    W4[p] = np.abs(W[max_ind3])
    W3[p] = np.concatenate((W1[p],-W2[p]), axis = 0)
    clim = [-0.7, 0.7]
    fig, axes = plt.subplots(1,1,figsize = (10,10))
    im1 = axes.imshow(W3[p][:,:], clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im1, cax=cbar_ax)


fig, axes = plt.subplots(1,1,figsize = (10,10))

lstyles = ['solid','dotted','dashed']
for p in [0,1,2]:
    y1 =np.mean(W3[p],0)
    e1 = np.std(W3[p],0)/np.sqrt(np.size(W3[p],0))
    axes.plot(xaxis,np.mean(W3[p],0),linestyle = lstyles[p]) 
    axes.fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
  # re plot weights by list1 and list2

max_ind4 = np.concatenate((max_ind1,max_ind2))
max_ind4 = max_ind3
for ind in [0,1,2]:
    W0 = CD[ind][f][d_list,:]
    W = ndimage.uniform_filter(W0[max_ind4,:],[0,3], mode = "mirror")
    W = np.abs(W)
    print(np.sum(np.mean(W,1)>0))

    clim = [-0.1, 0.7]
    fig, axes = plt.subplots(1,1,figsize = (10,10))
    im1 = axes.imshow(W, clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im1, cax=cbar_ax)
    
# %% 

fig, axes  = plt.subplots(1,1, figsize = (10,10))
lstyles = ['solid','dotted','dashed']
for p in [0,1,2]:
    y1 = np.mean(W3[p],0)
    y1 = ndimage.gaussian_filter1d(y1,2)
    e1 = np.std(W3[p],0)/np.sqrt(np.size(W3[p],0))
    axes.plot(xaxis,y1,linestyle = lstyles[p])
    axes.fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)



# %% plot positive and negative weights separately.
STD = {}
STD = 0.14626
# STD = 0.11 

fig, axes = plt.subplots(2,1,figsize = (10,10),sharex = "all")
fig.subplots_adjust(hspace=0)
for p in [0,1,2]:
    for pp in [0,1]:
        if pp == 0:
            y1 = np.sum(W1[p],0)
            e1 = np.std(W1[p],0)/np.sqrt(np.size(W1[p],0))
            # for t in np.arange(np.size(y1)):               
            #     s,prob = stats.ttest_1samp(W1[p][:,t],STD,alternative = 'greater')
            #     if prob<0.05:
            #         axes[pp].scatter([(t*window-prestim)*1e-3],[0.30],marker = '*', color = 'black')
        elif pp ==1:
            y1 = np.sum(W2[p],0)
            e1 = np.std(W2[p],0)/np.sqrt(np.size(W2[p],0))
            # for t in np.arange(np.size(y1)):               
            #     s,prob = stats.ttest_1samp(W2[p][:,t],STD,alternative = 'less')
            #     if prob<0.05:
            #         axes[pp].scatter([(t*window-prestim)*1e-3],[0.30],marker = '*', color = 'black')
        y1 = ndimage.gaussian_filter1d(y1,2)/np.size(W4[p],0)
        axes[pp].plot(xaxis,y1,linestyle = lstyles[p])
        axes[pp].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)

axes[0].set_ylim([-0.05,0.4])
axes[1].set_ylim([-0.4,0.05])

p= 0
test1 = {};
fig, axes = plt.subplots(1,1,figsize = (5,5))
for p in [0,1,2]:
    test1[p] = np.mean(W1[p][:,45:60],1)
    axes.bar([p],np.mean(test1[p]))
    axes.errorbar([p],np.mean(test1[p]),np.std(test1[p])/np.sqrt(np.size(test1[p])),color="black")


axes.set_ylim([0,0.4])


s,p = stats.ks_2samp(test1[1], test1[0])

Cr ={};
# %%

for ind in[0,1]:
    tmpl = nlist[0][ind,2]
    nlist[0][ind,2] = nlist[0][ind,3]
    nlist[0][ind,3] = tmpl

# %% overlap in n_list

ind = 0
f = 2





templist = np.intersect1d(nlist[0][ind,f],nlist[1][ind,f])

test = [len(np.intersect1d(templist,nlist[2][ind,f])),
 len(np.intersect1d(nlist[0][ind,f],nlist[1][ind,f]))-len(np.intersect1d(templist,nlist[2][ind,f])),
 len(np.intersect1d(nlist[1][ind,f],nlist[2][ind,f]))-len(np.intersect1d(templist,nlist[2][ind,f])),
 len(np.intersect1d(nlist[0][ind,f],nlist[2][ind,f]))-len(np.intersect1d(templist,nlist[2][ind,f])),
 len(nlist[0][ind,f])+len(np.intersect1d(templist,nlist[2][ind,f]))-len(np.intersect1d(nlist[0][ind,f],nlist[1][ind,f]))-len(np.intersect1d(nlist[0][ind,f],nlist[2][ind,f])),
 len(nlist[1][ind,f])+len(np.intersect1d(templist,nlist[2][ind,f]))-len(np.intersect1d(nlist[0][ind,f],nlist[1][ind,f]))-len(np.intersect1d(nlist[1][ind,f],nlist[2][ind,f])),
 len(nlist[2][ind,f])+len(np.intersect1d(templist,nlist[2][ind,f]))-len(np.intersect1d(nlist[2][ind,f],nlist[1][ind,f]))-len(np.intersect1d(nlist[0][ind,f],nlist[2][ind,f])),
 ]


#%% correlation and similarities between TV encoding R1, RT, R2 240712
ind = 1
f = 5

import scipy.io as sio


list0 = np.concatenate((nlist[0][ind,f],nlist[1][ind,f],nlist[2][ind,f]),0)
list0 = np.unique(list0).astype(int)

sio.savemat('saved_struct.mat',{'CD0':CD[0][f],'CD1' :CD[1][f],'CD2' : CD[2][f], 'list0' : list0} )


W = {}
for p in [0,1,2]:
    W[p] = CD[p][f][list0,:]
    
wbin = 20
# 
# fig, axes = plt.subplots(3,2,figsize = (10,15))
# axes[0,t].scatter(np.mean(W[0][:,wbin:wbin+40],1),np.mean(W[1][:,wbin:wbin+40],1))
# axes[1,t].scatter(np.mean(W[1][:,wbin:wbin+40],1),np.mean(W[2][:,wbin:wbin+40],1))
# axes[2,t].scatter(np.mean(W[0][:,wbin:wbin+40],1),np.mean(W[2][:,wbin:wbin+40],1))
   


x = np.mean(W[0][:,wbin:wbin+40],1)
y = np.mean(W[2][:,wbin:wbin+40],1)
x = x.reshape(-1,1)
y = y.reshape(-1,1)
from sklearn import linear_model
ss= ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)

regr = linear_model.LinearRegression()

cv_results = cross_validate(regr,x,y, cv = ss , 
                                    return_estimator = True, 
                                    scoring = 'r2') 

lin_reg_score = cv_results['test_score']

reg = regr.fit(y, x)# print(reg.score(y,x))

# %%
# 
Cr[ind,f] = np.zeros((np.size(W[0],0),1))
for n in np.arange(np.size(W[0],0)):
    Cr[ind,f][n,0] = np.corrcoef(W[0][n,:],W[2][n,:])[1,0] 
    if np.isnan(np.corrcoef(W[0][n,:],W[2][n,:])[1,0]):
        Cr[ind,f][n,0] = 0;


s,p = stats.ks_2samp(Cr[0,f][:,0], Cr[1,f][:,0])

np.mean(Cr[1,f][:,0])
edges = np.arange(0,1,0.1)

fig, axes = plt.subplots(1,1,figsize = (5,5))
axes.hist(Cr[0,f],edges,density = True)
axes.set_ylim([0,6])