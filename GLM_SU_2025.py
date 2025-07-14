# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:10:08 2025

Update log 
20/01/2025
Currently the data format is the following:
    1. spike times(s)
    2. lick times (s)
    3. event times (s)
    4. outcome index : Hit, Miss, CR, FA
    5. trial id: 
        1 - R1 5kHz
        2 - R1 10kHz
        3 - R2 5kHz
        4 - R2 10kHz
        5 - R2-tr-R1 5kHz
        6 - R2-tr-R1 10kHz
        7 - R1-tr-R2 5kHz
        8 - R1-tr-R2 10kHz
    6. channel site number



31/01/2025
Attempt to integrate rule into GLM has failed. Although the model could capture general rule effects such as increase in FR,
It could not emulate rule specific outcome related encoding

Decided to separate Rule and run GLM 
@author: Jong Hoon Lee
"""

# import packages 

import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat,savemat
from scipy import ndimage, stats, sparse
from sklearn.linear_model import TweedieRegressor, Ridge, ElasticNet, Lasso, PoissonRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA, SparsePCA
import seaborn as sns
from os.path import join as pjoin
import statsmodels.api as sm
from sklearn import preprocessing

# %%

fname = 'SUdata_IC.mat'

fdir = 'D:\Python\Data'




np.seterr(divide = 'ignore') 
def load_matfile(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_dataset']
    return D_ppc 

def load_matfile_Ca(dataname = pjoin(fdir,fname)):
    
    MATfile = loadmat(dataname)
    D_ppc = MATfile['GLM_CaData']
    return D_ppc 

def find_good_data():
    D_ppc = load_matfile()
    good_list = []
    for n in range(np.size(D_ppc,0)):
        timepoint = int(max(D_ppc[n,2][:,0])*1e3)+t_period+100;
        bins = np.arange(0,timepoint,window)
        data =D_ppc[n,0][:,0]*1e3    
        [S_all,bins] = np.histogram(data,bins)                
        if np.mean(S_all)*1e3/window>1:
            good_list = np.concatenate((good_list,[n]))
    return good_list

def basis_function(total_length,stim_onset,delay,duration,sigma):  
    if np.size(delay)== 1:
        for d in np.arange(duration):
            gauss_y = sparse.csr_array(np.zeros(total_length))
            for st in stim_onset:
                start = st+delay+d            
                x = np.arange(-10,10)
                gauss_y[:,x+start] = np.exp(-((x) ** 2) / (2 * sigma**2))        
            if d ==0:
                Xvar= gauss_y
            else:
                Xvar= sparse.vstack([Xvar,gauss_y])
    else:
        for d in np.arange(duration):
            gauss_y = sparse.csr_array(np.zeros(total_length))
            for st_ind in np.arange(len(stim_onset)):
                start = stim_onset[st_ind]+delay[st_ind]+d            
                x = np.arange(-10,10)
                gauss_y[:,x+start] = np.exp(-((x) ** 2) / (2 * sigma**2))        
            if d ==0:
                Xvar= gauss_y
            else:
                Xvar= sparse.vstack([Xvar,gauss_y])
    return Xvar

def basis_function2(total_length,lick_time,sigma):
    gauss_y = sparse.csr_array(np.zeros(total_length))
    for st in lick_time:
        start = st    
        x = np.arange(-10,10)
        gauss_y[:,x+start] = gauss_y[:,x+start] + np.exp(-((x) ** 2) / (2 * sigma**2))        
    Xvar= gauss_y

    return Xvar
    
def rule_function(total_length,start,end,stim_onset):
    d = start
    while d  < end:
        gauss_y = sparse.csr_array(np.zeros(total_length))
        gauss_y[:,stim_onset[d]:stim_onset[d+1]] =1
        if d == start:
            Xvar = gauss_y
        else:
            Xvar= sparse.vstack([Xvar,gauss_y])
        d += 1
    return Xvar
        



def build_GLM(D_ppc,n,window,sigma,Xp,r):
    # D_ppc = load_matfile()
    # extract spike and lick times to bins
    
    timepoint = int(max(D_ppc[n,2][:,0])*1e3)+t_period+100;
    ttr = int(D_ppc[n,6][0])
    bins = np.arange(0,timepoint,window)
    data =D_ppc[n,0][:,0]*1e3    
    data_lick = D_ppc[n,1][:,0]*1e3    
    [S_all,bins] = np.histogram(data,bins)
    S_all = np.sqrt(S_all)
    S_all =ndimage.gaussian_filter1d(S_all,sigma)
    [L_all,bins] = np.histogram(data_lick,bins)
    # L_all = np.sqrt(L_all)
    # L_all - ndimage.gaussian_filter1d(L_all,sigma)
    stim_onset = np.floor(D_ppc[n,2][:,0]*(1e3/window)).astype(int)
    lick_time = np.floor(D_ppc[n,1][:,0]*(1e3/window)).astype(int)
    lick_time = lick_time[lick_time<np.size(S_all)]
    # r_onset = np.floor(D_ppc[n,2][:,1]*(1e3/window)).astype(int)


    ### Extract Lick ### 
    # lag window for Lick = -500, +500 ms       
    l_w = int(200/window)
    L_all2 = basis_function2(np.size(S_all),lick_time,1);
    L_all2 = L_all2.toarray()
    L_del = np.zeros((2*l_w,np.size(L_all)))
    for lag in np.arange(1,l_w):
        L_del[lag,lag:] = L_all2[0,:-lag]
    for lag in np.arange(1,l_w):
        L_del[lag+l_w,:-lag] = L_all2[0,lag:]
    L_del[0,:] = L_all2[0,:]
    # for l in np.floor(D_ppc[n,6][:,0]*(1e3/window)):
    #     Rt[0,int(l)-1] = 1  
    # scaler = preprocessing.StandardScaler(with_mean=True).fit(L_del)
    # L_del = scaler.transform(L_del)
    L_del = L_del/np.max(L_del)
    L_del = sparse.csr_matrix(L_del)
    ### Extract Lick End ###
    
    
    ### Extract reward onset ###
    # Since reward is given at the second lick during reward period, we define 
    # reward onset as the second lick after reward period onset (1.5seconds post stim)
    

    reward_time =[]
    for tr in np.arange(np.size(stim_onset)):
        if D_ppc[n,3][tr,0] ==1:
            r_temp = lick_time-stim_onset[tr]-1.5*1e3/window
            if np.sum(r_temp>0)>0: # first lick
                reward_time = np.append(reward_time, lick_time[np.argwhere(r_temp>0)[0]])
    reward_time= reward_time.astype(int)
    
    # reward encoding, post 500ms
    # r_w = int(500/window)
    # R_all2 = basis_function2(np.size(S_all),reward_time,1);
    # R_all2 = R_all2.toarray()
    # R_del = np.zeros((r_w,np.size(S_all)))
    # for lag in np.arange(1,r_w):
    #     R_del[lag,lag:] = R_all2[0,:-lag]
    # R_del[0,:] = R_all2[0,:]
    # R_del = R_del/np.max(R_del)
    # R_del = sparse.csr_matrix(R_del)

    # L_del = L_del.toarray()
    
    
    # Ymat = np.zeros((200,160))
    # for tr in np.arange(200):
    #     Ymat[tr,:] = S_all[stim_onset[tr]-20:stim_onset[tr]+140]
        
    # fig, axes = plt.subplots(1,1,figsize = (5,5))
    # axes.imshow(Ymat)
    
    # fig,axes = plt.subplots(1,1,figsize = (5,5))
    # axes.plot(np.mean(Ymat,0))

    if not Xp:     
        # Create discrete variables
        total_length = np.size(S_all)
        sigma = 1
        T1= D_ppc[n,3] # trialtype by outcome
        T2= D_ppc[n,4] # trialtype by trialtype
        rt_ind = np.where(T2[:,1]>10000)[0]
        for tr in rt_ind:
            if T1[tr,1] ==1:
                T1[tr,0] = 1
                T1[tr,1] = 0
        
        # stim 
        # only during stim
        duration = np.floor(1000/window).astype(int)
        X_5_1  =   basis_function(total_length,stim_onset[np.mod(T2[:,3],2) == 1],0,duration,sigma)
        X_10_1 =   basis_function(total_length,stim_onset[np.mod(T2[:,3],2) == 0],0,duration,sigma)
        
        # stim encoding during delay
        duration = np.floor(500/window).astype(int)
        delay = np.floor(500/window).astype(int)
        X_5    =   basis_function(total_length,stim_onset[np.mod(T2[:,3],2) == 1],delay,duration,sigma)        
        X_10   =   basis_function(total_length,stim_onset[np.mod(T2[:,3],2) == 0],delay,duration,sigma)
        # outcome 
        duration = np.floor(1000/window).astype(int)
        delay = np.floor(500/window).astype(int)
        # delay =0
        X_Hit_1 =  basis_function(total_length,stim_onset[(T1[:,0] == 1)],delay,duration,sigma)
        X_Miss_1 =  basis_function(total_length,stim_onset[(T1[:,1] == 1)],delay,duration,sigma)
        X_CR_1 =  basis_function(total_length,stim_onset[(T1[:,2] == 1)],delay,duration,sigma)
        X_FA_1 =  basis_function(total_length,stim_onset[(T1[:,3] == 1)],delay,duration,sigma)
        
        duration = np.floor(6000/window).astype(int)
        delay = np.floor(1500/window).astype(int)
        # X_Hit =  basis_function(total_length,stim_onset[(T1[:,0] == 1)],r_onset[(T1[:,0] == 1)],duration,sigma)
        X_Hit =  basis_function(total_length,stim_onset[(T1[:,0] == 1)],delay,duration,sigma)
        X_Miss =  basis_function(total_length,stim_onset[(T1[:,1] == 1)],delay,duration,sigma)
        X_CR =  basis_function(total_length,stim_onset[(T1[:,2] == 1)],delay,duration,sigma)
        X_FA =  basis_function(total_length,stim_onset[(T1[:,3] == 1)],delay,duration,sigma)
        
    
        # test= X_Hit.toarray()
        # rule 
        # note : this bit will need to change once we have multiple reversals
        # X_r1 = np.zeros(total_length)
        # X_cs = np.zeros(total_length)
        # X_r2 = np.zeros(total_length)
    
        # X_r1[:stim_onset[199]] = 1
        # X_r2[stim_onset[260]:] = 1
        # X_cs[stim_onset[200]:stim_onset[259]] = 1
        
        # X_r1 = basis_function(total_length,[stim_onset[0]],0,stim_onset[199]-stim_onset[0],sigma)
        # X_r2 = basis_function(total_length,[stim_onset[260]],0,stim_onset[-1]-stim_onset[260],sigma)
        # X_cs = basis_function(total_length,[stim_onset[199]],0,stim_onset[260]-stim_onset[199],sigma)
        
        # X_r1 = rule_function(total_length,0,200,stim_onset)
        # X_r2 = rule_function(total_length,260,np.min([460,len(stim_onset)]),stim_onset)
        # X_cs = rule_function(total_length,200,260,stim_onset)
        
        X = {}
        
        # Licking related parameters
        ED1 = 5 # 500ms pre, 1second post lag
        ED2 = 10

        
        # X3_Lick_onset = np.zeros((ED1+ED2+1,np.size(L_all,0)))
        # X3_Lick_offset = np.zeros_like(X3_Lick_onset)
        
        # X3_Lick_onset[0,:] = L_all_onset
        # X3_Lick_offset[0,:] = L_all_offset

        # for lag in np.arange(ED1):
        #     X3_Lick_onset[lag+1,:-lag-1] = L_all_onset[lag+1:]
        #     X3_Lick_offset[lag+1,:-lag-1] = L_all_offset[lag+1:]
        
        # for lag in np.arange(ED2):
        #     X3_Lick_onset[lag+ED1+1,lag+1:] = L_all_onset[:-lag-1]
        #     X3_Lick_offset[lag+ED1+1,lag+1:] = L_all_offset[:-lag-1]
        
        
        # X[0]= np.reshape(L_all,(1,-1))
        X[0]= L_del
        X[1] = X_5_1
        X[2] = X_10_1
        X[3] = X_Hit
        X[4] = X_Miss
        X[5] = X_CR
        X[6] = X_FA
        # X[7] = R_del
        # X[7] = X3_Lick_onset
        # X[8] = X3_Lick_offset
        X[7] = X_5
        X[8] = X_10
        X[9] = X_Hit_1
        X[10] = X_Miss_1
        X[11] = X_CR_1
        X[12] = X_FA_1


        

    else : 
        X = Xp
    
    cs = sum(T2[:,1]== 12112) + sum(T2[:,1]== 12122)
    Xr = {}
    if r == 1: # Rule 1
        start = 0
        end = 150
    elif r == 2: # Rule 2
        start = 200 + np.max([ttr + 15,cs]) 
        end = np.min([410,len(stim_onset)])
    elif r == 3: # Rule 3
        start = 200
        end = 200 + ttr + 15
    t_start = stim_onset[start]-int(prestim/window)
    t_end = stim_onset[end]+int(t_period/window)
        
    for a in np.arange(len(X)):
        Xr[a] =  X[a][:,t_start:t_end]    
    S_all = S_all[t_start:t_end]
    L_all = L_all[t_start:t_end]
    L_del = L_del[:,t_start:t_end]
    R_del = [] #R_del[:,t_start:t_end]
    return X,Xr, S_all, L_del,R_del, stim_onset[start:end] ,reward_time[start:end],start,end,t_start
    


    
# %% GLM

def run_GLM(n,fig_on,Xp,r):    
    Xall,X,Y, L_all,R_all, stim_onset,r_onset,x_start,x_end,t_start = build_GLM(D_ppc,n,window,sigma,Xp,r)
    ss= ShuffleSplit(n_splits=k, test_size=0.30, random_state=0)
    T1= D_ppc[n,3][x_start:x_end,:] # trialtype by outcome
    T2= D_ppc[n,4][x_start:x_end,:]# trialtype by trialtyper
    stim_onset = stim_onset-t_start
    r_onset = r_onset-t_start
    
    # finding alpha value
    X4 = sparse.csr_array(np.zeros_like(Y))
    Nvar= len(X)
    for a in np.arange(Nvar):
        X4 = sparse.vstack([X4,X[a]])
    alphas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3,1e-2,1e-1]
    compare_score = np.zeros(len(alphas))
    for a in np.arange(len(alphas)):
        reg = PoissonRegressor(alpha = alphas[a], fit_intercept=True,max_iter = 1000) #Using a linear regression model with Ridge regression regulator set with alpha = 1

        cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
                                        return_estimator = True, 
                                        scoring = 'r2') 
        
        compare_score[a] = np.mean(cv_results['test_score'])
    alp = alphas[np.argmax(compare_score)] # best alpha, to store for later

    # finding alpha  value end
    
    # ### initial run, compare each TV ###
    reg = PoissonRegressor(alpha = alp, fit_intercept=True, max_iter = 1000)
    
    Nvar= len(X)
    compare_score = {}
    for a in np.arange(Nvar+1):
        
        X4 = sparse.csr_array(np.zeros_like(Y))


        if a < Nvar:
            X4 = sparse.vstack([X4,X[a]])

        cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
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
            
    elif np.max(score_mean) < 0.02:
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
                    X4 = sparse.csr_array(np.zeros_like(Y))
                    for a in m_ind:
                        X4 = sparse.vstack([X4,X[a]])        
                        cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
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
                    
        # rebuildingg max_it
        X3 =X
        if np.size(max_it) == 1:
                max_it = [max_it,max_it]
        for tv_ind in [1,2,3,4,5,6]:
            if (tv_ind+6 in max_it) and (tv_ind in max_it):
                    max_it = np.append(max_it, [tv_ind])
                            # X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
                    X3[tv_ind] = sparse.vstack([X3[tv_ind],X3[tv_ind+6]]);
            elif (tv_ind+6 in max_it) and(tv_ind not in max_it):
                    max_it = np.append(max_it, [tv_ind])
                    X3[tv_ind] = sparse.vstack([sparse.csr_array(np.zeros_like(X3[tv_ind].toarray())),X3[tv_ind+6]]);
            elif (tv_ind+6 not in max_it) and(tv_ind in max_it):
                            # max_it = np.append(max_it, [tv_ind])
                    X3[tv_ind] = sparse.vstack([X3[tv_ind],sparse.csr_array(np.zeros_like(X3[tv_ind+6].toarray()))]);
        # for tv_ind in [1,2]:
        #     if (tv_ind+8 in max_it) and (tv_ind in max_it):
        #             max_it = np.append(max_it, [tv_ind])
        #                     # X3[tv_ind] = np.concatenate((np.zeros_like(X3[tv_ind]),X3[tv_ind+4]),0);
        #             X3[tv_ind] = sparse.vstack([X3[tv_ind],X3[tv_ind+8]]);
        #     elif (tv_ind+8 in max_it) and(tv_ind not in max_it):
        #             max_it = np.append(max_it, [tv_ind])
        #             X3[tv_ind] = sparse.vstack([sparse.csr_array(np.zeros_like(X3[tv_ind].toarray())),X3[tv_ind+8]]);
        #     elif (tv_ind+8 not in max_it) and(tv_ind in max_it):
        #                     # max_it = np.append(max_it, [tv_ind])
        #             X3[tv_ind] = sparse.vstack([X3[tv_ind],sparse.csr_array(np.zeros_like(X3[tv_ind+8].toarray()))]);

        # === running regression with max_it ===
        X4 = sparse.csr_array(np.zeros_like(Y))
        for a in max_it:
            X4 = sparse.vstack([X4,X3[a]])        
            
        cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
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
            # yhat = yhat + Y0
    
    
    max_it = np.setdiff1d(max_it,[7,8,9,10,11,12,13,14])
    max_it = np.unique(max_it)
    TT = {}
    lg = 1
    X3 = {}
    for a in max_it:
        try:
            X3[a] = X[a].toarray()     
        except:
            X3[a] = X[a]
            X3[a] = np.reshape(X3[a],(1,-1))
    if np.size(max_it) ==1:
            a = np.empty( shape=(0, 0) )
            max_it = np.append(a, [int(max_it)]).astype(int)
    try:
        for t in max_it:
            TT[t] = X3[t].T@theta[lg:lg+np.size(X3[t],0),:]  
            lg = lg+np.size(X3[t],0)
    except: 
        TT[max_it] = X3[max_it].T@theta[lg:lg+np.size(X3[max_it],0),:]  
    
    # === figure === 
    if fig_on ==1:
        w_prestim = int(prestim/window)
        w_t_period = int(t_period/window)
        
        y = np.zeros((w_t_period,np.size(stim_onset)))
        yh = np.zeros((w_t_period,np.size(stim_onset)))
        l = np.zeros((w_t_period,np.size(stim_onset))) 
        weight = {}
        for a in np.arange(Nvar):
           weight[a] = np.zeros((w_t_period,np.size(stim_onset))) 
        
        yhat_mean = np.mean(yhat,1).T  
        for st in np.arange(np.size(stim_onset)):
            y[:,st] = Y[stim_onset[st]-w_prestim: stim_onset[st]+w_t_period-w_prestim]
            yh[:,st] = yhat_mean[stim_onset[st]-w_prestim: stim_onset[st]+w_t_period-w_prestim]
            
            # if np.size(max_it)>1:
            for t in max_it:
                weight[t][:,st] = np.mean(TT[t][stim_onset[st]-w_prestim: stim_onset[st]+w_t_period-w_prestim,:],1)
            # else:
            #     weight[max_it][:,st] = np.mean(TT[max_it][stim_onset[st]-prestim: stim_onset[st]+t_period,:],1)
            
    
        
        xaxis = np.arange(w_t_period)
        xaxis = xaxis*window-prestim
        xaxis = xaxis*1e-3
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10, 10))        
        cmap = ['black','tab:blue','tab:blue','tab:red','tab:grey','tab:green','tab:orange',
                'tab:red','tab:grey','tab:green','tab:orange','tab:blue','tab:blue','black','black']
        clabels = ["lick","5kHz","10kHz","Hit","Miss","CR","FA","Hit","Miss","CR","FA","5kHz","10kHz",'lick_onset','lick_offset']
        lstyles = ['solid','solid','dotted','solid','dotted','solid','solid','solid','solid','dotted','solid','solid','dotted','dashed','dotted']
        
        ### plot y and y hat
        stim_ind1 = np.mod(T2[:,3],2) ==1
        stim_ind2 = np.mod(T2[:,3],2) ==0
    
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
        fname = 'D:/GitHub/LeeLab/img/IC/'
        plt.savefig(fname + 'r'+ str(r) +'neuron'+str(n+1) + '.svg')
        plt.savefig(fname + 'r'+ str(r) +'neuron'+str(n+1) + '.png')
        plt.show()
    return Xall, inter, TT, Y, max_it, score3, init_compare_score, yhat,X4, theta, alp, stim_onset
    
# %% Initialize
"""     
Each column of X contains the following information:
    0 : contingency 
    1 : lick vs no lick
    2 : correct vs wrong
    3 : stim 1 vs stim 2
    4 : if exists, would be correct history (previous correct ) 

"""



t_period = 8000
prestim = 2000

window = 50 # averaging firing rates with this window. for Ca data, maintain 50ms (20Hz)
window2 = 500
k = 20 # number of cv
ca = 0
sigma = 1

# define c index here, according to comments within the "glm_per_neuron" function
c_list = [2]



if ca ==0:
    D_ppc = load_matfile()
    good_list = find_good_data()
else:
    D_ppc = load_matfile_Ca()
    # good_list = find_good_data_Ca(t_period)
    
good_list = np.arange(np.size(D_ppc,0))
# good_list = good_list[good_list>43].astype(int)
# %% Run GLM

Data = {}
# Data = np.load('SU_PPC2_06262025.npy',allow_pickle= True).item()

Xall = []
len_lick = 0
len_event = 0
for n in good_list:
    if len(D_ppc[n,2]) == len_lick and len(D_ppc[n,3]) == len_event:
        Xp = []; #Xall
    else:
        Xp = []
    counter = 0
    for r in [1,2,3]:
        try:                
            Xall, inter, TT, Y, max_it, score3, init_score, yhat,X4, theta, alp,stim_onset = run_GLM(n,1,Xp,r)
            Data[n,r] = {"X":Xall,"coef" : TT, "score" : score3, 'Y' : Y,'init_score' : init_score,
                               "intercept" : inter,"yhat" : yhat, "X4" : X4, "theta": theta,"alpha":alp,"stim_onset" : stim_onset}
            len_lick = len(D_ppc[n,2])
            len_event = len(D_ppc[n,3])
        except KeyboardInterrupt:
            break
        except:
            print("Break, no fit") 
            counter += 1
        
        if counter == 3:
            Xall = []
             

           

np.save('SU_IC3_06262025.npy', Data,allow_pickle= True)     
# Data = np.load('SU_PPC2_06262025.npy',allow_pickle= True).item()
# # test = Data2.item()

# # test1 =test(7,2)

# make goodlist per rule 
good_list_rule = {}
good_list_rule[1] = []
good_list_rule[2] = []
good_list_rule[3] = []
I2 = {}
for n in good_list:
    for r in [1,2,3]:
        if (n,r) in list(Data.keys()):
            if np.mean(Data[n,r]["score"]) >0.05:
                good_list_rule[r].append(int(n))
                I2[n] = np.mean(Data[n,r]["score"])
             
for r in [1,2,3]:
    good_list_rule[r] = np.array(good_list_rule[r])

# good_list_rule[r] = np.delete(good_list_rule[r],[68,79])

# %%

cmap = ['black','tab:blue','tab:blue','tab:red','tab:grey','tab:green','tab:orange','tab:red','tab:grey','tab:green','tab:orange','black']
clabels = ["lick","5kHz","10kHz","Hit","Miss","CR","FA","Hit","Miss","CR","FA",'lick']

def make_RS(r_ind):
    good_list_sep = good_list_rule[r_ind]
    ax_sz = len(cmap)
    I = np.zeros((np.size(good_list_sep),ax_sz+1))
       
        
    for n in np.arange(np.size(good_list_sep,0)):
        nn = int(good_list_sep[n])
        # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
        Model_score = Data[nn, r_ind]["score"]
        init_score =  Data[nn, r_ind]["init_score"]
        for a in np.arange(ax_sz):
            I[n,a] = np.mean(init_score[a])
        I[n,ax_sz] = np.mean(Model_score)*1.
        
    
    fig, axes = plt.subplots(1,1, figsize = (10,8))
        # Rsstat = {}
    for a in np.arange(ax_sz):
        Rs = I[:,a]
        Rs = Rs[Rs>0.02]
        axes.scatter(np.ones_like(Rs)*(a-0.3),Rs,facecolors=cmap[a], edgecolors= cmap[a])
        axes.scatter([(a-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')    
            # Rs = Rs/(Rmax+0.03)
            # Rsstat[c_ind,f] = Rs
    
                # axes.boxplot(Rs,positions= [f+(c_ind+1)*-0.3])
    Rs = I[:,ax_sz]
    Rs = Rs[Rs>0.02]
    axes.scatter(np.ones_like(Rs)*(ax_sz-0.3),Rs,c = 'k',)
    axes.scatter([(ax_sz-0.3)],np.mean(Rs),c = 'k',s = 500, marker='_')
    axes.set_ylim([0,0.75])
    axes.set_xlim([-1,len(cmap)])
    
    
    return I   

I1 = make_RS(1)
I2 = make_RS(2)
I3 = make_RS(3)

I1 = I1[:,-1]
I2 = I2[:,-1]
I3 = I3[:,-1]

bins = np.arange(0,0.8, 0.01)
fig, axs= plt.subplots(1,1,figsize = (5,5))
axs.hist(I1[I1>0.01],bins = bins,density=True, histtype="step",
                               cumulative=True)
axs.hist(I2[I2>0.01],bins = bins,density=True, histtype="step",
                               cumulative=True)
axs.hist(I3[I3>0.01],bins = bins,density=True, histtype="step",
                               cumulative=True)
axs.set_xlim([-.05,0.7])

# %% histogram of TV encoding
edgec = cmap
# edgec = ['tab:orange','tab:orange','tab:blue','tab:blue','tab:red','tab:red','black','green','tab:purple','tab:purple']


def TV_hist(r_ind):
    good_list_sep = good_list_rule[r_ind]
    TV = np.empty([1,1])
    for n in np.arange(np.size(good_list_sep,0)):
            nn = int(good_list_sep[n])
            Model_coef = Data[nn, r_ind]["coef"]
            max_it = [key for key in Model_coef]
            TV = np.append(TV, max_it)
    
    TV = TV[1:]
    ax_sz = 8
    B = np.zeros((1,ax_sz))
    for f in np.arange(ax_sz):
        B[0,f] = np.sum(TV == f)
        
    B = B/np.size(good_list_rule[r_ind])
    fig, axes = plt.subplots(1,1, figsize = (15,5))
    axes.grid(visible=True,axis = 'y')
    axes.bar(np.arange(ax_sz)*3,B[0,:], color = "white", edgecolor = edgec, alpha = 1, width = 0.5, linewidth = 2,hatch = '/')
    # axes.bar(np.arange(ax_sz)*3,B[0,:], color = cmap, edgecolor = edgec, alpha = 1, width = 0.5, linewidth = 2,hatch = '/')
    axes.set_ylim([0,0.8])
            
TV_hist(1)




def extract_onset_times(D_ppc,n,r):
    stim_onset = np.floor(D_ppc[n,2][:,0]*(1e3/window)).astype(int)

    
    if r == 1: # Rule 1
        start = 0
        end = 150
    elif r == 2: # Rule 2
        start = 260
        end = np.min([410,len(stim_onset)])
    elif r == 3: # Rule 3
        start = 200
        end = 260
    t_start = stim_onset[start]-int(prestim/window)
    t_end = stim_onset[end]+int(t_period/window)

    stim_onset2 = stim_onset[start:end]-t_start

    return stim_onset2


# r = 1
    
# for n in np.arange(np.size(good_list_rule[r],0)):
#     nn = int(good_list_rule[r][n])
#     # X, Y, Lm, L_on, L_off, stim_onset, r_onset, Xstim = import_data_w_Ca(D_ppc,nn,window,c_ind)
#     stim_onset = extract_onset_times(D_ppc,nn,r)
#     Data[nn,r]["stim_onset"] = stim_onset
    
listOv = {}

def extract_r_onset_times(D_ppc,n,r):
    ttr = int(D_ppc[n,6][0])
    timepoint = int(max(D_ppc[n,2][:,0])*1e3)+t_period+100;
    bins = np.arange(0,timepoint,window)  
    [S_all,bins] = np.histogram(D_ppc[n,0][:,0]*1e3 ,bins)
    stim_onset = np.floor(D_ppc[n,2][:,0]*(1e3/window)).astype(int)
    lick_time = np.floor(D_ppc[n,1][:,0]*(1e3/window)).astype(int)
    lick_time = lick_time[lick_time<np.size(S_all)] 
        ### Extract reward onset ###
        # Since reward is given at the second lick during reward period, we define 
        # reward onset as the second lick after reward period onset (1.5seconds post stim)
        

    reward_time =[]
    for tr in np.arange(np.size(stim_onset)):
        if D_ppc[n,3][tr,0] ==1 : # or D_ppc[n,3][tr,3] ==1 :
            r_temp = lick_time-stim_onset[tr]-1.5*1e3/window
            if np.sum(r_temp>0)>0: # first lick
                reward_time = np.append(reward_time, lick_time[np.argwhere(r_temp>0)[0]])

        
    reward_time= reward_time.astype(int)
    T2= D_ppc[n,4]
    cs = sum(T2[:,1]== 12112) + sum(T2[:,1]== 12122)
    if r == 1: # Rule 1
        start = 0
        end = 150
    elif r == 2: # Rule 2
        start = 200 + np.max([ttr + 15,cs]) 
        end = np.min([410,len(stim_onset)])
    elif r == 3: # Rule 3
        start = 200
        end = 200 + ttr + 15
    elif r == 0:
        start = 0 
        end = np.min([410,len(stim_onset)])
    # t_start = stim_onset[start]-int(prestim/window)
    # t_end = stim_onset[end]+int(t_period/window)
    
    # if r == 1: # Rule 1
    #     start = 0
    #     end = 150
    # elif r == 2: # Rule 2
    #     start = 260
    #     end = np.min([410,len(stim_onset)])
    # elif r == 3: # Rule 3
    #     start = 200
    #     end = 260
    start2 = np.min(np.where(reward_time > stim_onset[start]))
    end2 = np.max(np.where(reward_time < stim_onset[end]))
    reward_time = reward_time - (stim_onset[start]-int(prestim/window))
    return reward_time[start2:end2+1]
        



# %%
cmap = ['black','tab:blue','tab:blue','tab:red','tab:grey','tab:green','tab:orange']
clabels = ['lick',"5kHz","10kHz","Hit","Miss","CR","FA",'Rew']
lstyles = ['solid','solid','dotted','solid','dotted','solid','solid','solid']
ax_sz = len(cmap)

W5 = {};
b_count = {}
Conv_all = {}
for r_ind in [1,2,3]: # 0 is PPCIC, 1 is PPCAC
    b_count[r_ind] = np.zeros((2,ax_sz))
    for f in np.arange(ax_sz):
            W5[r_ind,f] = {}
            
            
# %% Normalized population average of task variable weights
# c_ind = 1
r = 3
region = 'IC'

good_list_sep =np.unique(np.concatenate((good_list_rule[1],good_list_rule[2],good_list_rule[3])))
weight_thresh = 5*1e-2



w_length = [8,30,30,140,140,140,140] # window lengths for GLM 
# w_length = [16,16,11,11,60,60,60,60] # window lengths for GLM 


Convdata = {}
Convdata2 = {}
pre = 40 # 10 40 
post = 120 # 50 20
xaxis = np.arange(post+pre)- pre
xaxis = xaxis*5e-2



for a in np.arange(ax_sz):
    Convdata[a] = np.zeros((np.size(good_list_sep),pre+post))
    Convdata2[a] = np.zeros(((np.size(good_list_sep),pre+post,w_length[a])))


good_list5 = [];
for n in np.arange(np.size(good_list_sep,0)):
    nn = int(good_list_sep[n])
    # Xall,X,Y, L_all, stim_onset,x_start,x_end,t_start = build_GLM(D_ppc,n,window,sigma,Xp,r)
    if nn in good_list_rule[r]:
        Model_coef = Data[nn, r]["coef"]
        theta = Data[nn,r]["theta"]
        X4 = Data[nn,r]["X4"].toarray()
        Model_score = Data[nn, r]["score"]
        stim_onset2 =  Data[nn, r]["stim_onset"]
        stim_onset =  Data[nn, r]["stim_onset"]
        # stim_onset = extract_r_onset_times(D_ppc,nn,r)
        # stim_onset= L_data[nn,1].T
        # stim_onset = stim_onset[0,1:-1]
        [T,p] = stats.ttest_1samp(np.abs(theta),0.05,axis = 1, alternative = 'greater') # set weight threshold here
        p = p<0.05
        Model_weight = np.multiply([np.mean(theta,1)*p],X4.T).T
        Model_weight = Model_weight[1:,:]
        maxC2 = np.max([np.abs(np.mean(theta,1))*p])+0.2
        
        
        weight = {}
        weight2 = {}
        max_it = [key for key in Model_coef]
        # max_it = np.setdiff1d(max_it,[8,9,10,11])
        for a in max_it:
            weight[a] = np.zeros((pre+post,np.size(stim_onset))) 
            weight2[a] = np.zeros((pre+post,np.size(stim_onset),w_length[a]) )  
                                  
        for st in np.arange(np.size(stim_onset)-1):
            lag = 0
            for a in max_it:
                if stim_onset[st] <pre:
                    stim_onset[st] = pre+1
                weight[a][:,st] = np.mean(Model_coef[a][stim_onset[st]-pre: stim_onset[st]+post,:],1)
                weight2[a][:,st,:] = Model_weight[lag:lag+w_length[a],stim_onset[st]-pre: stim_onset[st]+post].T
                    
                lag = lag+w_length[a]
            
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
                        Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/(1*maxC2)
                        # Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/2
                    else:                       
                        Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)/maxC2
                        # Convdata2[a][n,:,:] = np.mean(weight2[a][:,nz_ind,:],1)
        
    
# fig, axes = plt.subplots(1,1,figsize = (10,8))         
# axes.plot(xaxis,np.mean(weight[7],1))
# axes.plot(xaxis,np.mean(np.sum(weight2[a][:,nz_ind,:],1),1))

# adding Miss reward to Hit

if r == 3:
    Convdata[3] = Convdata[3] + Convdata[4]
lstyles = ['solid','solid','dotted','solid','dotted','solid','solid','solid']

fig, axes = plt.subplots(1,1,figsize = (10,8))       
for a in np.arange(ax_sz):
    list0 = (np.mean(Convdata[a],1) != 0)
    error = np.std(Convdata[a],0)/np.sqrt(np.size(good_list_sep))
    y = ndimage.gaussian_filter(np.mean(Convdata[a][list0,:],0),0)

    # W = ndimage.gaussian_filter(np.mean(Convdata[a],0),2)   
    # W = ndimage.uniform_filter(np.sum(Convdata2[a][list0,:,:],2),[0,5], mode = "mirror")
    # y = np.mean(W,0)
    # y = np.abs(np.mean(W,0))
    # error = np.std(W,0)/np.sqrt(np.sqrt(np.sum(list0)))
    axes.plot(xaxis,y,c = cmap[a],linestyle = lstyles[a])
    axes.fill_between(xaxis,y-error,y+error,facecolor = cmap[a],alpha = 0.3)
    axes.set_ylim([-0.5,0.5])
    # axes.set_ylim([-0.1,1])
    
# fig, axes = plt.subplots(1,1,figsize=(50,30))
# axes.imshow(theta,cmap = 'bwr_r',clim = [-0.5,0.5], aspect='auto')    
    
Conv_all[r] = Convdata

# % plotting weights by peak order

# Convdata2 = Model_weight

f = 3
W5AC= {}
W5IC = {}
max_peak3 ={}
tv_number = {}
cmap = ['black','tab:blue','tab:blue','tab:red','tab:grey','tab:green','tab:orange']
clabels = ['lick',"5kHz","10kHz","Hit","Miss","CR","FA","Rew"]
clabels2 = ['lick',"five","ten","hit","miss","cr","fa",'rew']
lstyles = ['solid','solid','dotted','solid','dotted','solid','solid','solid']
ax_sz = len(cmap)
listOv1 = {}
listOv2 = {}
# w_length1 = [16,16,11,11,30,30,20,20]
# w_length2 = [0,0,0,0,31,31,21,21]


for f in  np.arange(ax_sz):
        list0 = (np.mean(Convdata[f],1) != 0)
        # list0ind = good_list_rule[r][list0]
        list0ind = good_list_sep[list0]
        W = ndimage.uniform_filter(Convdata[f][list0,:],[0,0], mode = "mirror")
        # W = ndimage.uniform_filter(np.sum(Convdata2[f][list0,:,:],2),[0,0], mode = "mirror")
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
        max_peak3[r,f] = max_peak[list3]
        
        listOv[r,f] = list0ind[list3]
        listOv1[f] = list0ind[list1]
        listOv2[f] = list0ind[list2]
        W1 = W[max_ind1]
        W2 = W[max_ind2]    
        W4 = np.abs(W[max_ind3])
        s ='+' + str(np.size(W1,0)) +  '-' + str(np.size(W2,0))
        print(s)
        b_count[r][0,f] = np.size(W1,0)
        b_count[r][1,f] = np.size(W2,0)
        W3 = np.concatenate((W1,W2), axis = 0)
        tv_number[r,f] = [np.size(W1,0),np.size(W2,0)]
        # W3[:,0:8] = 0
        # W3[:,68:] = 0
        W5[r,f][0] = W1
        W5[r,f][1] = W2
        # if f in [3,6]:
        #     clim = [-.5, .5]
        #     fig, axes = plt.subplots(1,1,figsize = (10,10))
        #     im1 = axes.imshow(W3[:,:],clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
        #     # im2 = axes[1].imshow(W2, aspect = "auto", interpolation = "None")
        #     # axes.set_xlim([,40])
        #     fig.subplots_adjust(right=0.85)
        #     cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        #     fig.colorbar(im1, cax=cbar_ax)


# %%  overlap between RT miss and Hit
ind = 3
f1 = 3 
f2 = 4
list_overlap = np.intersect1d(listOv[ind,f1],listOv[ind,f2])

list_miss = np.setdiff1d(listOv[3,4], list_overlap)

ind2 = 2
np.size(listOv[ind2,3])
np.size(np.intersect1d(listOv[ind2,3],list_miss))
np.size(np.intersect1d(listOv[ind2,3],listOv[3,3]))

np.size(np.intersect1d(listOv[2,3],listOv[1,3]))

test1 = np.intersect1d(listOv[1,3],listOv[3,3])
test2 = np.intersect1d(listOv[2,3],listOv[3,3])
np.size(np.intersect1d(test1,test2))



conv_list = [];
for n in list_overlap:
    conv_list = np.append(conv_list,np.where(good_list_rule[ind] == n)[0])
    
W = ndimage.uniform_filter(Convdata[f2][conv_list.astype(int),:],[0,0], mode = "mirror")
max_peak = np.argmax(np.abs(W),1)
max_ind = max_peak.argsort()
fig, axes = plt.subplots(1,1,figsize = (10,10))
clim = [-.5, .5]
im1 = axes.imshow(W[max_ind,:],clim = clim, aspect = "auto", interpolation = "None",cmap = "viridis")
     

# %% Save encoding neuron list
# savemat('D:\DATA\listTV3_IC_RT.mat',{clabels2[f]:listOv[r,f] for f in np.arange(ax_sz)})
if r == 1:
    rulename= '_R1'
elif r == 2:
    rulename= '_R2'
else:
    rulename= '_RT'
savemat('D:\DATA\listTV3_'+region+rulename+'.mat',{clabels2[f]:listOv[r,f] for f in np.arange(ax_sz)})

savemat('D:\DATA\listTV3_'+region+rulename+'p.mat',{clabels2[f]:listOv1[f] for f in np.arange(ax_sz)})
savemat('D:\DATA\listTV3_'+region+rulename+'n.mat',{clabels2[f]:listOv2[f] for f in np.arange(ax_sz)})

# %% calculate nb of neurons encodin

# create list of all neurons that encode at least 1 variable
ind = r
test = [];
for ind in [r]:
    for f in np.arange(ax_sz):
        test = np.concatenate((test,listOv[ind,f]))

test_unique, counts = np.unique(test,return_counts= True)

# fig, axes = plt.subplots(1,1,figsize = (10,10))

# sns.histplot(data = counts)

labels = np.arange(1,6)
sizes = np.zeros((1,5))
for l in labels:
    sizes[0,l-1] = np.sum(counts == l) 

sizes[0,4] = np.sum(counts > 4)

# fig, axes = plt.subplots(1,1,figsize = (10,10))
# axes.pie(sizes[0,:], labels=labels, autopct='%1.1f%%')

# %% for each timebin, calculate the number of neurons encoding each TV

cmap = ['black','tab:blue','tab:blue','tab:red','tab:grey','tab:green','tab:orange']
clabels = ['lick',"5kHz","10kHz","Hit","Miss","CR","FA","Rew"]
lstyles = ['solid','solid','dotted','solid','dotted','solid','solid','solid']
ax_sz = len(cmap)

fig, axes = plt.subplots(1,1,figsize = (10,5))
y_all = np.zeros((ax_sz,pre+post))
p = 0 # positive vs negative
for f in np.arange(ax_sz):
    list0 = (np.mean(Convdata[f],1) != 0)
    # W = ndimage.uniform_filter(Convdata[f][list0,:],[0,2], mode = "mirror")
    # W = Convdata[f][list0,:]
    W = ndimage.uniform_filter(np.sum(Convdata2[f][list0,:,:],2),[0,0], mode = "mirror")
    SD = np.std(W[:,:])
    # test = np.abs(W5[ind,f][p])>1*SD
    test = np.abs(W5[r,f][p])>1*SD
    y = np.sum(test,0)/np.size(good_list_sep)
    if r == 3:
        if f == 4:
            y = 0
        
    y_all[f,:] = y
    y = ndimage.uniform_filter(y,2, mode = "mirror")
    if p == 0:
        axes.plot(y,c = cmap[f], linestyle = lstyles[f], linewidth = 3, label = clabels[f] )
        axes.set_ylim([-0.01,.5])
        axes.legend()
    elif p == 1:
        axes.plot(-y,c = cmap[f], linestyle = lstyles[f], linewidth = 3 )
        axes.set_ylim([-0.6,0.01])


# %%
fig, axes = plt.subplots(1,1,figsize = (10,5), sharex = True)
fig.tight_layout()
fig.subplots_adjust(hspace=0)


# Transition

# Lic =57 #64
# Lg = 213-57 #182
# Lic = 110
# Lg = 394-110
b11 = b_count[r][0,:]/np.size(good_list_sep)
b12 = b_count[r][1,:]/np.size(good_list_sep)


# axes.grid(visible=True,axis = 'y')
# axes[1].grid(visible=True,axis = 'y')
if r == 1:
    axes.bar(np.arange(ax_sz)*3+1,b11+b12, color = 'white', edgecolor = cmap, alpha = 1, width = 0.5, linewidth = 2) #, hatch = '/')
elif r ==2:
    axes.bar(np.arange(ax_sz)*3+1,b11+b12, color = 'white', edgecolor = cmap, alpha = 1, width = 0.5, linewidth = 2, hatch = '/')
else:
    axes.bar(np.arange(ax_sz)*3+1,b11+b12, color = cmap, edgecolor = cmap, alpha = 1, width = 0.5, linewidth = 2) #, hatch = '/')
        


# fname = 'D:/Python/Figures/fraction_'
# # axes[0].bar(np.arange(1)*2+0.7,b21, color = cmap3, alpha = 1, width = 0.5)
# # axes[0].bar(np.arange(4)*3+`1.4,b31, color = cmap3, alpha = 0.5, width = 0.5)
# axes.set_ylim([0,1])
# # axes.set_ylim([0,100])
# plt.savefig(fname + region + 'r'+ str(r) + '.svg')
        




# %% plot positive and negative weights separately.
# cmap = ['black','gray','tab:blue','tab:cyan','tab:red','tab:orange','black','green','tab:red','tab:orange','black','green',]

pp = 0

lstyles = ["","solid","dotted","dashed"]

maxy = np.zeros((2,10))
fig, axes = plt.subplots(2,ax_sz,figsize = (50,10),sharex = "all")
fig.subplots_adjust(hspace=0)
for p in [0,1]:
    for f in np.arange(ax_sz):
        if np.size(W5[r,f][p],0) >5:
            y1 = ndimage.gaussian_filter1d(np.sum(W5[r,f][p],0),1)
            y1 = y1/(np.size(W5[r,f][0],0)+np.size(W5[r,f][1],0))
            e1 = np.std(W5[r,f][p],0)/np.sqrt((np.size(W5[r,f][0],0)+np.size(W5[r,f][1],0)))
            axes[p,f].plot(xaxis,y1,c = cmap[f],linestyle = lstyles[r], linewidth = 3)
            axes[p,f].fill_between(xaxis,y1-e1,y1+e1,facecolor = cmap[f],alpha = 0.3)
            maxy[p,f] = np.max(np.abs(y1)+np.abs(e1))
            maxy[p,f] = np.max([maxy[p,f],.5])
        else:
            for tr in np.arange(np.size(W5[r,f][p],0)):
                y1 = ndimage.gaussian_filter1d((W5[r,f][p][tr,:]),1)
                y1 = y1/(np.size(W5[r,f][0],0)+np.size(W5[r,f][1],0))
                axes[p,f].plot(xaxis,y1,c = cmap[f],linestyle = lstyles[r], linewidth = 1)
                maxy[p,f] = np.max(np.abs(y1)+np.abs(e1))
                maxy[p,f] = np.max([maxy[p,f],.5])

            # if np.size(W5[ind,f][p],0 > 4):
            #     for t in np.arange(80):
            #         if p == 0:
            #             s1,p1 = stats.ttest_1samp(W5[ind,f][p][:,t],np.mean(e1),alternative = 'greater')
            #         else:
            #             s1,p1 = stats.ttest_1samp(W5[ind,f][p][:,t],-np.mean(e1),alternative= 'less')
            #         if p1 < 0.05:
            #             scat[0,t] = True
            #             pcat[0,t] = p1
            #     c1 = pcat[0,scat[0,:]>0]
            #     if p == 0:
            #         axes[p,f].scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*maxy[p,f] + 0.1,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,-1])
            #     elif p ==1:
            #         axes[p,f].scatter(xaxis[scat[0,:]>0],np.ones_like(xaxis[scat[0,:]>0])*-maxy[p,f] - 0.1,marker='s',c = np.log10(c1),cmap = 'Greys_r',clim = [-3,-1])

    
    for f in np.arange(7):
            axes[0,f].set_ylim([0, np.nanmax(maxy[:,f]+0.2)])
            axes[1,f].set_ylim([-np.nanmax(maxy[:,f]+0.2),0])
    





# %% Ca trace for selected units
# removing lick weights from Hit | Miss-Reward units

r = 1
n_list = np.unique(np.append(listOv[r,3],listOv[r,4]))


def concat_Y(n_list,r):
    
    Y_all = {}
    Y_all[0] = np.zeros((np.size(n_list),pre+post)) # 0 is hit, 1 is Miss-reward
    Y_all[1] = np.zeros((np.size(n_list),pre+post))
    # Y_all[2] = np.zeros((np.size(n_list),pre+post))
    nc=0
    for nn in n_list:
        Model_coef = Data[nn, r]["coef"]
        stim_onset =  Data[nn, r]["stim_onset"]
        X = D_ppc[nn,3]
        if r == 3:
            X = X[200:260,:]
        X[:,2] = X[:,3];
        max_it = [key for key in Model_coef]
        if 0 in max_it:
            Y = Data[nn,r]["Y"] - np.mean(Model_coef[0],1)/2
        else:
            Y = Data[nn,r]["Y"]
        for ind in [0,1]:
            tempC = np.zeros((np.size(stim_onset[X[:,ind]==1]),(pre+post)))
            sc = 0
            for st in stim_onset[X[:,ind]==1]:
                tempC[sc,:] = Y[st-pre:st+post]
                sc += 1
            Y_all[ind][nc,:] = np.nanmean(tempC,0)/(max(Y)+.5)
        nc += 1
    return Y_all
            
        
Y_all = concat_Y(n_list,r)

cmap2 = ['tab:red','tab:purple','tab:orange']

fig, axes = plt.subplots(1,1,figsize = (10,5))
for ind in [0,1]:
    y = np.nanmean(Y_all[ind],0)
    e = np.nanstd(Y_all[ind],0)/np.sqrt(np.size(Y_all[ind],0))
    axes.plot(xaxis,y,color = cmap2[ind])
    axes.fill_between(xaxis,y-e,y+e,facecolor = cmap2[ind],alpha = 0.3)

fig,axes = plt.subplots(1,2,figsize = (10,5))
for ind in [0,1]:
    max_peak = np.argmax(np.abs(Y_all[1]),1)
    max_ind = max_peak.argsort()
    im1 = axes[ind].imshow(Y_all[ind][max_ind,:],clim = [-0.4,0.4], aspect = "auto", interpolation = "None",cmap = "viridis")


# %% helper functions for PCA and subspace overlap

def subspace_overlap(W4,n_cv,t1,t2,ntv):
    R = {}
    pca = {}
    for g in  np.arange(ntv):
        pca[g] = PCA(n_components=max_k)
        R[g] = W4[g][:,t1:t2].T
        test = pca[g].fit_transform(ndimage.gaussian_filter(R[g],[sm,0]))        
        test = test.T
    Overlap = np.zeros((len(W4),len(W4),n_cv)); # PPC_IC
    # Overlap_across = np.zeros((trmax,trmax,n_cv));
    total_n = np.size(W4[0],0)
    
    k1 = n_cv
    
    for cv in np.arange(n_cv):
        n_list = np.random.choice(np.arange(total_n),int(np.floor(total_n*0.75)),replace = False)    
        for g in  np.arange(len(W4)):
            pca[g] = PCA(n_components=k1)
            test = pca[g].fit_transform(ndimage.gaussian_filter(R[g][:,n_list],[1,0]))  
            
        
        for g1 in np.arange(len(W4)): #np.arange(trmax):
           for g2 in np.arange(len(W4)): # np.arange(trmax):
               S_value = np.zeros((1,k1))
               for d in np.arange(0,k1):
                   S_value[0,d] = np.abs(np.dot(pca[g1].components_[d,:], pca[g2].components_[d,:].T))
                   S_value[0,d] = S_value[0,d]/(np.linalg.norm(pca[g1].components_[d,:])*np.linalg.norm(pca[g2].components_[d,:]))
               Overlap[g1,g2,cv] = np.max(S_value)
    return Overlap

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm


def draw_traj5(traj,v,ntv):
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    # styles = ['solid', 'dotted', 'solid','dotted']
    # cmap_names = ['autumn','autumn','winter','winter']
    styles = ['solid','solid','solid','dotted','solid','dotted','dashed','dashed']
    cmap_names = ['autumn','winter','winter']
    for ind in np.arange(ntv): # np.arange(trmax):
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
                colors = cm.autumn(np.linspace(0,1,160))
                ax.auto_scale_xyz(x,y,z)
            elif ind == 2:
                colors = cm.winter(np.linspace(0,1,160))
                # ax.auto_scale_xyz(x,y,z)
            else:
                colors = cm.gray(np.linspace(0,1,160))
                # ax.set_ylim([-5,5])
            

            lc = Line3DCollection(segments, color = colors,alpha = 1,linestyle = styles[ind])#linestyle = styles[tr])
    
            lc.set_array(time)
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            
            
            for m in [0]:
                ax.scatter(x[m], y[m], z[m], marker='o', color = "black")

# %% subspace overlap

pca = {}
max_k = 20;
sm = 5
t1 = 60
t2 = 160

# O = subspace_overlap(Y_all,max_k,t1,t2)

def make_traj():
    R = np.concatenate((Y_all[0],Y_all[1]),1)
    # R = np.concatenate((R,Y_all[2]),1)
    traj ={}
    pca = PCA(n_components=max_k)
    test = pca.fit_transform(ndimage.gaussian_filter(R[:,:].T,[1,0])) 
    # test =test
    for ind in [0,1]:
        traj[ind] = ndimage.gaussian_filter(test[ind*160:(ind+1)*160,:],[sm,0])
        traj[ind] = traj[ind]- np.mean(traj[ind][10:30,:],0)
    return traj


traj = make_traj()
draw_traj5(traj,1,2)

distance = {}
for ind in [0,1]:
    distance[ind] = np.linalg.norm(traj[ind][:,0:3],axis = 1)# np.sqrt(traj[ind][0,:]**2+traj[ind][1,:]**2+traj[ind][2,:]**2)

fig, axes = plt.subplots(figsize = (5,5))
axes.plot(distance[0],color ='tab:red' )
axes.plot(distance[1],color ='black')

# %% rastermap for W2
from rastermap import Rastermap, utils
from scipy.stats import zscore

    # W3[ind] = W2[ind][listind[0],:],[0,3],mode = "mirror")
                             
# W3[2] = (W3[2] + W3[3])/2
# W3[0] = W3[1]
ind =1
# fit rastermap
# note that D_r is already normalized
model = Rastermap(n_PCs=64,
                  locality=.75,
                  time_lag_window=7,
                  n_clusters = 10,
                  grid_upsample=1, keep_norm_X = False).fit(Y_all[ind])
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
# X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)

for ind in [0,1]:
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(zscore(Y_all[ind][isort, :],axis = 1), cmap="gray_r", vmin=0.5, vmax=1.2,aspect = "auto")
    ax.set(xticks=np.arange(0, pre+post, 20), xticklabels=np.arange(-pre/20, post/20, 1));


# %% subspace overlap using model weights

W_h = {}
pca = {}
max_k = 20;
sm = 5
# list_hit = np.unique(np.concatenate((listOv[1,3],listOv[3,3],listOv[2,3])))

# for ind in [0,1,2]:
#     W_h[ind] = np.zeros((np.size(list_hit),pre+post))
#     c = 0
#     for n in listOv[ind+1,3]:
#         nn = np.where(list_hit == listOv[ind+1,3][n])[0][0]
#         W_h[ind][nn,:] = W5[ind+1,3][c,:]
#         c += 1
        


f1 = 0
f2 = 3
f3 = 6
f4 = 0

W_h[0] = Conv_all[1][f1]
W_h[1] = Conv_all[3][f1]
W_h[2] = Conv_all[2][f1]

W_h[3] = Conv_all[1][f2]
W_h[4] = Conv_all[3][f2]
W_h[5] = Conv_all[2][f2]

# W_h[6] = Conv_all[1][f3]
# W_h[7] = Conv_all[3][f3]
# W_h[8] = Conv_all[2][f3]

# W_h[9] = Conv_all[1][f4]
# W_h[10] = Conv_all[3][f4]
# W_h[11] = Conv_all[2][f4]
lstyles = ["","solid","dotted","dashed"]

fig, axes = plt.subplots(figsize = (10,5))
for ind in [0,1,2]:
    y = np.mean(W_h[ind][np.sum(W_h[ind],1)>-10,:],0)
    e = np.std(W_h[ind][np.sum(W_h[ind],1)>-10,:],0)/np.sqrt(len(np.sum(W_h[ind],1)>-10))
    axes.plot(xaxis,y,c = cmap[f1],linestyle = lstyles[ind+1])
    axes.fill_between(xaxis,y-e,y+e,facecolor = cmap[f1],alpha = 0.3)
    axes.set_ylim([-0.01,0.5])
fig, axes = plt.subplots(figsize = (10,5))
for ind in [3,4,5]:
    y = np.mean(W_h[ind][np.sum(W_h[ind],1)>-10,:],0)
    e = np.std(W_h[ind][np.sum(W_h[ind],1)>-10,:],0)/np.sqrt(len(np.sum(W_h[ind],1)>-10))
    axes.plot(xaxis,y,c = cmap[f2],linestyle = lstyles[ind-2])
    axes.fill_between(xaxis,y-e,y+e,facecolor = cmap[f2],alpha = 0.3)
    axes.set_ylim([-0.05,0.3])

O2 = subspace_overlap(W_h, 20, 0, 160,len(W_h))

fig, axes = plt.subplots(figsize = (5,5))
imshowobj = axes.imshow(np.mean(O2[:,:,:],2),cmap = "hot_r")
imshowobj.set_clim(0.2, 0.9) #correct
plt.colorbar(imshowobj) #adjusts scale to value range, looks OK

# savemat('D:\DATA\subspace_'+region+'.mat',{'lick1':O2[0,1,:],'lick2':O2[0,2,:],'lick3':O2[1,2,:],'hit1':O2[3,4,:],'hit2':O2[3,5,:],'hit3':O2[4,5,:]})



def make_traj2(ntv):
    # R = np.concatenate((W_h[3],W_h[4],W_h[5]),1)
    R = np.concatenate((W_h[0],W_h[1],W_h[2]),1)
    # R = np.concatenate((R,Y_all[2]),1)
    traj ={}
    pca = PCA(n_components=max_k)
    test = pca.fit_transform(ndimage.gaussian_filter(R[:,:].T,[1,0])) 
    expvar = pca.explained_variance_ratio_
    # test =test
    for ind in np.arange(ntv):
        traj[ind] = ndimage.gaussian_filter(test[ind*160:(ind+1)*160,:],[sm,0])
        traj[ind] = traj[ind]- np.mean(traj[ind][10:20,:],0)
    return traj, expvar


traj,expvar = make_traj2(5)
draw_traj5(traj,1,3)


# fig, axes = plt.subplots(1,1)
# for rr in [0,1,2]:
#     axes.plot(traj[rr][:,0])

# %%










for f in [0,3,5,6]:
    fig, axes= plt.subplots(2,1,figsize =(10,10))
    for ind in [0,1]:
        for rr in [1,2,3]:
            y = np.sum(W5[rr,f][ind],0)/(np.size(W5[rr,f][0],0)+np.size(W5[rr,f][1],0))
            e = np.std(W5[rr,f][ind],0)/np.sqrt((np.size(W5[rr,f][0],0)+np.size(W5[rr,f][1],0)))
            axes[ind].plot(xaxis,y,c = cmap[f],linestyle = lstyles[rr])
            axes[ind].fill_between(xaxis,y-e,y+e,facecolor = cmap[f],alpha = 0.3)
            
            if ind ==1:
                axes[ind].set_ylim(-0.2,0.02)
            else:
                axes[ind].set_ylim(-0.02,0.25)


# %% how many units are pre-lick? run after everything

# r = 1

def ana_lick_theta(r):
    list_lick = np.array([])
    for r2 in [1,2,3]:
        for n in np.arange(np.size(good_list_sep,0)):
            nn = int(good_list_sep[n])
            # Xall,X,Y, L_all, stim_onset,x_start,x_end,t_start = build_GLM(D_ppc,n,window,sigma,Xp,r)
            if nn in good_list_rule[r2]:
                Model_coef = Data[nn, r2]["coef"]
                theta = Data[nn,r2]["theta"]
            
                if 0 in [key for key in Model_coef]:
                    list_lick = np.append(list_lick,nn)
    list_lick = np.unique(list_lick)
    mat_theta = np.zeros((7,np.size(list_lick)))
    mat_stat =  np.zeros((7,np.size(list_lick)))
    pn = 0
    for n in list_lick:
        nn = int(n)
        if nn in good_list_rule[r]:
            Model_coef = Data[nn, r]["coef"]
            theta = Data[nn,r]["theta"]
            test = np.flip(theta[1:5,:],0)
            test2 =  theta[6:9,:] 
            theta_lick = np.flip(np.concatenate((test,test2),0),0)
            mat_theta[:,pn] = np.mean(theta_lick,1)
            for t in np.arange(7):
                [T,p] = stats.ttest_1samp(np.abs(theta_lick[t,:]),np.std(theta[:,:]),alternative = 'greater')
                if p < 0.05:
                    mat_stat[t,pn] =1
        pn += 1
    return mat_stat, mat_theta, list_lick

mat_stat = {}
mat_theta = {}
for r in [1,2,3]:
    mat_stat[r],mat_theta[r], list_lick = ana_lick_theta(r)
    # mat_stat[r][10:-1,:] = mat_stat[r][11:,:]
    # mat_stat[r] = mat_stat[r][:-1,:]


l_xaxis = np.arange(-200,150,50)
fig, axes = plt.subplots(figsize = (10,5))
for r in [1,2,3]:
    y = np.mean(mat_stat[r],1)
    e = np.std(mat_stat[r],1)/np.sqrt(np.size(mat_stat[r],1))
    axes.plot(l_xaxis,y,linestyle = lstyles[r])
    axes.fill_between(l_xaxis,y-e,y+e,alpha = 0.3)
    axes.set_ylim([0,1])

mat_stat[4] = (mat_stat[1] + mat_stat[2] + mat_stat[3])/3

fig, axes = plt.subplots(figsize = (10,5))
y = np.mean(mat_stat[4],1)
e = np.std(mat_stat[4],1)/np.sqrt(np.size(mat_stat[4],1))
axes.plot(l_xaxis,y)
axes.fill_between(l_xaxis,y-e,y+e,alpha = 0.3)
axes.set_ylim([0,1])

axes.set_ylim([0,1])
# axes.set_ylim([0,100])
# fname = 'D:/Python/Figures/lick_theta2_'
# plt.savefig(fname + region + '.svg')
        


# test = np.mean(mat_stat,1)


# fig, axes = plt.subplots(1,1)
# axes.plot(np.mean(np.concatenate((mat_theta[0:10,:],mat_theta[11:-1,:])),1))
# %% Lick encoding, pre vs post


# %% finding alpha value, GLM part


    # ### initial run, compare each TV ###
    # reg = PoissonRegressor(alpha = alp, fit_intercept=True)
    
    # Nvar= len(X)
    # compare_score = {}
    # for a in np.arange(Nvar+1):
        
    #     X4 = sparse.csr_array(np.zeros_like(Y))


    #     if a < Nvar:
    #         X4 = sparse.vstack([X4,X[a]])

    #     cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
    #                                 return_estimator = True, 
    #                                 scoring = 'r2') 
    #     compare_score[a] = cv_results['test_score']
    
    # f = np.zeros((1,Nvar))
    # p = np.zeros((1,Nvar))
    # score_mean = np.zeros((1,Nvar))
    # for it in np.arange(Nvar):
    #     f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[Nvar],alternative = 'less')
    #     score_mean[0,it] = np.median(compare_score[it])

    # max_it = np.argmax(score_mean)
    # init_score = compare_score[max_it]
    # init_compare_score = compare_score
    
    # if p[0,max_it] > 0.05:
    #         max_it = []
    # else:  
    #     # === stepwise forward regression ===
    #     step = 0
    #     while step < Nvar:
    #             max_ind = {}
    #             compare_score2 = {}
    #             f = np.zeros((1,Nvar))
    #             p = np.zeros((1,Nvar))
    #             score_mean = np.zeros((1,Nvar))
    #             for it in np.arange(Nvar):
    #                 m_ind = np.unique(np.append(max_it,it))
    #                 X4 = sparse.csr_array(np.zeros_like(Y))
    #                 for a in m_ind:
    #                     X4 = sparse.vstack([X4,X[a]])        
    #                     cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
    #                                                 return_estimator = True, 
    #                                                 scoring = 'r2') 
    #                 compare_score2[it] = cv_results['test_score']
    
    #                 f[0,it], p[0,it] = stats.ks_2samp(compare_score2[it],init_score,alternative = 'less')
    #                 score_mean[0,it] = np.mean(compare_score2[it])
    #             max_ind = np.argmax(score_mean)
    #             if p[0,max_ind] > 0.05 or p[0,max_ind] == 0:
    #                 step = Nvar
    #             else:
    #                 max_it = np.unique(np.append(max_it,max_ind))
    #                 init_score = compare_score2[max_ind]
    #                 step += 1
    
    
    
    
# 

    
    # if np.max(compare_score) < 0.02:
    #     max_it = []
    # else:
        
    #     reg = PoissonRegressor(alpha = alp, fit_intercept=True)
    
    #     # initial run, get score values for all
    #     for a in np.arange(Nvar):
    #         X4 = sparse.vstack([X4,X[a]])
    #     cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
    #                                         return_estimator = True, 
    #                                         scoring = 'r2') 
    #     init_score = cv_results['test_score']
    
    # ### initial run, compare all minus one models
    
    #     Nvar= len(X)
    #     compare_score = {}
    #     f = np.zeros((1,Nvar))
    #     p = np.zeros((1,Nvar))
    #     for a in np.arange(Nvar):
            
    #         X4 = sparse.csr_array(np.zeros_like(Y))
            
    #         m_ind = np.delete(np.arange(Nvar),a)
    #         for b in m_ind:
    #             X4 = sparse.vstack([X4,X[b]])        
    #             cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
    #                                                     return_estimator = True, 
    #                                                     scoring = 'r2') 
    #             compare_score[a] = cv_results['test_score']
    #         f[0,a], p[0,a] = stats.ks_2samp(compare_score[a],init_score,alternative = 'greater')
    #         f[0,a] = np.mean(init_score-compare_score[a]) 
    #     # find max_it #
    #     max_it = np.arange(Nvar)[(f[0,:]>0.01)*(p[0,:]<0.01)]
    # ### initial run, end ###

    #     for a in np.arange(Nvar+1):
            
    #         X4 = sparse.csr_array(np.zeros_like(Y))


    #         if a < Nvar:
    #             X4 = sparse.vstack([X4,X[a]])

    #         cv_results = cross_validate(reg, X4.T, Y.T, cv = ss , 
    #                                     return_estimator = True, 
    #                                     scoring = 'r2') 
    #         compare_score[a] = cv_results['test_score']
        
    #     f = np.zeros((1,Nvar))
    #     p = np.zeros((1,Nvar))
    #     score_mean = np.zeros((1,Nvar))
    #     for it in np.arange(Nvar):
    #         f[0,it], p[0,it] = stats.ks_2samp(compare_score[it],compare_score[Nvar],alternative = 'less')
    #         score_mean[0,it] = np.median(compare_score[it])
    