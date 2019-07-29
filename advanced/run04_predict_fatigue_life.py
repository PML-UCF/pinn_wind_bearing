# ______          _           _     _ _ _     _   _      
# | ___ \        | |         | |   (_) (_)   | | (_)     
# | |_/ / __ ___ | |__   __ _| |__  _| |_ ___| |_ _  ___ 
# |  __/ '__/ _ \| '_ \ / _` | '_ \| | | / __| __| |/ __|
# | |  | | | (_) | |_) | (_| | |_) | | | \__ \ |_| | (__ 
# \_|  |_|  \___/|_.__/ \__,_|_.__/|_|_|_|___/\__|_|\___|
# ___  ___          _                 _                  
# |  \/  |         | |               (_)                 
# | .  . | ___  ___| |__   __ _ _ __  _  ___ ___         
# | |\/| |/ _ \/ __| '_ \ / _` | '_ \| |/ __/ __|        
# | |  | |  __/ (__| | | | (_| | | | | | (__\__ \        
# \_|  |_/\___|\___|_| |_|\__,_|_| |_|_|\___|___/        
#  _           _                     _                   
# | |         | |                   | |                  
# | |     __ _| |__   ___  _ __ __ _| |_ ___  _ __ _   _ 
# | |    / _` | '_ \ / _ \| '__/ _` | __/ _ \| '__| | | |
# | |___| (_| | |_) | (_) | | | (_| | || (_) | |  | |_| |
# \_____/\__,_|_.__/ \___/|_|  \__,_|\__\___/|_|   \__, |
#                                                   __/ |
#                                                  |___/ 
#														  
# MIT License
# 
# Copyright (c) 2019 Probabilistic Mechanics Laboratory
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

from pinn_model import create_model

# =============================================================================
#     PINN PREDICTION
# =============================================================================

if __name__ == "__main__":
    
    def arrangeTable(table):
        data = np.transpose(np.asarray(np.transpose(table))[1:])
        if data.shape[1] == 1:
            data = np.repeat(data,2,axis=1)
        data = np.expand_dims(data,0)
        data = np.expand_dims(data,-1)
        space = np.asarray([np.asarray(table.iloc[:,0]),np.asarray([float(i) for i in table.columns[1:]])])
        table_shape = data.shape
        bounds = np.asarray([[np.min(space[0]),np.min(space[1])],[np.max(space[0]),np.max(space[1])]])
        return {'data':data, 'bounds':bounds, 'table_shape':table_shape}
    
    # Training
    parent_dir = os.path.dirname(os.getcwd())
    
    dfLoad = pd.read_csv(parent_dir+'\data\\DynamicLoad_30Years_adv.csv', index_col = None)
    dfLoad = dfLoad[:6*24*180*40]
    dfLoad = np.asarray(dfLoad).reshape((40,6*24*180))
    PLogFleet = np.log10(dfLoad)
    nFleet, n10min = PLogFleet.shape
    
    dfTemp = pd.read_csv(parent_dir+'\data\\BearingTemp_30Years_adv.csv', index_col = None)
    dfTemp = dfTemp[:6*24*180*40]
    dfTemp = np.asarray(dfTemp).reshape((40,6*24*180))
    BTempFleet = dfTemp
    
    dfCyc = pd.read_csv(parent_dir+'\data\\Cycles_30Years_adv.csv', index_col = None)
    dfCyc = dfCyc[:6*24*180*40]
    dfCyc = np.asarray(dfCyc).reshape((40,6*24*180))
    CycFleet = dfCyc
    
    dfdKappa = pd.read_csv(parent_dir+'\data\\Dkappa_30Years_adv.csv', index_col = None, header = None)
    dKappaFleet = np.asarray(dfdKappa.transpose())
    
    inputArray = np.dstack((dKappaFleet, CycFleet, PLogFleet, BTempFleet))    
    batch_input_shape = inputArray.shape
    
    myDtype = 'float32'
    d0RNN = np.asarray([0.0])
    selectdKappa = [len(d0RNN)]
    selectCycle = [len(d0RNN)+1]
    selectLoad = [len(d0RNN)+2]
    selectBTemp = [len(d0RNN)+3]
    d0RNN = d0RNN * np.ones((inputArray.shape[0], 1), dtype=myDtype)

    #Preliminaries
    a1 = 1.0
    C = 6000
    Pu = 750
    a = -10/3                  # Slope of linearized SN-Curve in log10-log10 space
    b = (10/3)*np.log10(C)+np.log10(1e6)+np.log10(a1)  # Interception of linearized SN-Curve in log10-log10 space
    
    # Load and manipulate required tables
    df = pd.read_csv(parent_dir+'\\tables\\aSKF.csv')
    aSKFTable = arrangeTable(df)
    df = pd.read_csv(parent_dir+'\\tables\\kappa.csv')
    kappaTable = arrangeTable(df)
    df = pd.read_csv(parent_dir+'\\tables\\etac.csv')
    etacTable = arrangeTable(df)
    
    
    # PINN Model
    model = create_model(a, b, Pu, 
                         aSKFTable['data'], aSKFTable['bounds'], aSKFTable['table_shape'],
                         kappaTable['data'], kappaTable['bounds'], kappaTable['table_shape'],
                         etacTable['data'], etacTable['bounds'], etacTable['table_shape'],
                         d0RNN, batch_input_shape,
                         selectdKappa, selectCycle, selectLoad, selectBTemp,
                         myDtype, return_sequences = True)
    
    
    result = model.predict(inputArray)
    summedPrediction = list(result[0,:,0])
    for halfyear in range(1,result.shape[0]):
        summedPrediction = summedPrediction + list(summedPrediction[-1]+result[halfyear,:,0])
    plt.plot(range(len(summedPrediction)),summedPrediction,'--',label = 'PINN Prediction')            
            
        
    virginDamage = pd.read_csv(parent_dir+'\data\\fatigueDamage_Virgin_adv.csv', index_col = None).dropna()
    degradedDamage = pd.read_csv(parent_dir+'\data\\fatigueDamage_Degraded_adv.csv', index_col = None).dropna()
    actualDamage = pd.read_csv(parent_dir+'\data\\fatigueDamage_Actual_adv.csv', index_col = None).dropna()
    
            
    plt.plot(range(virginDamage.shape[0]),virginDamage,'g-',label = 'Virgin Grease')
    plt.plot(range(degradedDamage.shape[0]),degradedDamage,'r-',label = 'Degraded Grease')
    plt.plot(range(actualDamage.shape[0]),actualDamage,'k-',label = 'Actual')

    plt.xticks(np.arange(0,365*6*24*25,step = 365*6*24),np.arange(0,25,step = 1))
    plt.xlim(0, 365*6*24*24)
    plt.ylim(0, 1.0)
    plt.xlabel('Time (Years)',fontsize = 14)
    plt.ylabel('Bearing Fatigue Damage',fontsize = 14)
    plt.legend()
    plt.grid(which='both')
    plt.tight_layout()
    plt.savefig('./plots/FatiguePrediction.png')
    