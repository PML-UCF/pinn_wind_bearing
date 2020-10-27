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

"Predicting bearing fatigue damage accumulation"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rc('font', size=14)
import pandas as pd

from models import create_pinn_model

from utils import arrange_table

if __name__ == "__main__":
    
    # Inspection time stamps and turbine indices
    inspectionArray = np.asarray([6*24*30*1,6*24*30*2,6*24*30*3,6*24*30*4,6*24*30*5,6*24*30*6-1])  
    TurbineIndex = range(1,121)
    ndata = 6*24*360*20
    
    regint = 6
    inspection_bias = 'baseline'
    RNN_EPOCHS = 2500    
    
    # Import and pre-process input data
    print('Loading input data..')
    dfLoad = pd.read_csv('./data/DynamicLoad_20Years.csv', index_col = None, nrows = ndata)[['Turbine'+str(i) for i in TurbineIndex]]
    PLogFleet = np.log10(np.transpose(np.asarray(dfLoad)))
    
    dfTemp = pd.read_csv('./data/BearingTemp_20Years.csv', index_col = None, nrows = ndata)[['Turbine'+str(i) for i in TurbineIndex]]
    BTempFleet = np.transpose(np.asarray(dfTemp))
    
    dfCyc = pd.read_csv('./data/Cycles_20Years.csv', index_col = None, nrows = ndata)[['Turbine'+str(i) for i in TurbineIndex]]
    CycFleet = np.transpose(np.asarray(dfCyc))
    
    dfGrs = pd.read_csv('./data/Pred_GreaseDamage_'+str(regint)+'MReg_'+inspection_bias+'_'+str(RNN_EPOCHS)+'epochs.csv', index_col = None, nrows = ndata)[['Turbine'+str(i) for i in TurbineIndex]]
    dGrsFleet = np.transpose(np.asarray(dfGrs))
    inputArray = np.dstack((dGrsFleet, CycFleet, PLogFleet, BTempFleet))
    
    myDtype = 'float32'
    batch_input_shape = inputArray.shape    
    
    #Preliminaries
    a1 = 1.0
    C = 6000
    Pu = 750
    a = -10/3                  # Slope of linearized SN-Curve in log10-log10 space
    b = (10/3)*np.log10(C)+np.log10(1e6)+np.log10(a1)  # Interception of linearized SN-Curve in log10-log10 space
      
    # Load and manipulate required tables
    df = pd.read_csv('./tables/askf_table.csv')
    aSKFTable = arrange_table(df)
    df = pd.read_csv('./tables/kappa_table.csv')
    kappaTable = arrange_table(df)
    df = pd.read_csv('./tables/etac_table.csv')
    etacTable = arrange_table(df)
    
    # Initial damage and variable selection masks    
    d0RNN = np.asarray([0.0])
    selectdKappa = [len(d0RNN)]
    selectCycle = [len(d0RNN)+1]
    selectLoad = [len(d0RNN)+2]
    selectBTemp = [len(d0RNN)+3]
    d0RNN = d0RNN * np.ones((inputArray.shape[0], 1))
        
    # Physics-informed neural network model
    model = create_pinn_model(a, b, Pu, 
                             aSKFTable['data'], aSKFTable['bounds'], aSKFTable['table_shape'],
                             kappaTable['data'], kappaTable['bounds'], kappaTable['table_shape'],
                             etacTable['data'], etacTable['bounds'], etacTable['table_shape'],
                             d0RNN, batch_input_shape,
                             selectdKappa, selectCycle, selectLoad, selectBTemp,
                             myDtype, return_sequences = True)
  
    print('Predicting..')   
    result = model.predict(inputArray)
   
    # Plot bearing fatigue damage propagation 
    dfBrg = pd.read_csv('./data/True_FatigueDamage_20Years.csv', index_col = None, nrows = ndata)[['Turbine'+str(i) for i in TurbineIndex]]
    fatigueTrue = np.transpose(np.asarray(dfBrg))
    turb = 1
    fig = plt.figure()
    plt.xlim(0,6*24*360*20)
    plt.ylim(0.0,1.0)
    plt.xticks(np.arange(0,6*24*360*22,step = 6*24*360*2),np.arange(0,22,step = 2))
    plt.plot(np.transpose(result[turb-1,:,0]), '--',label = 'prediction')
    plt.plot(fatigueTrue[turb-1,:], 'k-',label = 'actual')
    plt.xlabel('years')
    plt.ylabel('a$_{BRG}$')
    plt.legend()
    plt.grid(which='both')
