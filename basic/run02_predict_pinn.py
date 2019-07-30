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

from models_and_functions import create_pinn_model, arrange_table

# =============================================================================
#     PINN PREDICTION
# =============================================================================

if __name__ == "__main__":
    
    parent_dir = os.path.dirname(os.getcwd())
    
    # Import and manipulate input data 
    dfLoad = pd.read_csv(parent_dir+'\data\\DynamicLoad_6Months.csv', index_col = None)['Turbine1']
    PLogFleet = np.log10(np.asarray(dfLoad))
    
    dfTemp = pd.read_csv(parent_dir+'\data\\BearingTemp_6Months.csv', index_col = None)['Turbine1']
    BTempFleet = np.asarray(dfTemp)
    
    dfCyc = pd.read_csv(parent_dir+'\data\\Cycles_6Months_bsc.csv', index_col = None)['Turbine1']
    CycFleet = np.asarray(dfCyc)
    
    dfdKappa = pd.read_csv(parent_dir+'\data\\Dkappa_Prediction_6Months_bsc.csv', index_col = None, header = None)
    dKappaFleet = np.asarray(dfdKappa.transpose())
    
    inputArray = np.dstack((dKappaFleet, CycFleet, PLogFleet, BTempFleet))
    batch_input_shape = inputArray.shape
    
    myDtype = 'float32'
    # Create masks for input selection
    d0RNN = np.asarray([0.0])
    selectdKappa = [len(d0RNN)]
    selectCycle = [len(d0RNN)+1]
    selectLoad = [len(d0RNN)+2]
    selectBTemp = [len(d0RNN)+3]
    d0RNN = d0RNN * np.ones((inputArray.shape[0], 1), dtype=myDtype)

    # Preliminaries
    a1 = 1.0                                           # 90% reliability life modification factor
    C = 6000                                           # Basic dynamic load rating of the bearing (kN)
    Pu = 750                                           # Fatigue load limit of the bearing (kN)
    a = -10/3                                          # Slope of linearized SN-Curve in log10-log10 space
    b = (10/3)*np.log10(C)+np.log10(1e6)+np.log10(a1)  # Interception of linearized SN-Curve in log10-log10 space
    
    # Import and manipulate required tables
    df = pd.read_csv(parent_dir+'\\tables\\aSKF.csv')
    aSKFTable = arrange_table(df)
    df = pd.read_csv(parent_dir+'\\tables\\kappa.csv')
    kappaTable = arrange_table(df)
    df = pd.read_csv(parent_dir+'\\tables\\etac.csv')
    etacTable = arrange_table(df)
    
    # Create PINN Model
    model = create_pinn_model(a, b, Pu, 
                              aSKFTable['data'], aSKFTable['bounds'], aSKFTable['table_shape'],
                              kappaTable['data'], kappaTable['bounds'], kappaTable['table_shape'],
                              etacTable['data'], etacTable['bounds'], etacTable['table_shape'],
                             d0RNN, batch_input_shape,
                             selectdKappa, selectCycle, selectLoad, selectBTemp,
                             myDtype, return_sequences = True)
    
    # Predict fatigue damage
    result = model.predict(inputArray)
    
    # Plot prediction against true damage
    actualDamage = pd.read_csv(parent_dir+'\data\\True_FatigueDamage_6Months_bsc.csv', index_col = None)

    plt.plot(range(actualDamage.shape[0]),actualDamage,'k-',label = 'Actual')
    plt.plot(range(result.shape[1]),result[0,:,:],'--',label = 'PINN Prediction')

    plt.xticks(np.arange(0,210*6*24*25,step = 30*6*24),np.arange(0,7,step = 1))
    plt.xlim(0, 180*6*24)
    plt.ylim(0, 0.05)
    plt.xlabel('Time (Months)',fontsize = 14)
    plt.ylabel('Bearing Fatigue Damage',fontsize = 14)
    plt.legend()
    plt.grid(which='both')
    plt.tight_layout()
    plt.show()
        