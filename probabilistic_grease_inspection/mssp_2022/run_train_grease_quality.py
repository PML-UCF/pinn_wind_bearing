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

"Training RNN for grease degradation with stacked classifier and 30 years grease prediction"

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=14)

from pinn.layers import getScalingDenseLayer

from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
from tensorflow.random import set_seed
from models import build_mlp, build_rnn_model

np.random.seed(57)
set_seed(57)
    
if __name__ == "__main__":
    
    myDtype = 'float32'
    
    # Inspection time stamps
    inspectionArray = np.asarray([6*24*30*1,6*24*30*2,6*24*30*3,6*24*30*4,6*24*30*5,6*24*30*6-1])


    MeanGreaseDamage = np.asarray([np.asarray(pd.read_csv('./data/grease_data/Turbine'+str(turb)+'_101PRCTL_GreaseDamage_6Months.csv')['Percentile 0.5']) for turb in range(1,101)])
       
    SortedTurbines = np.flip(np.argsort(MeanGreaseDamage[:,-1]) + 1)
    
    no_trb = 20
    
    print("Number of training turbines: {}".format(no_trb))
    if no_trb == 100:
        TrainingTurbineIndex = np.arange(1,101,1, dtype='int32')
    elif no_trb == 50:
        TrainingTurbineIndex = SortedTurbines[1:100:2]
    elif no_trb == 20:
        TrainingTurbineIndex = SortedTurbines[1:100:5]
    elif no_trb == 10:
        TrainingTurbineIndex = SortedTurbines[1:100:10]


    # Import and pre-process input load and temperature data
    print("Importing and pre-processing input load and temperature data")
    dfLoad_Trg = pd.read_csv('./data/All_DynamicLoad_6Months.csv', index_col = None)[['Turbine'+str(i) for i in TrainingTurbineIndex]]
    PFleetInv = 1/np.transpose(np.asarray(dfLoad_Trg))
    
    dfTemp_Trg = pd.read_csv('./data/All_BearingTemp_6Months.csv', index_col = None)[['Turbine'+str(i) for i in TrainingTurbineIndex]]
    BTempFleet = np.transpose(np.asarray(dfTemp_Trg))
    
    inputArray_Trg = np.dstack((PFleetInv, BTempFleet))
    batch_input_shape = inputArray_Trg.shape
            
    # Initial damage
    d0RNN = np.asarray([0.0])
    d0RNN = d0RNN * np.ones((inputArray_Trg.shape[0], 1))
    
        
    # Assign grease quality quantile for training turbines
    prctl_all = np.round(np.linspace(2.5,97.5,101)/100, 10)
    prct_iters = np.random.choice(prctl_all, no_trb)

    # Collect true grease damage and inspection data for training turbines
    AllGreaseInspections = []
    AllGreaseDamage = []
    TurbineGreasePrctl = np.round(prct_iters,10)
    for turb in TrainingTurbineIndex:
        turb_idx = np.where(TrainingTurbineIndex == turb)[0][0]
        TurbineGreaseDamage = np.asarray(pd.read_csv('./data/grease_data/Turbine'+str(turb)+'_101PRCTL_GreaseDamage_6Months.csv')['Percentile '+str(TurbineGreasePrctl[turb_idx])])
        TurbineGreaseInspections = TurbineGreaseDamage[inspectionArray]
        AllGreaseInspections.append(TurbineGreaseInspections)
        AllGreaseDamage.append(TurbineGreaseDamage)
    AllGreaseInspections = np.asarray(AllGreaseInspections)
    AllGreaseDamage = np.asarray(AllGreaseDamage)
        
        
    # Load the initialized multi-layer perceptron
    mlp_weight_path = "./trained_models/approximate_plane_case10/cp.ckpt"
    
    trainingSet_delgrs = pd.read_csv('./data/approximate_plane_set.csv')
    inputsMLPTrain = trainingSet_delgrs[['Dkappa','dynamicLoads','bearingTemp']]
    inputsMLPTrain_min = inputsMLPTrain.min(axis=0)
    inputsMLPTrain_range = inputsMLPTrain.max(axis=0) - inputsMLPTrain_min
    dLInputScaling = getScalingDenseLayer(inputsMLPTrain_min,inputsMLPTrain_range)

    lowBounds_delgrs = np.asarray([0.0])
    upBounds_delgrs = np.asarray([1e-4])

    mlp_model = build_mlp(dLInputScaling)
    mlp_model.build((inputArray_Trg.shape[0], inputArray_Trg.shape[1], inputArray_Trg.shape[2]+1))
    mlp_model.load_weights(mlp_weight_path)
    mlp_model.trainable = True

    
    # Build RNN model
    RNNmodel = build_rnn_model(inspectionArray, mlp_model, d0RNN, batch_input_shape, lowBounds_delgrs, upBounds_delgrs,
                 myDtype, 5e-4, return_sequences = True, unroll = False)
        

    # Set training config
    RNN_EPOCHS = 200
    rnn_weight_path = "./trained_models/rnn_"+str(len(TrainingTurbineIndex))+"turb_"+str(RNN_EPOCHS)+"EP_MAE/cp.ckpt"
        
    ModelCheckpoint_LR = ModelCheckpoint(filepath=rnn_weight_path, monitor='loss',
                                                         verbose=1, save_best_only=True,
                                                         mode='min', save_weights_only=True) 
    
    ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.7, min_delta=1e-8,
                                       min_lr = 1e-15, patience=10, verbose=1, mode='min')
    ToNaN = TerminateOnNaN()
    callbacks_list = [ReduceLR, ToNaN, ModelCheckpoint_LR]
        
    # Train RNN
    history = RNNmodel.fit(inputArray_Trg, np.expand_dims(AllGreaseInspections,-1), epochs=RNN_EPOCHS, verbose=1, steps_per_epoch=1 , callbacks=callbacks_list)
        
    
    # Predict with trained RNN
    RNNmodel.load_weights(rnn_weight_path)
    result_after = RNNmodel.predict(inputArray_Trg, batch_size=no_trb)
    
    plt.figure()
    plt.plot(result_after[0,:,0], 'r-', label ='prediction median')
    plt.plot(AllGreaseDamage[0,:], 'k-', label ='true quantile')
    plt.plot(MeanGreaseDamage[TrainingTurbineIndex[0]-1,:].transpose(), 'b-', label ='true median')
    plt.plot(result_after[:,:,0].transpose(), 'r-')
    plt.plot(AllGreaseDamage.transpose(), 'k-')
    plt.plot(MeanGreaseDamage[TrainingTurbineIndex-1,:].transpose(), 'b-')
    plt.xlim(0,6*24*180)
    plt.ylim(0.0,1)
    plt.xticks(np.arange(0,25920+4320,step = 4320),np.arange(0,6+1,step = 1))
    plt.xlabel('months')
    plt.ylabel('grease damage')
    plt.grid(which='both')
    plt.legend()
    plt.tight_layout()
    
    

    # Extract true quantile ratio distribution for training set
    delta_ratio_true = np.zeros((no_trb,25919))
    for j in range(1,25920):
        delta_ratio_true[:,j-1] = (AllGreaseDamage[:,j] - AllGreaseDamage[:,j-1]) / (MeanGreaseDamage[TrainingTurbineIndex-1,j] - MeanGreaseDamage[TrainingTurbineIndex-1,j-1])
    true_quantile_c = np.mean(delta_ratio_true[:,600:],axis=1)


    # Predict quantile ratio using median grease damage predictions and inspection data
    def mn_sq_er(x, *args):
        return np.mean((args[0] - x * args[1])**2)
    

    pred_quantile_c = []
    for trb in range(len(TrainingTurbineIndex)):
        opt_res = minimize(fun = mn_sq_er,x0=1, args=(AllGreaseInspections[trb,:],result_after[trb,inspectionArray]))
        pred_quantile_c.append(opt_res.x[0])
    pred_quantile_c =np.asarray(pred_quantile_c)


    plt.figure()
    plt.step(np.sort(pred_quantile_c), np.linspace(0.025,0.975,no_trb), '-', linewidth=3, label='predicted')
    plt.step(np.sort(true_quantile_c), np.linspace(0.025,0.975,no_trb),'--', linewidth=3, label='true')
    plt.ylim(0.0,1.0)
    plt.xlabel('quantile ratio')
    plt.ylabel('cumulative distribution')
    plt.grid(which='both')
    plt.legend()
    plt.tight_layout()
 