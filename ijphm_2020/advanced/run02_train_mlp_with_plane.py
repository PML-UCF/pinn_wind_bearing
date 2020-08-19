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

import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau

from pinn.layers import getScalingDenseLayer

# =============================================================================
#     MLP TRAINING WITH PLANE
# =============================================================================

if __name__ == "__main__":
           
    def build_model(dLInputScaling):
        model = Sequential([
                dLInputScaling,
                Dense(40,activation = 'sigmoid'),
                Dense(20,activation = 'elu'),
                Dense(10,activation = 'elu'),
                Dense(5,activation = 'elu'),
                Dense(1,activation = 'sigmoid')
                ], name='plane_delgrs_mlp')
        optimizer = RMSprop(0.01)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model
    
    parent_dir = os.path.dirname(os.getcwd())
    
    dfPlane = pd.read_csv(parent_dir+'\data\\random_plane_set_500_adv.csv')

    inputsMLPTrain = dfPlane[['Dkappa','dynamicLoads','bearingTemp']]
    inputsMLPTrain_min = inputsMLPTrain.min(axis=0)
    inputsMLPTrain_range = inputsMLPTrain.max(axis=0) - inputsMLPTrain_min
    dLInputScaling = getScalingDenseLayer(inputsMLPTrain_min,inputsMLPTrain_range)
        
    MLPmodel = build_model(dLInputScaling)

    outputsMLPTrain = dfPlane[['delDkappa']]
    outputsMLPTrain_min = outputsMLPTrain.min(axis=0)
    outputsMLPTrain_range = outputsMLPTrain.max(axis=0) - outputsMLPTrain_min
    outputsMLPTrain_norm = (outputsMLPTrain - outputsMLPTrain_min)/outputsMLPTrain_range
    
    dfTrueDOE = pd.read_csv(parent_dir+'\data\\true_set_500_adv.csv')

    inputsMLPPred = dfTrueDOE[['Dkappa','dynamicLoads','bearingTemp']]
    
    outputsMLPPred = dfTrueDOE[['delDkappa']]
    outputsMLPPred_min = outputsMLPPred.min(axis=0)
    outputsMLPPred_range = outputsMLPPred.max(axis=0) - outputsMLPPred_min
    outputsMLPPred_norm = (outputsMLPPred - outputsMLPPred_min)/outputsMLPPred_range
    
    ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.85,
                                       min_lr = 1e-15, patience=30, verbose=1, mode='min')
    callbacks_list = [ReduceLR]
    MLP_EPOCHS = 500
    MLPhistory = MLPmodel.fit(inputsMLPTrain, outputsMLPTrain_norm, epochs=MLP_EPOCHS, verbose=1, callbacks=callbacks_list, validation_data= (inputsMLPPred,outputsMLPPred_norm))
    
    MLPmodel.save('.\models\MLP_RANDOM_PLANE.h5py')
    
    MLPresults = MLPmodel.predict(inputsMLPPred)
    
    fig = plt.figure(figsize =[5.0, 4.0])
    
    yLB = min(list(MLPresults)+list(outputsMLPPred_norm['delDkappa']))
    yUB = max(list(MLPresults)+list(outputsMLPPred_norm['delDkappa']))
    
    plt.plot(outputsMLPPred_norm['delDkappa'],MLPresults,'ro')
    plt.plot([yLB, yUB], [yLB, yUB], '--k')
    plt.xlabel('Actual',fontsize=14)
    plt.ylabel('Predicted',fontsize=14)
    plt.xlim(yLB, yUB)
    plt.ylim(yLB, yUB)
    plt.grid(which='both')
    plt.tight_layout()
    plt.savefig('./plots/Plane_ActualvsPredict.png')
    