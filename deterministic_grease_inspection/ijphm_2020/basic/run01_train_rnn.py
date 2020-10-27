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
import numpy as np
import os

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model

from models_and_functions import create_rnn_model

# =============================================================================
#     RNN TRAINING
# =============================================================================

if __name__ == "__main__":
    
    parent_dir = os.path.dirname(os.getcwd())
    
    # Import and manipulate input data 
    dfLoad = pd.read_csv(parent_dir+'\data\\DynamicLoad_6Months.csv', index_col = None)
    dfLoad = dfLoad.dropna()
    PFleet = np.transpose(np.asarray(dfLoad))
    PFleetInv = 1/PFleet
    
    dfTemp = pd.read_csv(parent_dir+'\data\\BearingTemp_6Months.csv', index_col = None)
    dfTemp = dfTemp.dropna()
    BTempFleet = np.transpose(np.asarray(dfTemp))
    
    inputArray = np.dstack((PFleetInv, BTempFleet))
    batch_input_shape = inputArray.shape
    
    myDtype = 'float32'
    # Set initial damage value 
    d0RNN = np.asarray([0.0])
    d0RNN = d0RNN * np.ones((inputArray.shape[0], 1), dtype=myDtype)
    
    # Import and set inspection data
    dfVsc = pd.read_csv(parent_dir+'\data\\ViscDamage_6Months.csv', index_col = None)
    dfVsc = np.asarray(dfVsc.dropna())
    
    inspectionArray = np.asarray([6*24*30*1,6*24*30*2,6*24*30*3,6*24*30*4,6*24*30*5,6*24*30*6-1])
    multipleInspections = np.transpose(np.asarray([dfVsc[inspectionArray,:]]))
    
    # Load MLP Model
    mlp_model = load_model('.\models\MLP_PLANE.h5py')
    mlp_model.trainable = True
    
    # Set upper and lower bounds for rescaling of MLP output
    dfPlane = pd.read_csv(parent_dir+'\data\\random_plane_set_500_bsc.csv', index_col = None)
    trainingSet_delgrs = dfPlane
    lowBounds_delgrs = np.asarray([np.min(trainingSet_delgrs['delDkappa'])])
    upBounds_delgrs = np.asarray([np.max(trainingSet_delgrs['delDkappa'])])
    
    # Create model
    RNNmodel = create_rnn_model(mlp_model, d0RNN, batch_input_shape,
                             lowBounds_delgrs, upBounds_delgrs,
                             myDtype, return_sequences = True, unroll = False)
    
    # Set training parameters
    ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.7,
                                       min_lr = 1e-15, patience=5, verbose=1, mode='min')
    
    callbacks_list = [ReduceLR]
    
    EPOCHS = 50
    
    # Train RNN
    history = RNNmodel.fit(inputArray, multipleInspections, epochs=EPOCHS, verbose=1, steps_per_epoch=1 , callbacks=callbacks_list)
    
    df = pd.DataFrame.from_dict(history.history)
    df.insert(loc = 0, column='epoch', value = history.epoch)
    df.to_csv("./models/lossHistory.csv", index = False)
    
    RNNmodel.save_weights('.\models\RNN_WEIGHTS.h5py')

    # Save the Dkappa output to use for prediction
    result = RNNmodel.predict(inputArray)

    dfres = pd.DataFrame(data=result[0,:,0].transpose())
    dfres.to_csv(parent_dir+'\data\\Dkappa_Prediction_6Months_bsc.csv',index=False, header=False)
    