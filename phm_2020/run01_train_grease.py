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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=14)

from pinn.layers import getScalingDenseLayer

from tensorflow.keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint

from models import build_mlp, create_stacked_rnn, rnn_pred_model, discrete_ordinal_classifier

from utils import plot_confusion_matrix

    
if __name__ == "__main__":
    
    myDtype = 'float32'
    
    # Inspection time stamps and turbine indices
    inspectionArray = np.asarray([6*24*30*1,6*24*30*2,6*24*30*3,6*24*30*4,6*24*30*5,6*24*30*6-1])
    TrainingTurbineIndex = [1, 6, 7, 11, 14, 20, 30, 36, 43, 63]
    TestTurbineIndex = range(1,121)
    
    # Import and pre-process input load and temperature data
    print("Importing and pre-processing input load and temperature data")
    dfLoad = pd.read_csv('./data/DynamicLoad_20Years.csv', index_col = None, nrows = 6*24*360*20)[['Turbine'+str(i) for i in TestTurbineIndex]]
    dfLoad_Trg = dfLoad[['Turbine'+str(i) for i in TrainingTurbineIndex]].head(6*24*180)
    PFleetInv = 1/np.transpose(np.asarray(dfLoad_Trg))
    
    dfTemp = pd.read_csv('./data/BearingTemp_20Years.csv', index_col = None, nrows = 6*24*360*20)[['Turbine'+str(i) for i in TestTurbineIndex]]
    dfTemp_Trg = dfTemp[['Turbine'+str(i) for i in TrainingTurbineIndex]].head(6*24*180)
    BTempFleet = np.transpose(np.asarray(dfTemp_Trg))
    
    inputArray = np.dstack((PFleetInv, BTempFleet))
    
    batch_input_shape = inputArray.shape
    
    # Initial damage
    d0RNN = np.asarray([0.0])
    d0RNN = d0RNN * np.ones((inputArray.shape[0], 1))
    
    # Define the inspection bias and import inspection data
    inspection_bias = 'baseline'
    
    dfVsc = pd.read_csv('./data/GreaseInspection_'+inspection_bias+'_6Months.csv')[['Turbine'+str(i) for i in TrainingTurbineIndex]]
    multipleInspections = np.asarray(dfVsc, dtype='int32')

    # Load the initialized multi-layer perceptron
    mlp_weight_path = "./trained_models/approximate_plane_case10/cp.ckpt"
    
    trainingSet_delgrs = pd.read_csv('./data/approximate_plane_set.csv')
    inputsMLPTrain = trainingSet_delgrs[['Dkappa','dynamicLoads','bearingTemp']]
    inputsMLPTrain_min = inputsMLPTrain.min(axis=0)
    inputsMLPTrain_range = inputsMLPTrain.max(axis=0) - inputsMLPTrain_min
    dLInputScaling = getScalingDenseLayer(inputsMLPTrain_min,inputsMLPTrain_range)

    lowBounds_delgrs = np.asarray([0.05/25920])
    upBounds_delgrs = np.asarray([np.max(trainingSet_delgrs['delDkappa'])])

    mlp_model = build_mlp(dLInputScaling)
    mlp_model.load_weights(mlp_weight_path)
    mlp_model.trainable = True
    
    # Get the discrete ordinal classifier
    cl_model = discrete_ordinal_classifier(np.expand_dims(np.transpose(multipleInspections), -1).shape, myDtype)   
    cl_model.build(np.expand_dims(np.transpose(multipleInspections), -1).shape)
    cl_layer = cl_model.layers[0]
    cl_layer.trainable = True

    # Build stacked RNN model
    RNNmodel = create_stacked_rnn(cl_layer, inspectionArray, mlp_model, d0RNN, batch_input_shape,
                                  lowBounds_delgrs, upBounds_delgrs,
                                  myDtype, return_sequences = True, unroll = False)
    
    # Set training config
    RNN_EPOCHS = 2500
    rnn_weight_path = "./trained_models/rnn_"+inspection_bias+"_"+str(RNN_EPOCHS)+"EP/cp.ckpt"
    
    ModelCheckpoint = ModelCheckpoint(filepath=rnn_weight_path, monitor='loss',
                                                         verbose=1, save_best_only=True,
                                                         mode='min', save_weights_only=True) 
    ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.85,
                                       min_lr = 1e-15, patience=20, verbose=1, mode='min')
    ToNaN = TerminateOnNaN()
    callbacks_list = [ReduceLR, ToNaN, ModelCheckpoint]

    # Train RNN
    history = RNNmodel.fit(inputArray, np.expand_dims(np.transpose(multipleInspections), -1), epochs=RNN_EPOCHS, verbose=1, steps_per_epoch=1 , callbacks=callbacks_list)
    
    # Predict with trained RNN
    RNNmodel.load_weights(rnn_weight_path)
    result_after = RNNmodel.predict(inputArray)
    
    # Plot confusion matrix
    conf_mat = np.zeros((5,5))
    for turb in range(10):
        for month in range(6):
            conf_mat[np.round(result_after[turb,inspectionArray[month],0]).astype('int32')-1,multipleInspections[month,turb]-1] += 1
            
    plot_confusion_matrix(np.flip(conf_mat,axis=0), range(1,6), normalize=False)

    # Extract the multi-layer perceptron weights
    CDMWeights = RNNmodel.layers[0].get_weights()[:12]
    
    grease_mlp = build_mlp(dLInputScaling)
    grease_mlp.set_weights(RNNmodel.layers[0].cell.model.layers[1].get_weights())
    
    # Predict and export grease damage for entire fleet
    print("Predicting and exporting grease damage for entire fleet")
    df_grs = pd.DataFrame()
    regint = 6 # Regreasing interval
    dummy_machines = int(20*12/regint)
    dummy_time = 6*24*30*regint
    for trb in TestTurbineIndex:
        PFleetInv = 1/np.transpose(np.asarray(dfLoad['Turbine'+str(trb)])).reshape((dummy_machines,dummy_time))
        BTempFleet = np.transpose(np.asarray(dfTemp['Turbine'+str(trb)])).reshape((dummy_machines,dummy_time))
        inputArray_Trb = np.dstack((PFleetInv, BTempFleet))
        
        batch_input_shape = inputArray_Trb.shape
                
        # Build model for grease damage predictions
        d0RNN = np.asarray([0.0])
        d0RNN = d0RNN * np.ones((dummy_machines, 1))
        
        RNNPredModel = rnn_pred_model(inspectionArray, grease_mlp, d0RNN, batch_input_shape, lowBounds_delgrs, upBounds_delgrs,
                         myDtype, return_sequences = True, unroll = False)
    
        result_training = RNNPredModel.predict(inputArray_Trb)
        result_reshaped = result_training[:,:,0].reshape((dummy_machines*dummy_time))
        df_grs['Turbine'+str(trb)] = result_reshaped
        
    # Export grease damage data
    df_grs.to_csv('./data/Pred_GreaseDamage_'+str(regint)+'MReg_'+inspection_bias+'_'+str(RNN_EPOCHS)+'epochs.csv', index=False)
        
    # Plot grease damage propagation
    fig = plt.figure()
    plt.title('prediction')
    plt.xlim(0,6*24*180)
    plt.ylim(0.0,1.0)
    plt.xticks(np.arange(0,25920+4320,step = 4320),np.arange(0,6+1,step = 1))
    plt.plot(np.asarray(df_grs)[:6*24*180,np.asarray(TrainingTurbineIndex)-1])
    plt.xlabel('months')
    plt.ylabel('$a_{GRS}$')
    plt.grid(which='both')
    plt.tight_layout()
    
    grease_true = np.asarray(pd.read_csv('./data/True_GreaseDamage_6Months.csv',nrows=6*24*180)[['Turbine'+str(i) for i in TrainingTurbineIndex]])
    fig = plt.figure()
    plt.title('true')
    plt.xlim(0,6*24*180)
    plt.ylim(0.0,1.0)
    plt.xticks(np.arange(0,25920+4320,step = 4320),np.arange(0,6+1,step = 1))
    plt.plot(grease_true)
    plt.xlabel('months')
    plt.ylabel('$a_{GRS}$')
    plt.grid(which='both')
    plt.tight_layout()
