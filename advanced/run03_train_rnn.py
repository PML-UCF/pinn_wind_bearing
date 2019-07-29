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
import matplotlib.pyplot as plt
import os

from pinn.layers import CumulativeDamageCell

from tensorflow import concat, expand_dims
from tensorflow.keras import Sequential
from tensorflow.keras.losses import mean_squared_error as mse
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda, RNN

# =============================================================================
#     RNN TRAINING
# =============================================================================

if __name__ == "__main__":
        
        inspectionArray = np.asarray([6*24*30*1,6*24*30*2,6*24*30*3,6*24*30*4,6*24*30*5,6*24*30*6-1])
    
        def maskedLoss(y_true, y_pred, inspectionArray=inspectionArray):
            y_true_masked = y_true
            y_pred_masked = concat([[[y_pred[0,inspectionArray[0],0],y_pred[0,inspectionArray[1],0],y_pred[0,inspectionArray[2],0],y_pred[0,inspectionArray[3],0],y_pred[0,inspectionArray[4],0],y_pred[0,inspectionArray[5],0]]]],1)
            for batch in range(1,y_pred.shape[0]):
                y_pred_masked = concat([y_pred_masked, concat([[[y_pred[batch,inspectionArray[0],0],y_pred[batch,inspectionArray[1],0],y_pred[batch,inspectionArray[2],0],y_pred[batch,inspectionArray[3],0],y_pred[batch,inspectionArray[4],0],y_pred[batch,inspectionArray[5],0]]]],1)],0)
                
            y_pred_masked = expand_dims(y_pred_masked,-1)
            return mse(y_true_masked,y_pred_masked)
        
        def create_model(inspectionArray, greaseMLP, d0RNN, batch_input_shape, lowBounds_delgrs, upBounds_delgrs,
                         myDtype, return_sequences = False, unroll = False):
        
            placeHolder = Input(shape=(batch_input_shape[2]+1,)) #Adding states
            
            MLPOutputs = greaseMLP(placeHolder)
            
            scaledMLPOutputs = Lambda(lambda x, lowBounds_delgrs=lowBounds_delgrs, upBounds_delgrs=upBounds_delgrs:
                x*(upBounds_delgrs-lowBounds_delgrs)+lowBounds_delgrs)(MLPOutputs)

                
            functionalModel = Model(inputs = [placeHolder], outputs = [scaledMLPOutputs])
            
            "-------------------------------------------------------------------------"
            CDMCellHybrid = CumulativeDamageCell(model = functionalModel,
                                               batch_input_shape = batch_input_shape,
                                               dtype = myDtype,
                                               initial_damage = d0RNN)
            
            CDMRNNhybrid = RNN(cell = CDMCellHybrid,
                               return_sequences = return_sequences,
                               return_state = False,
                               batch_input_shape = batch_input_shape,
                               unroll = unroll)
            
            model = Sequential()
            model.add(CDMRNNhybrid)
            model.compile(loss= maskedLoss, optimizer=RMSprop(5e-4), metrics=[maskedLoss])
            return model

        parent_dir = os.path.dirname(os.getcwd())
        
        dfLoad = pd.read_csv(parent_dir+'\data\\DynamicLoad_6Months.csv', index_col = None)
        dfLoad = dfLoad.dropna()
        PFleet = np.transpose(np.asarray(dfLoad))
        PFleetInv = 1/PFleet
        
        dfTemp = pd.read_csv(parent_dir+'\data\\BearingTemp_6Months.csv', index_col = None)
        dfTemp = dfTemp.dropna()
        BTempFleet = np.transpose(np.asarray(dfTemp))
        
        inputArray = np.dstack((PFleetInv, BTempFleet))
        
        myDtype = 'float32'
        batch_input_shape = inputArray.shape
        
        d0RNN = np.asarray([0.0])
        d0RNN = d0RNN * np.ones((inputArray.shape[0], 1), dtype=myDtype)
        
        dfVsc = pd.read_csv(parent_dir+'\data\\ViscDamage_6Months.csv', index_col = None)
        dfVsc = np.asarray(dfVsc.dropna())
            
        multipleInspections = np.transpose(np.asarray([dfVsc[inspectionArray,:]]))
        
        ReduceLR = ReduceLROnPlateau(monitor='loss', factor=0.7,
                                           min_lr = 1e-15, patience=5, verbose=1, mode='min')
        
        callbacks_list = [ReduceLR]
        
        EPOCHS = 20
        
        mlp_model = load_model('.\models\MLP_RANDOM_PLANE.h5py')
        mlp_model.trainable = True
        
        dfPlane = pd.read_csv(parent_dir+'\data\\random_plane_set_500_adv.csv', index_col = None)
        trainingSet_delgrs = dfPlane
        lowBounds_delgrs = np.asarray([np.min(trainingSet_delgrs['delDkappa'])])
        upBounds_delgrs = np.asarray([np.max(trainingSet_delgrs['delDkappa'])])
        
        RNNmodel = create_model(inspectionArray, mlp_model, d0RNN, batch_input_shape,
                                 lowBounds_delgrs, upBounds_delgrs,
                                 myDtype, return_sequences = True, unroll = False)
        
        result_before = RNNmodel.predict(inputArray)
        
        InspectionsBefore = np.asarray([np.transpose(result_before)[0][inspectionArray,:]])
                
        history = RNNmodel.fit(inputArray, multipleInspections, epochs=EPOCHS, verbose=1, steps_per_epoch=1 , callbacks=callbacks_list)
        
        df = pd.DataFrame.from_dict(history.history)
        df.insert(loc = 0, column='epoch', value = history.epoch)
        df.to_csv("./models/lossHistory.csv", index = False)
        
        RNNmodel.save_weights('.\models\RNN_WEIGHTS_RANDOM_PLANE.h5py')
    
        
        result = RNNmodel.predict(inputArray)

        
        InspectionsPreds = np.asarray([np.transpose(result)[0][inspectionArray,:]])
        
        dfLoad = pd.read_csv(parent_dir+'\data\\DynamicLoad_6Months_Val_adv.csv', index_col = None)
        dfLoad = dfLoad.dropna()
        PFleet = np.transpose(np.asarray(dfLoad))
        PFleetInv = 1/PFleet
        
        dfTemp = pd.read_csv(parent_dir+'\data\\BearingTemp_6Months_Val_adv.csv', index_col = None)
        dfTemp = dfTemp.dropna()
        BTempFleet = np.transpose(np.asarray(dfTemp))
        
        
        InputArrayVal = np.dstack((PFleetInv, BTempFleet))
        batch_input_shape = InputArrayVal.shape
        
        d0RNN = np.asarray([0.0])
        d0RNN = d0RNN * np.ones((InputArrayVal.shape[0], 1), dtype=myDtype)
        
        RNNValmodel = create_model(inspectionArray, mlp_model, d0RNN, batch_input_shape,
                                 lowBounds_delgrs, upBounds_delgrs,
                                 myDtype, return_sequences = True, unroll = False)
        RNNValmodel.load_weights('.\models\RNN_WEIGHTS_RANDOM_PLANE.h5py')
        resultVal = RNNValmodel.predict(InputArrayVal)
        
        dfVscVal = pd.read_csv(parent_dir+'\data\\ViscDamage_6Months_Val_adv.csv', index_col = None)
        dfVscVal = np.asarray(dfVscVal.dropna())
            
        multipleValInspections = np.asarray([dfVscVal[inspectionArray,:]])
        InspectionsVal = np.asarray([np.transpose(resultVal)[0][inspectionArray,:]])
        
        fig = plt.figure()
        
        yLB = min(list(InspectionsVal.flatten())+list(multipleValInspections.flatten())+list(InspectionsPreds.flatten())+list(InspectionsBefore.flatten())+list(multipleInspections.flatten()))
        yUB = max(list(InspectionsVal.flatten())+list(multipleValInspections.flatten())+list(InspectionsPreds.flatten())+list(InspectionsBefore.flatten())+list(multipleInspections.flatten()))
        
        plt.plot(np.transpose(multipleInspections).flatten(),InspectionsBefore.flatten(),'bo',label = 'Before Training')
        plt.plot(np.transpose(multipleInspections).flatten(),InspectionsPreds.flatten(),'ro',label = 'After Training')
        plt.plot(multipleValInspections.flatten(),InspectionsVal.flatten(),'ko',label = 'Validation')
        plt.plot([yLB, yUB], [yLB, yUB], '--k')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.xlim(yLB, yUB)
        plt.ylim(yLB, yUB)
        plt.grid(which='both')
        plt.legend()
        plt.savefig('./plots/ActualvsPredict.png')
        
        dfLoad = pd.read_csv(parent_dir+'\data\\DynamicLoad_30Years_adv.csv', index_col = None)
        dfLoad = dfLoad[:6*24*180*40]
        dfLoad = np.asarray(dfLoad).reshape((40,6*24*180))
        PFleetInv = 1/dfLoad
        
        dfTemp = pd.read_csv(parent_dir+'\data\\BearingTemp_30Years_adv.csv', index_col = None)
        dfTemp = dfTemp[:6*24*180*40]
        dfTemp = np.asarray(dfTemp).reshape((40,6*24*180))
        BTempFleet = dfTemp

        InputArray30 = np.dstack((PFleetInv, BTempFleet))
        batch_input_shape = InputArray30.shape
        
        d0RNN = np.asarray([0.0])
        d0RNN = d0RNN * np.ones((InputArray30.shape[0], 1), dtype=myDtype)
        
        RNN30model = create_model(inspectionArray, mlp_model, d0RNN, batch_input_shape,
                                 lowBounds_delgrs, upBounds_delgrs,
                                 myDtype, return_sequences = True, unroll = False)
        RNN30model.load_weights('.\models\RNN_WEIGHTS_RANDOM_PLANE.h5py')
        
        result30 = RNN30model.predict(InputArray30)

        dfres = pd.DataFrame(data=result30[:,:,0].transpose())
        dfres.to_csv(parent_dir+'\data\\Dkappa_30Years_adv.csv',index=False, header=False)
        