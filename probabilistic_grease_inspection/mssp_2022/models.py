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

"Model functions"

import numpy as np

from pinn.layers import CumulativeDamageCell

from tensorflow.keras.losses import mean_absolute_error as mae
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input, Lambda, Dense, RNN
from tensorflow import concat, expand_dims


inspectionArray = np.asarray([6*24*30*1,6*24*30*2,6*24*30*3,6*24*30*4,6*24*30*5,6*24*30*6-1])


def build_mlp(dLInputScaling):
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


def maskedMAE(y_true, y_pred, inspectionArray=inspectionArray):
    y_true_masked = y_true
    y_pred_masked = concat([[[y_pred[0,inspectionArray[0],0],y_pred[0,inspectionArray[1],0],
                              y_pred[0,inspectionArray[2],0],y_pred[0,inspectionArray[3],0],
                              y_pred[0,inspectionArray[4],0],y_pred[0,inspectionArray[5],0]]]],1)
    for batch in range(1,y_pred.shape[0]):
        y_pred_masked = concat([y_pred_masked, concat([[[y_pred[batch,inspectionArray[0],0],y_pred[batch,inspectionArray[1],0],
                                                         y_pred[batch,inspectionArray[2],0],y_pred[batch,inspectionArray[3],0],
                                                         y_pred[batch,inspectionArray[4],0],y_pred[batch,inspectionArray[5],0]]]],1)],0)
        
    y_pred_masked = expand_dims(y_pred_masked,-1)
    
    return mae(y_true_masked,y_pred_masked)


def build_rnn_model(inspectionArray, greaseMLP, d0RNN, batch_input_shape, lowBounds_delgrs, upBounds_delgrs,
                 myDtype, lr, return_sequences = False, unroll = False):

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

    model.compile(loss=maskedMAE, optimizer=RMSprop(lr), metrics=[maskedMAE])
    return model
