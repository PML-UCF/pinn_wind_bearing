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

"Grease sampling based on true grease degradation data"

import pandas as pd
import numpy as np
import scipy.stats as stat

if __name__ == "__main__":
    
    # Inspection time stamps
    inspectionArray = np.asarray([6*24*30*1,6*24*30*2,6*24*30*3,6*24*30*4,6*24*30*5,6*24*30*6-1])
    
    # Indices of training turbines
    TrainingTurbineIndex = [1, 6, 7, 11, 14, 20, 30, 36, 43, 63]
    
    # Import ground true grease damage data
    df_grs = pd.read_csv('./data/True_GreaseDamage_6Months.csv',nrows=6*24*180)[['Turbine'+str(i) for i in TrainingTurbineIndex]]
    true_grs = np.clip(np.asarray(df_grs)[inspectionArray],0,0.9999999999)
    
    # Define the inspection bias
    inspection_bias = 'baseline'
    
    if inspection_bias == 'baseline':
        ProbabilityArray = np.array([[0.7,0.3,0.0,0.0,0.0],
                                     [0.1,0.4,0.3,0.15,0.05],
                                     [0.05,0.1,0.35,0.3,0.2],
                                     [0.02,0.08,0.1,0.3,0.5],
                                     [0.0,0.0,0.0,0.2,0.8]])
    elif inspection_bias == 'conservative':
        ProbabilityArray = np.array([[0.2,0.5,0.15,0.1,0.05],
                                     [0.0,0.1,0.4,0.3,0.2],
                                     [0.0,0.0,0.2,0.5,0.3],
                                     [0.0,0.0,0.0,0.2,0.8],
                                     [0.0,0.0,0.0,0.0,1.0]])
    
    # Subroutine for discrete visual grease damage rank sampling
    inspectionResult = np.zeros((len(TrainingTurbineIndex),6,5))
    for turb in range(len(TrainingTurbineIndex)):
        turbineInspection = np.zeros((6,5))
        for month in range(6):
            lab_result = stat.multinomial(1,ProbabilityArray[int(true_grs[month,turb]/0.2)]).rvs()
            turbineInspection[month] = lab_result
        inspectionResult[turb] = turbineInspection
    
    # Post-process and export the grease sample data
    inspectionRanks = np.transpose(np.asarray([np.where(inspectionResult[i,:,:] == 1)[-1] for i in range(len(TrainingTurbineIndex))])) + 1
    df_insp = pd.DataFrame(inspectionRanks,index = inspectionArray, columns = ['Turbine'+str(i) for i in TrainingTurbineIndex])
    df_insp.to_csv('./data/GreaseInspection_'+inspection_bias+'_6Months.csv')


