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

from pyDOE import lhs

if __name__ == "__main__":
    
    # =============================================================================
    #   RANDOM PLANE GENERATION
    # =============================================================================
    
    def deltaGreaseDamagePlane(DOE,coefs):
        """Function to generate delta grease damage outputs for given DOE and plane coefficients
        """
        lowBound = 1e-7
        upBound = 1.3e-4
        delGrsDmg = lowBound + (upBound - lowBound) * (coefs[0]+coefs[1]*np.transpose(DOE)[0]-coefs[0]*np.transpose(DOE)[1]+coefs[2]*np.transpose(DOE)[2])
        return delGrsDmg
    
    npnts = 500
    Xolhs = lhs(n = 3, samples = npnts, criterion = 'maximin', iterations = 10)
    lowerBounds = np.asarray([[0.0,1/1500,60.0]])
    upperBounds = np.asarray([[1.0,1/500,80.0]])
    
    scaledXolhs =  np.repeat(lowerBounds, npnts, axis = 0) + Xolhs * (upperBounds - lowerBounds)
    
    coefrand = np.random.random(3)
    print(coefrand)
    dfPlane = pd.DataFrame({'dynamicLoads':np.transpose(scaledXolhs)[1],'bearingTemp':np.transpose(scaledXolhs)[2],'Dkappa':np.transpose(scaledXolhs)[0],'delDkappa':deltaGreaseDamagePlane(Xolhs,coefrand)})
    dfPlane.to_csv('./data/random_plane_set_'+str(npnts)+'.csv', index = False)
    