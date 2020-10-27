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

""" Custom layers """


from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.constraints import MinMaxNorm

from tensorflow.python.ops import gen_math_ops


from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

from tensorflow.python.framework import tensor_shape

class DOrC(Layer):
    """ Discrete ordinal classifier layer          
    """
    def __init__(self,
                 kernel_initializer = 'glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DOrC, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint  = constraints.get(kernel_constraint)
        
    def build(self, input_shape, **kwargs):
        self.threshold1 = self.add_weight("threshold1",
                                      shape = [1],
                                      initializer = self.kernel_initializer,
                                      constraint = MinMaxNorm(min_value=0.0, max_value=0.3, rate=1.0),
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      **kwargs)
        
        self.threshold2 = self.add_weight("threshold2",
                                      shape = [1],
                                      initializer = self.kernel_initializer,
                                      constraint = MinMaxNorm(min_value=0.2, max_value=0.5, rate=1.0),
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      **kwargs)
        
        self.threshold3 = self.add_weight("threshold3",
                                      shape = [1],
                                      initializer = self.kernel_initializer,
                                      constraint = MinMaxNorm(min_value=0.4, max_value=0.8, rate=1.0),
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      **kwargs)
        
        self.threshold4 = self.add_weight("threshold4",
                                      shape = [1],
                                      initializer = self.kernel_initializer,
                                      constraint = MinMaxNorm(min_value=0.8, max_value=2.0, rate=1.0),
                                      dtype = self.dtype,
                                      trainable = self.trainable,
                                      **kwargs)
        self.built = True

    def call(self, inputs): 
        first_threshold = 1/(1 + gen_math_ops.exp(-5e1*(inputs-self.threshold1))) 
        second_threshold = 1/(1 + gen_math_ops.exp(-5e1*(inputs*first_threshold-self.threshold2))) 
        third_threshold = 1/(1 + gen_math_ops.exp(-5e1*(inputs*second_threshold-self.threshold3))) 
        fourth_threshold = 1/(1 + gen_math_ops.exp(-5e1*(inputs*third_threshold-self.threshold4))) 
        output = 1 + first_threshold +second_threshold + third_threshold + fourth_threshold
        return output

    def compute_output_shape(self, input_shape):
        aux_shape = tensor_shape.TensorShape((None,1))
        return aux_shape[:-1].concatenate(1) 
    