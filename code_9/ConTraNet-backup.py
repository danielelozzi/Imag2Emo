# -*- coding: utf-8 -*-

#*----------------------------------------------------------------------------*
#* Copyright (C) 2024 Ruhr University Bochum, Germany.                        *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Authors: Omair Ali, M. Saif-ur-Rehman, Tobias Glasmachers,                 *
#* Ioannis Iossifidis, Christian Klaes.                                       *
#*----------------------------------------------------------------------------*


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from keras.callbacks import LearningRateScheduler



from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.convolutional import Conv2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import SpatialDropout2D, Dropout
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm





#################################################
#
# Learning Rate Constant Scheduling
#
#################################################
def step_decay(epoch):
    if(epoch < 20):
        lr = 0.001
    elif(epoch < 50):
        lr = 0.001
    else:
        lr = 0.0001
    return lr
lrate = LearningRateScheduler(step_decay)

###################### implement multilayer perceptron (MLP) ##############################

def mlp(x, hidden_units, dropout_rate,regRate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.elu,kernel_constraint = max_norm(regRate))(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


############# implement patch creation as a layer #########################################

class Patches(layers.Layer):
    def __init__(self, patch_height,patch_width,patch_depth):
        super(Patches, self).__init__()
        self.patch_height = patch_height
        self.patch_width  = patch_width
        self.patch_depth  = patch_depth

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_height, self.patch_width, 1],
            strides=[1, self.patch_height, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID", 
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches



# #################### implement the patch encoding layer ###################################

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        #self.projection = layers.Conv1D(filters=projection_dim, kernel_size=1, activation='elu')
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
        
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded



########################### model #############################################

def ConTraNet(nb_classes,Chans,Samples,dropoutRate,regRate,
          kernLength,poolLength,numFilters,
          dropoutType,projection_dim,
          transformer_layers,num_heads,
          transformer_units,mlp_head_units, training=True):
    


    """
    Inputs: 
        
            nb_classes:          number of classes to categorize 
            Chans:               number of channels
            Samples:             sequence length
            dropoutRate:         0.5 for CNN block
            regRate:             0.25
            kernLength:          for EEG: sampling frequency/2
            poolLength:          8
            numFilters:          16
            dropoutType:         SpatialDropout2D/Dropout
            projection_dim:      32
            transformer_layers:  1
            num_heads:           8
            transformer_units:   [projection_dim * 2 , projection_dim]
            mlp_head_units:      112
            
    """
                  
                          
    F1 = numFilters
    D = 2
    F2= numFilters*2
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
     
###############################################################################
    input1  = Input(shape = (Chans, Samples,1))

#############################  CNN block  #####################################  

    block1 = Conv2D(F1, (1, kernLength), padding = 'same', 
                            input_shape = (Chans, Samples,1),
                            use_bias = False, kernel_constraint = max_norm(regRate))(input1) #
    
    block1 = BatchNormalization(axis = 1)(block1)
    print('shape after 1st conv2D is ', block1.shape)
    block1 = Activation('elu')(block1) 
    block1 = AveragePooling2D((1, poolLength))(block1) # changed from 4 to 8 pool_size=(2, 2), strides=None
    block1 = dropoutType(dropoutRate)(block1)
    print('shape of average pooling output is ', block1.shape)               

    
 #################### Transformer Block #######################################  
    shape = block1.shape 
    patch_height = shape[1] 
    patch_width  = 10
    patch_depth  = shape[3] 
    stride =  1
    
    num_patches  = (shape[1]//patch_height) *((shape[2] - patch_width)//stride + 1)
 
    # Create patches.
    patches = Patches(patch_height,patch_width,patch_depth)(block1)
    print('shape of patches = ', patches.shape)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches) #encoded_patches

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Create a multi-head attention layer.
        attention_output, attention_score  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.5
        )(x1, x1, return_attention_scores=True)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
       
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.7,regRate=regRate)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)  
    
 ######################################### MLP head ###########################   
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.7,regRate=regRate) 
    # Classify outputs.
    logits = layers.Dense(nb_classes,activation=tf.nn.elu,kernel_constraint = max_norm(regRate))(features)
    # Create the Keras model.
    softmax      = Activation('softmax', name = 'softmax')(logits)
    
    if not training:
        model = Model(input1, [softmax, attention_score])
    else:
        model = Model(inputs=input1, outputs=softmax)

    return model


