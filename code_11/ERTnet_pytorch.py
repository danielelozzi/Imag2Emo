import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def transformer_block_def(embed_dim, num_heads, rate=0.1):
    """
    Transformer Block in PyTorch functional style (def).
    """
    multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads)
    ffn = nn.Linear(embed_dim, embed_dim, bias=False) # activation='elu' is applied in forward
    layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
    layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
    dropout = nn.Dropout(rate)

    def forward(inputs):
        attn_output, _ = multi_head_attn(inputs, inputs, inputs) # PyTorch MHA returns attn_output and attn_output_weights
        out1 = layernorm1(inputs + attn_output)
        ffn_output = ffn(out1)
        ffn_output = F.elu(ffn_output) # Applying elu activation here
        ffn_output = dropout(ffn_output)
        out = layernorm2(out1 + ffn_output)
        return out

    return forward

def positional_encoding_def(d_model, max_len=500):
    """
    Positional Encoding Layer in PyTorch functional style (def).
    """
    def get_angles(pos, i):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angles

    def forward(inputs):
        length = inputs.shape[2] # length is now the last dimension
        pos_encoding = np.zeros((length, d_model))
        for pos in range(length):
            for i in range(d_model):
                if i % 2 == 0:
                    pos_encoding[pos, i] = np.sin(get_angles(pos, i))
                else:
                    pos_encoding[pos, i] = np.cos(get_angles(pos, i))
        pos_encoding = pos_encoding[np.newaxis, ...]
        pos_encoding = torch.from_numpy(pos_encoding).float() # Convert numpy array to torch tensor and cast to float
        return inputs + pos_encoding

    return forward


def ertnet_def(nb_classes, Chans=64, Samples=128,
             dropoutRate=0.5, kernLength=64, F1=8, heads=8,
             D=2, F2=16):
    """
    ERTNet (Emotion Recognition Transformer Network) model in PyTorch functional style (def).
    Input shape: (batch_size, n_channel, n_times)
    """
    def forward(input1):
        # input1 shape: (batch_size, n_channel, n_times)
        x = input1.unsqueeze(1) # add channel dimension -> (batch_size, 1, n_channel, n_times)

        block1 = F.conv2d(x, weight=torch.randn(F1, 1, 1, kernLength), padding='same', bias=None) # Conv2D, input shape is now (batch_size, 1, n_channel, n_times)
        block1 = nn.BatchNorm2d(F1)(block1) # BatchNormalization

        depthwise_weight = torch.randn(F1*D, 1, Chans, 1) # DepthwiseConv2D weights
        block1 = F.conv2d(block1, weight=depthwise_weight, bias=None, groups=F1, padding='valid', stride=1) # DepthwiseConv2D
        block1 = nn.BatchNorm2d(F1*D)(block1) # BatchNormalization
        block1 = F.elu(block1) # Activation('elu')
        block1 = F.avg_pool2d(block1, kernel_size=(1, 4)) # AveragePooling2D
        block1 = F.dropout(block1, p=dropoutRate, training=True) # Dropout

        block2 = F.conv2d(block1, weight=torch.randn(F2, F1*D, 1, 16), padding='same', bias=None) # SeparableConv2D
        block2 = nn.BatchNorm2d(F2)(block2) # BatchNormalization
        block2 = F.elu(block2) # Activation('elu')
        block2 = torch.squeeze(block2, dim=2) # Lambda (squeeze)

        pos_en_forward = positional_encoding_def(F2, 500) # PositionalEncoding
        pos_en = pos_en_forward(block2)

        attention1_forward = transformer_block_def(embed_dim=F2, num_heads=heads, rate=dropoutRate) # TransformerBlock
        transformer1 = attention1_forward(pos_en)

        outputs = F.adaptive_avg_pool1d(transformer1, 1).squeeze(dim=2) # GlobalAveragePooling1D and squeeze
        outputs = F.dropout(outputs, p=dropoutRate, training=True) # Dropout
        dense = nn.Linear(F2, nb_classes, bias=False)(outputs) # Dense
        softmax = F.softmax(dense, dim=1) # Activation('softmax')
        return softmax

    return forward


def deepconvnet_def(nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
    """
    DeepConvNet model in PyTorch functional style (def).
    Input shape: (batch_size, n_channel, n_times)
    """
    def forward(input_main):
        x = input_main.unsqueeze(1) # add channel dimension -> (batch_size, 1, n_channel, n_times)

        block1 = F.conv2d(x, weight=torch.randn(25, 1, 1, 2), kernel_constraint = lambda x: torch.clip(x, -2, 2)) # Conv2D, input shape is now (batch_size, 1, n_channel, n_times)
        block1 = F.conv2d(block1, weight=torch.randn(25, 25, Chans, 1), kernel_constraint = lambda x: torch.clip(x, -2, 2)) # Conv2D
        block1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1)(block1) # BatchNormalization
        block1 = F.relu(block1) # Activation('relu')
        block1 = F.max_pool2d(block1, kernel_size=(1, 2), stride=(1, 2)) # MaxPooling2D
        block1 = F.dropout(block1, p=dropoutRate, training=True) # Dropout

        block2 = F.conv2d(block1, weight=torch.randn(50, 25, 1, 2), kernel_constraint = lambda x: torch.clip(x, -2, 2)) # Conv2D
        block2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1)(block2) # BatchNormalization
        block2 = F.relu(block2) # Activation('relu')
        block2 = F.max_pool2d(block2, kernel_size=(1, 2), stride=(1, 2)) # MaxPooling2D
        block2 = F.dropout(block2, p=dropoutRate, training=True) # Dropout

        block3 = F.conv2d(block2, weight=torch.randn(100, 50, 1, 2), kernel_constraint = lambda x: torch.clip(x, -2, 2)) # Conv2D
        block3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1)(block3) # BatchNormalization
        block3 = F.relu(block3) # Activation('relu')
        block3 = F.max_pool2d(block3, kernel_size=(1, 2), stride=(1, 2)) # MaxPooling2D
        block3 = F.dropout(block3, p=dropoutRate, training=True) # Dropout

        block4 = F.conv2d(block3, weight=torch.randn(200, 100, 1, 5), kernel_constraint = lambda x: torch.clip(x, -2, 2)) # Conv2D
        block4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1)(block4) # BatchNormalization
        block4 = F.relu(block4) # Activation('relu')
        block4 = F.max_pool2d(block4, kernel_size=(1, 2), stride=(1, 2)) # MaxPooling2D
        block4 = F.dropout(block4, p=dropoutRate, training=True) # Dropout

        flatten = torch.flatten(block4, 1) # Flatten
        dense = nn.Linear(flatten.shape[1], nb_classes, bias=True) # Dense, kernel_constraint is applied in forward pass
        dense.weight.data = torch.clip(dense.weight.data, -0.5, 0.5) # kernel_constraint = max_norm(0.5)

        softmax = F.softmax(dense, dim=1) # Activation('softmax')
        return softmax

    return forward

def square_def(x):
    return torch.square(x)

def log_def(x):
    return torch.log(torch.clip(x, min=1e-7, max=10000))

def shallowconvnet_def(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """
    ShallowConvNet model in PyTorch functional style (def).
    Input shape: (batch_size, n_channel, n_times)
    """
    def forward(input_main):
        x = input_main.unsqueeze(1) # add channel dimension -> (batch_size, 1, n_channel, n_times)

        block1 = F.conv2d(x, weight=torch.randn(40, 1, 1, 13), kernel_constraint = lambda x: torch.clip(x, -2, 2)) # Conv2D, input shape is now (batch_size, 1, n_channel, n_times)
        block1 = F.conv2d(block1, weight=torch.randn(40, 40, Chans, 1), bias=False, kernel_constraint = lambda x: torch.clip(x, -2, 2)) # Conv2D
        block1 = nn.BatchNorm2d(40, eps=1e-05, momentum=0.9)(block1) # BatchNormalization
        block1 = square_def(block1) # Activation(square)
        block1 = F.avg_pool2d(block1, kernel_size=(1, 7), stride=(1, 3)) # AveragePooling2D
        block1 = log_def(block1) # Activation(log)
        block1 = F.dropout(block1, p=dropoutRate, training=True) # Dropout
        flatten = torch.flatten(block1, 1) # Flatten
        dense = nn.Linear(flatten.shape[1], nb_classes, bias=True) # Dense, kernel_constraint is applied in forward pass
        dense.weight.data = torch.clip(dense.weight.data, -0.5, 0.5) # kernel_constraint = max_norm(0.5)
        softmax = F.softmax(dense, dim=1) # Activation('softmax')
        return softmax
    return forward


def eegnet_def(nb_classes, Chans=64, Samples=128,
             dropoutRate=0.5, kernLength=64, F1=8,
             D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """
    EEGNet model in PyTorch functional style (def).
    Input shape: (batch_size, n_channel, n_times)
    """
    dropout_layer = nn.Dropout2d if dropoutType == 'SpatialDropout2D' else nn.Dropout # Define dropout type based on string input

    def forward(input1):
        x = input1.unsqueeze(1) # add channel dimension -> (batch_size, 1, n_channel, n_times)

        block1 = F.conv2d(x, weight=torch.randn(F1, 1, 1, kernLength), padding='same', bias=False) # Conv2D, input shape is now (batch_size, 1, n_channel, n_times)
        block1 = nn.BatchNorm2d(F1)(block1) # BatchNormalization

        depthwise_weight = torch.randn(F1*D, 1, Chans, 1) # DepthwiseConv2D weights
        block1 = F.conv2d(block1, weight=depthwise_weight, bias=None, groups=F1, padding='valid', stride=1) # DepthwiseConv2D
        block1 = nn.BatchNorm2d(F1*D)(block1) # BatchNormalization
        block1 = F.relu(block1) # Activation('relu')
        block1 = F.avg_pool2d(block1, kernel_size=(1, 4)) # AveragePooling2D
        block1_dropout = dropout_layer(dropoutRate)(block1) # dropoutType

        block2 = F.conv2d(block1_dropout, weight=torch.randn(F2, F1*D, 1, 16), padding='same', bias=False) # SeparableConv2D
        block2 = nn.BatchNorm2d(F2)(block2) # BatchNormalization
        block2 = F.relu(block2) # Activation('relu')
        block2 = F.avg_pool2d(block2, kernel_size=(1, 8)) # AveragePooling2D
        block2_dropout = dropout_layer(dropoutRate)(block2) # dropoutType

        flatten = torch.flatten(block2_dropout, 1) # Flatten

        dense = nn.Linear(flatten.shape[1], nb_classes, bias=True) # Dense, kernel_constraint is applied in forward pass
        dense.weight.data = torch.clip(dense.weight.data, -norm_rate, norm_rate) # kernel_constraint=max_norm(norm_rate)
        softmax = F.softmax(dense, dim=1) # Activation('softmax')
        return softmax
    return forward


def cnn_bilstm_def(nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8, num_lstm=64,
                 D=2, F2=16, dropoutType='Dropout'):
    """
    CNN_BiLSTM model in PyTorch functional style (def).
    Input shape: (batch_size, n_channel, n_times)
    """
    dropout_layer = nn.Dropout2d if dropoutType == 'SpatialDropout2D' else nn.Dropout # Define dropout type based on string input

    def forward(input1):
        x = input1.unsqueeze(1) # add channel dimension -> (batch_size, 1, n_channel, n_times)

        block1 = F.conv2d(x, weight=torch.randn(F1, 1, 1, kernLength), padding='same', bias=False) # Conv2D, input shape is now (batch_size, 1, n_channel, n_times)
        block1 = nn.BatchNorm2d(F1)(block1) # BatchNormalization

        depthwise_weight = torch.randn(F1*D, 1, Chans, 1) # DepthwiseConv2D weights
        block1 = F.conv2d(block1, weight=depthwise_weight, bias=None, groups=F1, padding='valid', stride=1) # DepthwiseConv2D
        block1 = nn.BatchNorm2d(F1*D)(block1) # BatchNormalization
        block1 = F.elu(block1) # Activation('elu')
        block1 = F.avg_pool2d(block1, kernel_size=(1, 4)) # AveragePooling2D
        block1_dropout = dropout_layer(dropoutRate)(block1) # dropoutType

        block2 = F.conv2d(block1_dropout, weight=torch.randn(F2, F1*D, 1, 16), padding='same', bias=False) # SeparableConv2D
        block2 = nn.BatchNorm2d(F2)(block2) # BatchNormalization
        block2 = F.elu(block2) # Activation('elu')
        reshaped = torch.reshape(block1_dropout, (block1_dropout.shape[0], block1_dropout.shape[2], -1)) # TimeDistributed(Flatten()) - reshape to (B, T, C*H)

        bilstm = nn.GRU(F1*D, num_lstm, bidirectional=False)(reshaped)[0] # GRU,  kernel_regularizer is not directly applicable in functional API

        dropout1 = F.dropout(bilstm, p=dropoutRate, training=True) # Dropout
        output_layer = nn.Linear(dropout1.shape[2], nb_classes)(dropout1) # Dense
        output_layer = output_layer[:, -1, :] # Take the output from the last time step
        softmax = F.softmax(output_layer, dim=1) # Activation('softmax')
        return softmax
    return forward


def gru_net_def(nb_classes, Chans=32, Samples=512, dropoutRate=0.3, L1=32, L2=16):
    """
    GRU_Net model in PyTorch functional style (def).
    Input shape: (batch_size, n_channel, n_times)
    """
    def forward(input_layer):
        timedist = torch.transpose(input_layer, 1, 2) # TimeDistributed(Flatten()) - reshape to (B, T, C) where T is now n_times and C is n_channel
        bigru_output = nn.GRU(Chans, L1, bidirectional=True, dropout=0)(timedist)[0] # Bidirectional(GRU), kernel_regularizer and dropout are not directly applicable in functional API
        dropout1 = F.dropout(bigru_output, p=dropoutRate, training=True) # Dropout
        gru_output = nn.GRU(L1*2, L2, dropout=0)(dropout1)[0] # GRU, kernel_regularizer and dropout are not directly applicable in functional API
        gru_output = gru_output[:, -1, :] # Take the output from the last time step
        output_layer = nn.Linear(L2, nb_classes)(gru_output) # Dense
        softmax = F.softmax(output_layer, dim=1) # Activation('softmax')
        return softmax
    return forward

"""
if __name__ == '__main__':
    # Example usage:
    nb_classes = 2
    Chans = 64
    Samples = 128
    batch_size = 2 # added batch size

    # ERTNet
    ertnet_model_def = ertnet_def(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    ertnet_input = torch.randn(batch_size, Chans, Samples) # input shape is now (batch_size, n_channel, n_times)
    ertnet_output = ertnet_model_def(ertnet_input)
    print("ERTNet output shape:", ertnet_output.shape)

    # DeepConvNet
    deepconvnet_model_def = deepconvnet_def(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    deepconvnet_input = torch.randn(batch_size, Chans, Samples) # input shape is now (batch_size, n_channel, n_times)
    deepconvnet_output = deepconvnet_model_def(deepconvnet_input)
    print("DeepConvNet output shape:", deepconvnet_output.shape)

    # ShallowConvNet
    shallowconvnet_model_def = shallowconvnet_def(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    shallowconvnet_input = torch.randn(batch_size, Chans, Samples) # input shape is now (batch_size, n_channel, n_times)
    shallowconvnet_output = shallowconvnet_model_def(shallowconvnet_input)
    print("ShallowConvNet output shape:", shallowconvnet_output.shape)

    # EEGNet
    eegnet_model_def = eegnet_def(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    eegnet_input = torch.randn(batch_size, Chans, Samples) # input shape is now (batch_size, n_channel, n_times)
    eegnet_output = eegnet_model_def(eegnet_input)
    print("EEGNet output shape:", eegnet_output.shape)

    # CNN_BiLSTM
    cnn_bilstm_model_def = cnn_bilstm_def(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    cnn_bilstm_input = torch.randn(batch_size, Chans, Samples) # input shape is now (batch_size, n_channel, n_times)
    cnn_bilstm_output = cnn_bilstm_model_def(cnn_bilstm_input)
    print("CNN_BiLSTM output shape:", cnn_bilstm_output.shape)

    # gru_net
    gru_net_model_def = gru_net_def(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    gru_net_input = torch.randn(batch_size, Chans, Samples) # input shape is now (batch_size, n_channel, n_times)
    gru_net_output = gru_net_model_def(gru_net_input)
    print("GRU_Net output shape:", gru_net_output.shape)
"""