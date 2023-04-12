# Libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from livelossplot import PlotLosses

#Encoder
class LayerBlockEncode(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2, p):
        super().__init__()
        self.layer = nn.Linear(hidden_size_1, hidden_size_2)
        self.activation = nn.Tanh()
        self.bn = nn.BatchNorm1d(hidden_size_2)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.dropout(x)
        return (x)

class Encoder(nn.Module):
    def __init__(self, input_shape, hidden_size, encoded_size, n_layers=3, drop_prob=0.5):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        
        self.e1 = nn.Linear(input_shape, hidden_size)
        self.activation1 = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        if (hidden_size // (2**n_layers)) > encoded_size:
            self.layers = nn.ModuleList([])
            for i in range(n_layers):
                self.layers.append(LayerBlockEncode(hidden_size//(2**i), hidden_size//(2**(i+1)), self.drop_prob))
        else:
            self.n_layers = 0
                                      
        self.e2 = nn.Linear((hidden_size//(2**n_layers)), encoded_size)
        
    def forward(self, input):
        x = self.e1(input)
        x = self.activation1(x)
        x = self.bn1(x)
        
        for i in range(self.n_layers):
            encode_block = self.layers[i]
            x = encode_block(x)
            
        #block1 = F.dropout(self.bn1(F.elu(self.e1(input))), p=self.drop_prob)
        #encoded_representation = torch.tanh(self.e2(block1))
        
        encoded_representation = self.e2(x)
        
        return encoded_representation
    
#Decoder:
class LayerBlockDecode(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2, p):
        super().__init__()
        self.layer = nn.Linear(hidden_size_1, hidden_size_2)
        self.activation = nn.Tanh()
        self.bn = nn.BatchNorm1d(hidden_size_2)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.dropout(x)
        return (x)

class Decoder(nn.Module):
    def __init__(self, output_shape, hidden_size, encoded_size, n_layers=3, drop_prob=0.5):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.second_last_layer_size = hidden_size // (2**n_layers)
        
        self.d1 = nn.Linear(encoded_size, self.second_last_layer_size)
        self.activation1 = nn.Tanh()
        self.bn1 = nn.BatchNorm1d(self.second_last_layer_size)
        
        if (self.second_last_layer_size) > encoded_size: 
            self.layers = nn.ModuleList([])
            for i in range(self.n_layers):
                self.layers.append(LayerBlockDecode(hidden_size//(2**(n_layers-i)), hidden_size//(2**(n_layers-i-1)), self.drop_prob))
        else:
            self.n_layers = 0
        
        self.d2 = nn.Linear(hidden_size, output_shape)
        
    
    def forward(self, input):
        x = self.d1(input)
        x = self.activation1(x)
        x = self.bn1(x)
        
        for i in range(self.n_layers):
            decode_block = self.layers[i]
            x = decode_block(x)
            
        #block = F.dropout(self.bn(F.elu(self.d(input))), p=self.drop_prob)
        #reconstruction = torch.tanh(self.d4(block))
        
        reconstruction = self.d2(x)
        
        return reconstruction
    
# Training AutoEncoders Function
def train_ae(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, hidden_size, encoded_size, n_layers, drop_prob, phase):
    
    if phase == 'train':
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()
        
    # clear the gradients in the optimizers
    if phase == 'train':
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
    # Forward pass through 
    
    encoded_representation = encoder(input_tensor)
    reconstruction = decoder(encoded_representation)
    
    # Compute the loss
    loss = criterion(reconstruction, target_tensor)
    
    if phase == 'train':
        # Compute the gradients
        loss.backward()
    
        # Step the optimizers to update the model weights
        encoder_optimizer.step()
        decoder_optimizer.step()
    
    # Return the loss value to track training progress
    return loss.item()    

# Training Loop
def trainIters(encoder, decoder, data_tensor, data_tensor_valid, epochs, 
               hidden_size, encoded_size, n_layers, drop_prob,
               print_every_n_batches=100, learning_rate=0.01,
               phases=["train", "validation"],):
    
    # Live Loss
    liveloss = PlotLosses()
    
    # keep track of losses
    train_plot_losses = []
    test_plot_losses = []
    
    # Initialize Encoder Optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Initialize Decoder Optimizer
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Specify loss function
    criterion = torch.nn.MSELoss(reduce=True)
    
    # Cycle through epochs
    for epoch in range(epochs):
        logs = {}
        
        for phase in phases:
            print(f'Epoch {epoch + 1}/{epochs}')
            
            if phase == 'train':
                loss = train_ae(data_tensor, data_tensor, encoder, decoder,
                             encoder_optimizer, decoder_optimizer, criterion, 
                            hidden_size, encoded_size, n_layers, drop_prob, phase)
                train_plot_losses.append(loss)
            else:
                loss = train_ae(data_tensor_valid, data_tensor_valid, encoder, decoder,
                             encoder_optimizer, decoder_optimizer, criterion, 
                            hidden_size, encoded_size, n_layers, drop_prob, phase)
                test_plot_losses.append(loss)
                
            print(loss)
            #plot_losses.append(loss)
        
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'
                
            logs[prefix + 'log loss'] = loss
        
        liveloss.update(logs) #liveloss
        liveloss.draw() #liveloss
                
    return train_plot_losses, test_plot_losses
