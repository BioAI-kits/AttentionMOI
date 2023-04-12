# Libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
from livelossplot import PlotLosses

# Create Fully Conn NN 
class RLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layers = 3, p = 0.1):
        super(RLModel, self).__init__()
        self.n_layers = n_layers
        self.p = p
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.activation = nn.ReLU()
        #self.layer = LayerBlock(hidden_size, self.p)
        #self.layers = [LayerBlock(hidden_size , self.p) for i in range(self.n_layers)]
        self.fc2 = nn.Linear(hidden_size, num_classes)
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        #x = self.layer(x)
        #for i in range(self.n_layers):
        #    x = self.layers[i](x)
        out = self.fc2(x)
        return out
    
def train_RLModel(input_tensor, 
                  target_tensor, 
                  rl_model,
                  rl_model_optimizer,
                #   criterion,
                  phase):
    
    if phase == 'train':
        rl_model.train()
    else:
        rl_model.eval()
        
    # clear gradients in the optimizers
    if phase == 'train':
        rl_model_optimizer.zero_grad()
    
    # Predict label
    prediction = rl_model(input_tensor)
    print(prediction.shape, target_tensor.shape)
    # compute the loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target_tensor)
    
    # # compute the Accuracy
    # _, predicted = torch.max(prediction,1)
    # correct = torch.sum(predicted==target_tensor)
    # total = input_tensor.size(0)
    # accuracy = 100 * correct / total
    
    # if phase == 'train':
    #     # Compute the gradients
    #     loss.backward()
        
    #     # Step the optimizers to update the model weights
    #     rl_model_optimizer.step()
        
    # # return the loss value to track training progress
    # return loss.item(), accuracy.item()

def train_RLModel_Iters(rl_model, 
                        data_tensor_train, 
                        data_tensor_valid,
                        label_tensor_train,
                        label_tensor_valid,
                        num_epochs,
                        print_every_n_batches=100,
                        learning_rate=0.01,
                        phases=['train', 'validation']):
    
    # Live Loss
    liveloss = PlotLosses()
    
    # Keep track of losses
    train_plot_losses = []
    test_plot_losses = []
    train_plot_accuracy = []
    test_plot_accuracy = []
    
    # Initialise RL_Model Optimizer
    rl_model_optimizer = torch.optim.SGD(rl_model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Specify loss function
    criterion = torch.nn.CrossEntropyLoss()
        
    # Cycle through epochs
    for epoch in range(num_epochs):
        logs = {}
        
        for phase in phases:
            print(f'Epoch {epoch +1}/{num_epochs}')
            
            if phase == 'train':
                loss, accuracy = train_RLModel(data_tensor_train, label_tensor_train, rl_model, rl_model_optimizer, criterion, phase)
                train_plot_losses.append(loss)
                train_plot_accuracy.append(accuracy)
            else:
                loss, accuracy = train_RLModel(data_tensor_valid, label_tensor_valid, rl_model, rl_model_optimizer, criterion, phase)
                test_plot_losses.append(loss)
                test_plot_accuracy.append(accuracy)
                
            print(loss)
            #plot_losses.append(loss)
            
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'
                
            logs[prefix + 'log loss'] = loss
            logs[prefix + 'accuracy'] = accuracy
            
            print ('Epoch [{}/{}], Test Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss))
            
        liveloss.update(logs)
        liveloss.draw()
        
    # Final accuracy count
    predictions_validation = rl_model(data_tensor_valid)
    _, predicted = torch.max(predictions_validation, 1)
    correct = (predicted==label_tensor_valid).sum().item()
    total = data_tensor_valid.size(0)
    valid_accuracy = 100*correct/total
    print ("Validation Accuracy:", 100*correct/total)
        
    return train_plot_losses, train_plot_accuracy, test_plot_losses, test_plot_accuracy


if __name__ == '__main__':
    X = torch.rand(size=(3, 16, 32))
    Y = np.random.choice(2, size=(3, 16, 1))
    Y = torch.Tensor(Y).to(torch.float)
    rlmodel = RLModel(input_size=32, hidden_size=32, num_classes=1)
    train_RLModel(input_tensor=X, target_tensor=Y, rl_model=rlmodel, 
                  rl_model_optimizer=torch.optim.Adam(rlmodel.parameters()),
                  phase='train'
                  )

