# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 01:46:40 2023

@author: Stephen Suryasentana - stephen@autogeolab.com

MIT License

Copyright (c) 2023 Stephen Suryasentana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the Low Fidelity Neural Network
class SNN(nn.Module):
    def __init__(self, num_neurons=32):
        """
        Initialize the Low Fidelity Neural Network.
        Args:
            num_neurons (int): Number of neurons in the hidden layers.
        """
        super(SNN, self).__init__()
        
        # Define the network architecture
        self.net_l = nn.Sequential(
            nn.Linear(3, num_neurons),  # Input layer
            nn.Tanh(),  # Activation function
            nn.Linear(num_neurons, num_neurons),  # Hidden layer
            nn.Tanh(),  # Activation function
            nn.Linear(num_neurons, 1)  # Output layer
        )

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor with positive values.
        """
        y = self.net_l(x)
        y = torch.exp(y)  # Constrain to positive-valued outputs
        return y

# Define the Multi-Fidelity Neural Network
class MFDF(nn.Module):
    def __init__(self, lf_model, num_neurons=128, num_inputs=3, dropout_prob=0.1):
        """
        Initialize the Multi-Fidelity Neural Network.
        Args:
            lf_model (nn.Module): Pre-trained low fidelity model.
            num_neurons (int): Number of neurons in the hidden layers.
            num_inputs (int): Number of input features.
            dropout_prob (float): Dropout probability.
        """
        super(MFDF, self).__init__()
        
        self.num_inputs = num_inputs
        
        # Store and freeze the pre-trained low fidelity model
        self.lfmodel = lf_model
        self.lfmodel.eval()
        for param in self.lfmodel.parameters():
            param.requires_grad = False
        
        # Define the scaling network
        self.net_scale = nn.Sequential()
        self.net_scale.add_module('layer_1', nn.Linear(num_inputs, num_neurons)) # Input layer
        self.net_scale.add_module('layer_2', nn.Tanh()) # Activation function
        self.net_scale.add_module('layer_3', nn.Linear(num_neurons, num_neurons)) # Hidden layer
        self.net_scale.add_module('layer_4', nn.Tanh()) # Activation function
        self.net_scale.add_module('layer_5', nn.Dropout(p=dropout_prob)) # Dropout layer
        self.net_scale.add_module('layer_6', nn.Linear(num_neurons, 1)) # Output layer
        
    
    def scale(self, x):
        """
        Scale function to constrain output to positive values.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Scaled output tensor.
        """
        scale = self.net_scale(x)
        return torch.exp(scale)  # Constrain to positive-valued outputs
    
    def forward(self, x):
        """
        Forward pass through the multi-fidelity network.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor combining low fidelity model and scale.
        """
        lf_output = self.lfmodel(x)        
        scale = self.scale(x)
        return scale * lf_output

def prob_predict(model, x, num_samples=1000):
    """
    Generate probabilistic predictions using the model.
    Args:
        model (nn.Module): Trained model.
        x (Tensor): Input tensor.
        num_samples (int): Number of prediction samples.
    Returns:
        Tuple: Mean, standard deviation, and percentiles of predictions.
    """
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(x)
            predictions.append(outputs)
            model.train()  # Enable dropout during prediction sampling

    predictions = torch.cat(predictions, dim=1)
    mean = predictions.mean(dim=1)
    std_dev = predictions.std(dim=1)
    p10 = np.percentile(predictions.detach().numpy(), 5, axis=1)
    p90 = np.percentile(predictions.detach().numpy(), 95, axis=1)
    percentiles = [p10, p90]
    
    model.eval()  # Disable dropout after prediction sampling
    
    return mean, std_dev, percentiles

    

# Load low-fidelity model
lfpath = "lfmodel-cowden.pt"
print(f"Loading low-fidelity model from {lfpath}")
lfmodel = SNN()
lfmodel.load_state_dict(torch.load(lfpath))
lfmodel.eval()

# Load dual-fidelity model
mfpath = "mfdfmodel-cowden-['CM2' 'CM9' 'CM3' 'CL2'].pt"
print(f"Loading DFNN model from {mfpath}")
model = MFDF(lfmodel, 128)
model.load_state_dict(torch.load(mfpath))
model.eval()

# Load tri-fidelity model
fpath = "fmodel-cowden-['CS2' 'CS3' 'CS4' 'CM2' 'CM9' 'CM3' 'CL2']-1-0.04.pt"
print(f"Loading TFNN model from {fpath}")
fmodel = MFDF(model, 256)
fmodel.load_state_dict(torch.load(fpath))
fmodel.eval()

# Set default figure settings for plots
plt.rcParams['figure.figsize'] = [5.0, 5.0]
plt.rcParams['figure.dpi'] = 600

def plot_mean_std(x, mean, std, mean_label, 
                  std_label='95% CI', 
                  axes=plt,
                  mean_pt='-',
                  mean_color='#555555',
                  std_color='#D3D3D3'):
    """
    Plot the mean and standard deviation with shaded confidence interval.
    
    Args:
        x (array): Input x values.
        mean (array): Mean values.
        std (array, float or None): Standard deviation values or percentiles.
        mean_label (str): Label for the mean line.
        std_label (str): Label for the confidence interval.
        axes (matplotlib axes): Axes to plot on.
        mean_pt (str): Line style for mean.
        mean_color (str): Color for mean line.
        std_color (str): Color for confidence interval.
    """
    axes.plot(x, mean, mean_pt, color=mean_color, label=mean_label)
    if std is None:
        return
    
    if isinstance(std, list):
        axes.fill_between(x, std[0], std[1], color=std_color, alpha=1, label=std_label)
    else:
        axes.fill_between(x, (mean - 1.96 * std), (mean + 1.96 * std), color=std_color, alpha=1, label=std_label)

# Define model parameters
ymaxs = {
    'CS2': 12, 
    'CS3': 30, 
    'CS4': 35, 
    'CM2': 40,
    'CM9': 140,
    'CM3': 400,
    'CL2': 2500,
}

data = {
    'CS2': {'D': 0.273, 'LD': 5.25},
    'CS3': {'D': 0.273, 'LD': 8.0},
    'CS4': {'D': 0.273, 'LD': 10.0},
    'CM2': {'D': 0.762, 'LD': 3.0},
    'CM9': {'D': 0.762, 'LD': 5.25},
    'CM3': {'D': 0.762, 'LD': 10.0},
    'CL2': {'D': 2.0, 'LD': 5.25}
}

# Plot model predictions
for pilename, values in data.items():
    uyD = np.arange(0., 0.11, 0.01/10)
    D = values['D']
    LD = values['LD']
    L = LD * D
    patm = 101325
    norm = patm * (D ** 2)
    ones = np.ones(uyD.shape[0])
    x = np.vstack([ones * D, ones * LD, uyD]).T
    xx = torch.tensor(x, requires_grad=False, dtype=torch.float32)
    ymax = ymaxs[pilename] / norm
    
    plt.figure()

    y_pred_m, _, _ = prob_predict(model, xx)
    plot_mean_std(uyD, y_pred_m.numpy() / norm, None, 'DFNN')
        
    y_pred_f, _, y_pred_f_pt = prob_predict(fmodel, xx)
    y_pred_f_pt2 = [np.array(p) / norm for p in y_pred_f_pt]
    plot_mean_std(uyD, y_pred_f.numpy() / norm, y_pred_f_pt2, 'TFNN', std_label='TFNN (P5-P95)', mean_color='coral', std_color='#FFE7C7')
    
    yD = 0.04
    plt.plot([yD, yD], [0, ymax], 'k:')
    plt.text(yD + 0.003, 0.02 * ymax, "Forecast region")
    
    plt.ylabel(r'$\frac{H}{p_{\mathrm{atm}}D^2}$')
    plt.xlabel(r'$u_H/D$')
    plt.xlim((0, 0.1))
    plt.ylim((0, ymax))
    plt.title(f'{pilename} (D={D}, L/D={LD})')
    plt.legend(loc='best', bbox_to_anchor=(0, 0.05, 1, 0.95), frameon=False)
    plt.show()

# Plot scale factor
plt.figure(figsize=(7, 7))
uyD = np.arange(0., 0.11, 0.01/10)
ones = np.ones(uyD.shape[0])
ymax = 1.4

for pilename, values in data.items():
    D = values['D']
    LD = values['LD']
    x = np.vstack([ones * D, ones * LD, uyD]).T
    xx = torch.tensor(x, requires_grad=False, dtype=torch.float32)
    scale_h = fmodel.scale(xx)
    
    plt.plot(x[:, 2], scale_h.detach().numpy(), '-', label=f'{pilename}: D={D}, L/D={LD}')

yD = 0.04
plt.plot([yD, yD], [0, ymax], 'k:')
plt.text(yD + 0.003, 0.04 * ymax, "Forecast region")

plt.ylabel('Scale Factor')
plt.xlabel(r'$u_H/D$')
plt.xlim((0, 0.1))
plt.ylim((0, 1.4))
plt.legend(loc='best', bbox_to_anchor=(0, 0.05, 1, 0.95), frameon=False)
plt.show()