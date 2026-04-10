import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) architecture for classification tasks.
    
    Args:
        input_size (int): Dimension of the input features (e.g., 784 for MNIST).
        hidden_sizes (List[int]): List containing the number of neurons for each hidden layer.
        num_classes (int): Number of output classes (e.g., 10 for MNIST).
        dropout (float): Dropout probability. Default is 0.3.
        activation (str): Activation function to use ('relu' or 'gelu'). Default is 'relu'.
        use_batch_norm (bool): Whether to include BatchNorm1d before activation. Default is False.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: List[int], 
        num_classes: int, 
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = False
    ) -> None:
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        self.hidden_layers = nn.ModuleList()
        
        in_dim = input_size
        for h_dim in hidden_sizes:
            layer_steps = [nn.Linear(in_dim, h_dim)]
            
            if use_batch_norm:
                layer_steps.append(nn.BatchNorm1d(h_dim))
                
            if activation.lower() == "gelu":
                layer_steps.append(nn.GELU())
            else:
                layer_steps.append(nn.ReLU())
                
            if dropout > 0:
                layer_steps.append(nn.Dropout(dropout))
                
            self.hidden_layers.append(nn.Sequential(*layer_steps))
            in_dim = h_dim
            
        self.output_layer = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 28, 28).
            
        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """

        x = self.flatten(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            
        logits = self.output_layer(x)
        return logits