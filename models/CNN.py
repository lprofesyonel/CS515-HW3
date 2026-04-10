import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network (CNN) designed to act as the 'Student' 
    in the Knowledge Distillation framework.

    This architecture consists of two convolutional layers followed by max pooling,
    and a classifier head with two fully connected layers. It is specifically designed 
    to process 32x32 RGB images (e.g., from the CIFAR-10 dataset).
    """
    
    def __init__(self, num_classes: int = 10) -> None:
        """
        Initializes the SimpleCNN model architecture.

        Args:
            num_classes (int): The number of output classes for the dataset (default: 10).
        """
        super(SimpleCNN, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reduces spatial dimensions: 32x32 -> 16x16
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Reduces spatial dimensions: 16x16 -> 8x8
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): A batch of input images of shape (N, 3, 32, 32).

        Returns:
            torch.Tensor: The output logits of the network of shape (N, num_classes).
        """
        return self.classifier(self.features(x))