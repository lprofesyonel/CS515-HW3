import torch
import random
import numpy as np

from parameters import get_params
from train import run_training
from models.resnet_transfer import TransferResNet18

from run_hw3_robustness import run_robustness_evaluations

def set_seed(seed: int) -> None:
    """
    Sets the random seed for various libraries to ensure reproducibility of the experiments.
    
    This function fixes the random seed for built-in Python 'random', 'numpy', 
    and PyTorch (both CPU and CUDA) so that random initializations and data 
    shuffling remain consistent across multiple runs.

    Args:
        seed (int): The integer value to be used as the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    """
    The main entry point for the HW3: Robustness and Adversarial Attacks pipeline.
    
    This function performs the following steps:
    1. Fetches hyperparameters and configurations from 'get_params()'.
    2. Sets the random seed for reproducibility.
    3. Detects and assigns the available computation device (CPU or CUDA).
    4. Initializes the specified Neural Network model (e.g., TransferResNet18).
    5. Dispatches the model to:
       - 'run_training' for standard or AugMix fine-tuning.
       - 'run_robustness_evaluations' for PGD attacks, CIFAR-10-C tests, and Grad-CAM.
    
    Raises:
        NotImplementedError: If the user selects a model architecture that has 
        not been implemented yet.
    """
    params = get_params()
    set_seed(params["seed"])
    print(f"Seed set to: {params['seed']}")
    
    # Check device
    device = torch.device(params["device"] if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {params['dataset']}  |  Model: {params['model']}  |  Transfer Mode: {params['transfer_mode']}")
    print(f"Using device: {device}")

    # Initialize the Transfer Learning Model
    if params["model"] == "resnet18":
        model = TransferResNet18(
            num_classes=params["num_classes"], 
            transfer_mode=params["transfer_mode"]
        )
    else:
        raise NotImplementedError(f"Model '{params['model']}' is not implemented yet!")

    model = model.to(device)

    # 1. Train the model (Standard or with AugMix)
    if params["mode"] in ["train", "both"]:
        run_training(model, params, device)
        
    # 2. Evaluate Robustness and Perform Adversarial Attacks
    if params["mode"] in ["test", "attack", "both"]:
        try:
            model.load_state_dict(torch.load(params["save_path"], map_location=device))
            print(f"\n[SUCCESS] Loaded trained weights from {params['save_path']}")
        except FileNotFoundError:
            print(f"\n[WARNING] Weights {params['save_path']} not found! Using random initialization.")
            
        run_robustness_evaluations(model, params, device)

if __name__ == "__main__":
    main()