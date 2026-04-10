import argparse
from typing import Dict, Any

def get_params() -> Dict[str, Any]:
    """
    Parses command-line arguments and defines default hyperparameters for the project.
    
    Updated for HW3: Robustness, AugMix Data Augmentation, and Adversarial Attacks 
    (PGD L_inf and L_2) on CIFAR-10 and CIFAR-10-C datasets.

    Returns:
        Dict[str, Any]: A dictionary containing all the necessary parameters.
    """
    parser = argparse.ArgumentParser(description="Robustness, AugMix & Adversarial Attacks on CIFAR-10")

    # --- Standard Execution Parameters ---
    parser.add_argument("--mode", choices=["train", "test", "both", "attack"], default="both", help="Execution mode")
    parser.add_argument("--model", choices=["resnet18", "simple_cnn", "mobilenet"], default="resnet18", help="NN model to use")
    parser.add_argument("--transfer_mode", choices=["resize_freeze", "modify_finetune", "scratch"], default="modify_finetune", help="Transfer learning strategy")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cpu or cuda)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and testing")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 penalty)")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

    # --- HW3 New Parameters: AugMix & Adversarial Attacks ---
    parser.add_argument("--use_augmix", action="store_true", help="Use AugMix data augmentation during training")
    parser.add_argument("--attack_type", choices=["none", "pgd_linf", "pgd_l2"], default="none", help="Type of adversarial attack to perform")
    parser.add_argument("--pgd_steps", type=int, default=20, help="Number of iterations for PGD attack")
    parser.add_argument("--eps_linf", type=float, default=4/255, help="Epsilon for L_inf PGD attack")
    parser.add_argument("--eps_l2", type=float, default=0.25, help="Epsilon for L_2 PGD attack")
    parser.add_argument("--alpha", type=float, default=1/255, help="Step size (alpha) for PGD attack")

    args = parser.parse_args()

    # CIFAR-10 normalization values (RGB)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    # Dynamic save path so normal and AugMix models don't overwrite each other
    aug_suffix = "_augmix" if args.use_augmix else ""
    save_path = f"best_{args.model}_{args.transfer_mode}{aug_suffix}.pth"

    return {
        "dataset":      "cifar10",
        "data_dir":     "./data",
        "cifar_c_dir":  "./data/CIFAR-10-C", # Path for the corrupted testing dataset
        "num_workers":  2,
        "mean":         mean,
        "std":          std,
        "model":        args.model,
        "transfer_mode": args.transfer_mode,
        "num_classes":  10,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  args.weight_decay,
        "patience":      args.patience,
        
        # HW3 Added Dictionary Keys
        "use_augmix":    args.use_augmix,
        "attack_type":   args.attack_type,
        "pgd_steps":     args.pgd_steps,
        "eps_linf":      args.eps_linf,
        "eps_l2":        args.eps_l2,
        "alpha":         args.alpha,
        
        "seed":         42,
        "device":       args.device,
        "save_path":    save_path,
        "log_interval": 100,
        "mode":         args.mode,
    }