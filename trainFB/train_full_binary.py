import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# Import the full binary dataset loader
from utils.dataset_full_binary import LoaderFullBinary
from utils.model import ECG_XNOR_Full_Bin_LP
from utils.engine import train
from utils.save_model import save_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    # Configuration
    classes_num = 5
    test_size = 0.2
    batch_size = 256
    lr = 0.005
    seed = 169
    num_epochs = 1500
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print(f"Training Full Binary ECG BNN")
    print(f"Classes: {classes_num}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {num_epochs}")
    print("-" * 50)
    
    # Load data with full binary preprocessing
    loader = LoaderFullBinary(batch_size=batch_size, classes_num=classes_num, device=device, test_size=test_size)
    labels, train_loader, test_loader = loader.loader()
    loader.plot_train_test_splits()
    
    # Network architecture
    kernel_size, padding, poolsize = 7, 5, 7
    padding_value = 1  # Use 1 for binary inputs
    
    blocks = [
        [1, 8, kernel_size, 2, padding, padding_value, poolsize, 2],
        [8, 16, kernel_size, 1, padding, padding_value, poolsize, 2],
        [16, 32, kernel_size, 1, padding, padding_value, poolsize, 2], 
        [32, 32, kernel_size, 1, padding, padding_value, poolsize, 2],
        [32, 64, kernel_size, 1, padding, padding_value, poolsize, 2],
        [64, classes_num, kernel_size, 1, padding, padding_value, poolsize, 2],
    ]
    
    # Create model
    model = ECG_XNOR_Full_Bin_LP(
        block1=blocks[0], block2=blocks[1], block3=blocks[2],
        block4=blocks[3], block5=blocks[4], block6=blocks[5],
        block7=None, device=device
    ).to(device)
    
    print(f"Model created: {model.name}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\nStarting training...")
    
    # Train
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader, 
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=num_epochs,
        device=device,
        writer=False,
        classes_num=classes_num
    )
    
    # Save final model
    print("\nSaving model...")
    
    # Save state dict
    torch.save(model.state_dict(), "ECG_BNN_Full_Binary_State_Dict.pth")
    
    # Save entire model (for easy loading)
    torch.save(model, "ECG_BNN_Full_Binary_Complete.pth")
    
    print("Models saved:")
    print("- ECG_BNN_Full_Binary_State_Dict.pth (state dict only)")
    print("- ECG_BNN_Full_Binary_Complete.pth (complete model)")
    
    print("\nTraining completed!")
    print("Next steps:")
    print("1. Use the saved model with your inference code")
    print("2. Update inference code to handle binary inputs")
    print("3. Test with your hardware-optimized inference")

if __name__ == "__main__":
    main()
