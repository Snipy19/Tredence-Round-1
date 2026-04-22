"""
Self-Pruning Neural Network for CIFAR-10
Tredence Analytics - AI Engineering Intern Case Study

Author: [Your Name]
Description: Implementation of a neural network with learnable gates that prune
unimportant weights during training using L1 sparsity regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PrunableLinear(nn.Module):
    """
    Custom Linear Layer with learnable pruning gates.
    
    Each weight has an associated gate parameter that is learned during training.
    The gate values (after sigmoid) multiply the weights element-wise, effectively
    allowing the network to learn which weights to keep and which to prune.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term (default: True)
        gate_init: Initial value for gate scores (default: 2.0 -> sigmoid(2.0) ≈ 0.88)
    """
    
    def __init__(self, in_features, out_features, bias=True, gate_init=2.0):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Gate scores - learnable parameters with same shape as weight
        # Initialize with positive values so gates start close to 1 (active)
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), gate_init))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weight and bias using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass with pruned weights.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Convert gate scores to gates in [0, 1] using sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Apply gates to weights (element-wise multiplication)
        pruned_weights = self.weight * gates
        
        # Standard linear transformation
        return F.linear(x, pruned_weights, self.bias)
    
    def get_gates(self):
        """Return the current gate values (after sigmoid)."""
        return torch.sigmoid(self.gate_scores)
    
    def get_sparsity(self, threshold=1e-2):
        """
        Calculate sparsity of this layer.
        
        Args:
            threshold: Gates below this value are considered pruned
            
        Returns:
            float: Fraction of pruned weights
        """
        gates = self.get_gates()
        pruned = (gates < threshold).float().sum().item()
        total = gates.numel()
        return pruned / total


class SelfPruningCNN(nn.Module):
    """
    Convolutional Neural Network with PrunableLinear layers for CIFAR-10.
    
    Architecture:
    - 3 Conv layers with batch norm and max pooling
    - 2 PrunableLinear layers for classification
    - Dropout for regularization
    """
    
    def __init__(self, num_classes=10, hidden_size=512, gate_init=2.0):
        super(SelfPruningCNN, self).__init__()
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
        )
        
        # Calculate feature size after conv layers
        self.feature_size = 128 * 4 * 4  # CIFAR-10: 32x32 -> 4x4 after 3 pooling layers
        
        # Prunable fully connected layers
        self.fc1 = PrunableLinear(self.feature_size, hidden_size, gate_init=gate_init)
        self.bn_fc = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = PrunableLinear(hidden_size, num_classes, gate_init=gate_init)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        # First prunable FC layer
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        return x
    
    def get_all_gates(self):
        """Collect all gate values from prunable layers."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates().flatten())
        return torch.cat(gates)
    
    def calculate_sparsity_loss(self):
        """
        Calculate L1 sparsity loss across all prunable layers.
        L1 norm encourages gates to become exactly zero.
        """
        gates = self.get_all_gates()
        return gates.sum()  # L1 norm (gates are positive)
    
    def get_total_sparsity(self, threshold=1e-2):
        """Calculate overall sparsity of the network."""
        total_pruned = 0
        total_params = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gates()
                total_pruned += (gates < threshold).float().sum().item()
                total_params += gates.numel()
        return (total_pruned / total_params * 100) if total_params > 0 else 0


class PruningTrainer:
    """
    Trainer class for Self-Pruning Neural Network.
    Handles training with sparsity regularization and evaluation.
    """
    
    def __init__(self, model, lambda_sparsity, device='cuda'):
        self.model = model.to(device)
        self.lambda_sparsity = lambda_sparsity
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.sparsity_history = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Training (λ={self.lambda_sparsity})')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Calculate losses
            classification_loss = self.criterion(output, target)
            sparsity_loss = self.model.calculate_sparsity_loss()
            total_loss = classification_loss + self.lambda_sparsity * sparsity_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += total_loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Sparsity': f'{self.model.get_total_sparsity():.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        sparsity = self.model.get_total_sparsity()
        
        return avg_loss, accuracy, sparsity
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                classification_loss = self.criterion(output, target)
                sparsity_loss = self.model.calculate_sparsity_loss()
                val_loss += (classification_loss + self.lambda_sparsity * sparsity_loss).item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        """Full training loop."""
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, sparsity = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.sparsity_history.append(sparsity)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'best_model_lambda_{self.lambda_sparsity}.pth')
            
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Sparsity: {sparsity:.2f}%')
        
        # Load best model
        self.model.load_state_dict(torch.load(f'best_model_lambda_{self.lambda_sparsity}.pth'))
        return best_val_acc


def get_data_loaders(batch_size=128):
    """Prepare CIFAR-10 data loaders with augmentation."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2470, 0.2435, 0.2616])
    ])
    
    # No augmentation for validation/test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2470, 0.2435, 0.2616])
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Split train into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def plot_gate_distribution(model, lambda_val, save_path=None):
    """Plot histogram of gate values."""
    gates = model.get_all_gates().cpu().detach().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0.01, color='r', linestyle='--', label='Pruning threshold (0.01)')
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Gate Values (λ={lambda_val})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    pruned_pct = (gates < 0.01).sum() / len(gates) * 100
    active_pct = (gates > 0.5).sum() / len(gates) * 100
    plt.text(0.7, 0.9, f'Pruned (<0.01): {pruned_pct:.1f}%', 
             transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.7, 0.85, f'Active (>0.5): {active_pct:.1f}%', 
             transform=plt.gca().transAxes, fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(trainer, lambda_val, save_path=None):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # Loss curves
    axes[0].plot(epochs, trainer.train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(epochs, trainer.val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss Curves (λ={lambda_val})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, trainer.train_accs, label='Train Acc', linewidth=2)
    axes[1].plot(epochs, trainer.val_accs, label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Accuracy Curves (λ={lambda_val})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Sparsity curve
    axes[2].plot(epochs, trainer.sparsity_history, label='Sparsity', color='green', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Sparsity (%)')
    axes[2].set_title(f'Sparsity Evolution (λ={lambda_val})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    """Main execution function."""
    print("=" * 60)
    print("Self-Pruning Neural Network for CIFAR-10")
    print("Tredence Analytics - AI Engineering Intern Case Study")
    print("=" * 60)
    
    # Get data loaders
    print("\n[1/4] Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=128)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Experiment with different lambda values
    lambda_values = [1e-5, 5e-5, 1e-4]  # Low, Medium, High
    results = []
    
    print("\n[2/4] Training models with different λ values...")
    print("-" * 60)
    
    for lambda_val in lambda_values:
        print(f"\nTraining with λ = {lambda_val}")
        print("-" * 40)
        
        # Create model
        model = SelfPruningCNN(num_classes=10, hidden_size=512, gate_init=2.0)
        
        # Create trainer
        trainer = PruningTrainer(model, lambda_val, device=device)
        
        # Train model
        best_val_acc = trainer.train(train_loader, val_loader, epochs=50)
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device=device)
        final_sparsity = model.get_total_sparsity()
        
        # Store results
        results.append({
            'lambda': lambda_val,
            'test_accuracy': test_acc,
            'sparsity': final_sparsity,
            'model': model,
            'trainer': trainer
        })
        
        print(f"\nResults for λ = {lambda_val}:")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Final Sparsity: {final_sparsity:.2f}%")
        
        # Plot gate distribution
        plot_gate_distribution(model, lambda_val, save_path=f'gate_distribution_lambda_{lambda_val}.png')
        
        # Plot training curves
        plot_training_curves(trainer, lambda_val, save_path=f'training_curves_lambda_{lambda_val}.png')
    
    # Print summary table
    print("\n" + "=" * 60)
    print("[3/4] Summary Results")
    print("=" * 60)
    print(f"{'Lambda':<12} {'Test Accuracy (%)':<20} {'Sparsity Level (%)':<20}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['lambda']:<12.0e} {result['test_accuracy']:<20.2f} {result['sparsity']:<20.2f}")
    
    # Find best model (highest accuracy with sparsity > 0)
    best_result = max(results, key=lambda x: x['test_accuracy'] if x['sparsity'] > 0 else 0)
    print("\n[4/4] Best Model Analysis")
    print("-" * 40)
    print(f"Best λ = {best_result['lambda']:.0e}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.2f}%")
    print(f"Sparsity Level: {best_result['sparsity']:.2f}%")
    
    # Calculate parameter reduction
    model = best_result['model']
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pruned_params = sum(1 for name, p in model.named_parameters() 
                       if 'gate_scores' not in name and p.requires_grad)
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Effective parameters (after pruning): {total_params * (1 - best_result['sparsity']/100):,.0f}")
    print(f"Parameter reduction: {best_result['sparsity']:.2f}%")
    
    # Save final model
    torch.save({
        'model_state_dict': best_result['model'].state_dict(),
        'lambda': best_result['lambda'],
        'test_accuracy': best_result['test_accuracy'],
        'sparsity': best_result['sparsity']
    }, 'best_self_pruning_model.pth')
    print(f"\nBest model saved as 'best_self_pruning_model.pth'")
    
    print("\n" + "=" * 60)
    print("Case Study Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()