import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.03),
            nn.Conv2d(10, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2)
        )
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.03),
            nn.Conv2d(16, 20, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, 2)
        )
        
        # Third convolution block with GAP
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 24, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.03),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 10, 1),  # 1x1 convolution
            nn.AdaptiveAvgPool2d(1)
        )
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(f'Epoch {epoch} Loss: {loss.item():.4f} Accuracy: {100*correct/processed:.2f}%')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def count_parameters(model):
    """
    Count and display parameters for each layer and total parameters
    """
    table = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.append([name, param])
        total_params += param
    
    print("\nModel Parameter Details:")
    print("-" * 60)
    print(f"{'Layer':<40} {'Parameters':>10}")
    print("-" * 60)
    for name, param in table:
        print(f"{name:<40} {param:>10,}")
    print("-" * 60)
    print(f"{'Total Parameters':<40} {total_params:>10,}")
    print("-" * 60)
    return total_params

def main():
    # Training settings
    batch_size = 128  # Reduced batch size for better generalization
    epochs = 20
    max_lr = 0.05    # Reduced max learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enhanced data transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation((-8.0, 8.0), fill=(1,)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    model = Model().to(device)
    
    # Add this line after model initialization
    total_params = count_parameters(model)
    if total_params > 20000:
        print(f"\nWarning: Model has {total_params:,} parameters, which exceeds the 20k limit!")
    
    optimizer = optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    
    # OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy='cos'
    )

    best_accuracy = 0
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        accuracy = test(model, device, test_loader)
        test_accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best Test Accuracy: {best_accuracy:.2f}%')

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), test_accuracies, 'bo-')
    plt.title(f'Test Accuracy vs. Epoch (Best: {best_accuracy:.2f}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.show()

if __name__ == '__main__':
    main() 