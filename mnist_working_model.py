from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomRotation(degrees=10),  # Rotate by up to 10 degrees
                        transforms.RandomAffine(degrees=0, shear=5),  # Shear by up to 5 degrees
                        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Add perspective transform
                        # transforms.ColorJitter(brightness=0.2),  # Add slight brightness variation
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)


class MNIST_CNN_Second(nn.Module):
    def __init__(self):
        super(MNIST_CNN_Second, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            ## Transition Layer
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 8, kernel_size=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            ## Transition Layer
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 8, kernel_size=1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Get the actual dimensions before reshaping
        x = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
        x = x.view(-1, 32)
        x = self.fc1(x) 

        x = F.log_softmax(x, dim=1)
        return x
    

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Epoch: {epoch} Loss={loss.item():.4f} Batch={batch_idx} Accuracy={100*correct/processed:0.2f}%')
    
    return train_loss/len(train_loader), 100*correct/processed

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
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.2f}%)')
    
    return test_loss, test_accuracy

if __name__ == "__main__":

    model = MNIST_CNN_Second().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,  # Higher max_lr for SGD
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,
    anneal_strategy='cos'
    )

    # Training loop with early stopping
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    early_stopping_accuracy = 99.4

    for epoch in range(1, 21):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, scheduler, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)
        
        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        
        # Check for early stopping
        if test_accuracy >= early_stopping_accuracy:
            print(f"Early stopping at epoch {epoch} with test accuracy: {test_accuracy:.2f}%")
            break
