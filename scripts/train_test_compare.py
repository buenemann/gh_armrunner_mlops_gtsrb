import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transformations for the training data and testing data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the GTSRB dataset
train_set = torchvision.datasets.GTSRB(root='./data', split='train',
download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.GTSRB(root='./data', split='test',
download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 43) # 43 classes in GTSRB dataset

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = TrafficSignNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
print('Set epoch number as %d' % num_epochs)


model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99: # Print every 100 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model into ./models/traffic_sign_net_ep10.pth
torch.save(model.state_dict(), './models/traffic_sign_net_ep3.pth')

print('Save trained Model')


# Test training model (num_epochs)
# Load the model
model = TrafficSignNet()
model.load_state_dict(torch.load('./models/traffic_sign_net_ep3.pth'))
model.eval()

# Define transformations for the testing data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the GTSRB test dataset
test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'\n\n [ Test Result ] Trained model epoch: {num_epochs}')
print(f' [ Test Result ] Accuracy of the test images: {100 * correct / total:.2f}%\n\n')


# Re-Test default epoch 10 model
# Load the model
model_1 = TrafficSignNet()
model_1.load_state_dict(torch.load('./models/traffic_sign_net.pth'))
model_1.eval()

# Define transformations for the testing data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the GTSRB test dataset
test_set = torchvision.datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model_1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n\n [ Test Result ] Default model epoch: 10')
print(f' [ Test Result ] Accuracy of the default model (epochs: 10) on the test images: {100 * correct / total:.2f}%\n\n')
