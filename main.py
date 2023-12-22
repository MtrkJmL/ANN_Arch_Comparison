import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import json

#parameters
epochs = 15
batch_size = 64
learning_rate = 0.01
input_size = 784
hidden_size = 64
num_classes = 10
RUN = 10
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])

# training set
train_data = torchvision.datasets.FashionMNIST('./data', train = True, download = True,
transform = transform)

# test set
test_data = torchvision.datasets.FashionMNIST('./data', train = False,
transform = transform)

indic = list(range(len(train_data)))
np.random.shuffle(indic)
split = int(np.floor(0.1*len(train_data)))
split9 = int(np.floor(0.9*len(train_data)))
valid_data = torch.utils.data.SubsetRandomSampler(indic[:split])
new_train_data = torch.utils.data.SubsetRandomSampler(indic[:split9])


train_generator = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler=new_train_data)
test_generator = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)
valid_generator = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler= valid_data)

#gpu = torch.device("cuda")
gpu = torch.device("cpu")
print(len(new_train_data))
print(len(valid_data))


#CNN Models
class mlp_1(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_1, self).__init__()

        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_size)

        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

class mlp_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_2, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        x = self.fc2(relu)
        output = self.fc3(x)
        return output

class cnn_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(cnn_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8,16,5)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 144)
        x = self.fc1(x)
        return x

class cnn_4(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(cnn_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8,8,3)
        self.conv4 = nn.Conv2d(8,16,5)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 144)
        x = self.fc1(x)
        return x

class cnn_5(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(cnn_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8,8,3)
        self.conv4 = nn.Conv2d(8,8,3)
        self.conv5 = nn.Conv2d(8, 16, 3)
        self.conv6 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 144)
        x = self.fc1(x)
        return x




#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_steps = len(train_generator)

def get_accuracy(pred, actual):
  assert len(pred) == len(actual)

  total = len(actual)
  _, predicted = torch.max(pred.data, 1)
  correct = (predicted == actual).sum().item()
  return correct / total

loss_arr = [[] for a in range(0,RUN)]
t_acc_arr = [[] for a in range(0,RUN)]
v_acc_arr = [[] for a in range(0,RUN)]
weights = [[] for a in range(0,RUN)]
test_accuracy = []

for a in range(RUN):
    model_cpu = cnn_5(input_size, hidden_size, num_classes)
    model = cnn_5(input_size, hidden_size, num_classes).to(gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        correct = 0
        for i, (images, labels) in enumerate(train_generator):
            model.train()
            images = images.to(gpu)
            labels = labels.to(gpu)

            # Forward pass
            outputs = model(images)
            _, pred = torch.max(outputs,1)
            correct += (pred == labels).sum().item()
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            samples = batch_size * 10
            if (i+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    n_correct = 0
                    n_samples = 0
                    for images1, labels1 in valid_generator:
                        images1 = images1.to(gpu)
                        labels1 = labels1.to(gpu)
                        outputs1 = model(images1)
                        _, preds = torch.max(outputs1,1)
                        n_samples += len(labels1)
                        n_correct += (preds == labels1).sum().item()
                    acc = n_correct / n_samples * 100
                    training_accuracy = correct / samples * 100
                    loss_arr[a] = loss_arr[a] + [loss.item()]
                    t_acc_arr[a] = t_acc_arr[a] + [training_accuracy]
                    v_acc_arr[a] = v_acc_arr[a] + [acc]
                    print (f'RUN:{a+1}, Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}, Training Accuracy:{training_accuracy:.2f}, Validation Accuracy:{acc:.2f}')
                correct = 0
                model.train()
    with torch.no_grad():
        n_correct2 = 0
        n_samples2 = 0
        for images2, labels2 in test_generator:
            model.eval()
            images2 = images2.to(gpu)
            labels2 = labels2.to(gpu)
            outputs2 = model(images2)
            # max returns (value ,index)
            _, predicted2 = torch.max(outputs2, 1)
            n_samples2 += len(labels2)
            n_correct2 += (predicted2 == labels2).sum().item()
    t_acc = 100.0 * n_correct2 / n_samples2
    test_accuracy = test_accuracy + [t_acc]
    print(f'Test Accuracy: {t_acc} %')


    torch.device("cpu")
    weight = model_cpu.fc1.weight.data.cpu().numpy().tolist()
    weights[a] = weights[a] + [weight]
    PATH = './weights1-{RUN}.pth'
    torch.save(weight, PATH)
    torch.device("cuda")
    del model
    del model_cpu

torch.save(loss_arr, './loss.pth' )
torch.save(t_acc_arr, './t_acc.pth' )
torch.save(v_acc_arr, './v_acc.pth' )

best_test_acc = 0
best_test_acc_index = 0
for i in range (0, len(test_accuracy)):
    if (test_accuracy[i] > best_test_acc):
        best_test_acc = test_accuracy[i]
        best_test_acc_index = i

training_loss = []
training_accuracy = []
valid_accuracy = []

training_loss = [sum(x) for x in zip(*loss_arr)]
training_loss = [element / float(RUN) for element in training_loss]

training_accuracy = [sum(x) for x in zip(*t_acc_arr)]
training_accuracy = [element / float(RUN) for element in training_accuracy]

valid_accuracy = [sum(x) for x in zip(*v_acc_arr)]
valid_accuracy = [element / float(RUN) for element in valid_accuracy]
print(type(training_loss))
print(type(training_accuracy))
print(type(valid_accuracy))
print(type(weights))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import json

#parameters
epochs = 15
batch_size = 64
learning_rate = 0.01
input_size = 784
hidden_size = 64
num_classes = 10
RUN = 10
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])

# training set
train_data = torchvision.datasets.FashionMNIST('./data', train = True, download = True,
transform = transform)

# test set
test_data = torchvision.datasets.FashionMNIST('./data', train = False,
transform = transform)

indic = list(range(len(train_data)))
np.random.shuffle(indic)
split = int(np.floor(0.1*len(train_data)))
split9 = int(np.floor(0.9*len(train_data)))
valid_data = torch.utils.data.SubsetRandomSampler(indic[:split])
new_train_data = torch.utils.data.SubsetRandomSampler(indic[:split9])


train_generator = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler=new_train_data)
test_generator = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False)
valid_generator = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler= valid_data)

gpu = torch.device("cuda")

print(len(new_train_data))
print(len(valid_data))


#CNN Models
class mlp_1(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_1, self).__init__()

        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_size)

        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

class mlp_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp_2, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        x = self.fc2(relu)
        output = self.fc3(x)
        return output

class cnn_3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(cnn_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8,16,5)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 144)
        x = self.fc1(x)
        return x

class cnn_4(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(cnn_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8,8,3)
        self.conv4 = nn.Conv2d(8,16,5)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 144)
        x = self.fc1(x)
        return x

class cnn_5(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(cnn_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8,8,3)
        self.conv4 = nn.Conv2d(8,8,3)
        self.conv5 = nn.Conv2d(8, 16, 3)
        self.conv6 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(144, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 144)
        x = self.fc1(x)
        return x




#criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_steps = len(train_generator)

def get_accuracy(pred, actual):
  assert len(pred) == len(actual)

  total = len(actual)
  _, predicted = torch.max(pred.data, 1)
  correct = (predicted == actual).sum().item()
  return correct / total

loss_arr = [[] for a in range(0,RUN)]
t_acc_arr = [[] for a in range(0,RUN)]
v_acc_arr = [[] for a in range(0,RUN)]
weights = [[] for a in range(0,RUN)]
test_accuracy = []

for a in range(RUN):
    model_cpu = cnn_5(input_size, hidden_size, num_classes)
    model = cnn_5(input_size, hidden_size, num_classes).to(gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        correct = 0
        for i, (images, labels) in enumerate(train_generator):
            model.train()
            images = images.to(gpu)
            labels = labels.to(gpu)

            # Forward pass
            outputs = model(images)
            _, pred = torch.max(outputs,1)
            correct += (pred == labels).sum().item()
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            samples = batch_size * 10
            if (i+1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    n_correct = 0
                    n_samples = 0
                    for images1, labels1 in valid_generator:
                        images1 = images1.to(gpu)
                        labels1 = labels1.to(gpu)
                        outputs1 = model(images1)
                        _, preds = torch.max(outputs1,1)
                        n_samples += len(labels1)
                        n_correct += (preds == labels1).sum().item()
                    acc = n_correct / n_samples * 100
                    training_accuracy = correct / samples * 100
                    loss_arr[a] = loss_arr[a] + [loss.item()]
                    t_acc_arr[a] = t_acc_arr[a] + [training_accuracy]
                    v_acc_arr[a] = v_acc_arr[a] + [acc]
                    print (f'RUN:{a+1}, Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}, Training Accuracy:{training_accuracy:.2f}, Validation Accuracy:{acc:.2f}')
                correct = 0
                model.train()
    with torch.no_grad():
        n_correct2 = 0
        n_samples2 = 0
        for images2, labels2 in test_generator:
            model.eval()
            images2 = images2.to(gpu)
            labels2 = labels2.to(gpu)
            outputs2 = model(images2)
            # max returns (value ,index)
            _, predicted2 = torch.max(outputs2, 1)
            n_samples2 += len(labels2)
            n_correct2 += (predicted2 == labels2).sum().item()
    t_acc = 100.0 * n_correct2 / n_samples2
    test_accuracy = test_accuracy + [t_acc]
    print(f'Test Accuracy: {t_acc} %')


    torch.device("cpu")
    weight = model_cpu.fc1.weight.data.cpu().numpy().tolist()
    weights[a] = weights[a] + [weight]
    PATH = './weights1-{RUN}.pth'
    torch.save(weight, PATH)
    torch.device("cuda")
    del model
    del model_cpu

torch.save(loss_arr, './loss.pth' )
torch.save(t_acc_arr, './t_acc.pth' )
torch.save(v_acc_arr, './v_acc.pth' )

best_test_acc = 0
best_test_acc_index = 0
for i in range (0, len(test_accuracy)):
    if (test_accuracy[i] > best_test_acc):
        best_test_acc = test_accuracy[i]
        best_test_acc_index = i

training_loss = []
training_accuracy = []
valid_accuracy = []

training_loss = [sum(x) for x in zip(*loss_arr)]
training_loss = [element / float(RUN) for element in training_loss]

training_accuracy = [sum(x) for x in zip(*t_acc_arr)]
training_accuracy = [element / float(RUN) for element in training_accuracy]

valid_accuracy = [sum(x) for x in zip(*v_acc_arr)]
valid_accuracy = [element / float(RUN) for element in valid_accuracy]
print(type(training_loss))
print(type(training_accuracy))
print(type(valid_accuracy))
print(type(weights))
