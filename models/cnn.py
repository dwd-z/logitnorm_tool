import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        num_neurons_monitored = 40
        self.conv1 = nn.Conv2d(1, 40, 5) # input 28*28*1->conv->24*24*40
        self.pool = nn.MaxPool2d(2, 2) # 24*24*40->maxpool->12*12*40
        self.conv2 = nn.Conv2d(40, 20, 5) # 12*12*40->conv->8*8*20-
        self.fc1 = nn.Linear(20*4*4, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, num_neurons_monitored)
        self.fc4 = nn.Linear(num_neurons_monitored, num_classes)
    
    # Relu is used as asctivation function.
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # input 28*28*1->conv->24*24*40->maxpool->12*12*40
        x = self.pool(F.relu(self.conv2(x))) #       12*12*40->conv->8*8*20->maxpool->4*4*20
        x = x.view(-1, 20*4*4)             # flatten it to an array
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        return out

    def forwardWithIntermediate(self, x):
        x = self.pool(F.relu(self.conv1(x)))        # input 28*28*1->conv->24*24*40->maxpool->12*12*40
        x = self.pool(F.relu(self.conv2(x)))        # 12*12*40->conv->8*8*20->maxpool->4*4*20
        x = x.view(-1, 20 * 4 * 4)      # flatten it to an array
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        intermediateValues = x # return the value of the layer monitored
        x = F.relu(x)
        out = self.fc4(x)
        return out, intermediateValues