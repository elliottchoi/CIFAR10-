import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

#hyperParam
c_Class=10
num_epochs=2
batch_size=5
learning_rate=0.001
momentum_rate=0.9

#Print Images for Visual check:
def normalize_image(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

#train dataloader:
#set up the transform and normalize the range to [-1,1]
#Input sequence of means and SD for each channel, we have 3 and we use 0.5 SD and U
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Finding appropriate data sets, pixel size is 32x32x3
trainset =dsets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = dsets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


#Loading data sets into data loader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True,)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False,)

#define the classes in the dataset
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



#Models are defined as a subclass to the nn.Module
class neuralNet(nn.Module):

    #intializing the layers to be used
    def __init__(self,c_Class):
        super(neuralNet,self).__init__()
        #Convolution Layer (input 3, out 6)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5,stride=1,padding=0)
        #Max Pool to get rid of 75%, 2 by 2 matrix with a skip of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Convolution Layer takes in 6, outputs 16
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5,stride=1,padding=0)
        #Connection conv2 output, 16 layers of 5x5 kernals
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #Scales down 120 nodes to 84
        self.fc2 = nn.Linear(120, 84)
        #Scales down 84 to 10 classes
        self.fc3 = nn.Linear(84, c_Class)

    #Specify the connection of the layers
    def forward(self, x):
        #Recall Relu is a max function of (0,Rx) then we pool to remove 75%
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        in_size=x.size(0)
        #Interpert tensor into 1 row by 400
        x = x.view(-1, 400)
        #Feed fully connected to activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


neuralNet = neuralNet(c_Class)

#Classifier optimizer and method of learning
#Cross Entropy is better than using soft_max and sigmoid function
criterion = nn.CrossEntropyLoss()
#will compare SGD to Adam
optimizer = optimizer.SGD(neuralNet.parameters(), lr=learning_rate, momentum=momentum_rate)

#training the neuralnet
for epochs in range(num_epochs):
    aggregate_epoch_loss=0
    placeholder_i=0
    for epoch, data in enumerate(train_loader,0):
        image,label=data
        images,labels=Variable(image),Variable(label)
        #Reset the gradients
        optimizer.zero_grad()

        net_Output=neuralNet(images)
        #compare net output to training set labels
        loss=criterion(net_Output,labels)
        #backwards propogation to get gradient
        loss.backward()
        optimizer.step()

        if (epoch+1)%batch_size==0:
            print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.6f'
                  % (epochs + 1, num_epochs, (epoch + 1)*len(images), len(trainset), loss.data[0]))
            aggregate_epoch_loss += loss.data[0]
        placeholder_i = epoch
    print('Mean Loss for Epoch: %.6f' % (aggregate_epoch_loss / ((placeholder_i+1) / batch_size)))

'''
Testing specific sets to see accuracy
'''

#interator
image_iter=iter(test_loader)
#fit the image and corresponding label to next set in iterator
image,label=image_iter.next()

#Output for visual check
normalize_image(torchvision.utils.make_grid(image))
#Output labels:
#iterates up to batch_size and puts out the lable in a string size 5 spaces
#.join is the equivalent to append
print('Labels: ', ' '.join('%5s' % classes[label[j]] for j in range(batch_size)))

#Comparison to what the model thinks
net_Output=neuralNet(Variable(image))
#we want to ONLY output the highest probability of since each class with have a probability
_,predicted_class=torch.max(net_Output.data,1)
print('Predicted: ', ' '.join('%5s' % classes[predicted_class[j]]
                              for j in range(batch_size)))

'''
Testing the entire set
'''
correct = 0
total = 0
for data in test_loader:
    images, labels = data
    outputs = neuralNet(Variable(images))
    _, predicted_class = torch.max(outputs.data, 1)
    total += labels.size(0)
    #if they match add to correct
    correct += (predicted_class == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

'''
Granular breakdown of testing set
'''

class_correct = list(0. for i in range(c_Class))
class_total = list(0. for i in range(c_Class))
for data in test_loader:
    images, labels = data
    outputs = neuralNet(Variable(images))
    _, highest_predict = torch.max(outputs.data, 1)
    #If we are correct, we resize into a 1-D row
    c = (highest_predict == labels).squeeze()
    #iterate through our batch
    for i in range(batch_size):
        #for the ith label in batch
        label = labels[i]
        #if class_correct matches,
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(c_Class):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
