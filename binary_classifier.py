from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data.sampler import WeightedRandomSampler

BATCH_SIZE = 35

trans = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
#trans = transforms.Compose([transforms.ToTensor()])

train_data = ImageFolder(root = 'baby_class', transform=trans)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)

test_data = ImageFolder(root = 'baby_test', transform=trans)
test_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)

class FaceCNN(nn.Module):
	def __init__(self):
		super(FaceCNN, self).__init__()

		self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride = 1, padding = 2)
		self.pool = nn.MaxPool2d(2,2) # go to 24x24
		self.conv2 = nn.Conv2d(6, 12, kernel_size = 5, stride = 1, padding = 2)
		self.conv3 = nn.Conv2d(12, 24, kernel_size = 5, stride = 1, padding = 2)
		self.fc1 = nn.Linear(6 * 6 * 24, 2)
		
	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = self.pool(out)
		out = F.relu(self.conv2(out))
		out = self.pool(out) # 12x12
		out = F.relu(self.conv3(out))
		out = self.pool(out)
		out = out.view(-1, 6 * 6 * 24)
		out = self.fc1(out)

		return out

print("Made it through loading...\n")

face_cnn = FaceCNN()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

# POTENTIAL SOLUTIONS 
# try smaller kernel size (did not work) 
# can try outputting a probabiltiy instead of 1 hot. sigmoid
# try printing tgradients. check for divergence there.
# try printing accuracy at each iteration... is it even getting better there?
# try changing batch size... -> smaller first? (did not work)
# dropout helps with noise, but maybe 2 small
# try sampling at each epoch from larger set. helps w robust

check_pt = {}
best = 100
optimizer = torch.optim.Adam(face_cnn.parameters(), lr = learning_rate)
num_epochs = 30
for epoch in range(num_epochs):
	train_loss = 0
	n_iter = 0
	for i, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		output = face_cnn(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		train_loss += loss.data[0]
		n_iter += 1
	print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, num_epochs, train_loss/n_iter))
	if (train_loss/n_iter < best):
                best = train_loss/n_iter
                check_pt['state_dict'] = face_cnn.state_dict()
                check_pt['epoch'] = epoch+1

torch.save(check_pt['state_dict'], 'best_binary.pth')

face_cnn.eval()
correct = 0
total = 0

for i, (data, target) in enumerate(test_loader):
    output = face_cnn(data)
    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)

    total += target.size(0)
    correct += (predicted==target).sum()
    #print("predicted = {}, target = {}".format(predicted, target))

print('Accuracy on test set: {}%'.format(100* correct/total))
