from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler 

BATCH_SIZE = 20
class_train_count = [3995, 436, 4097, 7214, 4830, 3171, 4965]
class_test_count = [467, 56, 496, 895, 653, 415, 607]
class_valid_count = [491, 55, 528, 879, 594, 416, 626]

train_weights = 1/torch.Tensor(class_train_count)
test_weights = 1/torch.Tensor(class_test_count)
valid_weights = 1/torch.Tensor(class_valid_count)

train_data = ImageFolder(root = 'Data_Images_Facial_Expressions/Training', transform=ToTensor())
test_data = ImageFolder(root = 'Data_Images_Facial_Expressions/PublicTest', transform = ToTensor())
valid_data = ImageFolder(root = 'Data_Images_Facial_Expressions/PrivateTest', transform = ToTensor())

train_sampler = WeightedRandomSampler(train_weights, 2800)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 10, shuffle = True, sampler = train_sampler)

test_sampler = WeightedRandomSampler(test_weights, 100)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 10, shuffle = True, sampler = test_sampler)

test_sampler = WeightedRandomSampler(test_weights, 100, replacement = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 10, shuffle = True, sampler = test_sampler)

valid_sampler = WeightedRandomSampler(valid_weights, 100, replacement = True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 10, shuffle = True, sampler = valid_sampler)



class FaceCNN(nn.Module):
	def __init__(self):
		super(FaceCNN, self).__init__()

		self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride = 1, padding = 2)
		self.pool = nn.MaxPool2d(2,2) # go to 24x24
		self.conv2 = nn.Conv2d(6, 12, kernel_size = 5, stride = 1, padding = 2)
		self.fc1 = nn.Linear(12 * 12 * 12, 7)
	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = self.pool(out)
		out = F.relu(self.conv2(out))
		out = self.pool(out) # 12x12
		out = out.view(-1, 12 * 12 * 12)
		out = self.fc1(out)

		return out

print("Made it through loading...\n")

face_cnn = FaceCNN()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1

optimizer = torch.optim.SGD(face_cnn.parameters(), lr = learning_rate)
num_epochs = 3
for epoch in range(num_epochs):
	train_loss = 0
	n_iter = 0
	for i, (data, target) in enumerate(trainloader):
		optimizer.zero_grad()
		output = face_cnn(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		train_loss += loss.data[0]
		n_iter += 1
		if (i % 50 == 0):
			print(i)
	print("Epoch: {}/{}, Loss: {:.4f}".format(epoch+1, num_epochs, train_loss/n_iter))
