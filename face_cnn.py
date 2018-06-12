import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceCNN(nn.Module):
	def __init__(self):
		super(FaceCNN, self).__init__()

		self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride = 1, padding = 2)
		self.pool = nn.MaxPool2d(2,2) # go to 24x24
		self.conv2 = nn.Conv2d(6, 12, kernel_size = 5, stride = 1, padding = 2)
		self.conv3 = nn.Conv2d(12, 24, kernel_size = 5, stride = 1, padding = 2)
		#self.conv2_drop = nn.Dropout2d(p=0.2)
		self.fc1 = nn.Linear(6 * 6 * 24, 100)
		self.fc2 = nn.Linear(100, 2)
		#self.conv4 = nn.Conv2d(24, 36, kernel_size = 5, stride = 1, padding = 2)
		#self.fc1 = nn.Linear(3 * 3 * 36, 2)
		
	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = self.pool(out)
		out = F.relu(self.conv2(out))
		out = self.pool(out) # 12 x 12
		out = F.relu((self.conv3(out)))
		out = self.pool(out) # 6 x 6
		out = out.view(-1, 6 * 6 * 24)
		out = self.fc1(out)
		out = self.fc2(out)
		#out = F.relu(self.conv4(out))
		#out = self.pool(out)
		#out = out.view(-1, 3 * 3 * 36)

		return out
