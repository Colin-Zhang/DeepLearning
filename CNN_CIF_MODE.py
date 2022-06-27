import torch
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear

class CNN_CIF_MODE(nn.Module):
	def __init__(self):
		super(CNN_CIF_MODE,self).__init__()
		self.model =  nn.Sequential(
			nn.Conv2d(3,32,5,1,2),
			nn.MaxPool2d(2),
			nn.Conv2d(32,32,5,1,2),
			nn.MaxPool2d(2),
			nn.Conv2d(32,64,5,1,2),
			nn.MaxPool2d(2),
			nn.Flatten(),
			nn.Linear(64*4*4,64),
			nn.Linear(64,10)
			)
	def forward(self,x):
		x = self.model(x)
		return x

