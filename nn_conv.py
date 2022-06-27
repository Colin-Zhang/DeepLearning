import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
dataset = torchvision.datasets.CIFAR10('../data',train = False,transform=torchvision.transforms.ToTensor(),download=True)
dataLoader =  DataLoader(dataset,batch_size=64)

class Cnn(nn.Module) :
	def __init__(self):
		super(Cnn,self).__init__()
		self.conv1 =Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
		self.conv2 =Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
		self.conv3 =Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
		self.conv4 =Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
		self.conv5 =Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
		self.MaxPool2d = MaxPool2d(kernel_size=3,ceil_mode=True)
		self.sigmoid1 = Sigmoid()
	def forward(self,x):
		print(x.shape)
		x = self.sigmoid1(x)
		return x


cnn = Cnn()

step = 0
writer = SummaryWriter("logs")
for data in dataLoader:
	imgs,targets = data
	output = cnn(imgs)
	print(imgs.shape)
	print(output.shape)
	writer.add_images("in_put",imgs,step)
	# output = torch.reshape(output,(-1,3,30,30))
	writer.add_images("out_put",output,step)
	step = step+1

writer.close()
