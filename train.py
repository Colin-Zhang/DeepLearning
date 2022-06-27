import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from CNN_CIF_MODE import *


train_data =  torchvision.datasets.CIFAR10(root="data",train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data =  torchvision.datasets.CIFAR10(root="data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_dataLoader = DataLoader(train_data,batch_size=64)

test_dataLoader = DataLoader(test_data,batch_size=64)

loss_fn = nn.CrossEntropyLoss()


cifModel = CNN_CIF_MODE()

learning_rate = 1e-2

optimizer = torch.optim.SGD(cifModel.parameters(),learning_rate)

epoch_num = 40

total_train_step = 0

total_test_step = 0

writer = SummaryWriter("logs_cif")

for i in range(epoch_num):
	
	print("---------第{}轮训练开始---------".format(i))
	for data in train_dataLoader:

		imgs,target = data
		
		outputs = cifModel(imgs)

		loss = loss_fn(outputs,target)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		total_train_step = total_train_step+1

		if total_train_step %100 ==0:

			print("训练次数: {},Loss: {}".format(total_train_step,loss.item()))
			
			writer.add_scalar("train_loss",loss.item(),total_train_step)
			
	total_test_loss = 0
	with torch.no_grad() :
		for data in test_dataLoader:
			imgs,target = data
			outputs = cifModel(imgs)
			loss = loss_fn(outputs,target)
			total_test_loss = total_test_loss+loss.item()
	writer.add_scalar("test_loss",total_test_loss,total_test_step)
	total_test_step = total_test_step+1
writer.close()