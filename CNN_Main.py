from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
def find_lable(str):
	first,last  =0,0
	for index in range(len(str)-1,-1,-1):
		if str[index] == '%' and str[index-1]=='.':
			last = index -1
		if (str[index]=='c' or str[index]=='d') and str[index-1]=='/':
			first = index
			break
	name = str[first:last]
	if name=='dog':
		return 1
	else :
		return 0


def init_process(path,lens):
	data = []
	name = find_lable(path)
	for index in range(lens[0],lens[1]):
		data.append([path%index,name])
	return data

def PicLoader(path):
	return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder
    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        print(img.shape)
        return img, label

    def __len__(self):
        return len(self.data)

def load_data():
	transform = transforms.Compose([transforms.CenterCrop(224),
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	train_dog_path = 'data//training_data//dogs//dog.%d.jpg'
	train_dog_data =  init_process(train_dog_path,[0,500])

	train_cat_path = 'data//training_data//cats//cat.%d.jpg'
	train_cat_data =  init_process(train_cat_path,[0,500])


	test_dog_path = 'data//testing_data//dogs//dog.%d.jpg'
	test_dog_data =  init_process(test_dog_path,[1000,1200])

	test_cat_path ='data//testing_data//cats//cat.%d.jpg'
	test_cat_data =init_process(test_cat_path,[1000,1200])

	print(test_dog_data)

	train_data = train_cat_data + train_dog_data +test_cat_data[0:150]+test_dog_data[0:150]
	test_data =test_dog_data[150:200] +test_cat_data[150:200]
	train = MyDataset(train_data,transform=transform,loder=PicLoader)

	test = MyDataset(test_data,transform=transform,loder=PicLoader)
	train_data = DataLoader(dataset=train,batch_size=5,shuffle=True,num_workers=0,pin_memory=True)
	test_data = DataLoader(dataset=test,batch_size=1,shuffle=True,num_workers=0,pin_memory=True)
	return train_data,test_data



