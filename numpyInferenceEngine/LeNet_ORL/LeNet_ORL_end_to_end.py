
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir = "./orl_face_dataset/training/"
    testing_dir = "./orl_face_dataset/testing/"
    train_batch_size = 64
    train_number_epochs = 100

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        # print("self.imageFolderDataset.imgs: ", self.imageFolderDataset.imgs)
        # print("img1_tuple: ", img1_tuple, ", img0_tuple: ", img0_tuple)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

folder_dataset = dset.ImageFolder(root=Config.training_dir)

siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
dataiter = iter(vis_dataloader)



class LeNet_Small_BN(nn.Module):
    def __init__(self):
        super(LeNet_Small_BN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.BN1 = nn.BatchNorm2d(6)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.BN2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1, bias=False) # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.act3 = nn.ReLU()
        self.BN3 = nn.BatchNorm2d(120)
        self.linear1 = nn.Linear(in_features=43320, out_features=512)
        self.act4 = nn.ReLU()
        # 10 => 
        # 256 => 51% accuracy
        self.linear2 = nn.Linear(in_features=512, out_features=10)
        # self.act5 = nn.ReLU()
        # self.linear3 = nn.Linear(in_features=500, out_features=10)
    def forward_once(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.BN1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.BN2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.BN3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        # x = self.act5(x)
        # x = self.linear3(x)
        return x
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2



class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)

net = LeNet_Small_BN().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )


def train():
    counter = []
    loss_history = [] 
    iteration_number= 0
    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        # torch.save(net.state_dict(), 'siamse_weight.pkl')

if os.path.exists('siamse_weight.pkl'):
    net.load_state_dict(torch.load('siamse_weight.pkl'))
    net.eval()
else:
    train()



folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)

def cosine_similarity(x, y):
    x = x.numpy()
    y = y.numpy()
    return x.dot(y)/(x.dot(x)**0.5 * y.dot(y)**0.5)

y_gt = []
y_pred = []

for idx in range(10):
    for data in enumerate(test_dataloader):
        x0, x1, label = data[1]
        
        output1,output2 = net(Variable(x0).cuda(),Variable(x1).cuda())
        # print("output1: ", output1, ", output2: ", output2)
        # euclidean_distance = F.pairwise_distance(output1, output2)
        cos = cosine_similarity(output1.detach().cpu().reshape((10,)), output2.detach().cpu().reshape((10,)))
        if cos > 0.5:
            y_pred.append(0) # The same person
        else:
            y_pred.append(1) # Different person
        y_gt.append(int(label.numpy()))

y_pred = np.array(y_pred)
y_gt = np.array(y_gt)

print("y_pred: ", y_pred)
print("y_gt: ", y_gt)

print((y_pred == y_gt).mean())

    
