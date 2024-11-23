import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

label_encode = preprocessing.LabelEncoder()
trans = transforms.ToTensor()

root = "/home/sysadm/Documents/BanglaWriting: A multi-purpose offline Bangla handwriting dataset/bengali_dataset"
pretrain_save_path = "results/Bengali_MOCO_model.pth"
finetune_save_path = "results/Bengali_MOCO_model_finetuned.pth"
results_df_save_path = "results/Bengali_MOCO_model_finetune_results.csv"
pretrain_image_save_path = "results/Bengali_MOCO_pretrain_loss.png"
finetune_image_save_path = "results/Bengali_MOCO_finetune_loss.png"

batch_size = 128

train_dir = root+"/train"
test_dir = root+"/test"

train_dataset = datasets.ImageFolder(root=train_dir)
test_dataset = datasets.ImageFolder(root=test_dir)

label_encode = LabelEncoder()
trans = transforms.ToTensor()
rand = transforms.RandomCrop((64,128))

class Bengali_Datamodule_Pretext(nn.Module):
    def __init__(self,dataset = train_dataset,ptsz=32):
        super().__init__()
        self.dataset = dataset
        self.ptsz = ptsz

        self.transformations = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.25,0.25, 0.2, 0.2)],p = 0.5),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                   ])

    def __len__(self):
        return len(self.dataset)

    def __white_pad__(self,img):

        if img.shape[0]<64 and img.shape[1]<128:

            deff_height = 64-img.shape[0]
            deff_height = deff_height//2
            deff_width = 128-img.shape[1]
            deff_width = deff_width//2
            if (img.shape[0]%2) == 0:
                deff_height = deff_height
            else:
                deff_height = deff_height+1
            
            if (img.shape[1]%2 == 0):
                deff_width = deff_width
            else:
                deff_width = deff_width+1
            result = cv2.copyMakeBorder(img,deff_height,deff_height,deff_width,deff_width,cv2.BORDER_CONSTANT,None,value=(255,255,255))
            

        elif img.shape[0]<=64 and img.shape[1]>=128:
            deff_height = 64-img.shape[0]
            deff_height = deff_height//2
            if (img.shape[0] % 2) == 0:
                deff_height=deff_height
            else:
                deff_height = deff_height+1
            result = cv2.copyMakeBorder(img,deff_height,deff_height,0,0,cv2.BORDER_CONSTANT,None,value=(255,255,255))

        elif img.shape[0]>=64 and img.shape[1]<=128:
            deff_width = 128-img.shape[1]
            deff_width = deff_width//2
            if (img.shape[1] % 2) == 0:
                deff_width=deff_width
            else:
                deff_width = deff_width+1

            result = cv2.copyMakeBorder(img,0,0,deff_width,deff_width,cv2.BORDER_CONSTANT,None,value=(255,255,255))

        else:
            result = img

        return result


    def __getpatches__(self, x):
        pts = []
        H,W,C = x.shape
        #Changes Made: Overlapping Removed
        numdelH = 64//(self.ptsz)
        numdelW = 128//(self.ptsz)

        for i in range(numdelH):
            for j in range(numdelW):
                sx = i*self.ptsz #//2)
                ex = sx + self.ptsz
                sy = j*self.ptsz #//2)
                ey = sy + self.ptsz
                temp = x[sx:ex,sy:ey,:]
                temp = np.transpose(temp, (2,0,1))
                temp = torch.from_numpy(temp)     
                pts.append(torch.unsqueeze(temp, 0))

        return torch.cat(pts, dim = 0)

    def __getitem__(self, idx):
        data_object = self.dataset[idx]
        PIL_img = data_object[0]
        orgpic = cv2.cvtColor(np.array(PIL_img), cv2.COLOR_RGB2BGR)
        orgpic = self.__white_pad__(orgpic)

        orgpic = orgpic/255.0
        orgpic = trans(orgpic)
        orgpic = rand(orgpic)
        orgpic = orgpic.float()
        
        orgpic1, orgpic2 = self.__augment__(orgpic) # 
        orgpic1 = orgpic1.numpy().transpose(1,2,0)
        orgpic2 = orgpic2.numpy().transpose(1,2,0)
        
        orgpic1pts = self.__getpatches__(orgpic1)
        orgpic2pts = self.__getpatches__(orgpic2)

        return orgpic1pts, orgpic2pts
    
    def __augment__(self,x):
        x1, x2 = self.transformations(x), self.transformations(x)
        return x1, x2

class Model(nn.Module):
    def __init__(self, Nh, Nw, bs, ptsz = 32, pout = 512):
        super().__init__()

        self.Nh = Nh
        self.Nw = Nw
        self.bs = bs
        self.ptsz = ptsz
        self.pout = pout
        
        self.base_encoder = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.base_encoder.append(module)
        # encoder
        self.base_encoder = nn.Sequential(*self.base_encoder)
        
        self.base_encoder.fc = nn.Identity()

        self.proj2 = nn.Sequential(*[nn.Linear(512, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, self.pout),
                                    nn.BatchNorm1d(self.pout)])
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.cntH = self.Nh//(self.ptsz)
        self.cntW = self.Nw//(self.ptsz)
    
    def forward(self, x):
        x = x.view((-1,3,self.ptsz,self.ptsz))
        x = self.base_encoder(x)
        x = x.view((self.bs, -1, self.cntH, self.cntW))
        x = self.gap(x).squeeze()
        x = torch.flatten(x, start_dim=1)
        x1 = self.proj2(x)
        return x, x1

train_data = Bengali_Datamodule_Pretext(dataset=train_dataset)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                          drop_last=True)

cos=True
epochs=1000
k=200
knn_t=0.1
moco_dim=128
moco_k=4096
momentum=0.99
moco_t=0.1
symmetric=False
wd=0.0005
m = 4096
temperature = 0.5

memory_queue = F.normalize(torch.randn(m, 512).cuda(), dim=-1)
model_q = Model(64, 128, bs=batch_size).cuda()
model_k = Model(64, 128, bs=batch_size).cuda()

for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
    param_k.data.copy_(param_q.data)
    param_k.requires_grad = False
    
optimizer = optim.Adam(model_q.parameters(), lr=1e-4, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, 1000)

def train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    for x_q, x_k in data_loader:
        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        _,query = encoder_q(x_q)

        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _,key = encoder_k(x_k[idx])
        key = key[torch.argsort(idx)]

        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)

    return total_loss / total_num

epochs = 5000

train_losses = []
best_loss = float("inf")
count = 0
early_stop = 50

bar = tqdm(range(epochs), desc=f"Training Loss: {best_loss:.4f}")
for epoch in bar:
    train_loss = train(model_q, model_k, train_loader, optimizer)
    scheduler.step()
    train_losses.append(train_loss)
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model_q.state_dict(), pretrain_save_path)
    else:
        count += 1
    bar.set_description(f"Training Loss: {train_loss:.4f}")
    if count > early_stop:
        break

plt.clf()
plt.plot(train_loss)
plt.savefig(pretrain_image_save_path)

class Bengali_Datamodule_Downstream(nn.Module):
    def __init__(self,dataset = train_dataset, ptsz = 32):
        super().__init__()
        self.dataset = dataset
        self.ptsz = ptsz
    
    def __len__(self):
        return len(self.dataset)

    def __white_pad__(self,img):

        if img.shape[0]<64 and img.shape[1]<128:

            deff_height = 64-img.shape[0]
            deff_height = deff_height//2
            deff_width = 128-img.shape[1]
            deff_width = deff_width//2
            if (img.shape[0]%2) == 0:
                deff_height = deff_height
            else:
                deff_height = deff_height+1
            
            if (img.shape[1]%2 == 0):
                deff_width = deff_width
            else:
                deff_width = deff_width+1
            result = cv2.copyMakeBorder(img,deff_height,deff_height,deff_width,deff_width,cv2.BORDER_CONSTANT,None,value=(255,255,255))

        elif img.shape[0]<=64 and img.shape[1]>=128:
            deff_height = 64-img.shape[0]
            deff_height = deff_height//2
            if (img.shape[0] % 2) == 0:
                deff_height=deff_height
            else:
                deff_height = deff_height+1
            result = cv2.copyMakeBorder(img,deff_height,deff_height,0,0,cv2.BORDER_CONSTANT,None,value=(255,255,255))

        elif img.shape[0]>=64 and img.shape[1]<=128:
            deff_width = 128-img.shape[1]
            deff_width = deff_width//2
            if (img.shape[1] % 2) == 0:
                deff_width=deff_width
            else:
                deff_width = deff_width+1

            result = cv2.copyMakeBorder(img,0,0,deff_width,deff_width,cv2.BORDER_CONSTANT,None,value=(255,255,255))

        else:
            result = img

        return result

    def __getpatches__(self, x):
        pts = []

        numdelH = 64//(self.ptsz)
        numdelW = 128//(self.ptsz)

        for i in range(numdelH):
            for j in range(numdelW):
                sx = i*self.ptsz#//2) 
                ex = sx + self.ptsz
                sy = j*self.ptsz#//2)
                ey = sy + self.ptsz
                temp = x[:,sx:ex,sy:ey]     
                pts.append(torch.unsqueeze(temp, 0))

        return torch.cat(pts, dim = 0)

    
    def __getitem__(self, idx):
        data_object = self.dataset[idx]
        PIL_img = data_object[0]
        pic = cv2.cvtColor(np.array(PIL_img), cv2.COLOR_RGB2BGR)
        pic = self.__white_pad__(pic)
        pic = pic/255.0
        pic = trans(pic)
        pic = rand(pic)
        pic = pic.float()
        pic = (-0.5 + pic)/0.5

        writer = torch.tensor(data_object[1])
        picpts = self.__getpatches__(pic)
        
        return picpts, writer

class DSModel(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(DSModel, self).__init__()

        # encoder
        self.f = Model(64, 128)
        # classifier
        self.fc1 = nn.Linear(512, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x, _ = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc1(feature)
        return out

train_data = Bengali_Datamodule_Downstream(dataset=train_dataset)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                        drop_last=True)

test_data = Bengali_Datamodule_Downstream(dataset=test_dataset)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True,
                        drop_last=True)

model_ds = DSModel(len(train_data), pretrain_save_path).cuda()

loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ds.parameters(), lr=1e-5, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, 1000)

def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0.0, 0
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100

epochs = 300
best_acc = 0.
count = 0
early_stop = 200

results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

bar = tqdm(range(epochs), desc=f"Test Accuracy: {best_acc:.4f}")

for epoch in bar:
    train_loss, train_acc_1, train_acc_5 = train_val(model_ds, train_loader, optimizer)
    scheduler.step()
    results['train_loss'].append(train_loss)
    results['train_acc@1'].append(train_acc_1)
    results['train_acc@5'].append(train_acc_5)
    test_loss, test_acc_1, test_acc_5 = train_val(model_ds, test_loader, None)
    results['test_loss'].append(test_loss)
    results['test_acc@1'].append(test_acc_1)
    results['test_acc@5'].append(test_acc_5)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv(results_df_save_path, index_label='epoch')
    if test_acc_1 > best_acc:
        best_acc = test_acc_1
        torch.save(model_ds.state_dict(), finetune_save_path)
    else:
        count += 1
    bar.set_description(f"Training Loss: {train_loss:.4f}")
    if count > early_stop:
        break

plt.clf()
plt.plot(results['train_acc@1'])
plt.plot(results['test_acc@1'])
plt.savefig(finetune_image_save_path)
