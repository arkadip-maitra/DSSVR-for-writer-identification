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

root = "/home/sysadm/Music/arka/handwriting/arabic_dataset/"
pretrain_save_path = "results_arabic/Arabic_SIMCLR_model.pth"
finetune_save_path = "results_arabic/Arabic_SIMCLR_model_finetuned.pth"
results_df_save_path = "results_arabic/Arabic_SIMCLR_model_finetune_results_arabic.csv"
pretrain_image_save_path = "results_arabic/Arabic_SIMCLR_pretrain_loss.png"
finetune_image_save_path = "results_arabic/Arabic_SIMCLR_finetune_loss.png"

batch_size = 32

train_dir = root+"/train"
test_dir = root+"/test"

train_dataset = datasets.ImageFolder(root=train_dir)
test_dataset = datasets.ImageFolder(root=test_dir)

label_encode = LabelEncoder()
trans = transforms.ToTensor()
rand = transforms.RandomCrop((64,128))

class Arabic_Datamodule_Pretext(nn.Module):
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

class SIMCLR(nn.Module):
    def __init__(self, batch_size, temperature = 0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z_i_ = z_i / torch.sqrt(torch.sum(torch.square(z_i),dim = 1, keepdim = True))
        z_j_ = z_j / torch.sqrt(torch.sum(torch.square(z_j),dim = 1, keepdim = True))
        z = torch.cat((z_i_, z_j_), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

train_data = Arabic_Datamodule_Pretext(dataset=train_dataset)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                          drop_last=True)

loss_fn = SIMCLR(batch_size)

model_pr = Model(64, 128, bs=batch_size).cuda()

optimizer = optim.Adam(model_pr.parameters(), lr=1e-4, weight_decay=1e-6)
scheduler = CosineAnnealingLR(optimizer, 1000)

def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num = 0.0, 0
    for pos_1, pos_2 in data_loader:
        pos_1, pos_2 = pos_1.cuda(), pos_2.cuda()
        _, out_1 = net(pos_1)
        _, out_2 = net(pos_2)
        
        loss = loss_fn(out_1, out_2)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

    return total_loss / total_num

# epochs = 5000

# train_losses = []
# best_loss = float("inf")
# count = 0
# early_stop = 300

# bar = tqdm(range(epochs), desc=f"Training Loss: {best_loss:.4f}")
# for epoch in bar:
#     train_loss = train(model_pr, train_loader, optimizer)
#     scheduler.step()
#     train_losses.append(train_loss)
#     if train_loss < best_loss:
#         best_loss = train_loss
#         torch.save(model_pr.state_dict(), pretrain_save_path)
#     else:
#         count += 1
#     bar.set_description(f"Training Loss: {train_loss:.4f}")
#     if count > early_stop:
#         break


# plt.plot(train_losses)
# plt.savefig(pretrain_image_save_path)

class Arabic_Datamodule_Downstream(nn.Module):
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
        self.f = Model(64, 128, bs=batch_size)
        # classifier
        self.fc1 = nn.Linear(512, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x, _ = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc1(feature)
        return out

train_data = Arabic_Datamodule_Downstream(dataset=train_dataset)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                        drop_last=True)

test_data = Arabic_Datamodule_Downstream(dataset=test_dataset)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True,
                        drop_last=True)

model_ds = DSModel(82, pretrain_save_path).cuda()

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

epochs = 10000
best_acc = 0.
count = 0
early_stop = 5000

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
    data_frame = pd.DataFrame(data=results)
    data_frame.to_csv(results_df_save_path, index_label='epoch')
    if test_acc_1 > best_acc:
        best_acc = test_acc_1
        torch.save(model_ds.state_dict(), finetune_save_path)
    else:
        count += 1
    bar.set_description(f"Training Loss: {train_loss:.4f}")
    if count > early_stop:
        break

plt.close()
plt.plot(results['train_acc@1'])
plt.plot(results['test_acc@1'])
plt.savefig(finetune_image_save_path)
