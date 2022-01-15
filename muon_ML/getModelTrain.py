import ROOT as root
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

##############################################################

#  this macro saved the model with the best hyper parameters

##############################################################

# use the multi GPU 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"

# check the GPU 
print("=====================================")
print("Avliable to GPU : ", torch.cuda.is_available())
print('cuda index:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print("=====================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# option
tuneVersion = "3d_1"
Total_epoch = 3

# the best parameters of tune version
# config = {
#     'epoch': Total_epoch,
#     'drop': 0.4702389525651125,
#     'l2': 2048,
#     'l3': 512,
#     'l4': 128,
#     'lr': 0.009169758796010664,
#     'batchSize': 1024
#     # conv1 : 60, conv2: 150, l1: 5000
#     }

config = {
'epoch': Total_epoch,
'drop': 0.5492848137201868,
'l2': 2048,
'l3': 512,
'l4': 128,
'lr': 0.0027884421029951037,
'batchSize': 512
}


class MuonPadDataset(Dataset): 
  def __init__(self):
    self.x_data = np.load('./dataTPC/array/3d/trainPad.npy')
    self.y_data = np.load('./dataTPC/array/3d/trainPadTrack.npy')

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx]).to(device)
    y = torch.FloatTensor(self.y_data[idx]).to(device)
    return x, y



class MuonPadDataSetTest(Dataset): 
  def __init__(self):
    self.x_data = np.load('./dataTPC/array/3d/validationPad.npy')
    self.y_data = np.load('./dataTPC/array/3d/validationPadTrack.npy')

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx]).to(device)
    y = torch.FloatTensor(self.y_data[idx]).to(device)
    return x, y



class TrackingModel(torch.nn.Module):

    def __init__(self, drop=0.5, l2=5000, l3=2500, l4=500):
        super(TrackingModel, self).__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 120, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(120, 200, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1))

        self.fc1 = torch.nn.Linear(3 * 15 * 200, 5000, bias=True)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        self.layer3 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop))
        
        self.fc2 = torch.nn.Linear(5000, 4, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) 
        out = self.layer3(out)
        out = self.fc2(out)
        # out = self.layer4(out)
        # out = self.layer5(out)
        # out = self.fc4(out)
        return out



def train_func(config):

    DataSetPad = MuonPadDataset()
    DataSetPadTest = MuonPadDataSetTest()
    DataPad = DataLoader(DataSetPad, batch_size=int(config['batchSize']), shuffle=True, drop_last=True)
    DataPadvaild = DataLoader(DataSetPadTest, batch_size=int(config['batchSize']), shuffle=True, drop_last=True)

    _model = TrackingModel(config["drop"], config["l2"], config["l3"], config["l4"]).cuda()
    model = nn.DataParallel(_model).to(device)
    loss_func = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=4,)
    

    valid_loss =0
    for epoch in range(config['epoch']):

        print("------------ epoch : ", epoch ,"---------------")
        # train loop
        trainSize = len(DataPad.dataset)
        for batch, (X, y) in enumerate(DataPad ,0):

            pred = model(X)
            loss = loss_func(pred, y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)

            loss, current = loss.item(), batch * len(X)
            if(batch%100 == 0):
                print(f"train loss: {loss:>7f}  data[{current:>5d}/{trainSize:>5d}]")



        # vaildation loop
        fit = root.TF1("fit","gaus",-2,2)
        fit2 = root.TF1("fit2","gaus",-2,2)

        h1 = root.TH1D("Pad","",1000,-0.5, 0.5)
        h2 = root.TH1D("Drift","",1000,-0.5, 0.5)
        test_loss = 0
        num_batches = len(DataPadvaild)


        with torch.no_grad():
            for X, y in DataPadvaild:

                model.eval()
                pred = model(X)
                test_loss += loss_func(pred, y).item()
                
                for i in range(len(pred)):
                    y0 = pred[i][0]*100 - y[i][0]*100
                    y100 = pred[i][1]*100 - y[i][1]*100
                    z0 = pred[i][2]*150 - y[i][2]*150
                    z100 = pred[i][3]*150 - y[i][3]*150

                    h1.Fill(y0)
                    h1.Fill(y100)
                    h2.Fill(z0)
                    h2.Fill(z100)
        
        h1.Fit(fit)
        h2.Fit(fit2)
        root.gStyle.SetOptFit(11)

        test_loss /= num_batches
        valid_loss = test_loss

        mean = fit.GetParameter(1)
        sigma = fit.GetParameter(2)
        mean2 = fit2.GetParameter(1)
        sigma2 = fit2.GetParameter(2)

        print("-------------     Validation    ------------------")
        print("Error mean: ", mean, "  Error sigma: ", sigma)
        print("hight Error mean: ", mean2, "  hight Error sigma: ", sigma2)
        print(f"Avg loss: {test_loss:>8f} \n")

    torch.save(model.module.state_dict(), "./model/bestModel_v{}.pth".format(tuneVersion))


if __name__ == '__main__':
    train_func(config)
    print("-------------  Model Save done  -------------")


