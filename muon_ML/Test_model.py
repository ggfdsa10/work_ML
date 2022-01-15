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


# use the multi GPU 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

# check the GPU 
print("=====================================")
print("Avliable to GPU : ", torch.cuda.is_available())
print('cuda index:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print("=====================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tuneVersion = "3d_layer2"
Total_epoch = 1
dataPath = './dataTPC/array/3d/test'

config = {
    # 'epoch': Total_epoch,
    # 'drop': 0.5492848137201868,
    # 'l2': 2048,
    # 'l3': 512,
    # 'l4': 128,
    # 'lr': 0.0027884421029951037,
    # 'batchSize': 512

    'epoch': Total_epoch ,
    'conv1': 100,
    'conv2': 80,
    'drop1': 0.5359439814869938,
    'drop2': 0.4868094676872776,
    'l1': 5000,
    'l2': 200,
    'lr': 0.00018474486938752717,
    # 'lr': 0.001,
    'batchSize': 64

}

class MuonPadDataSetTest(Dataset): 
  def __init__(self):
    self.x_data = np.load('{}Pad.npy'.format(dataPath))
    self.y_data = np.load('{}PadTrack.npy'.format(dataPath))

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx]).to(device)
    y = torch.FloatTensor(self.y_data[idx]).to(device)
    return x, y


DataSetPadTest = MuonPadDataSetTest()
DataPadTest = DataLoader(DataSetPadTest, batch_size=config['batchSize'], shuffle=True,drop_last=True)


# class TrackingModel(torch.nn.Module):


    # def __init__(self, drop=0.5, l2=5000, l3=2500, l4=500):
    #     super(TrackingModel, self).__init__()
        
    #     self.layer1 = torch.nn.Sequential(
    #         torch.nn.Conv2d(4, 120, kernel_size=3, stride=1, padding=1),
    #         torch.nn.ReLU(),
    #         torch.nn.MaxPool2d(kernel_size=2, stride=2))


    #     self.layer2 = torch.nn.Sequential(
    #         torch.nn.Conv2d(120, 200, kernel_size=3, stride=1, padding=1),
    #         torch.nn.ReLU(),
    #         torch.nn.MaxPool2d(kernel_size=2, stride=1))

    #     self.fc1 = torch.nn.Linear(3 * 15 * 200, 5000, bias=True)

    #     torch.nn.init.xavier_uniform_(self.fc1.weight)
        
    #     self.layer3 = torch.nn.Sequential(
    #         self.fc1,
    #         torch.nn.ReLU(),
    #         torch.nn.Dropout(p=drop))
        
    #     self.fc2 = torch.nn.Linear(5000, 4, bias=True)
    #     torch.nn.init.xavier_uniform_(self.fc2.weight)
        

    # def forward(self, x):
    #     out = self.layer1(x)
    #     out = self.layer2(out)
    #     out = out.view(out.size(0), -1) 
    #     out = self.layer3(out)
    #     out = self.fc2(out)
    #     return out






class TrackingModel(torch.nn.Module):
    def __init__(self, conv1, conv2, drop1, drop2, l1, l2):
        super(TrackingModel, self).__init__()
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, conv1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(conv1, conv2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=1))

        self.fc1 = torch.nn.Linear(3 * 15 * conv2, l1, bias=True)   # 3 * 15 = donated kernel and srtide of Maxpooling 
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        self.layer3 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop1))
        
        self.fc2 = torch.nn.Linear(l1, l2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        self.layer4 = torch.nn.Sequential(
            self.fc2,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=drop2))
        
        self.fc3 = torch.nn.Linear(l2, 4, bias=True)
        torch.nn.init.xavier_uniform_(self.fc3.weight)



    def forward(self, x): 
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) 
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc3(out)
        return out 


model = TrackingModel(config['conv1'], config['conv2'], config["drop1"], config["drop2"], config['l1'], config['l2'])
model.load_state_dict(torch.load('./model/bestModel_v{}.pth'.format(tuneVersion)))
model.to(device)
model.eval()
loss_func = torch.nn.MSELoss().to(device)

h0 = root.TH1D("","",200,-3,3)
h1 = root.TH1D("","",200,-3,3)
h2 = root.TH1D("","",200,-3,3)
h3 = root.TH1D("","",200,-3,3)
h4 = root.TH1D("","",200,-3,3)
h5 = root.TH1D("","",200,-3,3)
h6 = root.TH1D("","",200,-3,3)
h7 = root.TH1D("","",200,-3,3)

angle0 = root.TH1D("","",200,-3,3)
angle1 = root.TH1D("","",200,-3,3)
angle2 = root.TH1D("","",200,-3,3)
angle3 = root.TH1D("","",200,-3,3)
angle4 = root.TH1D("","",200,-3,3)
angle5 = root.TH1D("","",200,-3,3)
angle6 = root.TH1D("","",200,-3,3)
angle7 = root.TH1D("","",200,-3,3)

predictTrack = torch.FloatTensor()
hXY = root.TH1D("","",200,-3,3)
hYZ = root.TH1D("","",200,-3,3)
hAngle = root.TH1D("","",200,-3,3)

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0


    eventNum =0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # print("\ntest_loss : ", loss_fn(pred, y).item())

            for i in range(len(pred)):
                y0 = pred[i][0]*100 - y[i][0]*100
                y100 = pred[i][1]*100 - y[i][1]*100
                z0 = pred[i][2]*150 - y[i][2]*150
                z100 = pred[i][3]*150 - y[i][3]*150

                predict = torch.FloatTensor([pred[i][0].to("cpu")*100, pred[i][1].to("cpu")*100])
                predict = predict.unsqueeze(0)

                if(eventNum==0):
                    predictTrack = predict
                   
                else:
                    predictTrack = torch.cat([predictTrack, predict], dim=0)

                predictAngle = torch.arctan(np.abs(pred[i][0].to("cpu")*100-pred[i][1].to("cpu")*100)/100)*180/3.141592
                realAngle = torch.arctan(np.abs(y[i][0].to("cpu")*100-y[i][1].to("cpu")*100)/100)*180/3.141592

                hXY.Fill(y0)
                hXY.Fill(y100)
                hYZ.Fill(z0)
                hYZ.Fill(z100)
                hAngle.Fill(realAngle - predictAngle)


                if(realAngle >=0.000001 and realAngle <=1.):
                    h0.Fill(y0)
                    h0.Fill(y100)
                    angle0.Fill(realAngle-predictAngle)

                if(realAngle >1. and realAngle <=2.):
                    h1.Fill(y0)
                    h1.Fill(y100)
                    angle1.Fill(realAngle-predictAngle)

                if(realAngle >2. and realAngle <=3.):
                    h2.Fill(y0)
                    h2.Fill(y100)
                    angle2.Fill(realAngle-predictAngle)

                if(realAngle >3. and realAngle <=4.):
                    h3.Fill(y0)
                    h3.Fill(y100)
                    angle3.Fill(realAngle-predictAngle)

                if(realAngle >4. and realAngle <=5.):
                    h4.Fill(y0)
                    h4.Fill(y100)
                    angle4.Fill(realAngle-predictAngle)

                if(realAngle >5. and realAngle <=6.):
                    h5.Fill(y0)
                    h5.Fill(y100)
                    angle5.Fill(realAngle-predictAngle)

                if(realAngle >6. and realAngle <=7.):
                    h6.Fill(y0)
                    h6.Fill(y100)
                    angle6.Fill(realAngle-predictAngle)

                if(realAngle >7. and realAngle <=8.):
                    h7.Fill(y0)
                    h7.Fill(y100)
                    angle7.Fill(realAngle-predictAngle)

                # if(eventNum >=0 and eventNum <10000):
                #     h0.Fill(y0)
                #     h0.Fill(y100)
                #     angle0.Fill(realAngle-predictAngle)

                # if(eventNum >=10000 and eventNum <20000):
                #     h1.Fill(y0)
                #     h1.Fill(y100)
                #     angle1.Fill(realAngle-predictAngle)

                # if(eventNum >=20000 and eventNum <30000):
                #     h2.Fill(y0)
                #     h2.Fill(y100)
                #     angle2.Fill(realAngle-predictAngle)

                # if(eventNum >=30000 and eventNum <40000):
                #     h3.Fill(y0)
                #     h3.Fill(y100)
                #     angle3.Fill(realAngle-predictAngle)

                # if(eventNum >=40000 and eventNum <50000):
                #     h4.Fill(y0)
                #     h4.Fill(y100)
                #     angle4.Fill(realAngle-predictAngle)

                # if(eventNum >=50000 and eventNum <60000):
                #     h5.Fill(y0)
                #     h5.Fill(y100)
                #     angle5.Fill(realAngle-predictAngle)

                # if(eventNum >=60000 and eventNum <70000):
                #     h6.Fill(y0)
                #     h6.Fill(y100)
                #     angle6.Fill(realAngle-predictAngle)

                # if(eventNum >=70000 and eventNum <80000):
                #     h7.Fill(y0)
                #     h7.Fill(y100)
                #     angle7.Fill(realAngle-predictAngle)

                eventNum +=1

    test_loss /= num_batches
    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")
    
    np.save('{}predictTrack'.format(dataPath), predictTrack)


print(f"-------------- Model predicting.... ------------------")
test_loop(DataPadTest, model, loss_func)
print("Done!")


hResol = [0,0,0,0,0,0,0,0]
hResolErr = [0,0,0,0,0,0,0,0]
angleResol = [0,0,0,0,0,0,0,0]
angleResolErr = [0,0,0,0,0,0,0,0]

c1 = root.TCanvas("","",1800,1800)
c1.Divide(3,3)

c1.cd(1)
f0 = root.TF1("","gaus",-3,3)
h0.SetTitle("Predicted difference of track (0<#theta<1);#Delta X [mm];")
h0.Fit(f0)
h0.Draw()
hResol[0] = f0.GetParameter(2)
hResolErr[0] = f0.GetParError(2)
root.gStyle.SetOptFit(11)

c1.cd(2)
f1 = root.TF1("","gaus",-3,3)
h1.SetTitle("Predicted difference of track (1<#theta<2);#Delta X [mm];")
h1.Fit(f1)
h1.Draw()
hResol[1] = f1.GetParameter(2)
hResolErr[1] = f1.GetParError(2)
root.gStyle.SetOptFit(11)

c1.cd(3)
f2 = root.TF1("","gaus",-3,3)
h2.SetTitle("Predicted difference of track (2<#theta<3);#Delta X [mm];")
h2.Fit(f2)
h2.Draw()
hResol[2] = f2.GetParameter(2)
hResolErr[2] = f2.GetParError(2)
root.gStyle.SetOptFit(11)

c1.cd(4)
f3 = root.TF1("","gaus",-3,3)
h3.SetTitle("Predicted difference of track (3<#theta<4);#Delta X [mm];")
h3.Fit(f3)
h3.Draw()
hResol[3] = f3.GetParameter(2)
hResolErr[3] = f3.GetParError(2)
root.gStyle.SetOptFit(11)

c1.cd(5)
f4 = root.TF1("","gaus",-3,3)
h4.SetTitle("Predicted difference of track (4<#theta<5);#Delta X [mm];")
h4.Fit(f4)
h4.Draw()
hResol[4] = f4.GetParameter(2)
hResolErr[4] = f4.GetParError(2)
root.gStyle.SetOptFit(11)

c1.cd(6)
f5 = root.TF1("","gaus",-3,3)
h5.SetTitle("Predicted difference of track (5<#theta<6);#Delta X [mm];")
h5.Fit(f5)
h5.Draw()
hResol[5] = f5.GetParameter(2)
hResolErr[5] = f5.GetParError(2)
root.gStyle.SetOptFit(11)

c1.cd(7)
f6 = root.TF1("","gaus",-3,3)
h6.SetTitle("Predicted difference of track (6<#theta<7);#Delta X [mm];")
h6.Fit(f6)
h6.Draw()
hResol[6] = f6.GetParameter(2)
hResolErr[6] = f6.GetParError(2)
root.gStyle.SetOptFit(11)

c1.cd(8)
f7 = root.TF1("","gaus",-3,3)
h7.SetTitle("Predicted difference of track (7<#theta<8);#Delta X [mm];")
h7.Fit(f7)
h7.Draw()
hResol[7] = f7.GetParameter(2)
hResolErr[7] = f7.GetParError(2)
root.gStyle.SetOptFit(11)

c1.Draw()
# c1.SaveAs("result_Resol_{}.pdf".format(tuneVersion))




c2 = root.TCanvas("","",1800,1800)
c2.Divide(3,3)

c2.cd(1)
fa0 = root.TF1("","gaus",-3,3)
angle0.SetTitle("Predicted difference of angle (0<#theta<1);#Delta #theta [#circ];")
angle0.Fit(fa0)
angle0.Draw()
angleResol[0] = fa0.GetParameter(2)
angleResolErr[0] = fa0.GetParError(2)
root.gStyle.SetOptFit(11)

c2.cd(2)
fa1 = root.TF1("","gaus",-3,3)
angle1.SetTitle("Predicted difference of angle (1<#theta<2);#Delta #theta [#circ];")
angle1.Fit(fa1)
angle1.Draw()
angleResol[1] = fa1.GetParameter(2)
angleResolErr[1] = fa1.GetParError(2)
root.gStyle.SetOptFit(11)

c2.cd(3)
fa2 = root.TF1("","gaus",-3,3)
angle2.SetTitle("Predicted difference of angle (2<#theta<3);#Delta #theta [#circ];")
angle2.Fit(fa2)
angle2.Draw()
angleResol[2] = fa2.GetParameter(2)
angleResolErr[2] = fa2.GetParError(2)
root.gStyle.SetOptFit(11)

c2.cd(4)
fa3 = root.TF1("","gaus",-3,3)
angle3.SetTitle("Predicted difference of angle (3<#theta<4);#Delta #theta [#circ];")
angle3.Fit(fa3)
angle3.Draw()
angleResol[3] = fa3.GetParameter(2)
angleResolErr[3] = fa3.GetParError(2)
root.gStyle.SetOptFit(11)

c2.cd(5)
fa4 = root.TF1("","gaus",-3,3)
angle4.SetTitle("Predicted difference of angle (4<#theta<5);#Delta #theta [#circ];")
angle4.Fit(fa4)
angle4.Draw()
angleResol[4] = fa4.GetParameter(2)
angleResolErr[4] = fa4.GetParError(2)
root.gStyle.SetOptFit(11)

c2.cd(6)
fa5 = root.TF1("","gaus",-3,3)
angle5.SetTitle("Predicted difference of angle (5<#theta<6);#Delta #theta [#circ];")
angle5.Fit(fa5)
angle5.Draw()
angleResol[5] = fa5.GetParameter(2)
angleResolErr[5] = fa5.GetParError(2)
root.gStyle.SetOptFit(11)

c2.cd(7)
fa6 = root.TF1("","gaus",-3,3)
angle6.SetTitle("Predicted difference of angle (6<#theta<7);#Delta #theta [#circ];")
angle6.Fit(fa6)
angle6.Draw()
angleResol[6] = fa6.GetParameter(2)
angleResolErr[6] = fa6.GetParError(2)
root.gStyle.SetOptFit(11)

c2.cd(8)
fa7 = root.TF1("","gaus",-3,3)
angle7.SetTitle("Predicted difference of angle (7<#theta<8);#Delta #theta [#circ];")
angle7.Fit(fa7)
angle7.Draw()
angleResol[7] = fa7.GetParameter(2)
angleResolErr[7] = fa7.GetParError(2)
root.gStyle.SetOptFit(11)

c2.Draw()
# c2.SaveAs("result_Angle_{}.pdf".format(tuneVersion))

c3 = root.TCanvas("","",1800,600)
c3.Divide(3,1)

c3.cd(1)
hXY.SetTitle("Predicted difference of track in X-Y plane;#Delta X [mm];")
hXY.Fit("gaus")
hXY.Draw()
root.gStyle.SetOptFit(11)

c3.cd(2)
hYZ.SetTitle("Predicted difference of track in Y-Z plane;#Delta X [mm];")
hYZ.Fit("gaus")
hYZ.Draw()
root.gStyle.SetOptFit(11)

c3.cd(3)
hAngle.SetTitle("Predicted difference of angle in X-Y plane;#Delta #theta [#circ];")
hAngle.Fit("gaus")
hAngle.Draw()
root.gStyle.SetOptFit(11)

c3.Draw()
# c3.SaveAs("result_Total_{}.pdf".format(tuneVersion))

# f = open("eventDisplay.txt", 'w')
# for i in range(0,8):
#     f.write("{} {} {} {}\n".format(hResol[i], hResolErr[i], angleResol[i], angleResolErr[i]))

# f.close()
