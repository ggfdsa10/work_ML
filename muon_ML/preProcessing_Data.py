import ROOT as root
import numpy as np
import torch

# Muon data 
dataType = -1         # 0=trainning,   1=validation,   2=test,  -1=other file
padHitsCut = 18      # event cut from number of pad hits
padIndex = np.array([8,32])  # pad number of rows and columns
# dataPath = './dataTPC/rootfile/'
dataPath = '~/kebi/at-tpc/output_data/'
savePath = './dataTPC/array/3d/'
saveName = 'v1'

### Normalization parameters
ADCNormal = 4096.0
TimeNormal = 512.0
PosNormal = 100.0
DriftPosNormal = 150.0



arrayName = ''
if(dataType ==0):
    arrayName = 'train'
if(dataType ==1):
    arrayName = 'validation'
if(dataType ==2):
    arrayName = 'test'
if(dataType ==-1):
    arrayName = ''

dataName = ['%sSimPadData' %arrayName, '%sSimTrackData' %arrayName]
input = root.TFile("{}{}.root".format(dataPath, dataName[0]),"read")
inputSol = root.TFile("{}{}.root".format(dataPath, dataName[1]),"read")

tree = input.Get("Pad")
tree1 = inputSol.Get("Track")

TotalEvent = tree.GetEntries()
eventCutNum =0

print("======================================================================")
print("This program is converted Root file to Numpy array for Machine Leaning")
print("Preprocessing for %s data" %arrayName)
print("Total event : ", TotalEvent) 
print("======================================================================")

Pad = torch.FloatTensor() # dim=0 -> event, dim=1 -> ADC, dim=2 -> Row, dim=3 -> Column,
PadTrack = torch.FloatTensor() # dim=0 -> event, dim=1 -> (xfromY0, xfromY100)


# 텐서 전처리1
for entryNum in range (0 ,TotalEvent):
    if(entryNum %100 ==0):
        print("Converting event : ",entryNum)

    tree.GetEntry(entryNum)
    tree1.GetEntry(entryNum)

    hitADC = torch.FloatTensor(getattr(tree, "HitPad"))
    hitADC = hitADC.view(padIndex[0],padIndex[1])
    hitADC = hitADC.unsqueeze(0)

    random = root.TRandom3(0)
    checkPadNum = 0
    for i in range(0,8):
        for j in range(0,32):
            if(hitADC[0][i][j] > 0.1):
                checkPadNum +=1
                hitADC[0][i][j] += random.Gaus(0,17)
                if(hitADC[0][i][j] < 0.):
                    hitADC[0][i][j] = 0.
                
    if(checkPadNum <= padHitsCut):
        print("number of pad hits of this event : ",checkPadNum ,"  So event " ,entryNum ," pass.")
        eventCutNum +=1
        continue


    hitTime = torch.FloatTensor(getattr(tree, "TimePad"))
    hitTime = hitTime.view(padIndex[0],padIndex[1])
    hitTime = hitTime.unsqueeze(0)

    position = torch.FloatTensor(getattr(tree, "PositionPad"))
    position = position.view(padIndex[0],padIndex[1],2)
    position = position.permute(2,0,1)

    hitADC = torch.cat([hitADC/ADCNormal, hitTime/TimeNormal], dim=0)
    hitADC = torch.cat([hitADC, position/PosNormal], dim=0)
    hitADC = hitADC.unsqueeze(0)

    XfromY0 = getattr(tree1, "XfromY0")
    XfromY100 = getattr(tree1, "XfromY100")
    ZfromY0 = getattr(tree1, "ZfromY0")
    ZfromY100 = getattr(tree1, "ZfromY100")

    track = torch.FloatTensor([XfromY0/PosNormal, XfromY100/PosNormal, ZfromY0/DriftPosNormal, ZfromY100/DriftPosNormal]) 
    # track = torch.FloatTensor([XfromY0/PosNormal, XfromY100/PosNormal])
    track = track.unsqueeze(0)


    if(entryNum == 0):
        Pad = hitADC
        PadTrack = track
        
    else:
        Pad = torch.cat([Pad, hitADC], dim=0)
        PadTrack = torch.cat([PadTrack, track], dim=0)


# save numpy data
# np.save('./dataTPC/array/3d/%sPad' %arrayName, Pad)
# np.save('./dataTPC/array/3d/%sPadTrack' %arrayName, PadTrack)

np.save('{}{}{}Pad'.format(savePath, saveName, arrayName), Pad)
np.save('{}{}{}PadTrack'.format(savePath, saveName, arrayName), PadTrack)



print("======================================================================")
print("Pad size : ", np.shape(Pad))
print("Pad Track size : ", np.shape(PadTrack))
print(" ")
print("Total preprocessing event : ", TotalEvent-eventCutNum)
print("%s data preprocessing is done." %arrayName)
print("======================================================================")
