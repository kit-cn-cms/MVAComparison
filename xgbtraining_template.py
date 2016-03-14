#!/usr/bin/python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#
# arguments: signalfile.root bkgfile.root nMax nTestSignal nTestBkg outputtree
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import sys
import numpy as np
from array import array


from ROOT import *#TFile, TTree, TH1F, TCanvas, TColor

import xgboost as xgb

#-------------------------------------------------------------------------------------------------------#
# Functions

def makeROC( ):
    
    histo = TH1F("ROC","ROC",40,0,1)
    
    
    
    return histo

def makeMVAOutput( pred, label, hsig, hbkg ):
    #pred is a numpy 1d Array
    for i,e in enumerate(pred):
        if label[i] == 1:
            hsig.Fill(e)
        elif label[i] == 0:
            hbkg.Fill(e)
        else:
            print "CRASSHHHHHHHHHHHHHHHHHHHH"
            exit()

#
#-------------------------------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------------------------------#
#
maxEvt = int(sys.argv[3])

nTrainSig = int(sys.argv[4])
nTrainBkg = int(sys.argv[5])

substruct = ""

inputjetvars = [#["Jet_Pt",'f'],
                #["Jet_Eta",'f'],
                #["Jet_Phi",'f']]
		]
inputevtvars = [['X','d'],['Y','d']]
#Define cuts on dataset -> [name,=/>/</>=/<=,val,type]
evtcuts = [["Evt_Odd","=",int(1), 'i']]
jetcuts = []
#
#-------------------------------------------------------------------------------------------------------#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Get Data from Tree and save in numpy array
#
# TODO: Somewhere should a shuffling of all events happen
#
signalfile = TFile(sys.argv[1])
signaltree = signalfile.Get("S")

#backgroundfile = TFile(sys.argv[2])
backgroundtree = signalfile.Get("B")

#files = [backgroundfile,signalfile]
trees = [backgroundtree,signaltree]
names = ["backgroundtree","signaltree"]

ouputfile = TFile(sys.argv[6]+".root","RECREATE")

#inputvars = inputjetvars + inputevtvars
inputvars = inputevtvars
cuts = evtcuts + jetcuts
varnames = []
for l in inputvars:
    varnames.append(l[0])

#initialize numpy array
data = [np.array((len(inputvars)+1)*[-99],ndmin = 2),np.array((len(inputvars)+1)*[-99],ndmin = 2)]

for itree,tree in enumerate(trees):
    print "Processing "+names[itree]
    treevals = {}
    cutvals = {}
    evtvals = {}
    #set branch address for training vars and generate dic for the values in the event 
    for ivar in range(len(inputvars)):
	print inputvars[ivar][0]
        treevals.update({inputvars[ivar][0] : array(inputvars[ivar][1],20*[-1])})
        tree.SetBranchAddress(inputvars[ivar][0], treevals[inputvars[ivar][0]])
        evtvals.update({inputvars[ivar][0]: 0})
    #set branch address for cuts --> handling, when cut is inputvalue and cut. Root can only set branch address to one array
    #for icut in range(len(cuts)):
    #    cutvals.update({cuts[icut][0] : array(cuts[icut][3],20*[0])})
    #    tree.SetBranchAddress(cuts[icut][0], cutvals[cuts[icut][0]])
    nEvts = tree.GetEntries()
    
    for iev in range(nEvts):
        if iev == maxEvt:
            break
        if iev%10000 == 0:
            print ('Event: %d' % iev)
        tree.GetEvent(iev)
        cutspassed = False
        for cut in cuts:
            cutspassed = True
            #DO CUT STUFF
        if cutspassed:
            #Set truth for training -> first tree is bkg, second is signal
            classifier = itree
            #Get weight << needs to be implemented
            evt_weight = 1

            #Get Evt values
	    #print tree.X
	    #print tree.Y
	    #evtvals["X"] = tree.X
	    #evtvals["Y"] = tree.Y
            for ivar in range(len(inputevtvars)):
	#	print inputevtvars[ivar][0]
                evtvals[inputevtvars[ivar][0]] = treevals[inputevtvars[ivar][0]][0]
#		print evtvals[inputevtvars[ivar][0]]
#		print treevals[inputevtvars[ivar][0]][0]
	    #raw_input("")
            #Get Jet values
            flag = False
            if substruct == "N_Jets":             #This could be done nice --> Later
                for njet in range(tree.N_Jets):
                    evtdata = [classifier]
                    for ivar in range(len(inputjetvars)):
                        evtvals[inputjetvars[ivar][0]] = treevals[inputjetvars[ivar][0]][njet]
                    for iv in range(len(inputvars)):
                        vname = inputvars[iv][0]
                        evtdata.append(evtvals[vname])
                    data[itree] = np.append(data[itree],[evtdata],axis=0)
                    flag = True
            #if njets == 0: Write evt vars and 0s for jet vars
            if not flag:
                evtdata = [classifier]
                for iv in range(len(inputvars)):
                    vname = inputvars[iv][0]
                    evtdata.append(evtvals[vname])
                data[itree] = np.append(data[itree],[evtdata],axis=0)

    data[itree] = np.delete(data[itree],0,0)
                    


#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
raw_input("~~~~~~~~~~ Start Training ~~~~~~~~~~")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
# Training with xgboost

#Random shuffle the signal and background datasets
sigdata = data[1]
np.random.shuffle(sigdata)
bkgdata = data[0]
np.random.shuffle(bkgdata)

#Get shuffled trainingdataset and testingdataset and split signla/background truth
tmp = np.append(sigdata[0:nTrainSig],bkgdata[0:nTrainBkg],axis=0)
traindata = np.array((len(inputvars))*[-99],ndmin = 2)
trainlabel = np.array([-99])
#Loop over data and write fist element (signal/background) in label array and the remaining elements in data array
for row in tmp:
    #print row[1:], len(row[1:])
    trainlabel = np.append(trainlabel,row[0])
    traindata = np.append(traindata,[row[1:]],axis=0)
trainlabel = np.delete(trainlabel,0,0)
traindata = np.delete(traindata,0,0)
#np.random.shuffle(traindata)
del tmp

tmp = np.append(sigdata[nTrainSig:],bkgdata[nTrainBkg:],axis=0)
testdata = np.array((len(inputvars))*[-99],ndmin = 2)
testlabel = np.array([-99])
#Loop over data and write fist element (signal/background) in label array and the remaining elements in data array
for row in tmp:
    testlabel = np.append(testlabel,row[0])
    testdata = np.append(testdata,[row[1:]],axis=0)
testlabel = np.delete(testlabel,0,0)
testdata = np.delete(testdata,0,0)
#np.random.shuffle(testdata)
del tmp

#s. https://xgboost.readthedocs.org/en/latest/python/python_intro.html#data-interface


dtrain = xgb.DMatrix(traindata, label=trainlabel)
dtest = xgb.DMatrix(testdata, label=testlabel)

param = {'bst:max_depth':4, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param["nthreads"] = 1
param['eval_metric'] = 'auc'

evallist  = [(dtest,'eval'), (dtrain,'train')]

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist )

bst.save_model('0001.model')

# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
#bst.dump_model('dump.raw.txt','featmap.txt')

#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
flag = raw_input("~~~~~~~~~~ Draw Plots ->Press press ret to skip otherwise type 'drawthem!'   ~~~~~~~~~~")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
# Make Plots

if flag == "drawthem!":
    c1 = TCanvas()
    c1.cd()

trainpred = bst.predict(dtrain)
MVAOut_train_sig = TH1F("pred_train_sig","pred_train_sig",40,0,1)
MVAOut_train_bkg = TH1F("pred_train_bkg","pred_train_bkg",40,0,1)
makeMVAOutput(trainpred,trainlabel, MVAOut_train_sig, MVAOut_train_bkg)

if flag == "drawthem!":
    print "Drawing MVAOutput for training sample"
    MVAOut_train_sig.Draw()
    MVAOut_train_bkg.SetLineColor(kRed)
    MVAOut_train_bkg.Draw("same")
    c1.Update()

if flag == "drawthem!":
    raw_input("press ret")

testpred = bst.predict(dtest)
MVAOut_test_sig = TH1F("pred_test_sig","pred_test_sig",40,0,1)
MVAOut_test_bkg = TH1F("pred_test_bkg","pred_test_bkg",40,0,1)
makeMVAOutput(testpred,testlabel, MVAOut_test_sig, MVAOut_test_bkg)

if flag == "drawthem!":
    print "Drawing MVAOutput for test sample"
    MVAOut_test_sig.Draw()
    MVAOut_test_bkg.SetLineColor(kRed)
    MVAOut_test_bkg.Draw("same")
    c1.Update()

MVAOut = [[MVAOut_train_sig,MVAOut_train_bkg], [MVAOut_test_sig,MVAOut_test_bkg]]
preds  = [trainpred,testpred]
labels = [trainlabel,testlabel]


#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
raw_input("~~~~~~~~~~ Write Output to ROOT-File  ~~~~~~~~~~")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
# Generate Ouputtrees

testtree = TTree("testtree","testtree")
traintree = TTree("traintree","traintree")
trees = [testtree,traintree]

#Write training and test datasets in root file
ouputfile.cd()
#bvars = [["signal",'i']] + inputvars +[["BDTOut",'f']]
bvars = inputvars
bvals = []
for var in bvars:
    bvals.append(array('f',[0]))
BDTval = array('f',[0])
IsSig = array('f',[0])
#loop over training and test tree
for idata,data in enumerate([traindata,testdata]):
    #Set branches in output TTree
    for iv,var in enumerate(bvars):
        trees[idata].Branch(var[0],bvals[iv],var[0]+"/f")
    trees[idata].Branch("BDTG",BDTval,"BDTG/f")
    trees[idata].Branch("Is_Signal",IsSig,"Is_Signal/f")
    #Fill TTree
    for ir,row in enumerate(data):
        for i in range(len(row)):
            bvals[i][0] = row[i]
        BDTval[0] = preds[idata][ir]
        IsSig[0] = labels[idata][ir]
        trees[idata].Fill()
    trees[idata].Write()
    MVAOut[idata][0].Write()
    MVAOut[idata][1].Write()
