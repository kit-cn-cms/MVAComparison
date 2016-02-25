
#import sklearn
import ROOT
import math
import sys
import os
import datetime
from array import array
from subprocess import call
sys.path.insert(0, '../pyroot-plotscripts')
from plotutils import *
from mvautils import *
from time import *

class Trainer:
    def __init__(self, variables, variables_to_try=[], verbose=False):
        self.best_variables=variables
        self.variables_to_try=variables_to_try
        self.verbose=verbose

        self.ntrainings=0
        self.verbose=verbose
        self.stopwatch=ROOT.TStopwatch()
        self.weightfile='weights/weights.xml'
        weightpath='/'.join((self.weightfile.split('/'))[:-1])
        if not os.path.exists( weightpath ):
            os.makedirs(weightpath)
        self.rootfile='outfile/autotrain.root'
        outfilepath='/'.join((self.rootfile.split('/'))[:-1])
        if not os.path.exists( outfilepath ):
            os.makedirs(outfilepath)

        self.Streename='S'
	self.Btreename='B'
        self.weightexpression='1'
        self.equalnumevents=True
        self.selection=''
        self.factoryoptions="V:!Silent:Color:DrawProgressBar:AnalysisType=Classification:Transformations=I;D;P;G,D"
        self.bdtoptions= "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:!UseBaggedBoost:BaggedSampleFraction=0.6:nCuts=20:MaxDepth=2:NegWeightTreatment=IgnoreNegWeightsInTraining"     
        self.setVerbose(verbose)

    def setVerbose(self,v=True):
        self.verbose=v
        if self.verbose:
            self.setFactoryOption('!Silent')
        else:
            self.setFactoryOption('Silent')
           
    def addSamples(self, signal_train,background_train,signal_test,background_test):
        self.signal_train=signal_train
        self.signal_test=signal_test
        self.background_train=background_train
        self.background_test=background_test
        
    def setSelection(self, selection):
        self.selection=selection

    def setFactoryOption(self, option):
        self.factoryoptions=replaceOption(option,self.factoryoptions)

    def setBDTOption(self, option):
        self.bdtoptions=replaceOption(option,self.bdtoptions)

    def setEqualNumEvents(self, b=True):
        self.equalnumevents=b

    def setWeightExpression(self, exp):
        self.weightexpression=exp

    def setTreeName(self, treename):
        self.treename=treename

    def setReasonableDefaults(self):
        self.setBDTOption('MaxDepth=2')
        self.setBDTOption('nCuts=60')
        self.setBDTOption('Shrinkage=0.02')
        self.setBDTOption('NTrees=1000')
        self.setBDTOption('NegWeightTreatment=IgnoreNegWeightsInTraining')
        self.setBDTOption('UseBaggedBoost')
        self.equalnumevents=True

    def useTransformations(self, b=True):
        # transformation make the training slower
        if b:
            self.setFactoryOption('Transformations=I;D;P;G,D')
        else:
            self.setFactoryOption('Transformations=I')

    def showGui(self):
        ROOT.gROOT.SetMacroPath( "./tmvascripts" )
        ROOT.gROOT.Macro       ( "./TMVAlogon.C" )    
        ROOT.gROOT.LoadMacro   ( "./TMVAGui.C" )

    def printVars(self):
        print self.best_variables


    # trains a without changing the defaults of the trainer
    def trainBDT(self,variables_=[],bdtoptions_="",factoryoptions_=""):
        if not hasattr(self, 'signal_train') or not hasattr(self, 'signal_test') or not hasattr(self, 'background_train')  or not hasattr(self, 'background_test'):
            print 'set training and test samples first'
            return
        fout = ROOT.TFile(self.rootfile,"RECREATE")
        # use given options and trainer defaults if an options is not specified
        newbdtoptions=replaceOptions(bdtoptions_,self.bdtoptions)
        newfactoryoptions=replaceOptions(factoryoptions_,self.factoryoptions)
        factory = ROOT.TMVA.Factory("TMVAClassification",fout,newfactoryoptions)
        # add variables
        variables=variables_
        if len(variables)==0:
            variables = self.best_variables
        for var in variables:
            factory.AddVariable(var)
        # add signal and background trees
        inputS = ROOT.TFile( self.signal_train.path )
        inputB = ROOT.TFile( self.background_train.path )          
        treeS = inputS.Get(self.Streename)
        treeB = inputB.Get(self.Btreename)

        #inputS_test = ROOT.TFile( self.signal_test.path )
        #inputB_test = ROOT.TFile( self.background_test.path )          
        #treeS_test     = inputS_test.Get(self.Streename)
        #treeB_test = inputB_test.Get(self.Btreename)

        # use equal weights for signal and bkg
        signalWeight     = 1.
        backgroundWeight = 1.
        factory.AddSignalTree    ( treeS, signalWeight )
        factory.AddBackgroundTree( treeB, backgroundWeight)
        #factory.AddSignalTree    ( treeS_test, signalWeight,ROOT.TMVA.Types.kTesting )
        #factory.AddBackgroundTree( treeB_test, backgroundWeight,ROOT.TMVA.Types.kTesting)
        factory.SetWeightExpression(self.weightexpression)
        # make cuts
        mycuts = ROOT.TCut(self.selection)
        mycutb = ROOT.TCut(self.selection)
        # train and test all methods
        normmode="NormMode=NumEvents:"
        if self.equalnumevents:
            normmode="NormMode=EqualNumEvents:"
        factory.PrepareTrainingAndTestTree( mycuts, mycutb,
                                            "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:!V:"+normmode )
        #norm modes: NumEvents, EqualNumEvents
        factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDTG",newbdtoptions )
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()
        fout.Close()
        weightfile=self.weightfile
        dt=datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
        weightfile=weightfile.replace('.xml','_'+dt+'.xml')
        call(['cp','weights/TMVAClassification_BDTG.weights.xml',weightfile])
        movedfile=self.rootfile
        movedfile=movedfile.replace('.root','_'+dt+'.root')
        #call(['cp',self.rootfile,movedfile])

    def evaluateLastTraining(self):
        f = ROOT.TFile(self.rootfile)
    
        histoS = f.FindObjectAny('MVA_BDTG_S')
        histoB = f.FindObjectAny('MVA_BDTG_B')
        histoTrainS = f.FindObjectAny('MVA_BDTG_Train_S')
        histoTrainB = f.FindObjectAny('MVA_BDTG_Train_B')
        histo_rejBvsS = f.FindObjectAny('MVA_BDTG_rejBvsS')
        histo_effBvsS = f.FindObjectAny('MVA_BDTG_effBvsS')
        histo_effS = f.FindObjectAny('MVA_BDTG_effS')
        histo_effB = f.FindObjectAny('MVA_BDTG_effB')
        histo_trainingRejBvsS = f.FindObjectAny('MVA_BDTG_trainingRejBvsS')    

        rocintegral=histo_rejBvsS.Integral()/histo_rejBvsS.GetNbinsX()
        rocintegral_training=histo_trainingRejBvsS.Integral()/histo_trainingRejBvsS.GetNbinsX()
        bkgRej50=histo_rejBvsS.GetBinContent(histo_rejBvsS.FindBin(0.5))
        bkgRej50_training=histo_trainingRejBvsS.GetBinContent(histo_trainingRejBvsS.FindBin(0.5))
        ksS=histoTrainS.KolmogorovTest(histoS)
        ksB=histoTrainB.KolmogorovTest(histoB)
        
        ##implement better

        c1=ROOT.TCanvas("c1","c1",800,600)
        histoB.SetLineColor(ROOT.kBlue)
        histoTrainB.SetLineColor(ROOT.kRed)

        histoB.Draw("histo E")
        histoTrainB.Draw("SAME histo E")


        c1.SaveAs("Signal_07.pdf")
        
        outstr='ROC='+str(rocintegral)+'   ROC_tr='+str(rocintegral_training)+'   ksS='+str(ksS)+'   ksB'+str(ksB)+"\n"
        logfile = open("gaussopt_log.txt","a+")
        logfile.write('######'+str(localtime())+'#####'+"\n"+"\n"+"\n"+str(self.best_variables)+"\n"+"\n"+str(self.bdtoptions)+"\n"+"\n"+outstr+'###############################################\n\n\n\n\n')
        logfile.close()


        return rocintegral, ksS, ksB, rocintegral_training
    
    def drawBDT(self):
        f = ROOT.TFile(self.rootfile)

        histoS = f.FindObjectAny('MVA_BDTG_S')
        histoB = f.FindObjectAny('MVA_BDTG_B')
        histoTrainS = f.FindObjectAny('MVA_BDTG_Train_S')
        histoTrainB = f.FindObjectAny('MVA_BDTG_Train_B')
        
        histoS.SetLineColor(self.signal_test.color)
        histoS.Draw('histo')
        histoB.SetLineColor(self.background_test.color)
        histoB.Draw('samehisto')
        histoTrainS.SetLineColor(self.signal_train.color)
        histoTrainS.Draw('same')
        histoTrainB.SetLineColor(self.background_train.color)
        histoTrainB.Draw('same')

    def removeWorstUntil(self,length):
        if(len(self.best_variables)<=length):
            return 
        else:
            print "####### findig variable to remove, nvars is "+str(len(self.best_variables))+", removing until nvars is "+str(length)+"."
            bestscore=-1.
            bestvars=[]
            worstvar=""
            for i in range(len(self.best_variables)):
                # sublist excluding variables i
                sublist=self.best_variables[:i]+self.best_variables[i+1:]
                print 'training BDT without',self.best_variables[i]
                self.trainBDT(sublist)
                score=self.evaluateLastTraining()
                print 'score',score
                if score>bestscore:
                    bestscore=score
                    bestvars=sublist
                    worstvar=self.best_variables[i]
            print "####### removing ",
            print worstvar
            self.variables_to_try.append(worstvar)
            self.best_variables=bestvars
            self.removeWorstUntil(length)

    def addBestUntil(self,length):
        if(len(self.best_variables)>=length):
            return
        elif len(self.variables_to_try)==0:
            return        
        else:
            print "####### findig variable to add, nvars is "+str(len(self.best_variables))+", adding until nvars is "+str(length)+"."
            bestscore=-1.
            bestvar=""
            for var in self.variables_to_try:
                newlist=self.best_variables+[var]
                print 'training BDT with',var
                self.trainBDT(newlist)
                score=self.evaluateLastTraining()
                print 'score:',score
                if score>bestscore:
                    bestscore=score
                    bestvar=var
            print "####### adding ",
            print bestvar
            self.variables_to_try.remove(bestvar)
            self.best_variables=self.best_variables+[bestvar]
            self.addBestUntil(length)
        

    def optimizeOption(self,option,factorlist=[0.3,0.5,0.7,1.,1.5,2.,3.]):
        currentvalue=float(getValueOf(option,self.bdtoptions))
        print "####### optimizing "+option+", starting value",currentvalue
        valuelist=[x * currentvalue for x in factorlist] 
        print "####### trying values ",
        print valuelist
        best=valuelist[0]
        bestscore=-1
        for n in valuelist:
            theoption=option+'='+str(n)
            print 'training BDT with',theoption
            self.trainBDT([],theoption)
            score=self.evaluateLastTraining()
            print 'score:',score
            if score>bestscore:
                bestscore=score
                best=n
        print "####### optiminal value is", best
        print "####### yielding scroe ", bestscore
        self.setBDTOption(option+'='+str(best))
        if best==valuelist[-1] and len(valuelist)>2:
            print "####### optiminal value is highest value, optimizing again"
            highfactorlist=[f for f in factorlist if f > factorlist[-2]/factorlist[-1]]
            self.optimizeOption(option,highfactorlist)
        if best==valuelist[0]and len(valuelist)>2:
            print "####### optiminal value is lowest value, optimizing again"
            lowfactorlist=[f for f in factorlist if f < factorlist[1]/factorlist[0]]            
            self.optimizeOption(option,lowfactorlist)




    def suche(self,NTrees_min, NTrees_max, Shrin_min, Shrin_max, nCuts_min, nCuts_max, Schritte):


	self.setBDTOption("NTrees="+str(NTrees_min))
	self.setBDTOption("Shrinkage="+str(Shrin_min))
	self.setBDTOption("nCuts="+str(nCuts_min))
	test_max=0
	best_NT=0
	best_Sh=0
	best_nC=0
	ntrees=range(0,Schritte)
	shrin=range(0,Schritte)
	ncuts=range(0,Schritte)
	ijk=[0,0,0]
	
	mystyle=ROOT.gStyle.SetOptStat(0)
	
	roc_hist=ROOT.TH2F("roc_hist","roc_hist",Schritte,Shrin_min,Shrin_max,Schritte,nCuts_min,nCuts_max)
	roc_hist.SetXTitle("Shrinkage")
	roc_hist.SetYTitle("nCuts")
	#roc_hist.SetLineColor("kblue")
	roct_hist=ROOT.TH2F("roct_hist","roct_hist",Schritte,Shrin_min,Shrin_max,Schritte,nCuts_min,nCuts_max)
	roct_hist.SetXTitle("Shrinkage")
	roct_hist.SetYTitle("nCuts")
	ratio_hist=ROOT.TH2F("ratio_hist","ROC/ROCT",Schritte,Shrin_min,Shrin_max,Schritte,nCuts_min,nCuts_max)
	ratio_hist.SetXTitle("Shrinkage")
	ratio_hist.SetYTitle("nCuts")
	#roct_hist.SetLineColor("kred")
	c=ROOT.TCanvas("c","c",800,600)
	c.SetRightMargin(0.15)


#for k in range(0,Schritte):
  #ntrees[k]=NTrees_min+k*int((NTrees_max-NTrees_min)/Schritte)
  #shrin[k]=Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)
  #ncuts[k]=nCuts_min+k*int((nCuts_max-nCuts_min)/Schritte)

	for i in range(0,Schritte):
	  ntrees[i]=NTrees_min+i*int((NTrees_max-NTrees_min)/Schritte)
	  self.setBDTOption("NTrees="+str(NTrees_min+i*int((NTrees_max-NTrees_min)/Schritte)))
	  #self.trainBDT([],"")
	  #ROC, ksS, ksB, ROCT = self.evaluateLastTraining()
	  #test_tmp=10*ROC+min(ksS,ksB)
	  #if test_tmp>test_max:
	    #test_max=test_tmp
	    #best_NT=ntrees[i]
	    #ijk[0]=i
    
	  for k in range(0,Schritte):
	    shrin[k]=Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)
	    self.setBDTOption("Shrinkage="+str(Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)))
	    k1=Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)
	    #self.trainBDT([],"")
	    #ROC, ksS, ksB, ROCT = self.evaluateLastTraining()
	    #test_tmp=10*ROC+min(ksS,ksB)
	    #if test_tmp>test_max:
	      #test_max=test_tmp
	      #best_Sh=shrin[k]
	      #ijk[1]=k
     
	    for j in range(0,Schritte):
	      ncuts[j]=nCuts_min+j*int((nCuts_max-nCuts_min)/Schritte)
	      self.setBDTOption("nCuts="+str(nCuts_min+j*int((nCuts_max-nCuts_min)/Schritte)))
	      j1=nCuts_min+j*int((nCuts_max-nCuts_min)/Schritte)
	      self.trainBDT([],"")
	      ROC, ksS, ksB, ROCT = self.evaluateLastTraining()
	      roc_hist.SetBinContent(k+1,i+1,ROC)
	      roct_hist.SetBinContent(k+1,i+1,ROCT)
	      ratio_hist.SetBinContent(k+1,i+1,(ROC/ROCT))
	      test_tmp=10*ROC+min(ksS,ksB)
	      if test_tmp>test_max:
		test_max=test_tmp
		best_nC=ncuts[j]
		ijk[2]=j
	
	
	#if ijk[0]==NTrees_min:
	  #nt1=ntrees[0]
	#else:
	  #nt1=ntrees[ijk[0]-1]
	#if ijk[1]==Shrin_min:
	  #sh1=shrin[0]
	#else:
	  #sh1=shrin[ijk[1]-1]
	#if ijk[2]==nCuts_min:
	  #nc1=ncuts[0]
	#else:
	  #nc1=ncuts[ijk[2]-1]
  
	#if ijk[0]==NTrees_max:
	  #nt2=ntrees[Schritte]
	#else:
	  #nt2=ntrees[ijk[0]+1]
	#if ijk[1]==Shrin_max:
	  #sh2=shrin[Schritte]
	#else:
	  #sh2=shrin[ijk[1]+1]
	#if ijk[2]==nCuts_max:
	  #nc2=ncuts[Schritte]
	#else:
	  #nc2=ncuts[ijk[2]+1]
	  
	roc_hist.Draw("colz")
	c.Update()
	c.SaveAs("ROC_hist2.pdf(")
	c.Clear()
	roct_hist.Draw("colz")
	c.Update()
	c.SaveAs("ROC_hist2.pdf")
	c.Clear()
	diff_hist=roc_hist.Clone()
	diff_hist.Add(roct_hist,-1)
	diff_hist.SetTitle("Differenz ROC-ROCT")
	#c.SetRightMargin(0.2)
	c.Update()
	diff_hist.Draw("col")
	c.Update()
	c.Clear()
	diff_hist.Draw("colz")
	c.Update()
	c.SaveAs("ROC_hist2.pdf")
	c.Clear()
	c.Update()
	c.Clear()
	ratio_hist.Draw("colz")
	c.Update()
	c.SaveAs("ROC_hist2.pdf)")
	c.Clear()
	  
	#outstr="bestes NTrees=" + str(best_NT) + "beste Shrinkage=" + str(best_Sh) + "beste nCuts=" + str(best_nC)+"\n"+str(nt1)+"      "+str(nt2)+"   "+str(sh1)+"         "+str(sh2)+"   "+str(nc1)+"       "+str(nc2)+"\n"
        #logfile = open("bestlog_roc.txt","a+")
        #logfile.write('######'+str(localtime())+'#####'+"\n"+"\n"+"\n"+str(self.best_variables)+"\n"+"\n"+str(self.bdtoptions)+"\n"+"\n"+outstr+'###############################################\n\n\n\n\n')
        #logfile.close()
	
	#print "bestes NTrees=", best_NT, "beste Shrinkage=", best_Sh,"beste nCuts=", best_nC	
	#return nt1,nt2,sh1,sh2,nc1,nc2
       
