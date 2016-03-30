
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

ROOT.gDirectory.cd('PyROOT:/')

class Trainer:
    def __init__(self, variables, variables_to_try=[], verbose=False):
        self.best_variables=variables
        self.variables_to_try=variables_to_try
        self.verbose=verbose
	self.pdffile="printout.pdf"
        self.ntrainings=0
        self.verbose=verbose
        self.stopwatch=ROOT.TStopwatch()
        self.weightfile='weights/weights.xml'
	self.trainedweight='weights/weights.xml'
        weightpath='/'.join((self.weightfile.split('/'))[:-1])
        if not os.path.exists( weightpath ):
            os.makedirs(weightpath)
        self.rootfile='outfile/autotrain.root'
        outfilepath='/'.join((self.rootfile.split('/'))[:-1])
        if not os.path.exists( outfilepath ):
            os.makedirs(outfilepath)
	self.PlotFile='TMVA_PlotFile.pdf'
        self.Streename='S'
	self.Btreename='B'
        self.weightexpression='1'
        self.equalnumevents=True
        self.selection=''
        self.factoryoptions="V:!Silent:Color:DrawProgressBar:AnalysisType=Classification:Transformations=I;D;P;G,D"
        self.bdtoptions= "!H:!V:NTrees=1000:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:!UseBaggedBoost:BaggedSampleFraction=0.6:nCuts=20:MaxDepth=2:NegWeightTreatment=IgnoreNegWeightsInTraining"     
        self.setVerbose(verbose)


    def SetPlotFile(self):
        dt=datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
	self.PlotFile='TMVA_PlotFile_'+dt+'.pdf'

    def OpenPDF(self):
	c=ROOT.TCanvas('c','c',800,640)
	c.Print(self.PlotFile+'[')
	c.Close()

    def ClosePDF(self):
	c=ROOT.TCanvas('c','c',800,640)
	c.Print(self.PlotFile+']')
	c.Close()

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
	self.trainedweight=weightfile
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
	      self.testBST([],"")
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
	c.SaveAs(self.PlotFile)
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
	c.SaveAs(self.PlotFile)
	c.Clear()
	c.Update()
	c.Clear()
	ratio_hist.Draw("colz")
	c.Update()
	c.SaveAs("ROC_hist2.pdf)")
	c.SaveAs(self.PlotFile)
	c.Clear()
	  
	#outstr="bestes NTrees=" + str(best_NT) + "beste Shrinkage=" + str(best_Sh) + "beste nCuts=" + str(best_nC)+"\n"+str(nt1)+"      "+str(nt2)+"   "+str(sh1)+"         "+str(sh2)+"   "+str(nc1)+"       "+str(nc2)+"\n"
        #logfile = open("bestlog_roc.txt","a+")
        #logfile.write('######'+str(localtime())+'#####'+"\n"+"\n"+"\n"+str(self.best_variables)+"\n"+"\n"+str(self.bdtoptions)+"\n"+"\n"+outstr+'###############################################\n\n\n\n\n')
        #logfile.close()
	
	#print "bestes NTrees=", best_NT, "beste Shrinkage=", best_Sh,"beste nCuts=", best_nC	
	#return nt1,nt2,sh1,sh2,nc1,nc2
       


	#tests the BDT with a reader
    def testBDT(self,variables_=[],bdtoptions_="",factoryoptions_=""):
	ROOT.gStyle.SetOptStat(0)	#no legends in plots
	c1=ROOT.TCanvas('c1', 'c1', 800, 640)
	#self.PlotSaver.append(ROOT.TCanvas('c'+str(len(self.PlotSaver)),'c'+str(len(self.PlotSaver)), 800, 600))
        if not hasattr(self, 'signal_train') or not hasattr(self, 'signal_test') or not hasattr(self, 'background_train')  or not hasattr(self, 'background_test'):
            print 'set training and test samples first'
            return
	c1.SetRightMargin(0.2)
        newbdtoptions=replaceOptions(bdtoptions_,self.bdtoptions)
        newoptions=replaceOptions(factoryoptions_,self.factoryoptions)
        reader = ROOT.TMVA.Reader(newoptions)
        # add variables	
	varx = array('f',[0])
	vary = array('f',[0])
	localvar = [varx, vary]
        variables=variables_
        if len(variables)==0:
            variables = self.best_variables
        for i in range(len(variables)):
	    #localvar.append(None)
            reader.AddVariable(variables[i],localvar[i])
        # add signal and background trees
        input_test_S = ROOT.TFile( self.signal_test.path )
        input_test_B = ROOT.TFile( self.background_test.path )          
        test_treeS = input_test_S.Get(self.Streename)
        test_treeB = input_test_B.Get(self.Btreename)




        #signalWeight     = 1.
        #backgroundWeight = 1.
        #reader.AddSignalTree    ( test_treeS, signalWeight )
        #reader.AddBackgroundTree( test_treeB, backgroundWeight)
        #reader.SetWeightExpression(self.weightexpression)
        reader.BookMVA( "BDTG", self.trainedweight)

	mvaValue = reader.EvaluateMVA( "BDTG" )
	#print mvaValue
	varx[0] = -1
	vary[0] = 1
	we = reader.EvaluateMVA("BDTG")
	print we
	# create a new 2D histogram with fine binning
	histo2 = ROOT.TH2F("histo2","",200,-5,5,200,-5,5)
	maxout=0
	minout=0
	 
	# loop over the bins of a 2D histogram
	for i in range(1,histo2.GetNbinsX() + 1):
	    for j in range(1,histo2.GetNbinsY() + 1):
         
        	# find the bin center coordinates
	        varx[0] = histo2.GetXaxis().GetBinCenter(i)
	        vary[0] = histo2.GetYaxis().GetBinCenter(j)
	         
	        # calculate the value of the classifier
	        # function at the given coordinate
	        bdtOutput = reader.EvaluateMVA("BDTG")

		#get min max of bdtOutput for bdtOutput-Shape Histo
		if bdtOutput>maxout:
			maxout=bdtOutput
		if bdtOutput<minout:
	         	minout=bdtOutput

	        # set the bin content equal to the classifier output
	        histo2.SetBinContent(i,j,bdtOutput)
	 
	#self.PlotSaver.append(ROOT.TCanvas())
	histo2.SetTitle("BDT prediction")
	histo2.Draw("colz")
	 
	# draw sigma contours around means
#	for mean, color in (
#	    ((0,0), ROOT.kRed), # signal
#	    ((1,1), ROOT.kBlue), # background
#	    ):
	     
	    # draw contours at 1 and 2 sigmas
#	    for numSigmas in (1,2):
#	        circle = ROOT.TEllipse(mean[0], mean[1], numSigmas)
#	        circle.SetFillStyle(0)
#	        circle.SetLineColor(color)
#	        circle.SetLineWidth(2)
#	        circle.Draw()
#	        self.PlotSaver.append(circle)
	 
#	ROOT.gPad.Modified()
	
	c1.Print(self.PlotFile)

	ROOT.gStyle.SetOptStat(0)
	c1.Clear()

	c2=ROOT.TCanvas('c2', 'c2', 800, 800)
	ROOT.gROOT.SetBatch(True)
	f=ROOT.TFile("2D_test_scat.root")

	S=f.Get("S")
	B=f.Get("B")

	ns=0
	nb=0
	for Row in S:
		ns+=1
	for Row in B:
		nb+=1
	n=ns+nb

	#create Tree with BDT output
	T = ROOT.TTree('T','Tree with BDToutput')
	sx = array( 'f' , [0] )
	sy = array( 'f' , [0] )
	sz = array( 'i' , [0] )
	T.Branch('X', sx, 'X/F')
	T.Branch('Y', sy, 'Y/F')
	T.Branch('Z', sz, 'Z/I')

	#create Histos
	HS = ROOT.TH2F('HS','HS', 100, -5, 5, 100, -5, 5)
	HB = ROOT.TH2F('HB','HB', 100, -5, 5, 100, -5, 5)
	HWS = ROOT.TH2F('HWS','HWS', 100, -5, 5, 100, -5, 5)
	HWB = ROOT.TH2F('HWB','HWB', 100, -5, 5, 100, -5, 5)
	hxs = ROOT.TH1F('hxs','hxs', 25, -3, 4)
	hys = ROOT.TH1F('hys','hys', 25, -3, 4)
	hxb = ROOT.TH1F('hxb','hxb', 25, -3, 4)
	hyb = ROOT.TH1F('hyb','hyb', 25, -3, 4)
	hxws = ROOT.TH1F('hxws','hxws', 25, -3, 4)
	hyws = ROOT.TH1F('hyws','hyws', 25, -3, 4)
	hxwb = ROOT.TH1F('hxwb','hxwb', 25, -3, 4)
	hywb = ROOT.TH1F('hywb','hywb', 25, -3, 4)
	histshapeS = ROOT.TH1F('histshapeS', 'histshapeS', 50 , minout-0.02, maxout+0.02)
	histshapeB = ROOT.TH1F('histshapeB', 'histshapeB', 50 , minout-0.02, maxout+0.02)

	#Evaluate BDT with Testtree
	for Row in S:
		#print S.X
		varx[0]=S.X
		sx[0]=S.X
		vary[0]=S.Y
		sy[0]=S.Y
		z = reader.EvaluateMVA( "BDTG" )
		histshapeS.Fill(z)
		if z<0:
			sz[0]=-2
			HWS.Fill(sx[0],sy[0])
			hxws.Fill(sx[0])
			hyws.Fill(sy[0])
		else:
			sz[0]=1
			HS.Fill(sx[0],sy[0])
			hxs.Fill(sx[0])
			hys.Fill(sy[0])
		#print 'BDT Output=   '+str(sz[0])
		T.Fill()
	for Row in B:
		#print S.X
		varx[0]=B.X
		sx[0]=B.X
		vary[0]=B.Y
		sy[0]=B.Y
		z = reader.EvaluateMVA( "BDTG" )
		histshapeB.Fill(z)
		if z<0:
			sz[0]=-1
			HB.Fill(sx[0],sy[0])
			hxb.Fill(sx[0])
			hyb.Fill(sy[0])
		else:
			sz[0]=-2
			#print sx[0], sy[0]
			HWB.Fill(sx[0],sy[0])
			hxwb.Fill(sx[0])
			hywb.Fill(sy[0])
		T.Fill()
		#print 'BDT Output=   '+str(sz[0])

	#plot (Test-) Tree (Scatterplot)
	####----doesn't work, but Histos HS/HB/HWS/HWB plot the same----####
	#T.SetMarkerStyle(8)
	#T.SetMarkerColor(ROOT.kRed)
	#T.Draw('X:Y', 'Z==-2', 'SCAT')
	#T.SetMarkerColor(ROOT.kYellow+1)
	#T.Draw('X:Y', 'Z==1', 'SCAT SAME')
	#T.SetMarkerColor(ROOT.kBlue)
	#T.Draw('X:Y', 'Z==2', 'SCAT SAME')
	#T.SetMarkerColor(ROOT.kYellow+1)
	#T.Draw('X:Y', 'Z==-1', 'SCAT SAME')
	#c2.SaveAs(self.PlotFile)
	#c2.Clear()

	#Print OutputShape of BDT
	#ROOT.gStyle.SetOptStat('O U')
	#cs = ROOT.TCanvas('cs', 'cs', 800, 640)
	c, t = self.PreparePlot()
	histshapeS.SetLineColor(ROOT.kRed)
	histshapeS.Draw("HIST")
	histshapeB.SetLineColor(ROOT.kBlue)
	histshapeB.Draw('SAME HIST')
	t.Draw("SAME")
	self.FinishPlot(c)
	#cs.Print("BDTShape.pdf")
	ROOT.gStyle.SetOptStat(0)

	#Print Scatterplot of Testevents with correct/wrong classified Signal/Background
	cnew = ROOT.TCanvas('cnew', 'cnew', 800, 640)
	cnew.SetLeftMargin(0.1)
	HWS.SetMarkerColor(ROOT.kRed)
	HWS.SetMarkerStyle(8)
	HWS.SetMarkerSize(0.5)
	HWS.Draw()
	HWB.SetMarkerColor(ROOT.kBlue)
	HWB.SetMarkerStyle(8)
	HWB.SetMarkerSize(0.5)
	HWB.Draw('same')
	HS.SetMarkerColor(ROOT.kYellow+1)
	HS.SetMarkerStyle(8)
	HS.SetMarkerSize(0.5)
	HS.Draw('same')
	HB.SetMarkerColor(ROOT.kCyan)
	HB.SetMarkerStyle(8)
	HB.SetMarkerSize(0.5)
	HB.Draw('same')
	t = ROOT.TLatex()
	t.SetTextFont(43)
	p = cnew.GetPad(0)
	#textsize = t.GetTextSize()
	pad_width  = cnew.XtoPixel(cnew.GetX2())
	pad_height = cnew.YtoPixel(cnew.GetY1())
	print cnew.GetUymax()
	#if (pad_width < pad_height):
	#  charheight = textsize*pad_width
	#else:
        #  charheight = textsize*pad_height
	t.SetTextSize(10)
	p.Update()
	X=p.GetUxmin()
	print X
	Y=p.GetUymax()# - t.GetTextSize()
	print Y
	#t.AddText("#splitline{"+str(self.best_variables)+"}{"+str(self.bdtoptions)+"}")
	t.DrawLatex(X, Y ,"#splitline{"+str(self.best_variables)+"}{"+str(self.bdtoptions)+"}")
	#self.PlotOPTs(cnew)
	cnew.Print(self.PlotFile)
	cnew.Close()

	#print Output (in right order)
	#cs.Print(self.PlotFile)
	#cs.Close()
	
	#Print Histo for var 'X' with correct/wrong classified Signal/Background
	cx = ROOT.TCanvas('cx', 'cx', 800, 640)
	hxs.SetLineColor(ROOT.kYellow + 1)
	hxs.Draw("hist")
	hxb.SetLineColor(ROOT.kCyan)
	hxb.Draw("SAME HIST")
	hxws.SetLineColor(ROOT.kRed)
	hxws.Draw("SAME HIST")
	hxwb.SetLineColor(ROOT.kBlue)
	hxwb.Draw("SAME HIST")
	self.PlotOPTs(cx)
	cx.Update()
	cx.Print(self.PlotFile)
	cx.Close()

	#Print Histo for var 'Y' with correct/wrong classified Signal/Background
	cy = ROOT.TCanvas('cy', 'cy', 800, 640)
	hys.SetLineColor(ROOT.kYellow + 1)
	hys.Draw("hist")
	hyb.SetLineColor(ROOT.kCyan)
	hyb.Draw("SAME HIST")
	hyws.SetLineColor(ROOT.kRed)
	hyws.Draw("SAME HIST")
	hywb.SetLineColor(ROOT.kBlue)
	hywb.Draw("SAME HIST")
	cy.Print(self.PlotFile)
	cy.Close()

    def PrintCanvases(self):

	pdffile=self.pdffile
        dt=datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
        pdf=pdffile.replace('.pdf','_'+dt+'.pdf')

	c=ROOT.TCanvas("c","c",800,600)
	c.Print(pdf+"[")
	for can in self.PlotSaver:
		can.Print(pdf)
		print can
	c.Print(pdf+"]")

    def PrintTrees(self):
	c1 = ROOT.TCanvas('c1','c1',800,600)

	f=ROOT.TFile("corr_train.root")

	S=f.Get("S")
	B=f.Get("B")

	h = ROOT.TH2F('h','h',100, -4, 5, 100, -4, 5)

	S.SetMarkerColor(ROOT.kRed)	
	S.Draw('X:Y',"","SCAT")
	B.SetMarkerColor(ROOT.kBlue)	
	B.Draw('X:Y',"", 'SCAT SAME')

	c1.Print('input_sample_scatter.pdf')

	

    def EvolveBDTs(self):
	
	NTreeslist=range(1,10)
	NTreeslist.append(100)
	#NTreeslist.append(1000)
	#NTreeslist.append(10000)
	for NTree in NTreeslist:
		self.setBDTOption("NTrees="+str(NTree))
		self.trainBDT(self.best_variables)
		self.testBDT(self.best_variables)

    def PlotOPTs(self, c):
	t = ROOT.TLatex()
	t.SetTextFont(43)
	p = c.GetPad(0)
	p.Update()
	pad_width  = p.GetX2()-p.GetX1()
	pad_height = p.GetY2()-p.GetY1()
	t.SetTextSize(10)
	X = p.GetUxmin()-0.09*pad_width 
	Y=c.GetUymax() + 0.03*pad_height
	t.DrawLatex(X,Y,"#splitline{"+str(self.best_variables)+"}{"+str(self.bdtoptions)+"}")


    def PreparePlot(self):
	c = ROOT.TCanvas('c','c',800,800)
	c.SetTopMargin(0.2)
	t = ROOT.TLatex()
	t.SetTextFont(43)
	t.SetTextSize(10)
	X=c.GetUxmin()
	Y=c.GetUymax() - 0.05 #* pad_height
	t.DrawLatex(X,Y,"#splitline{"+str(self.best_variables)+"}{"+str(self.bdtoptions)+"}")
	return c, t

    def FinishPlot(self,c):
	c.Update()
	c.Print(self.PlotFile)
	c.Close()
####def
