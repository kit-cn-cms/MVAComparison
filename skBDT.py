# code mostly taken from
#http://betatim.github.io/posts/advanced-sklearn-for-TMVA/
#http://betatim.github.io/posts/sklearn-for-TMVA-users/

import sklearn
import numpy as np
from root_numpy import root2array, rec2array
from root_numpy import array2root
import ROOT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from mvautils import *
import math
import sys
import os
import datetime
from time import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#from root_numpy import tree2rec
import pickle
from matplotlib.backends.backend_pdf import PdfPages


def frange(start, stop, step):
	l=array('f',[])
	i=start
	j=start
	trigger=array('f',[])
	trigger=np.arange(start,stop,step)
	print ((stop-start)/step)
	for i in trigger:
		i+=step
		print i
		for j in trigger:
			j+=step
			print str(j) + '\n'
			l = np.concatenate(((l),([i,j])))
		#print l
	a=np.reshape(l,((len(l)/2),2))
	return a


class data:
  def __init__(self, variables):
	self.Streename='MVATree'
	self.Btreename='MVATree'
	self.weightfile='weights/weights.xml'
	self.X_Array=[]
	self.Y_Array=[]
	self.W_Array=[]
	self.test_X=[]
	self.test_y=[]
	self.SPath='/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root'
	self.StestPath='/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root'
	self.BPath='/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root'
	self.BtestPath='/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root'
	self.variables=variables
	self.weights='1.'
	#self.GradBoostOptions="learning_rate=0.1, n_estimators=100, max_depth=3, random_state=0"
	self.learning_rate=0.1
	self.n_estimators=100
	self.max_depth=3
	self.random_state=0
	self.loss='deviance'
	self.subsample=1.0
	self.min_samples_split=2
	self.min_samples_leaf=1
	self.min_weight_fraction_leaf=0.0
	self.init=None
	self.max_features=None
	self.verbose=0
	self.max_leaf_nodes=None
	self.warm_start=False
	self.presort='auto'
#	self.train
	self.train_Background=[]
	self.train_Signal=[]
	self.outname="SKout.root"
#	self.SKout=ROOT.TFile(self.outname,"RECREATE")
	self.sy_tr=[]
	self.ROC_Color=0
	self.ROC_fig = plt.figure()
	self.ROC_Curve=[[],[],[]]
	self.logfilename="log.txt"
	self.listoffigures=[]



  def SetLogfileName(self,NAME=''):
	self.logfilename = NAME

  def SetSPath(self,SPATH=''):
	self.SPath=SPATH

  def SetBPath(self, BPATH=''):
	self.BPath=BPATH

  def SetStestPath(self,SPATH=''):
	self.StestPath=SPATH

  def SetBtestPath(self, BPATH=''):
	self.BtestPath=BPATH

  def SetBTreename(self, TREENAME=''):
	self.Btreename=TREENAME

  def SetSTreename(self, TREENAME=''):
	self.Streename=TREENAME

  def SetWeight(self, WEIGHT=''):
	self.weights=weight

  def Convert(self):
	train_Signal=root2array(self.SPath, self.Streename, self.variables)
	train_Background=root2array(self.BPath, self.Btreename, self.variables)
	#train_Sweight=root2array(self.SPath, self.Streename, self.weights)
	#train_Bweight=root2array(self.BPath, self.Btreename, self.weights)
	train_Signal=rec2array(train_Signal)
	train_Background=rec2array(train_Background)
	#train_Sweight=rec2array(train_Sweight)
	#train_Bweight=rec2array(train_Bweight)
	X_train = np.concatenate((train_Signal, train_Background))
	y_train = np.concatenate((np.ones(train_Signal.shape[0]), np.zeros(train_Background.shape[0])))
	#w_train = np.concatenate((train_Sweight, train_Bweight))
	self.X_Array = X_train
	self.Y_Array = y_train
	#self.W_Array = w_train
	self.train_Background=train_Background
	self.train_Signal=train_Signal

#and testtree
	test_Signal=root2array(self.StestPath, self.Streename, self.variables)
	test_Background=root2array(self.BtestPath, self.Btreename, self.variables)
	test_Signal=rec2array(test_Signal)
	test_Background=rec2array(test_Background)
	self.test_X=np.concatenate((test_Signal,test_Background))
	self.test_y=np.concatenate((np.ones(test_Signal.shape[0]), np.zeros(test_Background.shape[0])))


  def SetGradBoostOption(self, option, value):
	if option=='n_estimators':
		self.n_estimators=value
	elif option=='learning_rate':
		self.learning_rate=value
	elif option=='max_depth':
		self.max_depth=value
	elif option=='random_state':
		self.random_state=value
	elif option=='loss':
		self.loss=value
	elif option=='subsample':
		self.subsample=value
	elif option=='min_samples_split':
		self.min_samples_split=value
	elif option=='min_samples_leaf':
		self.min_samples_leaf=value
	elif option=='min_weight_fraction_leaf':
		self.min_weight_fraction_leaf=value
	elif option=='init':
		self.init=value
	elif option=='max_features':
		self.max_features=value
	elif option=='verbose':
		self.verbose=value
	elif option=='max_leaf_nodes':
		self.max_leaf_nodes=value
	elif option=='warm_start':
		self.warm_start=value
	elif option=='presort':
		self.presort=value
	else:
		print "Keine GradBoostOption ==> Abbruch!"
		sys.exit()

  def SetGradBoostDefault(self):
	self.learning_rate=0.1
	self.n_estimators=100
	self.max_depth=3
	self.random_state=0
	self.loss='deviance'
	self.subsample=1.0
	self.min_samples_split=2
	self.min_samples_leaf=1
	self.min_weight_fraction_leaf=0.0
	self.init=None
	self.max_features=None
	self.verbose=0
	self.max_leaf_nodes=None
	self.warm_start=False
	self.presort='auto'

  def Classify(self):
	train = GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, loss=self.loss, subsample=self.subsample, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf, init=self.init, max_features=self.max_features, verbose=self.verbose, max_leaf_nodes=self.max_leaf_nodes, warm_start=self.warm_start, presort=self.presort).fit(self.X_Array,self.Y_Array)
	self.PrintLog(train)
	self.WriteOutTree(train)
        print "tree written"
	return train

#  def TestDefiance(self,train):
#	test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
#	for i, y_pred in enumerate(train.staged_decision_function(self.X_Array)):
#        # train.loss_ assumes that y_test[i] in {0, 1}
#		test_deviance[i] = train.loss_(Y_Array, y_pred)

  def Score(self,train):
	return train.score(self.X_Array, self.Y_Array)

  def PrintLog(self,train):
	gbo='learning_rate='+str(self.learning_rate)+', n_estimators='+str(self.n_estimators)+', max_depth='+str(self.max_depth)+', random_state='+str(self.random_state)+', loss='+str(self.loss)+', subsample='+str(self.subsample)+', min_samples_split='+str(self.min_samples_split)+', min_samples_leaf='+str(self.min_samples_leaf)+', min_weight_fraction_leaf='+str(self.min_weight_fraction_leaf)+', init='+str(self.init)+', max_features='+str(self.max_features)+', verbose='+str(self.verbose)+', max_leaf_nodes='+str(self.max_leaf_nodes)+', warm_start='+str(self.warm_start)+', presort='+str(self.presort)
	outstr='\n\n-----------------input variables:-----------------\n'+str(self.variables)+'\n\n-----------------weights:-----------------\n'+str(self.weights)+'\n\n-----------------Gradient Boost Options:-----------------\n'+gbo+'\n\n\n\n'+'--------------- ROC integral = '+str(self.ROCInt(train))+' -----------------'
	logfile = open(self.logfilename,"a+")
	logfile.write('######'+str(localtime())+'#####'+outstr+'###############################################\n\n\n\n\n')
	logfile.close()
	print outstr

  def PrintOpts(self):
	gbo='learning_rate='+str(self.learning_rate)+', n_estimators='+str(self.n_estimators)+', max_depth='+str(self.max_depth)+', random_state='+str(self.random_state)+', loss='+str(self.loss)+', subsample='+str(self.subsample)+', min_samples_split='+str(self.min_samples_split)+', min_samples_leaf='+str(self.min_samples_leaf)+', min_weight_fraction_leaf='+str(self.min_weight_fraction_leaf)+', init='+str(self.init)+', max_features='+str(self.max_features)+', verbose='+str(self.verbose)+', max_leaf_nodes='+str(self.max_leaf_nodes)+', warm_start='+str(self.warm_start)+', presort='+str(self.presort)
	outstr='\n\n-----------------input variables:-----------------\n'+str(self.variables)+'\n\n-----------------weights:-----------------\n'+str(self.weights)+'\n\n-----------------Gradient Boost Options:-----------------\n'+gbo+'\n\n\n\n'
	logfile = open(self.logfilename,"a+")
	logfile.write('######'+str(localtime())+'#####'+outstr+'###############################################\n\n\n\n\n')
	logfile.close()
	print outstr

  def RecreateOutRoot(self):
	SKout = ROOT.TFile(self.outname, "RECREATE")
	return SKout

  def WriteOutTree(self,train):#implement TestSamples 
	SKout = self.RecreateOutRoot()
	sy_trained = train.predict(self.train_Signal) #only gives binary prediction
	self.sy_tr=sy_trained
	sX_trained = train.decision_function(self.train_Signal)
	sy_trained.dtype = [('training_output_y', np.float64)]
	sX_trained.dtype = [('training_output', np.float64)]
	array2root(sy_trained, self.outname, "Signal_train", mode="recreate")
	array2root(sX_trained, self.outname, "Signal_train")
#  def WriteBackgroundTree(self)#implement TestSamples 
	by_trained = train.predict(self.train_Background) #only gives binary prediction
	bX_trained = train.decision_function(self.train_Background)
	by_trained.dtype = [('training_output_y', np.float64)]
	bX_trained.dtype = [('training_output', np.float64)]
	array2root(by_trained, self.outname, "Background_train")
	array2root(bX_trained, self.outname, "Background_train")
        SKout.Close()

  def ROCInt(self,train):
	return roc_auc_score(self.Y_Array, train.decision_function(self.X_Array))

  def ROCCurve(self,train):
	decisions = train.decision_function(self.X_Array)
# Compute ROC curve and area under the curve
	fpr, tpr, thresholds = roc_curve(self.Y_Array, decisions)
	#print fpr
	#print "\n\n\n\n\n"
	#print tpr
	fprfile = open("fpr.pkl","w")
	tprfile = open("tpr.pkl","w")
	pickle.dump(fpr,fprfile)
	pickle.dump(tpr,tprfile)
	tprfile.close()	
	fprfile.close()
	self.ROC_Curve[0] = 1-fpr
	self.ROC_Curve[1] = tpr
	self.ROC_Curve[2] = thresholds
	roc_auc = auc(fpr, tpr)
	fig = plt.figure()
	plt.plot((1-fpr), tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

	plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.grid()
	plt.show()
	fig.savefig("bspROC.pdf")


#  def KSTest(self,train):

  def TestGradBoostOptions(self, minestimators, maxestimators, minlearning_rate, maxlearning_rate, steps):
	self.SetLogfileName("BestROClog.txt")
	self.SetGradBoostOption('n_estimators', minestimators)
	self.SetGradBoostOption('learning_rate', minlearning_rate)
	estimators=[]
	rate=[]
	rocint=[]
	scatterplots=[]

	for i in range(steps):
		self.SetGradBoostOption('n_estimators', (minestimators+(i*((maxestimators-minestimators)/steps))))
#		estimators.append((minestimators+(i*((maxestimators-minestimators)/steps))))
		for k in range(steps):
			self.SetGradBoostOption('learning_rate', (minlearning_rate+(k*((maxlearning_rate-minlearning_rate)/steps))))
			estimators.append((minestimators+(i*((maxestimators-minestimators)/steps))))
			rate.append((minlearning_rate+(k*((maxlearning_rate-minlearning_rate)/steps))))
			train=self.Classify()
			#self.PrintOutput()
			rocint.append(self.ROCInt(train))
			#self.PrintOutput(train)
			#print str(rocint)
			scatterplots.append(self.PrintOutput(train))

	with PdfPages('scatterplots.pdf') as pdf:
		for plot in scatterplots:
	#	with PdfPages('multipage.pdf') as pdf:
			pdf.savefig(plot)
			#print "plot"
			#plt.close()

	rocint = np.array(rocint)
	estimators = np.array(estimators)
	rate = np.array(rate)
	print str(estimators) + str(rate) + str(rocint)
	fig = plt.figure()
	plt.hist2d(estimators,rate,bins=steps,weights=rocint)
	plt.colorbar()
	plt.xlabel("n_estimators")
	plt.xticks(estimators,estimators)
	plt.ylabel("learning_rate")
	plt.yticks(rate,rate)
	plt.title("ROC integrals")
	#plt.show()
	fig.savefig("bestROC.pdf")

	index=np.argmax(rocint)
	self.SetLogfileName("BestOptions.txt")
	self.SetGradBoostOption('n_estimators', estimators[index])
	self.SetGradBoostOption('learning_rate', rate[index])
	self.PrintOpts()


  def TestTwoOptions(self, name1, name2, min1, max1, min2, max2, steps):
	self.SetLogfileName("BestROClog_"+str(name1)+"_"+str(name2)+".txt")
	self.SetGradBoostOption(name1, min1)
	self.SetGradBoostOption(name2, min2)
	opt1=[]
	opt2=[]
	rocint=[]

	for i in range(steps):
		self.SetGradBoostOption(name1, int((min1+(i*((max1-min1)/steps)))))
#		estimators.append((minestimators+(i*((maxestimators-minestimators)/steps))))
		for k in range(steps):
			self.SetGradBoostOption(name2, int((min2+(k*((max2-min2)/steps)))))
			opt1.append(int((min1+(i*((max1-min1)/steps)))))
			opt2.append(int((min2+(k*((max2-min2)/steps)))))
			train=self.Classify()
			rocint.append(self.ROCInt(train))
			#print str(rocint)

	rocint = np.array(rocint)
	opt1 = np.array(opt1)
	opt2 = np.array(opt2)
	#print str(estimators) + str(rate) + str(rocint)
	fig = plt.figure()
	plt.hist2d(opt1,opt2,bins=steps,weights=rocint)
	plt.colorbar()
	plt.xlabel(name1)
	plt.xticks(opt1,opt1)
	plt.ylabel(name2)
	plt.yticks(opt2,opt2)
	plt.title("ROC integrals")
	plt.show()
	fig.savefig("bestROC_"+str(name1)+"_"+str(name2)+".pdf")

	index=np.argmax(rocint)
	self.SetLogfileName("BestOptions_"+str(name1)+"_"+str(name2)+".txt")
	self.SetGradBoostOption(name1, opt1[index])
	self.SetGradBoostOption(name2, opt2[index])
	self.PrintOpts()


  def PrintOutput(self,train):
	fig = plt.figure()
	x = self.test_X
	y = train.predict(x)
	plt.scatter(x[:,0], x[:,1], c=y, cmap='autumn', alpha = 1)
	plt.colorbar()
	plt.title('BDT Output for Testtree')
	plt.text(-4.1,4.9,self.ReturnOpts(),verticalalignment='top', horizontalalignment='left', fontsize=7)
	#fig.savefig("output.pdf")
	self.listoffigures.append(fig)
	return fig

  def Output(self,train):
	fig = plt.figure()
	#bin = 50
	x = frange(-5.,5.,0.1)
	print x
	print '---------------------------------------\n'
	b= x[:,0]
	a= x[:,1]
	bin = np.sqrt(len(a))
	print a
	print '---------------------------------------\n'
	print bin
	z = train.decision_function(x)
	print '---------------------------------------\n'
	print z
	plt.hist2d(a, b, bins=bin, weights=z)
	plt.colorbar()
	plt.title('BDT Output')
	plt.text(-4.8,4.9,self.ReturnOpts(),verticalalignment='top', horizontalalignment='left', fontsize=7)	
	fig.savefig("BDToutput.pdf")
	self.listoffigures.append(fig)
	return fig

  def ReturnOpts(self):
	gbo='learning_rate='+str(self.learning_rate)+', n_estimators='+str(self.n_estimators)+', max_depth='+str(self.max_depth)+', random_state='+str(self.random_state)+', loss='+str(self.loss)+',\nsubsample='+str(self.subsample)+', min_samples_split='+str(self.min_samples_split)+', min_samples_leaf='+str(self.min_samples_leaf)+', min_weight_fraction_leaf='+str(self.min_weight_fraction_leaf)+', \ninit='+str(self.init)+', max_features='+str(self.max_features)+', verbose='+str(self.verbose)+', max_leaf_nodes='+str(self.max_leaf_nodes)+', warm_start='+str(self.warm_start)+', presort='+str(self.presort)
	return gbo



  def CompareTrainTest(self, train, bins=30):
	decisions = []
	for X,y in ((self.X_Array, self.Y_Array), (self.test_X, self.test_y)):
	        d1 = train.decision_function(X[y>0.5]).ravel()
	        d2 = train.decision_function(X[y<0.5]).ravel()
	        decisions += [d1, d2]
        
	low = min(np.min(d) for d in decisions)
	high = max(np.max(d) for d in decisions)
	low_high = (low,high)

	fig = plt.figure()
    
	plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
	plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

	hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
	scale = len(decisions[2]) / sum(hist)
	err = np.sqrt(hist * scale) / scale
   
	width = (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
	hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
	scale = len(decisions[2]) / sum(hist)
	err = np.sqrt(hist * scale) / scale

	plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

	plt.xlabel("BDT output")
	plt.ylabel("Arbitrary units")
	plt.legend(loc='best')

	fig.savefig("bdt_output.pdf")
	return fig
