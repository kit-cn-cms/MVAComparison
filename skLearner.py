# code partly taken from
#http://betatim.github.io/posts/advanced-sklearn-for-TMVA/
#http://betatim.github.io/posts/sklearn-for-TMVA-users/

import sklearn
import numpy as np
from root_numpy import root2array, rec2array
from root_numpy import array2root
import ROOT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
#from mvautils import *
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
import itertools
from array import array



def TwoDRange(xmin, xmax, ymin, ymax, steps):
	#print xmin, xmax, ymin, ymax
	l=array('f',[])
	x=xmin
	y=ymin
	for i in range(steps):
		x+=((xmax-xmin)/steps)
		y=ymin
		for j in range(steps):
			y+=((ymax-ymin)/steps)
			l = np.concatenate(((l),([x,y])))
	a=np.reshape(l,((len(l)/2),2))
	return a

class sklearner:
  def __init__(self, variables):
#-------trees:-------#
	self.Streename='MVATree'
	self.Btreename='MVATree'
	#self.weightfile='weights/weights.xml'
#-------np.Arrays:-------#
	self.Var_Array=[]			#Training-Sample: Variables of ROOT.Tree converted to np.Array (shape= ['f',[var1,var2,var3,...]...])
	self.ID_Array=[]			#Training-Sample: IDs of Events (0 for Background; 1 for Signal)
	#self.Weights_Array=[]
	self.test_var=[]			#Test-Sample: Variables of ROOT.Tree converted to np.Array (shape= ['f',[var1,var2,var3,...]...])
	self.test_ID=[]				#Training-Sample: IDs of Events (0 for Background; 1 for Signal)
#-------file-paths:-------#			#File Path of Training/TestSamples; plotfile, LogFile; OutFile
	self.SPath='/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root'
	self.StestPath='/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root'
	self.BPath='/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root'
	self.BtestPath='/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root'
	self.PlotFile='SKlearn_PlotFile.pdf'
	self.logfilename="log.txt"
	self.outname="SKout.root"
#-------variables & weights:-------#
	self.variables=variables
	self.varpairs=[]
	self.varindex=[]
	self.weights='1.'
#-------BDT-Options:-------#
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
	self.options=['learning_rate', 'n_estimators', 'max_depth', 'random_state', 'loss', 'subsample', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'init', 'max_features', 'verbose', 'max_leaf_nodes', 'warm_start', 'presort']
#-------Storage:-------#
	self.listoffigures=[]			#List with all plots
	self.plot = False			#Flag if PLots are created or not

#-----------------------------------------------#
########unused########
#	self.train
	self.train_Background=[]
	self.train_Signal=[]

#	self.SKout=ROOT.TFile(self.outname,"RECREATE")
	self.sy_tr=[]
	self.ROC_Color=0
	self.ROC_fig = plt.figure()
	self.ROC_Curve=[[],[],[]]


	self.Class_ID=[]
########################

#create new Plotfile [use at the beginning of training/testing; all plots are printed there]
  def SetPlotFile(self):
        dt=datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
	self.PlotFile = 'SKlearn_PlotFile_'+dt+'.pdf'
	self.plot = True

#create new Plotfile [use at the beginning of training/testing; all options, variables and evaluationvalues are printed there]
  def SetLogfileName(self):
        dt=datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
	self.logfilename = 'SKlearn_LogFile_'+dt+'.pdf'

#use to change Path of (Training-)SignalTree
  def SetSPath(self,SPATH=''):
	self.SPath=SPATH

#use to change Path of (Training-)BackgroundTree
  def SetBPath(self, BPATH=''):
	self.BPath=BPATH

#use to change Path of (Test-)SignalTree
  def SetStestPath(self,SPATH=''):
	self.StestPath=SPATH

#use to change Path of (Test-)BackgroundTree
  def SetBtestPath(self, BPATH=''):
	self.BtestPath=BPATH

#use to change name of BackgroundTree
  def SetBTreename(self, TREENAME=''):
	self.Btreename=TREENAME

#use to change name of SignalTree
  def SetSTreename(self, TREENAME=''):
	self.Streename=TREENAME

#use to set Weights ###not impelmented yet###
#  def SetWeight(self, WEIGHT=''):
#	self.weights=weight


#convert .root Trees to numpy arrays
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
	self.Var_Array = X_train
	self.ID_Array = y_train
	#self.Weight_Array = w_train
	test_Signal=root2array(self.StestPath, self.Streename, self.variables)
	test_Background=root2array(self.BtestPath, self.Btreename, self.variables)
	test_Signal=rec2array(test_Signal)
	test_Background=rec2array(test_Background)
	self.test_var=np.concatenate((test_Signal,test_Background))
	self.test_ID=np.concatenate((np.ones(test_Signal.shape[0]), np.zeros(test_Background.shape[0])))
	self.permuteVars()

#Shuffle Signal and Background
  def Shuffle(self, values, IDs):
	permutation = np.random.permutation(len(IDs))
	x, y = [],[]
	for i in permutation:
		x.append(values[i])
		y.append(IDs[i])
	values, IDs = x, y


#change options
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


#sets default options
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


#trains on training sample and returns classifier fit
  def Classify(self):
	train = GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, loss=self.loss, subsample=self.subsample, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf, init=self.init, max_features=self.max_features, verbose=self.verbose, max_leaf_nodes=self.max_leaf_nodes, warm_start=self.warm_start, presort=self.presort).fit(self.Var_Array,self.ID_Array)
	self.PrintLog(train)
	return train

#Scikit-Learn Score function, the higher the better
  def Score(self,train):
	return train.score(self.Var_Array, self.ID_Array)


#print all figures in 1 pdf
  def PrintFigures(self):
	with PdfPages(self.PlotFile) as pdf:
		for fig in self.listoffigures:
			pdf.savefig(fig)


#prints logfile of the training
  def PrintLog(self,train):
	gbo='learning_rate='+str(self.learning_rate)+', n_estimators='+str(self.n_estimators)+', max_depth='+str(self.max_depth)+', random_state='+str(self.random_state)+', loss='+str(self.loss)+', subsample='+str(self.subsample)+', min_samples_split='+str(self.min_samples_split)+', min_samples_leaf='+str(self.min_samples_leaf)+', min_weight_fraction_leaf='+str(self.min_weight_fraction_leaf)+', init='+str(self.init)+', max_features='+str(self.max_features)+', verbose='+str(self.verbose)+', max_leaf_nodes='+str(self.max_leaf_nodes)+', warm_start='+str(self.warm_start)+', presort='+str(self.presort)
	outstr='\n\n-----------------input variables:-----------------\n'+str(self.variables)+'\n\n-----------------weights:-----------------\n'+str(self.weights)+'\n\n-----------------Gradient Boost Options:-----------------\n'+gbo+'\n\n\n\n'+'--------------- ROC integral = '+str(self.ROCInt(train))+' -----------------'
	logfile = open(self.logfilename,"a+")
	logfile.write('######'+datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")+'#####'+outstr+'###############################################\n\n\n\n\n')
	logfile.close()
	print outstr


#Compute ROC-Integral
  def ROCInt(self,train):
	return roc_auc_score(self.test_ID, train.decision_function(self.test_var))


#print Roccurve
  def ROCCurve(self,train):
	decisions = train.decision_function(self.test_var)
	fpr, tpr, thresholds = roc_curve(self.test_ID, decisions)
	#########################################
	#---store stuff to compare afterwards---#
	fprfile = open("fpr.pkl","w")		#
	tprfile = open("tpr.pkl","w")		#
	pickle.dump(fpr,fprfile)		#
	pickle.dump(tpr,tprfile)		#
	tprfile.close()				#
	fprfile.close()				#
	#########################################
	roc_auc = auc(fpr, tpr)
	fig = plt.figure(figsize=(10,8))
	plt.plot((1-fpr), tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
	plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.grid()
	axes = fig.gca()
	ymin, ymax = axes.get_ylim()
	xmin, xmax = axes.get_xlim()
	if xmin<0:
		xmark = xmin*1.07
	else:
		xmark = xmin*0.93
	if ymax>0:
		ymark = ymax*1.1
	else:
		ymark = ymax*0.9
	plt.text(xmark, ymark, self.ReturnOpts(), verticalalignment='top', horizontalalignment='left', fontsize=7 )
	self.listoffigures.append(fig)
	return fig


  def permuteVars(self):
	combs = itertools.combinations(self.variables, 2)
	varcombs = list(combs)
	self.varpairs = varcombs
	indexcombs = []
	for pair in varcombs:
		indexpair = []
		for var in pair:
			indexpair.append(self.variables.index(var))
			#print indexpair
		indexcombs.append(indexpair)
	self.varindex = indexcombs
	#print varcombs, indexcombs
	return varcombs, indexcombs


  def permuteOpts(self, options, steps):
	opts2test=[]
	testopts=[]
	opts=[]
	valuelist=[]
	for i in range(len(options)):
		opts2test.append(options[i][0])
	for opt in self.options:
		for i in range(len(options)):
			if opt == options[i][0]:
				opts.append(options[i])
				options[i][0]='fertig'
			if opt != options[i][0] and opt not in opts2test:
				opts.append([opt,eval('self.'+opt),eval('self.'+opt)])
				break
	for var in opts:
		valuelist.append([])
		name=var[0]
		minv=var[1]
		maxv=var[2]
		currentvalue=minv
		if type(minv) in (int, float) and (maxv-minv)==0:
			valuelist[-1].append(currentvalue)
		elif type(currentvalue)==int:
			dstep=int((maxv-minv)/steps)
		elif type(currentvalue)==float:
			dstep=float((maxv-minv)/steps)
		else:
			#print name + ' is not iterable --> set default'
			valuelist[-1].append(currentvalue)
		while currentvalue<maxv:
			valuelist[-1].append(currentvalue)
			currentvalue+=dstep
	combs=itertools.product(*valuelist)
	testlist=list(combs)
	#print testlist
	return testlist


#test different options by brute-force
### include best roc ###
  def testOpts(self, opts, steps):
	testlist = self.permuteOpts(opts,steps)
	ROCs = []
	bestroc = [0,self.ReturnOpts]
	for test in testlist:
		#print test
		for opt, val in zip(self.options, test):
			#print opt
			#print val
			self.SetGradBoostOption(opt, val)
		train = self.Classify()
		ROCs.append([self.ROCInt(train),self.ReturnOpts()])
		if self.ROCInt(train)>bestroc[0]:
			bestroc[0] = self.ROCInt(train)
			bestroc[1] = self.ReturnOpts()
		if self.plot == True:
			self.ROCCurve(train)
			self.PrintOutput(train)
			self.Output(train)


#makes scatterplot with classification of testevents and wrong classified Events
###--- add histos ---###
  def PrintOutput(self,train):
	value = self.test_var
	predict = train.predict(value)

	#compute decision_function	
	decisionsS=train.decision_function(value[:(len(value)/2)])
	decisionsB=train.decision_function(value[(len(value)/2):])
	decisions = np.concatenate((decisionsS,decisionsB))
	low = min(np.min(d) for d in decisions)
	high = max(np.max(d) for d in decisions)
	low_high = (low,high)

	#plot BDT ouput-shape
	shape = plt.figure(figsize=(10,8))
	plt.hist(decisionsS, color='r', range=low_high, bins=50, histtype='step', normed=False, label='Signal')
	plt.hist(decisionsB, color='b', range=low_high, bins=50, histtype='step', normed=False, label='Background')
	plt.xlabel('SKlearn decision_function')
	plt.ylabel('Events')
	plt.title('shape of BDT Output')
	plt.legend(loc='best')
	axes = shape.gca()
	ymin, ymax = axes.get_ylim()
	if low<0:
		xmark = low*1.07
	else:
		xmark = low*0.93
	if ymax>0:
		ymark = ymax*1.1
	else:
		ymark = ymax*0.9
	plt.text(xmark, ymark, self.ReturnOpts(), verticalalignment='top', horizontalalignment='left', fontsize=7 )
	self.listoffigures.append(shape)
	plt.close()

	#check if prediction is correct or not
	for i in range(len(predict)):
		if (self.test_ID[i] != predict[i] and self.test_ID[i] == 0):
			predict[i] = -1		#Background wrong classified
		elif (self.test_ID[i] != predict[i] and self.test_ID[i] == 1):
			predict[i] = 2		#Signal wrong classified

	#plot for every combination of variables
	for pair, index in zip(self.varpairs, self.varindex):
		fig, ax = plt.subplots(figsize=(10,8))

		#compute ax-limits for scatterplots
		if min(value[:,index[0]]) < 0:
			low_x = min(value[:,index[0]])*1.05
		else:
			low_x = min(value[:,index[0]])*0.95
		if max(value[:,index[0]]) < 0:
			high_x = max(value[:,index[0]])*0.95
		else:
			high_x = max(value[:,index[0]])*1.055
		if min(value[:,index[1]]) < 0:
			low_y = min(value[:,index[1]])*1.05
		else:
			low_y = min(value[:,index[1]])*0.95
		if max(value[:,index[1]]) < 0:
			high_y = max(value[:,index[1]])*0.95
		else:
			high_y = max(value[:,index[1]])*1.055
		ax.set_xlim([low_x, high_x])
		ax.set_ylim([low_y, high_y])
		ax.set_xlabel(pair[0])
		ax.set_ylabel(pair[1])

		#create lists with correct/wrong classified Signal/Background
		wsx, wsy, wbx, wby, sx, sy, bx, by = [],[],[],[],[],[],[],[]
		for i in range(len(predict)):
			if (predict[i] == -1):
				wby.append(value[i,index[1]])
				wbx.append(value[i,index[0]])		#Background wrong classified
			elif (predict[i] == 2):
				wsy.append(value[i,index[1]])
				wsx.append(value[i,index[0]])		#Signal wrong classified
			elif (predict[i] == 1):
				sy.append(value[i,index[1]])
				sx.append(value[i,index[0]])		#Signal correct classified
			elif (predict[i] == 0):
				by.append(value[i,index[1]])
				bx.append(value[i,index[0]])		#Background correct classified
			else:
				print "Whuaaaat??? - wrong classification in "+str(i)

		plt.scatter(value[:,index[0]::1000], value[:,index[1]::1000], c=predict, cmap='rainbow', alpha = 1)
		plt.colorbar()
		plt.title('BDT Output for Testtree')
		axes = fig.gca()
		ymin, ymax = axes.get_ylim()
		if low_x<0:
			xmark = low_x*1.07
		else:
			xmark = low_x*0.93
		if ymax>0:
			ymark = ymax*1.17
		else:
			ymark = ymax*0.83
		plt.text(xmark, ymark, self.ReturnOpts(), verticalalignment='top', horizontalalignment='left', fontsize=7 )
		self.listoffigures.append(fig)
		plt.close()

	#plot histos with Classified Signal/Background
	for var in self.variables:
		lowx = min(np.min(d) for d in value[:,self.variables.index(var)])
		highx = max(np.max(d) for d in value[:,self.variables.index(var)])
		lowx_highx = (lowx,highx)

		#create lists with correct/wrong classified Signal/Background
		ws, wb, cs, cb = [],[],[],[]
		for i in range(len(predict)):
			if (predict[i] == -1):
				wb.append(value[i,self.variables.index(var)])		#Background wrong classified
			elif (predict[i] == 2):
				ws.append(value[i,self.variables.index(var)])		#Signal wrong classified
			elif (predict[i] == 1):
				cs.append(value[i,self.variables.index(var)])		#Signal correct classified
			elif (predict[i] == 0):
				cb.append(value[i,self.variables.index(var)])		#Background correct classified
			else:
				print "Whuaaaat??? - wrong classification in "+str(i)
		s = np.concatenate((cs,wb))
		b = np.concatenate((cb,ws))

		histx = plt.figure(figsize=(10,8))
		plt.hist(s, color='r', range=lowx_highx, bins=25, histtype='stepfilled', alpha=0.5, normed=False, label='signal')
		plt.hist(b, color='b', range=lowx_highx, bins=25, histtype='stepfilled', alpha=0.5, normed=False, label='background')
		plt.hist(cs, color='orange', range=lowx_highx, bins=25, histtype='step', normed=False, label='correct signal')
		plt.hist(cb, color='c', range=lowx_highx, bins=25, histtype='step', normed=False, label='correct background')
		plt.hist(ws, color='darkred', range=lowx_highx, bins=25, histtype='step', normed=False, label='wrong signal')
		plt.hist(wb, color='navy', range=lowx_highx, bins=25, histtype='step', normed=False, label='wrong background')
		plt.xlabel(var)
		plt.ylabel("Events")
		plt.legend(loc='best')
		axes = histx.gca()
		ymin, ymax = axes.get_ylim()
		if lowx<0:
			xmark = lowx*1.07
		else:
			xmark = lowx*0.93
		if ymax>0:
			ymark = ymax*1.1
		else:
			ymark = ymax*0.9
		plt.title('Histogramm of Classification '+var)
		plt.text(xmark, ymark, self.ReturnOpts(), verticalalignment='top', horizontalalignment='left', fontsize=7 )	
		self.listoffigures.append(histx)
		plt.close()

	return fig, histx, shape


#return Classifier Options
  def ReturnOpts(self):
	gbo='learning_rate='+str(self.learning_rate)+', n_estimators='+str(self.n_estimators)+', max_depth='+str(self.max_depth)+', random_state='+str(self.random_state)+', loss='+str(self.loss)+',\nsubsample='+str(self.subsample)+', min_samples_split='+str(self.min_samples_split)+', min_samples_leaf='+str(self.min_samples_leaf)+', min_weight_fraction_leaf='+str(self.min_weight_fraction_leaf)+', \ninit='+str(self.init)+', max_features='+str(self.max_features)+', verbose='+str(self.verbose)+', max_leaf_nodes='+str(self.max_leaf_nodes)+', warm_start='+str(self.warm_start)+', presort='+str(self.presort)
	return gbo


#create sample with variation of al vars
#### num(vars)-dimensional sample! step^num(vars) values...
  def variateVars(self):
	var = []
	for i in range(len(self.variables)):
		exec('var_'+str(i)+' = list()')
		var.append(['var_'+str(i),eval('var_'+str(i))])
	#print var
	

#makes 2d-Histo with BDT output, appends to figurelist
  def Output(self,train):
	for pair, index in zip(self.varpairs, self.varindex):
		fig, ax = plt.subplots(figsize=(10,8))
		value = self.test_var
		#predict = train.predict(value)

		#compute ax-limits for scatterplots
		if min(value[:,index[0]]) < 0:
			low_x = min(value[:,index[0]])*1.05
		else:
			low_x = min(value[:,index[0]])*0.95
		if max(value[:,index[0]]) < 0:
			high_x = max(value[:,index[0]])*0.95
		else:
			high_x = max(value[:,index[0]])*1.055
		if min(value[:,index[1]]) < 0:
			low_y = min(value[:,index[1]])*1.05
		else:
			low_y = min(value[:,index[1]])*0.95
		if max(value[:,index[1]]) < 0:
			high_y = max(value[:,index[1]])*0.95
		else:
			high_y = max(value[:,index[1]])*1.055
		ax.set_xlim([low_x, high_x])
		ax.set_ylim([low_y, high_y])
		ax.set_xlabel(pair[0])
		ax.set_ylabel(pair[1])

		bin = 100
		#print low_x
		#print low_y
		#print high_x
		#print high_y

		x = TwoDRange(low_x, high_x, low_y, high_y, bin)
		#print x
		v = []
		#v = np.ndarray(shape=(len(x)/len(self.variables),len(self.variables)), dtype=float)
		l = range(len(self.variables))
		for i in range(len(x)):
			for n in range(len(self.variables)):
				if n == index[0]:
					v.append(x[i][0])
					#l[n] = x[i,n]
				elif n == index[1]:
					v.append(x[i][1])
					#l[n] = x[i,n]
				else:
					v.append(np.mean(value, axis=0)[n])
					#l[n] = np.mean(value, axis=0)
			#v = np.append((v),(l))
		#print value, type(value)
		#v = np.ndarray(v, dtype=float)
		v = np.reshape(v,(len(v)/len(self.variables),len(self.variables)))
		#print v
		a= v[:,index[0]]
		b= v[:,index[1]]
		z = train.decision_function(v)
		plt.hist2d(a, b, bins=bin, weights=z)
		plt.colorbar()
		plt.title('BDT prediction for '+pair[0]+', '+pair[1])
		xmark = low_x-(high_x-low_x)*0.07
		ymark = high_y+(high_y-low_y)*0.09
		plt.text(xmark,ymark,self.ReturnOpts(),verticalalignment='top', horizontalalignment='left', fontsize=7)	
		self.listoffigures.append(fig)
		plt.close()


  def KSTest(self, train):
	value = self.test_var
	predict = train.predict(value)
	good, bad = 0,0
	for i in range(len(predict)):
		if predict[i]==self.test_ID[i]:
			good+=1
		else:
			bad+=1
	KS=(good-bad)/(good+bad)
	return KS
