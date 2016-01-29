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

#from root_numpy import tree2rec

class data:
  def __init__(self, variables):
	self.Streename='MVATree'
	self.Btreename='MVATree'
	self.weightfile='weights/weights.xml'
	self.X_Array=[]
	self.Y_Array=[]
	self.W_Array=[]
	self.SPath='/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root'
	self.BPath='/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root'
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




  def SetSPath(self,SPATH=''):
	self.SPath=SPATH

  def SetBPath(self, BPATH=''):
	self.BPath=BPATH

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
	self.PrintLog()
	self.WriteOutTree(train)
	return train

#  def TestDefiance(self,train):
#	test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
#	for i, y_pred in enumerate(train.staged_decision_function(self.X_Array)):
#        # clf.loss_ assumes that y_test[i] in {0, 1}
#		test_deviance[i] = clf.loss_(Y_Array, y_pred)

  def Score(self,train):
	return train.score(self.X_Array, self.Y_Array)

  def PrintLog(self):
	gbo='learning_rate='+str(self.learning_rate)+', n_estimators='+str(self.n_estimators)+', max_depth='+str(self.max_depth)+', random_state='+str(self.random_state)+', loss='+str(self.loss)+', subsample='+str(self.subsample)+', min_samples_split='+str(self.min_samples_split)+', min_samples_leaf='+str(self.min_samples_leaf)+', min_weight_fraction_leaf='+str(self.min_weight_fraction_leaf)+', init='+str(self.init)+', max_features='+str(self.max_features)+', verbose='+str(self.verbose)+', max_leaf_nodes='+str(self.max_leaf_nodes)+', warm_start='+str(self.warm_start)+', presort='+str(self.presort)
	outstr='\n\n-----------------input variables:-----------------\n'+str(self.variables)+'\n\n-----------------weights:-----------------\n'+str(self.weights)+'\n\n-----------------Gradient Boost Options:-----------------\n'+gbo+'\n\n\n\n'
	logfile = open("log.txt","a+")
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

  def ROCInt(self,train):
	return roc_auc_score(self.Y_Array, train.decision_function(self.X_Array))
