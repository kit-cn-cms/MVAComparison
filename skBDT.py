# code mostly taken from
#http://betatim.github.io/posts/advanced-sklearn-for-TMVA/
#http://betatim.github.io/posts/sklearn-for-TMVA-users/

import sklearn
import numpy as np
from root_numpy import root2array, rec2array
from root_numpy import array2root
import ROOT
from sklearn.ensemble import GradientBoostingClassifier
from mvautils import *

#from root_numpy import tree2rec

class data:
  def __init__(self, variables):
	self.treename='MVATree'
	self.weightfile='weights/weights.xml'
	self.X_Array=[]
	self.Y_Array=[]
	self.SPath='/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root'
	self.BPath='/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root'
	self.variables=variables
	self.GradBoostOptions="learning_rate=0.1, n_estimators=100, max_depth=3, random_state=0"



  def SetSPath(self,SPATH=''):
	self.SPath=SPATH

  def SetBPath(self, BPATH=''):
	self.BPath=BPATH

  def SetTreename(self, TREENAME=''):
	self.treename=TREENAME

  def Convert(self):
	train_Signal=root2array(self.SPath, self.treename, self.variables)
	train_Background=root2array(self.BPath, self.treename, self.variables)
	train_Signal=rec2array(train_Signal)
	train_Background=rec2array(train_Background)
	X_train = np.concatenate((train_Signal, train_Background))
	y_train = np.concatenate((np.ones(train_Signal.shape[0]), np.zeros(train_Background.shape[0])))
	self.X_Array = X_train
	self.Y_Array = y_train

  def SetGradBoostOption(self, option):
	self.GradBoostOptions=self.GradBoostOptions.replace(', ',':')
	self.GradBoostOptions=replaceOption(option,self.GradBoostOptions)
	self.GradBoostOptions=self.GradBoostOptions.replace(':',', ')

  def SetGradBoostDefault(self):
	self.GradBoostOptions="loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'"

  def Classify(self):
	listofoptions=self.GradBoostOptions.split(', ')

#OPtionsliste Variabel...; evtl jede Option seperat....

	for i in listofoptions:
		options=listofoptions[i].split('=')
		print listofoptionsandvalues
	train = GradientBoostingClassifier(listofoptionsandvalues).fit(self.X_Array,self.Y_Array)
	return train

#  def TestDefiance(self,train):
#	test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
#	for i, y_pred in enumerate(train.staged_decision_function(self.X_Array)):
#        # clf.loss_ assumes that y_test[i] in {0, 1}
#		test_deviance[i] = clf.loss_(Y_Array, y_pred)

  def Score(self, train):
	return train.score(self.X_Array, self.Y_Array)

