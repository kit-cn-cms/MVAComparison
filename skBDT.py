import sklearn
import numpy as np
import ROOT

#from root_numpy import tree2rec

class data:
  def __init__(self, variables):
	self.treename='MVATree'
	self.weightfile='weights/weights.xml'
	self.signalArray=[]
	self.backgroundArray=[]
	self.SPath='/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root'
	self.Bpath='/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root'



  def SetSPath(self,SPATH=''):
	self.SPath=SPATH

  def SetBPath(self, BPATH=''):
	self.BPath=BPATH

  def SetTreename(self, TREENAME=''):
	self.treename=TREENAME

#  def Convert():
#	f1=ROOT.TFile(self.SPath)
#	f2=ROOT.TFile(self.BPath)
#	t1=f1.Get(self.treename)
#	t2=f2.Get(self.treename)
#	
#	for v in variables:
		


