
from trainer import Trainer
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


def suche(NTrees_min, NTrees_max, Shrin_min, Shrin_max, nCuts_min, nCuts_max, Schritte)


trainer.setBDTOption("NTrees="+str(NTrees_min))
trainer.setBDTOption("Shrinkage="+str(Shrin_min))
trainer.setBDTOption("nCuts="+str(nCuts_min))
test_max=0
best_NT=0
best_Sh=0
best_nC=0


#for k in range(0,Schritte):
  #ntrees[k]=NTrees_min+k*int((NTrees_max-NTrees_min)/Schritte)
  #shrin[k]=Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)
  #ncuts[k]=nCuts_min+k*int((nCuts_max-nCuts_min)/Schritte)

for i in range(0,Schritte):
  ntrees[i]=NTrees_min+i*int((NTrees_max-NTrees_min)/Schritte)
  ROC, ksS, ksB, ROCT = trainer.evaluateLastTraining()
  test_tmp=10*ROC+min(ksS,KsB)
  if test_tmp>test_max:
    test_max=test_tmp
    best_NT=ntrees[i]
    ijk=[i,j,k]
    
  for k in range(0,Schritte):
    shrin[k]=Shrin_min+k*((Shrin_max-Shrin_min)/Schritte)
    ROC, ksS, ksB, ROCT = trainer.evaluateLastTraining()
    test_tmp=10*ROC+min(ksS,KsB)
    if test_tmp>test_max:
      test_max=test_tmp
      best_Sh=shrin[k]
      ijk=[i,j,k]
     
     for j in range(0,Schritte):
       ncuts[k]=nCuts_min+k*int((nCuts_max-nCuts_min)/Schritte)
       ROC, ksS, ksB, ROCT = trainer.evaluateLastTraining()
       test_tmp=10*ROC+min(ksS,KsB)
       if test_tmp>test_max:
	test_max=test_tmp
	best_nC=ncuts[k]
	ijk=[i,j,k]
	
	
if ijk[0]==NTrees_min:
  nt1=ntrees[0]
else:
  nt1=ntrees[ijk[0]-1]
if ijk[1]==Shrin_min:
  sh1=shrin[0]
else:
  sh1=shrin[ijk[1]-1]
if ijk[2]==nCuts_min:
  nc1=ncuts[0]
else:
  nc1=ncuts[ijk[2]-1]
  
if ijk[0]==NTrees_max:
  nt2=ntrees[Schritte]
else:
  nt2=ntrees[ijk[0]+1]
if ijk[1]==Shrin_max:
  sh2=shrin[Schritte]
else:
  sh2=shrin[ijk[1]+1]
if ijk[2]==nCuts_max:
  nc2=ncuts[Schritte]
else:
  nc2=ncuts[ijk[2]+1]
	
print "bestes NTrees=", best_NT, "beste Shrinkage=", best_Sh,"beste nCuts=", best_nC	
return nt1,nt2,sh1,sh2,nc1,nc2
       