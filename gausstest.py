from gausstrainer import Trainer
import sys
sys.path.insert(0, '../pyroot-plotscripts')
from plotutils import *
from mvautils import *
#from interpol_vars import *
from time import *

t1=clock()

variables=["X","Y"]

addtional_variables=["BDTOhio_v2_input_h0",
                     "BDTOhio_v2_input_h1"
                     ]

#samples have a name, a color, a path, and a selection (not implemented yet for training)
#only the path is really relevant atm
cat='6j4t'
signal_test=Sample('t#bar{t}H test',ROOT.kBlue,'/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root','') 
signal_train=Sample('t#bar{t}H training',ROOT.kGreen,'/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root','')
background_test=Sample('t#bar{t} test',ROOT.kRed+1,'/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root','')
background_train=Sample('t#bar{t} training',ROOT.kRed-1,'/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root','')
trainer=Trainer(variables,addtional_variables)

trainer.addSamples(signal_train,background_train,signal_test,background_test) #add the sample defined above
#trainer.setTreeName('MVATree') # name of tree in files
trainer.setReasonableDefaults() # set some configurations to reasonable values
trainer.setEqualNumEvents(True) # reweight events so that integral in training and testsample is the same
trainer.useTransformations(True) # faster this way
trainer.setVerbose(True) # no output during BDT training and testing
#trainer.setWeightExpression('Weight')
#trainer.setSelection('N_Jets>=6&&N_BTagsM>=4') # selection for category (not necessary if trees are split)

#trainer.removeWorstUntil(12) # removes worst variable until only 10 are left 
#trainer.optimizeOption('NTrees') # optimizies the number of trees by trying more and less trees # you need to reoptimize ntrees depending on the variables and on other parameters
#trainer.optimizeOption('Shrinkage')
#trainer.optimizeOption('nCuts')
#trainer.addBestUntil(13) # add best variables until 12 are used
#trainer.optimizeOption('NTrees')
#trainer.optimizeOption('Shrinkage')
#trainer.optimizeOption('nCuts')
#trainer.removeWorstUntil(12)
#trainer.optimizeOption('NTrees')
#trainer.optimizeOption('Shrinkage')
#trainer.optimizeOption('nCuts')
#trainer.removeWorstUntil(10)
#trainer.optimizeOption('NTrees')
#trainer.optimizeOption('Shrinkage')
#trainer.optimizeOption('nCuts')
#print "these are found to be the 8 best variables and best bdt and factory options"

trainer.setBDTOption("NTrees=1200")
trainer.setBDTOption("Shrinkage=0.02")
#trainer.setBDTOption("nCuts=50")
trainer.setBDTOption("MaxDepth=2")
#trainer.setBDTOption("MinNodeSize=0.01%")
#trainer.optimizeOption('Shrinkage')
#trainer.optimizeOption('nCuts')

trainer.suche(1000, 2000, 0.01, 0.3, 50, 50, 2)
#nt3,nt4,sh3,sh4,nc3,nc4 = trainer.suche(nt1,nt2,sh1,sh2,nc1,nc2,2)

print trainer.best_variables
trainer.trainBDT(variables)

t2=clock()

ROC, ksS, ksB, ROCT = trainer.evaluateLastTraining()
print ROC, ROCT, ksS, ksB
print("Laufzeit: %f"%(t2-t1))
print trainer.bdtoptions
print trainer.factoryoptions
