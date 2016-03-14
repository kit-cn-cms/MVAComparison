from skBDT import data
import matplotlib.pyplot as plt
from time import *

SKL_time_01 = clock()

variables=["X","Y"]

DATA=data(variables)
DATA.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
DATA.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
DATA.SetStestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test.root')
DATA.SetBtestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test.root')
DATA.SetSTreename("S")
DATA.SetBTreename("B")


DATA.SetGradBoostOption('n_estimators', 2000)
DATA.SetGradBoostOption("max_depth", 3)
DATA.SetGradBoostOption("learning_rate", 0.05)
#DATA.SetGradBoostOption("min_samples_leaf", 250)

DATA.Convert()
#print DATA.X_Array
#DATA.PrintLog()

train1=DATA.Classify()
#DATA.CompareTrainTest(train1)

#print "classified"
DATA.Output(train1)
#DATA.WriteSignalTree()

#DATA.TestGradBoostOptions(1000,2000,0.01,0.1,2)

#DATA.TestTwoOptions("min_samples_split", "min_samples_leaf", 10, 90, 10, 30, 3)

SKL_time_02 = clock()
print("Laufzeit: %f"%(SKL_time_02 - SKL_time_01))
#DATA.ROCCurve(train1)

#print "ROC: %.4f"%(DATA.ROCInt(train1))

#print DATA.Score(train1)
