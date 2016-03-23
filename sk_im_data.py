from skBDT import data
import matplotlib.pyplot as plt
from time import *
from matplotlib.backends.backend_pdf import PdfPages

SKL_time_01 = clock()

variables=["X","Y"]

DATA=data(variables)
DATA.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
DATA.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
DATA.SetStestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_scat.root')
DATA.SetBtestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_scat.root')
DATA.SetSTreename("S")
DATA.SetBTreename("B")

estimatorslist=range(10)
plotlist=[]

DATA.SetGradBoostOption('n_estimators', 1500)
DATA.SetGradBoostOption("max_depth", 2)
DATA.SetGradBoostOption("learning_rate", 0.05)
#DATA.SetGradBoostOption("min_samples_leaf", 250)

DATA.Convert()
#print DATA.Y_Array
#DATA.PrintLog()

for estimator in estimatorslist:
	DATA.SetGradBoostOption('n_estimators', estimator+1)
	train1=DATA.Classify()
	f1, f2, f3 = DATA.PrintOutput(train1)
	plotlist.append(f1)
	plotlist.append(f2)
	plotlist.append(f3)
	plotlist.append(DATA.Output(train1))


with PdfPages('results.pdf') as pdf:
	for fig in plotlist:
		pdf.savefig(fig)
#DATA.PrintOutput(train2)

#DATA.PrintFigures()
#DATA.WriteSignalTree()

#DATA.TestGradBoostOptions(1000,2000,0.01,0.1,2)

#DATA.TestTwoOptions("min_samples_split", "min_samples_leaf", 10, 90, 10, 30, 3)

SKL_time_02 = clock()
print("Laufzeit: %f"%(SKL_time_02 - SKL_time_01))
#DATA.ROCCurve(train1)

#print "ROC: %.4f"%(DATA.ROCInt(train1))

#print DATA.Score(train1)
