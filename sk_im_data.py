from skBDT import data
import matplotlib.pyplot as plt

variables=["X","Y"]

DATA=data(variables)
DATA.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
DATA.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
DATA.SetSTreename("S")
DATA.SetBTreename("B")


DATA.SetGradBoostOption('n_estimators', 1200)
DATA.SetGradBoostOption("max_depth", 2)
DATA.SetGradBoostOption("learning_rate", 0.02)
#DATA.SetGradBoostOption("falsch", 1200)

DATA.Convert()
#DATA.PrintLog()

train1=DATA.Classify()
#DATA.WriteSignalTree()

print DATA.Score(train1)
