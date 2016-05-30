from skxgb import xgbLearner
from skLearner import sklearner
import matplotlib.pyplot as plt
from time import *
from matplotlib.backends.backend_pdf import PdfPages

SKL_time_01 = clock()

variables=["X","Y"]

#variables=[#"BDT_common5_input_avg_dr_tagged_jets",
	   #"BDT_common5_input_sphericity",
	   #"BDT_common5_input_third_highest_btag",
#	   "BDT_common5_input_h3",
	   #"BDT_common5_input_HT",
#	   "BDT_common5_input_fifth_highest_CSV",
	   #"BDT_common5_input_fourth_highest_btag",
	   #"Reco_Deta_Fn_best_TTBBLikelihood",
	   #"Reco_Higgs_M_best_TTLikelihood_comb",
#	   "Reco_LikelihoodRatio_best_Likelihood",
	   #"BDT_common5_input_avg_btag_disc_btags",
#	   "BDT_common5_input_pt_all_jets_over_E_all_jets",
	   #"BDT_common5_input_all_sum_pt_with_met",
	   #"BDT_common5_input_aplanarity",
#	   "BDT_common5_input_dr_between_lep_and_closest_jet",
	   #"BDT_common5_input_best_higgs_mass",
	   #"BDT_common5_input_fourth_jet_pt",
	   #"BDT_common5_input_min_dr_tagged_jets",
	   #"BDT_common5_input_second_highest_btag",
	   #"Evt_Deta_JetsAverage",
	   #"BDT_common5_input_third_jet_pt",
#	   "BDT_common5_input_closest_tagged_dijet_mass",
#	   "BDT_common5_input_tagged_dijet_mass_closest_to_125",
	   #"Reco_Deta_TopHad_BB_best_TTBBLikelihood",
	   #"Reco_Deta_TopLep_BB_best_TTBBLikelihood",
	   #"Reco_LikelihoodTimesMERatio_best_Likelihood",
	   #"Reco_LikelihoodTimesMERatio_best_LikelihoodTimesME",
#	   "Reco_MERatio_best_TTLikelihood_comb",
	   #"Reco_Sum_LikelihoodTimesMERatio",
	   #"Evt_4b3bLikelihoodRatio",
#	   "Evt_4b2bLikelihoodRatio"
#]

#LEARNER=xgbLearner(variables)
XGB=xgbLearner(variables)
XGB.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
XGB.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
XGB.SetStestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_01.root')#2D_test_scat.root')
XGB.SetBtestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_01.root')#2D_test_scat.root')
XGB.SetSTreename("S")
XGB.SetBTreename("B")
XGB.SetPlotFile()


LEARN=sklearner(variables)
LEARN.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
LEARN.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
LEARN.SetStestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss_test.root')#2D_test_scat.root')
LEARN.SetBtestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss_test.root')#2D_test_scat.root')
LEARN.SetSTreename("S")
LEARN.SetBTreename("B")
LEARN.SetPlotFile()

names=[]
classifiers=[]
#LEARNER.Convert()

XGB.Convert()
XGB.Shuffle(XGB.Var_Array,XGB.ID_Array)
LEARN.Convert()
#opts=[['learning_rate',0.05,0.07],['n_estimators',1200,1500]]
#varlst, indexlist = LEARNER.permuteVars()
#print varlst
#print indexlist

#LEARNER.variateVars()

#LEARNER.SetGradBoostOption('n_estimators', 1200)
#LEARNER.SetGradBoostOption('max_depth', 2)
#LEARNER.SetGradBoostOption('learning_rate', 0.05)

XGB.SetGradBoostOption('n_estimators', 1500)
XGB.SetGradBoostOption('max_depth', 2)
XGB.SetGradBoostOption('learning_rate', 0.05)

#LEARN.SetGradBoostOption('n_estimators', 1200)
#LEARN.SetGradBoostOption('max_depth', 3)
#LEARN.SetGradBoostOption('learning_rate', 0.05)

t = XGB.Classify()
#t2 = XGB.Classify()
classifiers.append(t)
names.append('GradientBoostingClassifier')
T = XGB.XGBClassify()
classifiers.append(T)
names.append('XGBoostClassifier')
#LEARN.Output(t)
#LEARN.PrintOutput(t)
#LEARNER.permuteVars()
#XGB.testOpts(opts, 5)
#LEARN.PrintFigures()
#LEARNER.SetGradBoostDefault()
#XGB.PrintOutput(t)
#XGB.PrintOutput(t2)
#XGB.PrintOutput(T)

XGB.ROCCurve(classifiers, names)
XGB.CLFsCorrelation(classifiers, names)
XGB.PrintOutput(t)
XGB.PrintOutput(T)
XGB.Output(t)
XGB.Output(T)


#XGB.PrintOutput(t)
#XGB.Output(t)
XGB.PrintFigures()
#T.evals_result()

#for var in XGB.test_var:
#	print T.predict(var)

#print LEARNER.KSTest(t)

#print XGB.KSTest(T)

#nsteps=10
#valuelist=[]
#for var in opts:
#  valuelist.append([])
#  name=var[0]
#  minv=var[1]
#  maxv=var[2]
#  currentvalue=minv
#  dstep=(maxv-minv)/nsteps
#  while currentvalue<=maxv:
#    valuelist[-1].append(currentvalue)
#    currentvalue+=dstep

#valuelist=[[0.1,...],[...]]

#combs



#compare decision_function and predict proba
#print t.decision_function(LEARN.test_var)
#print t.predict(LEARN.test_var)
#a = t.predict_proba(XGB.test_var)
#print a[:,1]
#b = a[:,1]
#print b[:(len(b)/2)]
#print b[(len(b)/2):]

#print t.predict_proba(XGB.test_var)[:,1]
#print t.decision_function(XGB.test_var)
