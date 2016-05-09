from skxgb import xgbLearner
import matplotlib.pyplot as plt
from time import *
from matplotlib.backends.backend_pdf import PdfPages

SKL_time_01 = clock()

#variables=["X","Y"]

variables=[#"BDT_common5_input_avg_dr_tagged_jets",
	   #"BDT_common5_input_sphericity",
	   #"BDT_common5_input_third_highest_btag",
	   "BDT_common5_input_h3",
	   #"BDT_common5_input_HT",
	   "BDT_common5_input_fifth_highest_CSV",
	   #"BDT_common5_input_fourth_highest_btag",
	   #"Reco_Deta_Fn_best_TTBBLikelihood",
	   #"Reco_Higgs_M_best_TTLikelihood_comb",
	   "Reco_LikelihoodRatio_best_Likelihood",
	   #"BDT_common5_input_avg_btag_disc_btags",
	   "BDT_common5_input_pt_all_jets_over_E_all_jets",
	   #"BDT_common5_input_all_sum_pt_with_met",
	   #"BDT_common5_input_aplanarity",
	   "BDT_common5_input_dr_between_lep_and_closest_jet",
	   #"BDT_common5_input_best_higgs_mass",
	   #"BDT_common5_input_fourth_jet_pt",
	   #"BDT_common5_input_min_dr_tagged_jets",
	   #"BDT_common5_input_second_highest_btag",
	   #"Evt_Deta_JetsAverage",
	   #"BDT_common5_input_third_jet_pt",
	   "BDT_common5_input_closest_tagged_dijet_mass",
	   "BDT_common5_input_tagged_dijet_mass_closest_to_125",
	   #"Reco_Deta_TopHad_BB_best_TTBBLikelihood",
	   #"Reco_Deta_TopLep_BB_best_TTBBLikelihood",
	   #"Reco_LikelihoodTimesMERatio_best_Likelihood",
	   #"Reco_LikelihoodTimesMERatio_best_LikelihoodTimesME",
	   "Reco_MERatio_best_TTLikelihood_comb",
	   #"Reco_Sum_LikelihoodTimesMERatio",
	   #"Evt_4b3bLikelihoodRatio",
	   "Evt_4b2bLikelihoodRatio"
]

#LEARNER=xgbLearner(variables)
XGB=xgbLearner(variables)
#LEARNER.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
#LEARNER.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
#LEARNER.SetStestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_scat.root')
#LEARNER.SetBtestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_scat.root')
#LEARNER.SetSTreename("S")
#LEARNER.SetBTreename("B")
XGB.SetPlotFile()

names=[]
classifiers=[]
#LEARNER.Convert()
XGB.Convert()
opts=[['learning_rate',0.05,0.07],['n_estimators',1200,1500]]
#varlst, indexlist = LEARNER.permuteVars()
#print varlst
#print indexlist

#LEARNER.variateVars()

#LEARNER.SetGradBoostOption('n_estimators', 1200)
#LEARNER.SetGradBoostOption('max_depth', 2)
#LEARNER.SetGradBoostOption('learning_rate', 0.05)

XGB.SetGradBoostOption('n_estimators', 1200)
XGB.SetGradBoostOption('max_depth', 2)
XGB.SetGradBoostOption('learning_rate', 0.05)

t = XGB.Classify()
classifiers.append(t)
names.append('GradientBoostingClassifier')
T = XGB.XGBClassify()
classifiers.append(T)
names.append('XGBoostClassifier')
#LEARNER.Output(t)
#LEARNER.permuteVars()
#LEARNER.testOpts(opts, 5)
#LEARNER.PrintFigures()
#LEARNER.SetGradBoostDefault()

XGB.ROCCurve(classifiers, names)
#XGB.PrintOutput(T)
XGB.PrintFigures()
#T.evals_result()

#for var in XGB.test_var:
#	print T.predict(var)

#print LEARNER.KSTest(t)



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
