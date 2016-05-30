from skxgb import xgbLearner
from skLearner import sklearner
import matplotlib.pyplot as plt
from time import *
from matplotlib.backends.backend_pdf import PdfPages

variables=["BDT_common5_input_avg_dr_tagged_jets",
	   #"BDT_common5_input_sphericity",
	   #"BDT_common5_input_third_highest_btag",
	   "BDT_common5_input_h3",
	   #"BDT_common5_input_HT",
	   "BDT_common5_input_fifth_highest_CSV",
	   #"BDT_common5_input_fourth_highest_btag",
	   #"Reco_Deta_Fn_best_TTBBLikelihood",
	   #"Reco_Higgs_M_best_TTLikelihood_comb",
	   #"Reco_LikelihoodRatio_best_Likelihood",
	   #"BDT_common5_input_avg_btag_disc_btags",
	   "BDT_common5_input_pt_all_jets_over_E_all_jets",
	   #"BDT_common5_input_all_sum_pt_with_met",
	   #"BDT_common5_input_aplanarity",
	   "BDT_common5_input_dr_between_lep_and_closest_jet",
	   "BDT_common5_input_best_higgs_mass",
	   #"BDT_common5_input_fourth_jet_pt",
	   #"BDT_common5_input_min_dr_tagged_jets",
	   #"BDT_common5_input_second_highest_btag",
	   #"Evt_Deta_JetsAverage",
	   "BDT_common5_input_third_jet_pt",
	   "BDT_common5_input_closest_tagged_dijet_mass",
	   "BDT_common5_input_tagged_dijet_mass_closest_to_125",
	   #"Reco_Deta_TopHad_BB_best_TTBBLikelihood",
	   #"Reco_Deta_TopLep_BB_best_TTBBLikelihood",
	   #"Reco_LikelihoodTimesMERatio_best_Likelihood",
	   #"Reco_LikelihoodTimesMERatio_best_LikelihoodTimesME",
	   #"Reco_MERatio_best_TTLikelihood_comb",
	   #"Reco_Sum_LikelihoodTimesMERatio",
	   #"Evt_4b3bLikelihoodRatio",
	   #"Evt_4b2bLikelihoodRatio"
]


XGB=xgbLearner(variables)
XGB.SetPlotFile()

XGB.SetSPath('/nfs/dust/cms/user/pkraemer/trees/Category_64/Even/ttHbb_nominal_even.root')
XGB.SetBPath('/nfs/dust/cms/user/pkraemer/trees/Category_64/Even/ttbar_nominal_even.root')
XGB.SetStestPath('/nfs/dust/cms/user/pkraemer/trees/Category_64/Odd/ttHbb_nominal_odd.root')#2D_test_scat.root')
XGB.SetBtestPath('/nfs/dust/cms/user/pkraemer/trees/Category_64/Odd/ttbar_nominal_odd.root')#2D_test_scat.root')

names=[]
classifiers=[]

XGB.Convert()
#no shuffling!! first Signal, then Background is important
#XGB.Shuffle(XGB.Var_Array,XGB.ID_Array)



XGB.SetGradBoostOption('n_estimators', 1500)
XGB.SetGradBoostOption('max_depth', 2)
XGB.SetGradBoostOption('learning_rate', 0.05)

t = XGB.Classify()

decfkt = t.decision_function(XGB.test_var)
proba = t.predict_proba(XGB.test_var)[:,1]

print decfkt
print proba

import numpy as np

decfkt_norm = []

ma = max(np.max(d) for d in decfkt)
mi = min(np.min(d) for d in decfkt)
diff = ma - mi

for i in range(len(decfkt)):
  decfkt_norm.append(decfkt[i]/diff+0.5)
  print proba[i], decfkt_norm[i]

print decfkt_norm

from sklearn.metrics import roc_curve, auc

#col = cm.rainbow(np.linspace(0, 1, len(trains)))
fig = plt.figure(figsize=(10,8))

fpr_1, tpr_1, thresholds_1 = roc_curve(XGB.test_ID, decfkt)
fpr_2, tpr_2, thresholds_2 = roc_curve(XGB.test_ID, proba)
roc_auc_1 = auc(fpr_1, tpr_1)
roc_auc_2 = auc(fpr_2, tpr_2)
plt.plot((1-fpr_1), tpr_1, lw=1, color='r', label='decision_function: ROC (area = %0.4f)'%(roc_auc_1))
plt.plot((1-fpr_2), tpr_2, lw=1, color='b', label='predict_proba: ROC (area = %0.4f)'%(roc_auc_2))
plt.legend(loc="lower right")
plt.show()
raw_input()

#fig2 = plt.figure(figsize=(10,8))

#classifiers.append(t)
#names.append('GradientBoostingClassifier')
#T = XGB.XGBClassify()
#classifiers.append(T)
#names.append('XGBoostClassifier')

#XGB.ROCCurve(classifiers, names)
#XGB.CLFsCorrelation(classifiers, names)

#XGB.PrintOutput(t)
#XGB.PrintOutput(T)
#XGB.PrintScatter(t)
#XGB.PrintScatter(T)
#XGB.PrintHistos(t)
#XGB.PrintHistos(T)
#XGB.Output(t)
#XGB.Output(T)

XGB.PrintFigures()

