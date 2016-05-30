from skxgb import xgbLearner
from skLearner import sklearner
import matplotlib.pyplot as plt
from time import *
from matplotlib.backends.backend_pdf import PdfPages
from ctypes import *

variables=["X","Y"]

#variables=["BDT_common5_input_avg_dr_tagged_jets",
	   ##"BDT_common5_input_sphericity",
	   ##"BDT_common5_input_third_highest_btag",
	   #"BDT_common5_input_h3",
	   ##"BDT_common5_input_HT",
	   #"BDT_common5_input_fifth_highest_CSV",
	   ##"BDT_common5_input_fourth_highest_btag",
	   ##"Reco_Deta_Fn_best_TTBBLikelihood",
	   ##"Reco_Higgs_M_best_TTLikelihood_comb",
	   ##"Reco_LikelihoodRatio_best_Likelihood",
	   ##"BDT_common5_input_avg_btag_disc_btags",
	   #"BDT_common5_input_pt_all_jets_over_E_all_jets",
	   ##"BDT_common5_input_all_sum_pt_with_met",
	   ##"BDT_common5_input_aplanarity",
	   #"BDT_common5_input_dr_between_lep_and_closest_jet",
	   #"BDT_common5_input_best_higgs_mass",
	   ##"BDT_common5_input_fourth_jet_pt",
	   ##"BDT_common5_input_min_dr_tagged_jets",
	   ##"BDT_common5_input_second_highest_btag",
	   ##"Evt_Deta_JetsAverage",
	   #"BDT_common5_input_third_jet_pt",
	   #"BDT_common5_input_closest_tagged_dijet_mass",
	   #"BDT_common5_input_tagged_dijet_mass_closest_to_125",
	   ##"Reco_Deta_TopHad_BB_best_TTBBLikelihood",
	   ##"Reco_Deta_TopLep_BB_best_TTBBLikelihood",
	   ##"Reco_LikelihoodTimesMERatio_best_Likelihood",
	   ##"Reco_LikelihoodTimesMERatio_best_LikelihoodTimesME",
	   ##"Reco_MERatio_best_TTLikelihood_comb",
	   ##"Reco_Sum_LikelihoodTimesMERatio",
	   ##"Evt_4b3bLikelihoodRatio",
	   ##"Evt_4b2bLikelihoodRatio"
#]


XGB=xgbLearner(variables)
XGB.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
XGB.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
XGB.SetStestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_01.root')#2D_test_scat.root')
XGB.SetBtestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_01.root')#2D_test_scat.root')
XGB.SetSTreename("S")
XGB.SetBTreename("B")
#XGB.SetPlotFile()

#XGB.SetSPath('/nfs/dust/cms/user/pkraemer/trees/Category_64/Even/ttHbb_nominal_even.root')
#XGB.SetBPath('/nfs/dust/cms/user/pkraemer/trees/Category_64/Even/ttbar_nominal_even.root')
#XGB.SetStestPath('/nfs/dust/cms/user/pkraemer/trees/Category_64/Odd/ttHbb_nominal_odd.root')#2D_test_scat.root')
#XGB.SetBtestPath('/nfs/dust/cms/user/pkraemer/trees/Category_64/Odd/ttbar_nominal_odd.root')#2D_test_scat.root')

names=[]
classifiers=[]

XGB.Convert()
#no shuffling!! first Signal, then Background is important
#XGB.Shuffle(XGB.Var_Array,XGB.ID_Array)



XGB.SetGradBoostOption('n_estimators', 1500)
XGB.SetGradBoostOption('max_depth', 3)
XGB.SetGradBoostOption('learning_rate', 0.1)

t = XGB.Classify()
path_1 = XGB.LastClassification
#classifiers.append(t)
#names.append('GradientBoostingClassifier')
T = XGB.XGBClassify()
import pickle
print 'XGBoost: ',type(T)
print 'SKlearn: ',type(t)
#with open('filename.pkl', 'wb') as f:
#    pickle.dump(T, f)
#s = pickle.dump(T,'filename.pkl')
#path_2 = XGB.LastClassification
#classifiers.append(T)
#names.append('XGBoostClassifier')

#XGB.ROCCurve(classifiers, names)
#XGB.CLFsCorrelation(classifiers, names)

T.save_model("01.model")
T.dump_model('dump.raw.txt')
T.dump_model('dump.raw.txt','featmap.txt')


#XGB.PrintOutput(t)
#XGB.PrintOutput(T)
#XGB.PrintScatter(t)
#XGB.PrintScatter(T)
#XGB.PrintHistos(t)
#XGB.PrintHistos(T)
#XGB.Output(t)
#XGB.Output(T)

#XGB.PrintFigures()

TEST = xgbLearner(variables)
TEST.SetSPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
TEST.SetBPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_Gauss.root')
TEST.SetStestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_01.root')#2D_test_scat.root')
TEST.SetBtestPath('/nfs/dust/cms/user/pkraemer/MVAComparison/2D_test_01.root')#2D_test_scat.root')
TEST.SetSTreename("S")
TEST.SetBTreename("B")
TEST.SetPlotFile()
TEST.Convert()
clf_1 = TEST.LoadClassifier(path_1)
#clf_2 = TEST.LoadClassifier(path_2)
#with open('filename.pkl', 'rb') as f:
#    clf_2 = pickle.load(f)
#clf_2 = pickle.loads(s)

classifiers = [clf_1, clf_2]
names = ['loaded GBC', 'XGB']

TEST.ROCCurve(classifiers, names)
TEST.CLFsCorrelation(classifiers, names)
TEST.PrintOutput(clf_1)
TEST.PrintOutput(T)
TEST.PrintScatter(clf_1)
TEST.PrintScatter(T)
TEST.PrintHistos(clf_1)
TEST.PrintHistos(T)
TEST.Output(clf_1)
TEST.Output(T)

TEST.PrintFigures()


###### Links with examples #####
#http://www.xavierdupre.fr/app/pymyinstall/helpsphinx/notebooks/example_xgboost.html
#https://github.com/dmlc/xgboost/commit/68444a06269013efd133ee6e5535faad203110b9
#https://github.com/dmlc/xgboost/commit/a4de0ebcd4e5a5b699794d3b78471b44890fea53
#https://xgboost.readthedocs.io/en/latest/python/python_intro.html#training