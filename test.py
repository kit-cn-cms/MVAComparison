from trainer import Trainer
import sys
sys.path.insert(0, '../pyroot-plotscripts')
from plotutils import *
from mvautils import *
#from interpol_vars import *


variables=[#"BDT_v4_input_avg_dr_tagged_jets",
	   #"BDT_v4_input_sphericity",
	   #"BDT_v4_input_third_highest_btag",
	   "BDT_v4_input_h3",
	   #"BDT_v4_input_HT",
	   "BDT_v4_input_fifth_highest_CSV",
	   #"BDT_v4_input_fourth_highest_btag",
	   #"Reco_Deta_Fn_best_TTBBLikelihood",
	   #"Reco_Higgs_M_best_TTLikelihood_comb",
	   "Reco_LikelihoodRatio_best_Likelihood",
	   #"BDT_v4_input_avg_btag_disc_btags",
	   "BDT_v4_input_pt_all_jets_over_E_all_jets",
	   #"BDT_v4_input_all_sum_pt_with_met",
	   #"BDT_v4_input_aplanarity",
	   "BDT_v4_input_dr_between_lep_and_closest_jet",
	   #"BDT_v4_input_best_higgs_mass",
	   #"BDT_v4_input_fourth_jet_pt",
	   #"BDT_v4_input_min_dr_tagged_jets",
	   #"BDT_v4_input_second_highest_btag",
	   #"Evt_Deta_JetsAverage",
	   #"BDT_v4_input_third_jet_pt",
	   "BDT_v4_input_closest_tagged_dijet_mass",
	   "BDT_v4_input_tagged_dijet_mass_closest_to_125",
	   #"Reco_Deta_TopHad_BB_best_TTBBLikelihood",
	   #"Reco_Deta_TopLep_BB_best_TTBBLikelihood",
	   #"Reco_LikelihoodTimesMERatio_best_Likelihood",
	   #"Reco_LikelihoodTimesMERatio_best_LikelihoodTimesME",
	   "Reco_MERatio_best_TTLikelihood_comb",
	   #"Reco_Sum_LikelihoodTimesMERatio",
	   #"Evt_4b3bLikelihoodRatio",
	   "Evt_4b2bLikelihoodRatio"
]

addtional_variables=["BDTOhio_v2_input_h0",
                     "BDTOhio_v2_input_h1"
                     ]

#samples have a name, a color, a path, and a selection (not implemented yet for training)
#only the path is really relevant atm
cat='6j4t'
signal_test=Sample('t#bar{t}H test',ROOT.kBlue,'/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root','') 
signal_train=Sample('t#bar{t}H training',ROOT.kGreen,'/nfs/dust/cms/user/pkraemer/trees/ttH_nominal.root','')
background_test=Sample('t#bar{t} test',ROOT.kRed+1,'/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root','')
background_train=Sample('t#bar{t} training',ROOT.kRed-1,'/nfs/dust/cms/user/pkraemer/trees/ttbar_nominal.root','')
trainer=Trainer(variables,addtional_variables)

trainer.addSamples(signal_train,background_train,signal_test,background_test) #add the sample defined above
trainer.setTreeName('MVATree') # name of tree in files
trainer.setReasonableDefaults() # set some configurations to reasonable values
trainer.setEqualNumEvents(True) # reweight events so that integral in training and testsample is the same
trainer.useTransformations(True) # faster this way
trainer.setVerbose(True) # no output during BDT training and testing
trainer.setWeightExpression('Weight')
trainer.setSelection('N_Jets>=6&&N_BTagsM>=4') # selection for category (not necessary if trees are split)

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
trainer.setBDTOption("nCuts=50")
trainer.setBDTOption("MaxDepth=2")

#trainer.optimizeOption('Shrinkage')
#trainer.optimizeOption('nCuts')

#trainer.suche(1200, 1200, 0.001, 0.05, 30, 60, 10)
#nt3,nt4,sh3,sh4,nc3,nc4 = trainer.suche(nt1,nt2,sh1,sh2,nc1,nc2,2)

print trainer.best_variables
trainer.trainBDT(variables)
ROC, ksS, ksB, ROCT = trainer.evaluateLastTraining()
print ROC, ROCT, ksS, ksB

print trainer.bdtoptions
print trainer.factoryoptions
