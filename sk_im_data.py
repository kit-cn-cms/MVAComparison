from skBDT import data

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


DATA=data(variables)
DATA.SetSPath('test')
print DATA.SPath
