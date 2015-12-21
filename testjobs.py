from os import path
import sys

Schritte = int(input("Anzahl der Schritte: "))
NTrees_min= int(input("kleinstes Anzahl Trees:  "))
NTrees_max= int(input("groesstes Anzahl Trees:  "))
Shrin_min= float(input("kleinste Shrinkage:  "))
Shrin_max= float(input("groesste Shrinkage:  "))
nCuts_min= int(input("kleinste Anzahl Cuts:  "))
nCuts_max= int(input("groesste Anzahl Cuts:  "))

print sys.path

for i in range(0,Schritte):
  for j in range(0,Schritte):
    for k in range(0,Schritte):
      file_path=path.relpath("shellskripts/job-i"+str(i)+"-j"+str(j)+"-k"+str(k)+".sh")
      new_sh=open(file_path,"w")
      new_sh.write("#!/bin/bash\n\
	/etc/profile.d/modules.sh\n\
	module use -a /afs/desy.de/group/cms/modulefiles/\n\
	module load cmssw/slc6_amd64_gcc491\n\
	export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch\n\
	export SCRAM_ARCH=slc6_amd64_gcc491\n\
	source $VO_CMS_SW_DIR/cmsset_default.sh\n\
	cd /afs/desy.de/user/p/pkraemer/CMSSW_7_4_15/src\n\
	eval `scram runtime -sh`\n\
	cd /nfs/dust/cms/user/pkraemer/MVAComparison\n\
	export NTREES="+str(NTrees_min+i*int((NTrees_max-NTrees_min)/Schritte))+"\n\
	export SHRINKAGE="+str(Shrin_min+k*((Shrin_max-Shrin_min)/Schritte))+"\n\
	export NCUTS="+str(nCuts_min+j*int((nCuts_max-nCuts_min)/Schritte))+"\n\
	export I="+str(i)+"\n\
	export J="+str(j)+"\n\
	export K="+str(k)+"\n\
	python best.py")
      new_sh.close()
  