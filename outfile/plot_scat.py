import ROOT
ROOT.gROOT.SetBatch(True)

f=ROOT.TFile("autotrain.root")
T=f.Get("TestTree")
L=T.GetListOfBranches()
print L
n=L.GetEntries()
#h=ROOT.TH2F()

c=ROOT.TCanvas("c","c",800,600)
c.Print("Scatter_Plots.pdf[")

for i in range(len(L)):
  for j in range(len(L)):
    st=L[i].GetName()+':'+L[j].GetName()+">>h"
    sts=L[i].GetName()+':'+L[j].GetName()+">>hs"
    #print st
    #h=ROOT.TH2F("h",st,50,T.GetMinimum(L[j].GetName()),T.GetMaximum(L[j].GetName()),50,T.GetMinimum(L[i].GetName()),T.GetMaximum(L[i].GetName()))
    #h.Clear()
   #h=ROOT.TH2F()
    if j!=i and ( L[i].GetName() not in ["classID" , "className" , "weight"] and L[j].GetName() not in ["classID" , "className" , "weight"]):
      h=ROOT.TH2F("h",st,50,T.GetMinimum(L[j].GetName()),T.GetMaximum(L[j].GetName()),50,T.GetMinimum(L[i].GetName()),T.GetMaximum(L[i].GetName()))
      hs=ROOT.TH2F("hs",sts,50,T.GetMinimum(L[j].GetName()),T.GetMaximum(L[j].GetName()),50,T.GetMinimum(L[i].GetName()),T.GetMaximum(L[i].GetName()))
      h.SetMarkerColor(ROOT.kRed)
      hs.SetMarkerColor(ROOT.kBlue)
      #st=L[i].GetName()+':'+L[j].GetName()+">>HIST"
      c.Clear()
    #c.Flush()
      T.Draw(st,"classID==1","SCAT")
      T.Draw(sts,"classID==0","SCAT SAME")
      
    #h=ROOT.TH2F()
      #c.Update()
      c.Print("Scatter_Plots.pdf")
    #st=""
      
c.Print("Scatter_Plots.pdf]")
      