import ROOT


f=ROOT.TFile("autotrain.root","r+")
HS=f.Get("CorrelationMatrixS")
HB=f.Get("CorrelationMatrixB")
HD=HS.Clone()
HD.Add(HB,-1)
HD.SetTitle("Correlation Matrix (signal-background)")

AXS=HS.GetXaxis()
AXS.SetLabelSize(0.03)
AYS=HS.GetYaxis()
AYS.SetLabelSize(0.03)

AXB=HB.GetXaxis()
AXB.SetLabelSize(0.03)
AYB=HB.GetYaxis()
AYB.SetLabelSize(0.03)

AXD=HD.GetXaxis()
AXD.SetLabelSize(0.03)
AYD=HD.GetYaxis()
AYD.SetLabelSize(0.03)


c=ROOT.TCanvas("c", "c", 2500, 1200)
HS.Draw("colz")
c.SetLeftMargin(0.225)
c.SetBottomMargin(0.175)
c.Print("Correlation_Matrix.pdf(")

#cB=ROOT.TCanvas("c", "c", 2500, 900)
HB.Draw("colz")
#c.SetLeftMargin(0.3)
c.Print("Correlation_Matrix.pdf")

#cD=ROOT.TCanvas("c", "c", 2500, 900)
HD.Draw("colz")
#cD.SetLeftMargin(0.3)
c.Print("Correlation_Matrix.pdf)")