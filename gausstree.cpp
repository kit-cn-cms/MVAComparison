#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TH2.h"
#include "TRandom.h"
#include "iostream"
#include "Riostream.h"
#include <fstream>
#include <stdio.h>

void gausstree(){

   Double_t xs=0;
   Double_t ys=0;
   Double_t xb=0;
   Double_t yb=0;

   TRandom3 *r = new TRandom3;

   TFile *f = new TFile("2D_test.root","recreate");
   cout<<"test"<<endl;
   TTree *S = new TTree("S","Signal_Tree");
   cout<<"test"<<endl;
   TTree *B = new TTree("B","Background_Tree");
   cout<<"test"<<endl;


	S->Branch("X", &xs);
	S->Branch("Y", &ys);
	B->Branch("X", &xb);
	B->Branch("Y", &yb);

//   FILE *fp;
//   fp = fopen("2d_gauss.dat","r");
//   string str = "";

//   while (!fp->eof()){
   //while(!fp.eof()){
//	fscanf(fp, "%lf:%lf:%lf:%lf", &xs, &ys, &xb, &yb);

   for(int i=0; i<=299; i++){
   cout<<"test"<<endl;
	xs = r->Gaus(0,1);
   cout<<"test5"<<xs<<endl;
	ys = r->Gaus(0.,1.);
	xb = r->Gaus(1.,1.);
	yb = r->Gaus(1.,1.);
	//S->Branch("X", xs);
	//S->Branch("Y", ys);
	//B->Branch("X", xb);
	//B->Branch("Y", yb);
	S->Fill();
	B->Fill();
	}
	
   S->Write();
   B->Write();
}
