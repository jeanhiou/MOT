#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include "LBFGS.h"
#include "LBFGSB.h"

using namespace LBFGSpp;
using namespace std;
using namespace Eigen;

struct bornes{
  double borne_basse;
  double borne_haute;
};

struct Sinkhorn{

  Sinkhorn(VectorXd support1, VectorXd support2, VectorXd loi1, VectorXd loi2, double epsilon,
  function<double(double const &, double const&)> payoff): support1(support1),support2(support2),loi1(loi1),loi2(loi2),epsilon(epsilon), payoff(payoff){};

  bornes resolution_sans_hedging();
  bornes resolution_avec_hedging(const bool,const bool );

private:
  VectorXd support1;
  VectorXd support2;
  VectorXd loi1;
  VectorXd loi2;
  double epsilon;
  function<double(double const &, double const&)> payoff;
};

VectorXd ln_v(VectorXd const &v){
  int n = v.size();
  VectorXd ln_(n);
  for (int i =0; i<n;i++){
    ln_(i)= log(v(i));
  };
  return ln_;
};

MatrixXd Payoffs(VectorXd support1, VectorXd support2, function<double(double const&, double const&)> payoff){
  int N1 = support1.size();
  int N2 = support2.size();
  MatrixXd payoffs(N1,N2);
  for (int i =0; i<N1;i++){
    for (int j=0;j<N2;j++){
      payoffs(i,j) = payoff(support1[i],support2[j]);
    };
  };
  return payoffs;
};


bornes Sinkhorn::resolution_sans_hedging(){

  int N1 = support1.size();
  int N2 = support2.size();

  int iter_max = 100;
  bornes m;
  double epsilon_depart= 1.;

  VectorXd ln_loi1=ln_v(loi1);
  VectorXd ln_loi2=ln_v(loi2);

  MatrixXd payoffs = Payoffs(support1,support2,payoff);

  VectorXd phi_basse = ArrayXd::Zero(N1);
  VectorXd phi_haute = ArrayXd::Zero(N1);
  VectorXd psi_basse = ArrayXd::Zero(N2);
  VectorXd psi_haute = ArrayXd::Zero(N2);

  double maximum = - 10;
  double minimum = + 10;

  for (int i = 0; i< N1;i++){
     maximum = max ( payoffs.row(i).maxCoeff(), maximum);
     minimum = min ( payoffs.row(i).minCoeff(), minimum);
  };



  for (int i = 0;i<N1;i++){
    for (int j =0;j<N2;j++){
        payoffs(i,j) = (payoffs(i,j)-minimum)/(maximum-minimum);
    };
  };
  for (int i = 0; i<N1;i++){
    phi_haute(i) = payoffs.row(i).maxCoeff();
    phi_basse(i) = payoffs.row(i).minCoeff();
  };

  for (int j =0; j<N2;j++){
    psi_basse(j) = payoffs.col(j).minCoeff();
    psi_haute(j) = payoffs.col(j).maxCoeff();
  };

  while (epsilon_depart > epsilon){
  for (int k = 0 ; k<iter_max;k++){
  for (int i = 0; i< N1; i++){
    double somme_haute = 0;
    double somme_basse = 0;
    double maximum = 0;
    double minimum = 0;
    for (int j = 0; j< N2;j++){
      maximum = max( maximum, + (payoffs(i,j)-psi_haute[j])/epsilon);
      minimum = max(minimum, + ( - payoffs(i,j)-psi_haute[j])/epsilon);
    };
    for (int j = 0 ; j< N2;j++){
      somme_haute += exp( + ( + payoffs(i,j)-psi_haute[j])/epsilon - maximum);
      somme_basse += exp( + ( - payoffs(i,j)-psi_basse[j])/epsilon - minimum);
    };
    phi_haute[i]= + epsilon * ( maximum + log(somme_haute) - ln_loi1[i] );
    phi_basse[i]= + epsilon * ( minimum + log(somme_basse) - ln_loi1[i] );
  };
  for (int j = 0; j< N2; j++){
    double somme_haute = 0;
    double somme_basse = 0;
    double maximum = 0;
    double minimum = 0;
    for (int i = 0; i< N1;i++){
      maximum = max (maximum, + (payoffs(i,j)-phi_haute[i])/epsilon);
      minimum = max (minimum, + (-payoffs(i,j)-phi_haute[i])/epsilon);
    };
    for (int i = 0 ; i< N1;i++){
      somme_haute += exp( + ( + payoffs(i,j)-phi_haute[i])/epsilon - maximum);
      somme_basse += exp( + ( - payoffs(i,j)-phi_basse[i])/epsilon - minimum);
    };
    psi_haute[j]=  + epsilon * ( maximum + log(somme_haute) - ln_loi2[j] );
    psi_basse[j]=  + epsilon * ( minimum + log(somme_basse) - ln_loi2[j] );
  };
};
epsilon_depart = epsilon_depart/2;
cout << " epsilon_current = " << epsilon_depart << endl;
};

  double esperance_phi_haute = 0.;
  double esperance_psi_haute = 0.;
  double esperance_phi_basse = 0.;
  double esperance_psi_basse = 0.;

  for (int i =0;i<N1;i++){
    esperance_phi_haute += phi_haute(i) * loi1(i);
    esperance_phi_basse += phi_basse(i) * loi1(i);
  };
  for (int i =0;i<N2;i++){
    esperance_psi_haute += psi_haute(i) * loi2(i);
    esperance_psi_basse += psi_basse(i) * loi2(i);
  };
  m.borne_basse = (maximum-minimum)*( - esperance_psi_basse - esperance_phi_basse)+minimum ;
  m.borne_haute = (maximum-minimum)*( + esperance_phi_haute + esperance_psi_haute)+minimum ;
// m.borne_basse = (maximum-minimum)*(esperance_psi_basse + esperance_phi_basse)+minimum ;
  // m.borne_haute = (maximum-minimum)*(esperance_phi_haute + esperance_psi_haute)+minimum ;
  return m;
};


std::ostream & operator<<(std::ostream& o, bornes m ){
  o << "borne_basse = " << m.borne_basse << " " << " borne_haute = " << m.borne_haute << endl;
  return o;
};

//////// Partie avec hedging ////////////////

double Newton_algorithm1D(function<double(double const &)> f,function<double(double const&)> df,double x0,int iter_max){
  double x_zero = x0;
  for (int i = 0;i<iter_max;i++){
    x_zero = x_zero - f(x_zero)/df(x_zero);
  };
  return x_zero;
};

/////  Derive essai_1 //////////::
//// trop d'explosions snif snif snif //////////
double Derivee_F_x_h(double h,int i,double pointx, double phi_x,VectorXd psi_y,VectorXd support2,
  double epsilon,MatrixXd payoffs){
  int N2 = support2.size();
  double somme = 0;
  for (int j = 0; j<N2;j++){
    somme += exp( - (+ psi_y[j] + h*(support2[j]-pointx) - payoffs(i,j))/epsilon) * (support2[j]-pointx);
  };
  return  pow(somme * exp( - phi_x /epsilon ),2);
};

double Derivee2_F_x_h(double h,int i,double pointx, double phi_x,VectorXd psi_y,VectorXd support2,double epsilon,MatrixXd payoffs)
  {
  int N2 = support2.size();
  double somme = 0;
  for (int j = 0; j<N2;j++){
    somme += exp( -  ( + psi_y[j] + h*(support2[j]-pointx) - payoffs(i,j))/epsilon) * pow(support2[j]-pointx,2);
  };
  return 1/epsilon * somme * exp( - phi_x / epsilon);
};

//// Derivee essai_2 par méthode implicite ////

double phi_i_h(double h,int i,double pointx,VectorXd support2, double epsilon,double ln_loi,VectorXd psi, MatrixXd payoffs,bool up_or_low){
  int N2 = support2.size();
  double signe = 1;
  if (up_or_low){ signe = -1;};
    double somme_haute = 0;
    double maximum = 0;
    for (int j = 0; j< N2;j++){
      maximum = max( maximum, signe * ( - payoffs(i,j) + h*(support2[j]-pointx) + psi[j])/epsilon);
    };
    for (int j = 0 ; j< N2;j++){
      somme_haute += exp( signe * ( - payoffs(i,j) + h*(support2[j]-pointx) + psi[j])/epsilon - maximum);
    };
  return (-signe) *epsilon * ( maximum + log(somme_haute) - ln_loi );
};

double Derivee_F_x_h_impli(double h,int i,double pointx,VectorXd support2, double epsilon,double ln_loi,VectorXd psi, MatrixXd payoffs,bool up_or_low){
  double signe= +1;;
  if (up_or_low){signe = -1;};
  int N2 = support2.size();
  double somme = 0;
  for (int j = 0; j<N2;j++){
    somme += exp( signe * ( + phi_i_h(h,i,pointx,support2,epsilon,ln_loi,psi,payoffs,up_or_low) + psi[j] + h*(support2[j]-pointx) - payoffs(i,j))/epsilon) * (support2[j]-pointx);
  }
  return signe * somme;
};

double Derivee2_F_x_h_impli(double h,int i,double pointx,VectorXd support2, double epsilon,double ln_loi,VectorXd psi, MatrixXd payoffs,bool up_or_low){
  double signe;
  if (up_or_low){signe = -1;};
  int N2 = support2.size();
  double somme = 0;
  for (int j = 0; j<N2;j++){
    somme += exp( signe* ( + phi_i_h(h,i,pointx,support2,epsilon,ln_loi,psi,payoffs,up_or_low) + psi[j] + h*(support2[j]-pointx) - payoffs(i,j))/epsilon) * pow(support2[j]-pointx,2);
  }
  return 1/epsilon * (somme - exp(-ln_loi) * pow(Derivee_F_x_h_impli(h,i,pointx,support2,epsilon,ln_loi,psi,payoffs,up_or_low),2) ) ;
};

class F_deri{
private:
  int i ;
  double pointx;
  double phi_x;
  VectorXd psi_y;
  VectorXd support2;
  double epsilon;
  MatrixXd payoffs;
  bool up_or_low;

public:
  F_deri(int i,double pointx,double phi_x,VectorXd psi_y,VectorXd support2,double epsilon,MatrixXd payoffs):
  i(i),pointx(pointx),phi_x(phi_x),psi_y(psi_y),support2(support2),epsilon(epsilon),payoffs(payoffs){};

  double operator()(const VectorXd x, VectorXd& grad){
    double fx;
    double derivee =Derivee_F_x_h(x[0],i,pointx,phi_x,psi_y,support2,epsilon,payoffs);
      fx = pow(derivee,2);
      grad[0] = 2 * derivee * Derivee2_F_x_h(x[0],i,pointx,phi_x,psi_y,support2,epsilon,payoffs);
    return fx;
  };
};


class F_deri_impli{
private:
  int i;
  double pointx;
  VectorXd psi_y;
  VectorXd support2;
  double epsilon;
  MatrixXd payoffs;
  double ln_loi;
  bool up_or_low;

public:
  F_deri_impli( int i,double pointx,VectorXd psi_y,VectorXd support2,double epsilon,MatrixXd payoffs,double ln_loi,bool up_or_low):
  i(i),pointx(pointx),psi_y(psi_y),support2(support2),epsilon(epsilon),payoffs(payoffs),ln_loi(ln_loi),up_or_low(up_or_low){};

  double operator()(VectorXd x, VectorXd& grad){
    double derivee = Derivee_F_x_h_impli(x[0],i,pointx,support2,epsilon,ln_loi,psi_y,payoffs,up_or_low);
    double fx = pow(derivee,2);
    grad[0] = derivee * Derivee2_F_x_h_impli(x[0],i,pointx,support2,epsilon,ln_loi,psi_y,payoffs,up_or_low);
    return fx;
  };
};

bornes Sinkhorn::resolution_avec_hedging(const bool hedge, const bool impli = false){

  int N1 = support1.size();
  int N2 = support2.size();

  int iter_max = 20;
  bornes m;
  double epsilon_depart= 1.;

  VectorXd ln_loi1=ln_v(loi1);
  VectorXd ln_loi2=ln_v(loi2);

  MatrixXd payoffs = Payoffs(support1,support2,payoff);

  VectorXd phi_basse = ArrayXd::Zero(N1);
  VectorXd phi_haute = ArrayXd::Zero(N1);
  VectorXd psi_basse = ArrayXd::Zero(N2);
  VectorXd psi_haute = ArrayXd::Zero(N2);

  VectorXd h_basse = ArrayXd::Zero(N1);
  VectorXd h_haute = ArrayXd::Zero(N1);


    double maximum = - 10;
    double minimum = + 10;

    for (int i = 0; i< N1;i++){
       maximum = max ( payoffs.row(i).maxCoeff(), maximum);
       minimum = min ( payoffs.row(i).minCoeff(), minimum);
    };

  for (int i = 0;i<N1;i++){
    for (int j =0;j<N2;j++){
        payoffs(i,j) = (payoffs(i,j)-minimum)/(maximum-minimum);
    };
  };

  for (int i =0; i<N1;i++){
    double maximum_2= - 1e10 ;
    double minimum_2= + 1e10 ;
    for (int j = 0; j<N2;j++){
      maximum_2 = max(payoffs(i,j),maximum_2);
      minimum_2 = min(payoffs(i,j),minimum_2);
    };
    phi_basse(i) = minimum_2;
    phi_haute(i) = maximum_2;
  };

  for (int j =0; j<N2;j++){
    double maximum=0;
    double minimum=0;
    for (int i = 0; i<N1;i++){
      maximum = max(payoffs(i,j),maximum);
      minimum = min(payoffs(i,j),minimum);
    };
    psi_basse(j) = minimum;
    psi_haute(j) = maximum;
  };

  while (epsilon_depart > epsilon){
  for (int k = 0 ; k<iter_max;k++){
// determination de phi //
  for (int i = 0; i< N1; i++){
    double somme_haute = 0;
    double somme_basse = 0;
    double maximum = 0;
    double minimum = 0;
    for (int j = 0; j< N2;j++){
      maximum = max( maximum, + ( + payoffs(i,j)-h_haute[i]*(support2[j]-support1[i])-psi_haute[j])/epsilon);
      minimum = max (minimum, + ( - payoffs(i,j)-h_basse[i]*(support2[j]-support1[i])-psi_basse[j])/epsilon);
    };
    for (int j = 0 ; j< N2;j++){
      somme_haute += exp( + ( + payoffs(i,j)-h_haute[i]*(support2[j]-support1[i])-psi_haute[j])/epsilon - maximum);
      somme_basse += exp( + ( - payoffs(i,j)-h_basse[i]*(support2[j]-support1[i])-psi_basse[j])/epsilon - minimum);
    };
    phi_haute[i]= + epsilon * ( maximum + log(somme_haute) - ln_loi1[i] );
    phi_basse[i]= + epsilon * ( minimum + log(somme_basse) - ln_loi1[i] );
  };
  // determination de h //
  if (hedge){
    if (impli){
  for (int i = 0; i< N1; i++){
    LBFGSParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 100;
    LBFGSSolver<double> solver(param);
    F_deri_impli haute(i,support1[i],psi_haute,support2,epsilon,payoffs,ln_loi1[i],true);
    F_deri_impli basse(i,support1[i],psi_basse,support2,epsilon,payoffs,ln_loi1[i],false);
    double fx_basse;
    double fx_haute;
    VectorXd x_haute(1);
    VectorXd x_basse(1);
    x_haute(0) = h_haute[i];
    x_basse(0) = h_basse[i];
    int niter  = solver.minimize( haute, x_haute, fx_basse);
    int niter2 = solver.minimize( basse, x_basse, fx_haute);
    h_haute[i] = x_haute.transpose()[0];
    h_basse[i] = x_basse.transpose()[0];
  };
}
  else
  {
      LBFGSParam<double> param;
      param.epsilon = 1e-6;
      param.max_iterations = 100;
      LBFGSSolver<double> solver(param);
      cout << " aaaaaaaa " << endl;
      for (int i = 0 ; i< N1 ;i ++){
        F_deri haute(i,support1[i],phi_haute[i],psi_haute,support2,epsilon, payoffs);
        F_deri basse(i,support1[i],phi_basse[i],psi_basse,support2,epsilon, -1 * payoffs);
        double fx_basse;
        double fx_haute;
        VectorXd x_haute(1);
        VectorXd x_basse(1);
        x_haute(0)= h_basse[i];
        x_basse(0)= h_haute[i];
        cout << " check = " << endl;
        int niter  = solver.minimize( haute , x_haute, fx_basse);
        int niter2 = solver.minimize( basse , x_basse, fx_haute);
        h_haute[i]= x_haute.transpose()[0];
        h_basse[i]= x_basse.transpose()[0];
      };
    //
    // function<double(double const &)> derivee_premiere_haute = [=](double h){return Derivee2_F_x_h(h,i,support1[i],phi_haute[i],psi_haute,support2,epsilon,payoffs,true);};
    // function<double(double const &)> derivee_seconde_haute = [=](double h){ return Derivee2_F_x_h(h,i,support1[i],phi_haute[i],psi_haute,support2,epsilon,payoffs,true);};
    // function<double(double const &)> derivee_premiere_basse = [=](double h){return Derivee2_F_x_h(h,i,support1[i],phi_basse[i],psi_basse,support2,epsilon,payoffs,false);};
    // function<double(double const &)> derivee_seconde_basse = [=](double h){ return Derivee2_F_x_h(h,i,support1[i],phi_basse[i],psi_basse,support2,epsilon,payoffs,false);};
};
};
  // determination de psi///

  for (int j = 0; j< N2; j++){
    double somme_haute = 0;
    double somme_basse = 0;
    double maximum = 0;
    double minimum = 0;
    for (int i = 0; i< N1;i++){
      maximum = max( maximum, + ( + payoffs(i,j)-h_haute[i]*(support2[j]-support1[i])-phi_haute[i])/epsilon);
      minimum = max (minimum, + ( - payoffs(i,j)-h_basse[i]*(support2[j]-support1[i])-phi_basse[i])/epsilon);
    };
    for (int i = 0 ; i< N1;i++){
      somme_haute += exp( + ( + payoffs(i,j)-h_haute[i]*(support2[j]-support1[i])-phi_haute[i])/epsilon - maximum);
      somme_basse += exp( + ( - payoffs(i,j)-h_basse[i]*(support2[j]-support1[i])-phi_basse[i])/epsilon - minimum);
    };
    psi_haute[j]=  + epsilon * ( maximum + log(somme_haute) - ln_loi2[j] );
    psi_basse[j]=  + epsilon * ( minimum + log(somme_basse) - ln_loi2[j] );
  };
};
epsilon_depart = epsilon_depart/2;
cout <<"epsilon_current = " <<  epsilon_depart << endl;

};

  double esperance_phi_haute = 0.;
  double esperance_psi_haute = 0.;
  double esperance_phi_basse = 0.;
  double esperance_psi_basse = 0.;

  for (int i =0;i<N1;i++){
    esperance_phi_haute += phi_haute(i) * loi1(i);
    esperance_phi_basse += phi_basse(i) * loi1(i);
  };
  for (int i =0;i<N2;i++){
    esperance_psi_haute += psi_haute(i) * loi2(i);
    esperance_psi_basse += psi_basse(i) * loi2(i);
  };

  m.borne_basse = (maximum-minimum)*( - esperance_psi_basse - esperance_phi_basse)+minimum ;
  m.borne_haute = (maximum-minimum)*( + esperance_phi_haute + esperance_psi_haute)+minimum ;

  return m;
};


VectorXd extract(const VectorXd full, vector<int> indices){
  int n_indices = indices.size();
  int n_full = full.size();
  assert(n_indices<n_full);
  VectorXd vex(n_indices);
  for (int i = 0; i<n_indices;i++){
    vex[i]=full[indices[i]];
  };
  return vex;
};

vector<int> sequence(int n1, int n2){
  int n = n2-n1;
  vector<int> v(n);
  for (int i = 0;i<n;i++){
    v[i]=n1+i ;
  }
  return v;
};

double Newton_objective( VectorXd x_y_h,VectorXd loi1,VectorXd loi2, MatrixXd payoffs,VectorXd support1, VectorXd support2,double epsilon)
{
  int N1 = loi1.size();
  int N2 = loi2.size();
  vector<int> seqx = sequence(0,N1);
  vector<int> seqy = sequence(N1,N1+N2);
  vector<int> seqh = sequence(N1+N2,2*N1+N2);
  VectorXd phi = extract(x_y_h,seqx);
  VectorXd psi = extract(x_y_h,seqy);
  VectorXd h = extract(x_y_h,seqh);
  double somme_1 = 0 ;
  double somme_2 = 0 ;
  double somme_3 = 0 ;
  for (int i = 0;i<N1;i++){
    somme_1 += phi[i]*loi1[i];
  };
  for (int i = 0;i<N2;i++){
    somme_2 += psi[i]*loi2[i];
  };
  for (int i =0;i<N1;i++){
    double somme_haute = 0;
    for (int j=0;j<N2;j++){
    somme_haute += exp( - ( - payoffs(i,j) + psi[j] + phi[i] + h[i]*(support2[j]-support1[i]))/epsilon );
    somme_3 += somme_haute;
  };
};
  return somme_1 + somme_2 + epsilon * somme_3;
};

VectorXd Newton_objective_grad(VectorXd x_y_h,VectorXd loi1,VectorXd loi2, MatrixXd payoffs,VectorXd support1, VectorXd support2,double epsilon)
{
  int N1 = loi1.size();
  int N2 = loi2.size();
  VectorXd grad=ArrayXd::Zero(2*N1+N2);

  vector<int> seqx = sequence(0,N1);
  vector<int> seqy = sequence(N1,N1+N2);
  vector<int> seqh = sequence(N1+N2,2*N1+N2);
  VectorXd phi = extract(x_y_h,seqx);
  VectorXd psi = extract(x_y_h,seqy);
  VectorXd h = extract(x_y_h,seqh);

  for (int i = 0 ; i<N1;i++){
    double somme_haute = 0;
    for (int j = 0;j<N2;j++){
      somme_haute += exp( - ( - payoffs(i,j) + psi[j] + h[i]*(support2[j]-support1[i]))/epsilon);
    };
    somme_haute *= exp( - phi[i]/epsilon);
    grad[i] =   - somme_haute + loi1[i];
  };
  for (int j = 0 ; j<N2;j++){
    double somme_haute = 0;
    for (int i = 0;i<N1;i++){
      somme_haute += exp( - ( - payoffs(i,j) + phi[i] + h[i]*(support2[j]-support1[i]))/epsilon);
    };
    somme_haute *= exp( - psi[j]/epsilon);
    grad[j+N1] =  - somme_haute + loi2[j];
  };
  for (int i = 0 ;i<N1;i++){
    double somme_h = 0;
    for (int j = 0 ;j < N2;j++){
      somme_h += exp( - ( - payoffs(i,j) + phi[i] + psi[j]+ h[i]*(support2[j]-support1[i]))/epsilon ) * (support2[j]-support1[i]);
    };
    grad[i+N1+N2] = - somme_h;
  };
  return grad;
};

MatrixXd Newton_objective_hessian(VectorXd x_y_h,VectorXd loi1,VectorXd loi2,MatrixXd payoffs,VectorXd support1, VectorXd support2,double epsilon){
  int N1 = loi1.size();
  int N2 = loi2.size();
  MatrixXd Hessian =MatrixXd::Zero(2*N1+N2,2*N1+N2);

  vector<int> seqx = sequence(0,N1);
  vector<int> seqy = sequence(N1,N1+N2);
  vector<int> seqh = sequence(N1+N2,2*N1+N2);
  VectorXd phi = extract(x_y_h,seqx);
  VectorXd psi = extract(x_y_h,seqy);
  VectorXd h = extract(x_y_h,seqh);

  for (int k = 0;k<N1;k++){
    double somme_haute = 0;
    for (int j = 0;j<N2;j++){
      somme_haute += exp(- ( - payoffs(k,j) + h[k]*(support2[j]-support1[k])+ psi[j])/epsilon);
    };
    Hessian(k,k) = -1/epsilon * exp(-phi[k]/epsilon)* somme_haute;
    for (int j = 0 ;j <N2;j ++){
      Hessian(j + N1,k) = - 1/epsilon * exp(-psi[j]/epsilon)*exp( - ( - payoffs(k,j) + h[k]*(support2[j]-support1[k]) + phi[k] )/epsilon);
    };
    for (int i = 0; i<N1 ; i++){
      double somme_haute = 0;
      for (int j = 0; j <N2 ; j++){
        somme_haute += - exp(- ( - payoffs(k,j) + h(k)*(support2[j]-support1[k]) + psi[j])/epsilon)*pow(support2[j]-support1[k],2);
      };
      Hessian(i + N1+N2,k) = -1/epsilon *exp(-phi[k]/epsilon)* somme_haute;
    };
  };
  for (int k = 0;k<N2;k++){
      for (int i = 0 ; i< N1 ; i++){
        Hessian(i,k+N1) = -1/epsilon * exp(- ( phi[i] + psi[k] + h[i]*(support2[i]-support1[k]) - payoffs(i,k))/epsilon);
      };
      double somme_haute = 0;
      for (int i = 0;i<N1;i++){
        somme_haute += exp(- ( phi[i] + psi[k] + h[i]*(support2[i]-support1[k]) - payoffs(i,k))/epsilon);
      };
      Hessian(k+N1,k+N1) = somme_haute * -1/epsilon;
      for (int j = 0;j<N1;j++){
        Hessian(j+N1+N2,k+N1) = 1/epsilon * (support2[k]-support1[j])*exp(-psi[k]/epsilon)*exp((-phi[j]-h[j]*(support2[k]-support1[j])+payoffs(j,k))/epsilon);
    };
  };
  for (int k = 0;k<N1;k++){
    for (int i = 0;i<N1;i++){
      double somme_haute = 0;
      for (int j = 0;j<N2;j++){
        somme_haute += exp(- ( + psi[j] + h[k]*(support2[j]-support1[k]) - payoffs(k,j) )/epsilon);
      };
      Hessian(i,k+N1+N2)= -1/epsilon * somme_haute * exp(-phi[k]/epsilon);
    };
    for (int j = 0; j < N2;j++){
      Hessian(j+N1,k+N1+N2) = -1/epsilon * exp( - ( phi[k ] + psi[j] + h[k]*(support2[j]-support1[k]) - payoffs(k,j))/epsilon) ;;
    };
    double somme_haute = 0;
    for (int j = 0;j<N2;j++){
        somme_haute += exp( - ( phi[k ] + psi[j] + h[k]*(support2[j]-support1[k]) - payoffs(k,j))/epsilon )* pow(support2[j]-support1[k],2);
      };
    Hessian(k+N1+N2,k+N1+N2) = somme_haute * 1/epsilon ;
  };
  return Hessian;
};

VectorXd CG_basic_method(MatrixXd A,VectorXd b, VectorXd x0, double criterium){
  VectorXd r0 = A * x0 - b ;
  VectorXd p0 = - r0 ;
  double alpha_0 = 0;
  double beta_0  = 0;
  VectorXd productA ;
  double productr0;
  VectorXd solution = x0;
  int k = 0 ;
  while (r0.norm()>criterium){
    k+=1;
    productA =  A * p0;
    productr0 = r0.dot(r0);
    alpha_0 = productr0 / (p0.dot(productA));
    solution = solution + alpha_0 * p0;
    r0 = r0 + alpha_0 * productA;
    beta_0 = r0.dot(r0)/productr0;
    p0 = - r0 + beta_0 *p0;
  };
  return solution;
};

VectorXd Newton_CG(function<VectorXd(const VectorXd&)> f,function<MatrixXd(const VectorXd)> grad_f, VectorXd x0){
  int iter_max = 100;
  VectorXd solution = x0;
  double criterium;
  for (int i = 0; i< iter_max; i++){
    criterium = min(0.5,sqrt( f(solution).norm() ) );
    cout << "critere = " << criterium << endl;
    solution = solution -  CG_basic_method(grad_f(solution),f(solution),solution,criterium);
  };
  return solution;
};

double Resolution_par_gradient(  VectorXd support1,VectorXd support2,VectorXd loi1,VectorXd loi2,
  double epsilon,function<double(double const &, double const&)> payoff)
  {
  MatrixXd payoffs = Payoffs(support1, support2, payoff);
  int N1 = loi1.size();
  int N2 = loi2.size();
  VectorXd x0 = ArrayXd::Zero(N1+N2+N1);
  for (int i = 0; i<N1;i++){
    x0(i) = payoffs.row(i).maxCoeff();
  };
  for (int j =0; j<N2;j++){
    x0(j+N1) = payoffs.col(j).minCoeff();
  };
  function<VectorXd(VectorXd x)> grad_f = [=](VectorXd x){return Newton_objective_grad(x,loi1,loi2,payoffs,support1,support2,epsilon);};
  function<MatrixXd(VectorXd x)> hessian_f = [=](VectorXd x ){return Newton_objective_hessian(x,loi1,loi2,payoffs,support1,support2,epsilon);};

  VectorXd solution =  Newton_CG(grad_f,hessian_f,x0);
  return Newton_objective(solution,loi1,loi2,payoffs,support1,support2,epsilon);
};
