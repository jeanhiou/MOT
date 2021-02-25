#include<iostream>
#include<random>
#include "sinkhorn.h"
#include "loi.h"

using namespace std;
using namespace Eigen;

// Standard normal probability density function
double norm_pdf(const double& x) {
    return (1.0/(pow(2*M_PI,0.5)))*exp(-0.5*x*x);
}

double norm_cdf(const double& x) {
    double k = 1.0/(1.0 + 0.2316419*x);
    double k_sum = k*(0.319381530 + k*(-0.356563782 + k*(1.781477937 + k*(-1.821255978 + 1.330274429*k))));

    if (x >= 0.0) {
        return (1.0 - (1.0/(pow(2*M_PI,0.5)))*exp(-0.5*x*x) * k_sum);
    } else {
        return 1.0 - norm_cdf(-x);
    }
}

double d_j(const int& j, const double& S, const double& K, const double& r, const double& v, const double& T) {
    return (log(S/K) + (r + (pow(-1,j-1))*0.5*v*v)*T)/(v*(pow(T,0.5)));
}

double call_price(const double& S, const double& K, const double& r, const double& v, const double& T) {
    return S * norm_cdf(d_j(1, S, K, r, v, T))-K*exp(-r*T) * norm_cdf(d_j(2, S, K, r, v, T));
}

double call(double x , double k ){
  if (x>k){
    return x-k;}
  else{
    return 0;}
  };

int main(){

  int N1 = 3;
  int N2 = 3;
  double sigma1 = 0.2;
  double sigma2 = 0.2 * sqrt(1.5);
  double mu1 =  + pow(sigma1,2)/2;
  double mu2 =  + pow(sigma2,2)/2;

  MatrixXd Loi1 = loi_log(sigma1,mu1,N1);

  MatrixXd Loi2 = loi_log(sigma2,mu2,N2);

  VectorXd support1 = Loi1.col(0) ;
  VectorXd support2 = Loi2.col(0);
  VectorXd loi1 = Loi1.col(1);
  VectorXd loi2 = Loi2.col(1);

  double epsilon = 0.01;

  double strike = 1.;

  cout << "call_price = " << call_price(1.,strike,0.,sigma2,1.) << endl;
  cout << endl;

  function<double(double const & ,double const& )> OT_test =[=](double x, double y){return call(y,strike) ;};
  function<double(double const & ,double const& )> OT_test2 =[=](double x, double y){return -pow(x-y,2) ;};

  Sinkhorn Sink_test( support1, support2, loi1, loi2, epsilon, OT_test);
  Sinkhorn Sink_test2(support1,support2,loi1,loi2,epsilon,OT_test2);

  bool hedge = true ;
  bool impli = false ;
  // bool no_hedge = false;
  MatrixXd Payoffs_test = MatrixXd::Identity(2*N1+N2,2*N1+N2);
  VectorXd x_y_h= ArrayXd::Zero(2*N1+N2);

  cout << " resolution par gradient directo subito mdr" << endl;
  cout << Resolution_par_gradient(support1,support2,loi1,loi2,epsilon,OT_test) << endl;
  cout << "sans hedging " << endl;

  cout << Sink_test.resolution_avec_hedging(false,false) << endl;
  cout << Sink_test2.resolution_avec_hedging(false,false) << endl;
  cout << endl;
  cout << " avec hedging " << endl;
  cout << endl;
  cout << Sink_test.resolution_avec_hedging(hedge,impli) << endl;
  cout << Sink_test2.resolution_avec_hedging(hedge,impli) << endl;
  cout << endl;

  return 0;
}
