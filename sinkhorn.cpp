#include<iostream>
#include<random>
#include "sinkhorn.h"
#include "loi.h"

using namespace std;
using namespace Eigen;

// Standard normal probability density function

int main(){

  int N1 = 10;
  int N2 = 10;
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

  // cout << " resolution par gradient" << endl;
  // cout << Resolution_par_gradient(support1,support2,loi1,loi2,epsilon,OT_test) << endl;
  cout << "sans hedging " << endl;
  cout << Sink_test.resolution_avec_hedging(false,false) << endl;
  cout << Sink_test2.resolution_avec_hedging(false,false) << endl;
  cout << endl;
  // cout << " avec hedging " << endl;
  // cout << endl;
  // cout << Sink_test.resolution_avec_hedging(hedge,impli) << endl;
  // cout << Sink_test2.resolution_avec_hedging(hedge,impli) << endl;
  // cout << endl;

  return 0;
}
