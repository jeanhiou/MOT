#include<iostream>
#include "Resolution_transport.h"


double call(double x){
  if (x>1.){
    return x-1.;
  }
  else{
    return 0;
  }
};

int main(){
  std::cout << " j'en veux joe" << std::endl;
  int N1 = 50;
  int N2 = 50;
  double sigma1 = 0.2;
  double sigma2 = 0.2 * sqrt(1.5);
  double mu = 1;

  MatrixXd Loi1 = loi_log(sigma1,mu,N1);

  MatrixXd Loi2 = loi_log(sigma2,mu,N2);

  VectorXd support1 = Loi1.col(0) ;
  VectorXd support2 = Loi2.col(0);
  VectorXd loi1 = Loi1.col(1);
  VectorXd loi2 = Loi2.col(1);

  double epsilon = 0.01;

  std::function<double(double const & ,double const& )> OT_test =[=](double x, double y){return pow(x+y,2) ;};
  std::function<double(double const & ,double const& )> OT_test2 =[=](double x, double y){return -pow(x-y,2) ;};

  Resolution_transport R_Test(support1,support2,loi1,loi2,OT_test);

  std::cout << R_Test.Resolution_sinkhorn() << std::endl;
  // std::cout << R_Test.Resolution_simplex()  << std::endl;
  //
  // std::cout << R_Test.Resolution_simplex_calls() << std::endl;

  return 0;
}
