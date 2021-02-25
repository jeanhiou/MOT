#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std ;
using namespace Eigen;

template <typename TState = double, typename TPayoff = VectorXd,typename Vec2= VectorXd(2)}

double max(double x , double y ){
  if (x>y){return x;}
  else
  {return y;
  }
};

double min( double x , double y){
  return  -. max( -. x , -. y);
}

struct Sinkhorn2D {

  Sinkhorn_shema() = default;
  Sinkhorn_shema(VectorXd phi, VectorXd psi, VectorXd h, bool hedging,
    double precision,function<Tpayoff(Tstate const &)> payoff,VectorXd loi1 , VectorXd loi2,int n ):
    phi(phi),psi(psi),h(h),bool(hedging),precision(precision),payoff(payoff),loi1(loi1),loi2(loi2),pas_discre(n) {};

  MatrixXd init_payoffs(function<Tpayoff(Tstate const &)> payoff, VectorXd points1,VectorXd points2);
  Sinkhorn_shema operator() ()
  {
    for (int i = 0;i<n;i++){
      double max_u = 0.;
      double max_l = 0.;

    }


  }



protected:
      VectorXd phi ;
      VectorXd psi ;
      VectorXd h ;
      bool hedging ;
      double precision;
      function<Tpayoff(Tstate const &)> payoff;
      VectorXd loi1;
      VectorXd loi2;
      int pas_discre;
  };
