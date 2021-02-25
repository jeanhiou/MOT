#include<iostream>
#include "solver.hpp"
#include "eigen.hpp"
#include "matrix.hpp"
#include "loi.h"
#include "sinkhorn.h"

struct Resolution_transport{
  Resolution_transport(VectorXd support1, VectorXd support2, VectorXd loi1, VectorXd loi2,
  function<double(double const &, double const&)> payoff): support1(support1),support2(support2),loi1(loi1),loi2(loi2), payoff(payoff){};


  bornes Resolution_simplex();
  bornes Resolution_sinkhorn();


private:
  Eigen::VectorXd support1;
  Eigen::VectorXd support2;
  Eigen::VectorXd loi1;
  Eigen::VectorXd loi2;
  double epsilon;
  std::function<double(double const &, double const&)> payoff;
};


bornes Resolution_transport::Resolution_simplex(){
  using EigenSolver = simplex::Solver<Eigen::MatrixXd>;
  bornes m_resolution;
  Eigen::MatrixXd constraints_basse =  creation_contrainte(support1,support2,payoff);
  Eigen::MatrixXd constraints_haute =  creation_contrainte(support1,support2,payoff);
  Eigen::VectorXd objectiveFunction= creation_objectif(loi1,loi2,false);

  EigenSolver solver1(EigenSolver::MODE_MAXIMIZE, objectiveFunction, constraints_basse);
  EigenSolver solver2(EigenSolver::MODE_MINIMIZE, objectiveFunction, constraints_haute);

  switch (solver1.hasSolution())
  {
    case EigenSolver::SOL_FOUND:
    m_resolution.borne_basse = solver1.getOptimum();
    m_resolution.borne_haute = solver2.getOptimum();
    break;

    case EigenSolver::SOL_NONE:
      std::cout << "The linear problem has no solution.\n";
      m_resolution.borne_basse = +1e100;
      m_resolution.borne_haute = -1e100;
      break;

    default:
      std::cout << "Some error occured\n";
      m_resolution.borne_basse = +1e100;
      m_resolution.borne_haute = -1e100;
      break;
  };
  return m_resolution;
};

bornes Resolution_transport::Resolution_sinkhorn(){
  Sinkhorn Sink_test(support1,support2,loi1,loi2,0.01,payoff);
  return Sink_test.resolution_sans_hedging();
};
