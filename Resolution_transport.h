#include<iostream>
#include "solver.hpp"
#include "eigen.hpp"
#include "matrix.hpp"
#include "loi.h"
#include "sinkhorn.h"

struct Resolution_transport{
  Resolution_transport(VectorXd support1, VectorXd support2, VectorXd loi1, VectorXd loi2,
  function<double(double const &, double const&)> payoff): support1(support1),support2(support2),loi1(loi1),loi2(loi2), payoff(payoff){};


  bornes Resolution_simplex(bool );
  bornes Resolution_sinkhorn();
  bornes Resolution_simplex_calls(bool );


private:
  Eigen::VectorXd support1;
  Eigen::VectorXd support2;
  Eigen::VectorXd loi1;
  Eigen::VectorXd loi2;
  double epsilon;
  std::function<double(double const &, double const&)> payoff;
};


bornes Resolution_transport::Resolution_simplex(bool hedging = false){
  using EigenSolver = simplex::Solver<Eigen::MatrixXd>;
  bornes m_resolution;

  Eigen::MatrixXd constraints_basse =  creation_contrainte_hedging(support1,support2,payoff,hedging);
  Eigen::MatrixXd constraints_haute =  creation_contrainte_hedging(support1,support2,payoff,hedging);
  Eigen::VectorXd objectiveFunction=   creation_objectif(loi1,loi2,hedging);

  // std::cout << "contraints_call = " << std::endl;
  // std:: cout << constraints_call << std::endl;
  // std::cout << std::endl;
  // std::cout << "constraints_haute = " << std::endl;
  // std::cout << constraints_haute << std::endl;
  // std::cout << "objective = " << std::endl;
  // std::cout << objectiveFunction.transpose() << std::endl;
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


bornes Resolution_transport::Resolution_simplex_calls(bool hedging = false ){
  using EigenSolver = simplex::Solver<Eigen::MatrixXd>;
  bornes m_resolution;
  int K_grid = 5;
  Eigen::VectorXd grid_strike_1 = creation_grid_strike(K_grid,support1);
  Eigen::VectorXd grid_strike_2 = creation_grid_strike(K_grid,support2);

  Eigen::MatrixXd constraints_call_basse = creation_contrainte_calls_hedging(support1,support2,grid_strike_1,grid_strike_2,payoff);
  Eigen::MatrixXd constraints_call_haute = creation_contrainte_calls_hedging(support1,support2,grid_strike_1,grid_strike_2,payoff);
  Eigen::VectorXd objective_call = creation_objectif_calls(loi1,loi2,grid_strike_1,grid_strike_2,support1,support2);

  EigenSolver solver1(EigenSolver::MODE_MAXIMIZE, objective_call, constraints_call_basse);
  EigenSolver solver2(EigenSolver::MODE_MINIMIZE, objective_call, constraints_call_haute);

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
