#include <eigen3/Eigen/Dense>
#include<random>

static double pi = 3.14132;

double log_normal_densite(double sigma, double mu, double x){
  return 1/(x*sigma*sqrt(2*pi)) * exp(-(pow(log(x)-mu,2)/(2 *sigma*sigma)));
};

double black_scholes_densite(double sigma, double mu,double x0,double t, double x ){
  return x0 * exp( mu * t + sqrt(t)*x * sigma );
}

Eigen::VectorXd sup(int N1){
  Eigen::VectorXd supp(N1);
  for (int i = 0;i<N1;i++){
    supp(i) = (i+1)/float(N1)*3.;
  };
  return supp;
};

Eigen::MatrixXd loi_log(double sigma, double mu,int N){
  std::normal_distribution<double> G(mu,sigma);
  std::random_device rd;
  std::mt19937_64 gen(rd());
  Eigen::MatrixXd loi_lo(N,2);
  for (int i = 0; i<N;i++){
    double g = G(gen);
    loi_lo(i,0)=exp(g);
    loi_lo(i,1)=black_scholes_densite(sigma,mu,1,1,g);
  };
  double somme = (loi_lo.col(1)).sum();
  loi_lo.col(1) = loi_lo.col(1)/somme;
  return loi_lo;
};

Eigen::VectorXd creation_objectif(Eigen::VectorXd loi1, Eigen::VectorXd loi2,bool hedging){
  int N1 = loi1.size();
  int N2 = loi2.size();
  if (hedging)
  {Eigen::VectorXd objective=Eigen::VectorXd::Zero(N1+N2+N1);
  for (int i = 0;i<N1;i++){
    objective(i)=loi1(i);};
  for (int i=0;i<N2;i++){
    objective(i+N1)=loi2(i);};
  return objective;}
  else
  {
  Eigen::VectorXd objective(N1+N2);
  for (int i = 0;i<N1;i++){
    objective(i)=loi1(i);};
  for (int i=0;i<N2;i++){
    objective(i+N1)=loi2(i);};
  return objective;
};
};


Eigen::MatrixXd creation_contrainte_hedging(Eigen::VectorXd support1,Eigen::VectorXd support2,std::function<double(double const& , double const&)> payoff,bool hedging)
  {
    int N1 = support1.size();
    int N2 = support2.size();
    if (hedging){
    Eigen::MatrixXd constraints=Eigen::MatrixXd::Zero(N1*N2,N1+N2+N1+1);
    int k =0;
    for (int i =0; i<N1;i++){
      for (int j=0;j<N2;j++){
        constraints(k,i)=1;
        constraints(k,j+N1)=1;
        constraints(k,i+N1+N1) = (support2[j]-support1[i]);
        constraints(k,N1+N1+N2)= payoff(support1[i],support2[j]);
        k+=1;
      };
    };
    return constraints;
  }
  else
  {
    Eigen::MatrixXd constraints(N1*N2,N1+N2+1);
    int k =0;
    for (int i =0; i<N1;i++){
      for (int j=0;j<N2;j++){
        constraints(k,i)=1;
        constraints(k,j+N1)=1;
        constraints(k,N1+N2)=payoff(support1[i],support2[j]);
        k+=1;
      };
    };
    return constraints;
  };
};
