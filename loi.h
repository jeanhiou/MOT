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
    double g = sigma * G(gen) + mu - pow(sigma,2)/2;
    loi_lo(i,0)=exp(g);
    loi_lo(i,1)=log_normal_densite(sigma,mu,exp(g));
  };
  double somme = (loi_lo.col(1)).sum();
  loi_lo.col(1) = loi_lo.col(1)/somme;
  return loi_lo;
};

Eigen::VectorXd creation_objectif(Eigen::VectorXd loi1, Eigen::VectorXd loi2,bool hedging){
  int N1 = loi1.size();
  int N2 = loi2.size();
  int d = 2;
  if (hedging)
  {Eigen::VectorXd objective=Eigen::VectorXd::Zero(N1*d+N2+N1);
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
    int d = 2 ;
    if (hedging){
    Eigen::MatrixXd constraints=Eigen::MatrixXd::Zero(N1*N2,N1*d+N2+N1+1);
    int k =0;
    for (int i =0; i<N1;i++){
      for (int j=0;j<N2;j++){
        constraints(k,i)=1;
        constraints(k,j+N1)=1;
        for (int m = 0 ;m<d;m++){
        constraints(k, m + d * i  + N2 +N1) = pow(support1[i],m+1) * (support2[j]-support1[i]);
        };
        constraints(k,N1*d+N2+N1)= payoff(support1[i],support2[j]);
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

double call(double x, double k){
  if (x>k){
    return x-k;
  }
  else{
    return 0;}
};

Eigen::MatrixXd creation_contrainte_calls_hedging(Eigen::VectorXd support1,Eigen::VectorXd support2,Eigen::VectorXd grid_strike_1, Eigen::VectorXd grid_strike_2,
  std::function<double(double const& , double const&)> payoff,bool hedging = false)
  {
    int N1 = support1.size();
    int N2 = support2.size();
    int d = 2 ;
    int K1 = grid_strike_1.size();
    int K2 = grid_strike_2.size();
    Eigen::MatrixXd constraints= Eigen::MatrixXd::Zero(N1*N2,K1 + K2 +  N1*d + 1);
    int k =0;
    for (int i =0; i<N1;i++){
      for (int j=0;j<N2;j++){

        for (int m = 0;m<K1;m++){
          constraints(k,m)= (support1[i] - grid_strike_1[m]);
        }
        for (int m = 0;m<K2;m++){
          constraints(k,m+K1)=(support2[j] - grid_strike_2[m]);
        };
        for (int m = 0 ;m<d;m++){
          constraints(k, m + d * i  + K2 + K1) = pow(support1[i],m+1) * (support2[j]-support1[i]);
        };

        constraints(k,N1*d+K2+K1)= payoff(support1[i],support2[j]);

        k+=1;
      };
    };
    return constraints;
};

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


Eigen::VectorXd creation_objectif_calls(Eigen::VectorXd loi1, Eigen::VectorXd loi2,Eigen::VectorXd grid_strike_1, Eigen::VectorXd grid_strike_2,
  Eigen::VectorXd support1 , Eigen::VectorXd support2 ){
  int N1 = loi1.size();
  int N2 = loi2.size();
  int K1 = grid_strike_1.size();
  int K2 = grid_strike_2.size();
  int d = 2;
  Eigen::VectorXd objective=Eigen::VectorXd::Zero(K1 + K2+ N1*d);
  for (int i = 0;i<K1;i++){
    double calls = 0;
    for ( int j = 0 ; j< N1 ; j++){
      calls += call(support1[j],grid_strike_1[i])*loi1[j];
    };
    objective[i] = calls;
  };
  for (int i = 0;i<K2;i++){
    double calls = 0;
    for (int j = 0;j<N2 ;j++){
      calls += call(support2[j],grid_strike_2[i])*loi2[j];
    };
    objective[K1+i] = calls ;
  };

  return objective;
};

Eigen::VectorXd creation_grid_strike(int K_grid,Eigen::VectorXd support){
  Eigen::VectorXd grid_strike(K_grid);
  double min = support.minCoeff();
  double max = support.maxCoeff();
  for (int i= 0;i<K_grid;i++){
    grid_strike(i) = max-min/K_grid * (i+1);
  };
  return grid_strike;
};
