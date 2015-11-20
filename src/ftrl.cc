
#include <stdlib.h>
#include <algorithm>
#include <rabit.h>
#include <dmlc/logging.h>
#include "ftrl.h"
#include "metrics.h"

namespace ftrl{

FtrlSolver::FtrlSolver(FtrlSolver::real_t lambda_1,
                       FtrlSolver::real_t lambda_2,
                       FtrlSolver::real_t alpha_init,
                       FtrlSolver::real_t beta_init,
                       std::size_t dim_init)
    : l_1_(lambda_1), l_2_(lambda_2), alpha_(alpha_init),
      beta_(beta_init), dim_(dim_init), ls_2_(0){

  weight_.resize(dim_);
  mid_weight_.resize(dim_);
  squared_sum_.resize(dim_);
  
  for(auto i = 0u; i < dim_; ++i){
    weight_[i] = 0;
    mid_weight_[i] = 0;
    squared_sum_[i] = 0;
  }
} 

FtrlSolver::FtrlSolver() {
}

FtrlSolver::~FtrlSolver() {
}

void FtrlSolver::Init(::admm::FtrlConfig& params) {
  l_1_ = params.l_1;
  l_2_ = params.l_2;
  ls_2_ = 0;
  alpha_ = params.alpha;
  beta_ = params.beta;
  dim_ = params.dim;
  
  weight_.resize(dim_);
  mid_weight_.resize(dim_);
  squared_sum_.resize(dim_);
  
  for(auto i = 0u; i < dim_; ++i) {
    weight_[i] = 0;
    mid_weight_[i] = 0;
    squared_sum_[i] = 0;
  }
}

FtrlSolver::real_t FtrlSolver::Predict(FtrlSolver::Row& x,
                                       const std::vector<FtrlSolver::real_t>& offset,
                                       const std::vector<FtrlSolver::real_t>& reg_offset) {
  for(auto i = 0u; i < x.length; ++i) {
    // w[i] =		 0													 if |z[i]| <= l_1
    //		  (sgn(z[i])*l_1 - z[i])/((beta + sqrt(n[i]))/alpha + l_2)   otherwise.
    //
    real_t sign = mid_weight_[x.index[i]] < 0? -1:1;

    if (sign * mid_weight_[x.index[i]] <= l_1_)
      weight_[x.index[i]] = 0;
    else
      weight_[x.index[i]] = (sign * l_1_ - mid_weight_[x.index[i]] + l_2_ * reg_offset[x.index[i]]) / ((beta_ + sqrt(squared_sum_[x.index[i]])) / alpha_ + l_2_);
  }
  
  auto inner_product = x.SDot(&weight_[0], dim_) + x.SDot(&offset[0], dim_);
  
  // P(y=1|x,w) = 1/(1 + exp(-<w,x>)) 
  return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
}

void FtrlSolver::Update(FtrlSolver::real_t predict, const FtrlSolver::Row& x, const std::vector<FtrlSolver::real_t>& reg_offset) {
  int label = x.label > 0? 1:0;
  //g[i] = (p - y)*x[i]
  auto pre_loss = predict - label;
  
  for(auto i = 0u; i < x.length; ++i) {
    auto loss = pre_loss; 
    auto sigma = (sqrt(squared_sum_[x.index[i]] + loss*loss) - sqrt(squared_sum_[x.index[i]]))/alpha_;
    //z[i] = z[i] + g[i] - sigma*w[i]
    mid_weight_[x.index[i]] += loss - sigma * weight_[x.index[i]];
    //n[i] = n[i] + g[i]^2;
    squared_sum_[x.index[i]] += loss * loss;
  }
}

std::vector<FtrlSolver::real_t> FtrlSolver::weight() const {
  return weight_;
}

void FtrlSolver::Assign(const std::vector<FtrlSolver::real_t>& x, const std::vector<FtrlSolver::real_t>& y) {
  for(auto i = 0u;  i < x.size(); ++i) {
    mid_weight_[i] = x[i];
    squared_sum_[i] = y[i];
  }
}

void FtrlSolver::Run(FtrlSolver::SampleSet& train_set,
                     FtrlSolver::SampleSet& test_set,
                     const std::vector<FtrlSolver::real_t>& offset,
                     const std::vector<FtrlSolver::real_t>& reg_offset) {

  train_set.Rewind();
  while(train_set.Next()) {
    Row x = train_set.GetData();

    auto predict = Predict(x, offset, reg_offset);
    Update(predict, x, reg_offset);
  }
}
} // namespace ftrl
