
#include <stdlib.h>
#include <algorithm>
#include <dmlc/logging.h>
#include "ftrl.h"

namespace ftrl{

FtrlSolver::FtrlSolver(FtrlSolver::real_t lambda_1,
                       FtrlSolver::real_t lambda_2,
                       FtrlSolver::real_t alpha_init,
                       FtrlSolver::real_t beta_init,
                       std::size_t dim_init)
    : l_1_(lambda_1), l_2_(lambda_2), alpha_(alpha_init),
      beta_(beta_init), dim_(dim_init) {

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

FtrlSolver::real_t FtrlSolver::Predict(FtrlSolver::Row& x) {
	for(auto i = 0u; i < x.length; ++i) {
		// w[i] =		 0													 if |z[i]| <= l_1
		//		  (sgn(z[i])*l_1 - z[i])/((beta + sqrt(n[i]))/alpha + l_2)   otherwise.
		//
		real_t sign = mid_weight_[x.index[i]] < 0? -1:1;
		weight_[x.index[i]] = abs(mid_weight_[x.index[i]]) <= l_1_? 0:(sign*l_1_ - mid_weight_[x.index[i]])/((beta_ + sqrt(squared_sum_[x.index[i]]))/alpha_ + l_2_);
	}
	real_t* ptr_weight = &weight_[0];
	auto inner_product = x.SDot(ptr_weight, dim_);

    // P(y=1|x,w) = 1/(1 + exp(-<w,x>)) 
	return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
}

FtrlSolver::real_t FtrlSolver::Predict(FtrlSolver::Row& x, std::vector<FtrlSolver::real_t>& offset) {
    for(auto i = 0u; i < x.length; ++i) {
        //LOG(INFO) << "x.index[" << i << "] = " << x.index[i] << "\n";
		real_t sign = mid_weight_[x.index[i]] < 0? -1:1;
		weight_[x.index[i]] = abs(mid_weight_[x.index[i]]) <= l_1_? 0:(sign*l_1_ - mid_weight_[x.index[i]])/((beta_ + sqrt(squared_sum_[x.index[i]]))/alpha_ + l_2_);
    }

    real_t* ptr_weight = &weight_[0];
    real_t* ptr_offset = &offset[0];

    auto inner_product = x.SDot(ptr_weight, dim_) + x.SDot(ptr_offset, dim_);

	return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
}

void FtrlSolver::Update(const FtrlSolver::Row& x, FtrlSolver::real_t predict) {
    int label = 0;
	//g[i] = (p - y)*x[i]
    if (x.label == 1) label = 1;

	auto loss = predict - label; 

	for(auto i = 0u; i < x.length; ++i) {
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

void FtrlSolver::Assign(const std::vector<FtrlSolver::real_t>& x) {
    dim_ = x.size();
    weight_.resize(dim_);
    LOG(INFO) << "The dim of weight is " << dim_ << " now in ftrl process." << "\n";
    for(auto i = 0u;  i < x.size(); ++i) {
        weight_[i] = x[i];
    }
}

void FtrlSolver::Run(FtrlSolver::SampleSet& sample_set, std::vector<FtrlSolver::real_t>& offset) {
  sample_set.Rewind();
  while(sample_set.Next()) {
  	Row x = sample_set.GetData();
    if(offset.size() == 0) {
  	  auto predict = Predict(x);
  	  Update(x,predict);
    } else {
  	  auto predict = Predict(x,offset);
  	  Update(x,predict);
    }
  }
}
} // namespace ftrl
