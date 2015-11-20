#include "sgd.h"
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <dmlc/logging.h>


namespace sgd {
 
SgdSolver::SgdSolver() {
}

SgdSolver::~SgdSolver() {
}

void SgdSolver::Init(std::size_t dim, 
                     SgdSolver::real_t step_size_init,
                     SgdSolver::real_t l_2_init,
                     const std::vector<SgdSolver::real_t>& offset_init,
                     const std::vector<SgdSolver::real_t>& reg_offset_init) {
  dim_ = dim;
  step_size_ = step_size_init;
  l_2_ = l_2_init;
  offset_ = offset_init;
  reg_offset_ = reg_offset_init;
  weight_.resize(dim);
  std::fill(weight_.begin(), weight_.end(), 0);
}

SgdSolver::real_t SgdSolver::Predict(const SgdSolver::Row& x) {
  
  auto inner_product = x.SDot(&weight_[0], dim_); //+ x.SDot(&offset_[0], dim_);
  
  return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
}

void SgdSolver::Run(SgdSolver::SampleSet& sample_set) {
  sample_set.Rewind();
  while(sample_set.Next()) {
    Row y = sample_set.GetData();
    auto x = sample_set.TranslateData(y);

    real_t predict = Predict(x);
    real_t label = x.label > 0?1:0;
    for (size_t i = 0; i < x.length; ++i) {
        weight_[x.index[i]] -= step_size_ * (predict - label + l_2_ * (weight_[x.index[i]] - reg_offset_[x.index[i]]));
    }
    
    delete x.index;
    delete x.value;
  }
}

std::vector<SgdSolver::real_t> SgdSolver::GetWeight() const {
  return weight_;
}

}
