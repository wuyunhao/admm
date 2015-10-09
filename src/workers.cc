
#include "workers.h"
#include <dmlc/logging.h>
#include <rabit.h>
#include "ftrl.h"
#include "config.h"

namespace admm{

void Worker::LogLoss(::admm::SampleSet &sample_set, const AdmmConfig& admm_params, bool T) {
  std::vector<float> final_weights(base_vec_.size());
  for (size_t i = 0; i < final_weights.size(); ++i) {
     final_weights[i] = bias_vec_[i] + base_vec_[i];
  }
  float sum = 0.0;
  int count = 0;
  sample_set.Rewind();
  while(sample_set.Next()) {
    dmlc::Row<std::size_t> x = sample_set.GetData();
    auto inner_product = x.SDot(&bias_vec_[0], bias_vec_.size()) + x.SDot(&base_vec_[0], base_vec_.size());
    auto predict = 1.0f/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35))));
    sum += (int)x.label == 1? -log(predict): -log(1.0f - predict);
    count++;
  }
  sum = sum/count ;
  if (T)
    LOG(INFO) << " train LOGLOSS is " << sum ;
  else
    LOG(INFO) << " test LOGLOSS is " << sum ;
}

Worker::Worker() {
}

void Worker::InitWorker(std::size_t fdim) {
  base_vec_.resize(fdim);
  bias_vec_.resize(fdim);
  langr_vec_.resize(fdim);
  std::fill(base_vec_.begin(), base_vec_.end(), 0.0f);    
  std::fill(bias_vec_.begin(), bias_vec_.end(), 0.0f);    
  std::fill(langr_vec_.begin(), langr_vec_.end(), 0.0f);    
}

Worker::~Worker() {
}

void Worker::BaseUpdate(SampleSet& sample_set, const AdmmConfig& admm_params, SampleSet& test_set) {
  FtrlConfig ftrl_params(admm_params);
  ftrl_params.l_2 = admm_params.step_size;
  ftrl_params.l_1 = 0;
  
  ::ftrl::FtrlSolver ftrl_processor;
  ftrl_processor.Init(ftrl_params);
  
  std::vector<::ftrl::FtrlSolver::real_t> offset(ftrl_params.dim, 0);
  rabit::TrackerPrintf("base ftrl\n");
  for (int i = 0; i < 20; ++i) { 
    //set the offset vector
    for (auto i = 0u; i < ftrl_params.dim; ++i) {
      offset[i] = admm_params.global_weights[i] - langr_vec_[i]/admm_params.step_size + bias_vec_[i]; 
    }
    //set the ftrl initial solution
    for (auto i = 0u; i < ftrl_params.dim; ++i) {
      base_vec_[i] += -admm_params.global_weights[i] + langr_vec_[i]/admm_params.step_size;
    }
    ftrl_processor.Assign(base_vec_);
    ftrl_processor.Run(sample_set, offset);
    base_vec_ = ftrl_processor.weight();
    //recover the modified base weights 
    for(auto i = 0u; i < ftrl_params.dim; ++i) {
      base_vec_[i] = base_vec_[i] + admm_params.global_weights[i] - langr_vec_[i]/admm_params.step_size; 
    }
    LogLoss(sample_set, admm_params, true);
    LogLoss(test_set, admm_params, false);
  }
  
}

void Worker::BiasUpdate(SampleSet& sample_set, const AdmmConfig& admm_params, SampleSet& test_set) {
  FtrlConfig ftrl_params(admm_params);
  ftrl_params.l_2 = 0;
  ftrl_params.l_1 = admm_params.bias_var;
  
  ::ftrl::FtrlSolver ftrl_processor;
  ftrl_processor.Init(ftrl_params);
  
  rabit::TrackerPrintf("bias ftrl\n");
  for (int i = 0; i < 20; ++i) {
    //set the offset vector
    std::vector<::ftrl::FtrlSolver::real_t> offset = base_vec_;
    //set the ftrl initial solution
    ftrl_processor.Assign(bias_vec_);
    ftrl_processor.Run(sample_set, offset);
    bias_vec_ = ftrl_processor.weight();
    LogLoss(sample_set, admm_params, true);
    LogLoss(test_set, admm_params, false);
  }
  
}

void Worker::LangrangeUpdate(const SampleSet& sample_set, const AdmmConfig& admm_params) {
  for (auto i = 0u; i < langr_vec_.size(); ++i) {
    langr_vec_[i] += admm_params.step_size*(base_vec_[i] - admm_params.global_weights[i]);
  }
}

void Worker::GetWeights(AdmmConfig& admm_params, std::vector<Worker::real_t>& ptr) const {
  for (auto i = 0u; i < ptr.size(); ++i) {
    ptr[i] = base_vec_[i] + langr_vec_[i]/admm_params.step_size;
  }
}

} // namespace admm
