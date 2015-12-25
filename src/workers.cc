
#include "workers.h"
#include <dmlc/logging.h>
#include <rabit.h>
#include "ftrl.h"
#include "sgd.h"
#include "config.h"

namespace admm{

void Worker::LogLoss(::admm::SampleSet &sample_set, const AdmmConfig& admm_params, bool T) {
  float sum = 0.0;
  int count = 0;
  sample_set.Rewind();
  while(sample_set.Next()) {
    dmlc::Row<std::size_t> x = sample_set.GetData();
    auto inner_product = x.SDot(&bias_vec_[0], bias_vec_.size()) + x.SDot(&base_vec_[0], base_vec_.size());
    auto predict = 1.0f/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35))));
    sum += x.label > 0? -log(predict): -log(1.0f - predict);
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
  nt_vec_.resize(fdim);
  zt_vec_.resize(fdim);

  std::fill(base_vec_.begin(), base_vec_.end(), 0);    
  std::fill(bias_vec_.begin(), bias_vec_.end(), 0);    
  std::fill(langr_vec_.begin(), langr_vec_.end(), 0);    
  std::fill(nt_vec_.begin(), nt_vec_.end(), 0);    
  std::fill(zt_vec_.begin(), zt_vec_.end(), 0);    

  load = false;
}

Worker::~Worker() {
}

void Worker::BaseUpdate(SampleSet& train_set, SampleSet& test_set, const AdmmConfig& admm_params) {
  FtrlConfig ftrl_params(admm_params);
  ftrl_params.l_2 = 200;
  ftrl_params.l_1 = 1;
  
  //set the reg_offset vector
  std::vector<::ftrl::FtrlSolver::real_t> reg_offset(ftrl_params.dim, 0);
  for (auto i = 0u; i < ftrl_params.dim; ++i) {
    reg_offset[i] = admm_params.global_weights[i] ;//- langr_vec_[i]/admm_params.step_size;
  }

  //set the ftrl initial solution
  ::ftrl::FtrlSolver ftrl_processor; //::sgd::SgdSolver sgd_processor;
  ftrl_processor.Init(ftrl_params); //sgd_processor.Init(ftrl_params.dim, 0.01, ftrl_params.l_2, bias_vec_, reg_offset);
  
  ftrl_processor.squared_sum_ = nt_vec_;
  ftrl_processor.mid_weight_ = zt_vec_; 

  std::string train_name = admm_params.train_path + "/" + psid_ + "_aggregated/part-00000"; 

  //rabit::TrackerPrintf("base ftrl\n");
  for (int i = 0; i < 1; ++i) { 
    for (int part = 0; part < num_part_; ++part) {
      CHECK(train_set.Initialize(train_name, part, num_part_));
      //rabit::TrackerPrintf("base stage :%s %d part \n", &psid_[0], part);
      ftrl_processor.Run(train_set, test_set, bias_vec_, reg_offset); //sgd_processor.Run(train_set);
    }
    base_vec_ = ftrl_processor.weight();
  }

  if (load) {
    nt_vec_ = ftrl_processor.squared_sum_;
    zt_vec_ = ftrl_processor.mid_weight_;
    load = false;
  }

}

void Worker::BiasUpdate(SampleSet& train_set, SampleSet& test_set, const AdmmConfig& admm_params) {
  FtrlConfig ftrl_params(admm_params);
  ftrl_params.l_2 = 200;
  ftrl_params.l_1 = 1000; // admm_params.bias_var;
  
  //set the reg_offset vector
  std::vector<::ftrl::FtrlSolver::real_t> reg_offset(ftrl_params.dim, 0);
  
  //set the ftrl initial solution
  ::ftrl::FtrlSolver ftrl_processor; //::sgd::SgdSolver sgd_processor;
  ftrl_processor.Init(ftrl_params); //sgd_processor.Init(ftrl_params.dim, 0.01, ftrl_params.l_2, base_vec_, reg_offset);

  std::string train_name = admm_params.train_path + psid_ + "_aggregated/part-00000";
  
  rabit::TrackerPrintf("bias ftrl\n");
  for (int i = 0; i < 1; ++i) {
    for (int part = 0; part < num_part_; ++part) {
      CHECK(train_set.Initialize(train_name, part, num_part_));
      //rabit::TrackerPrintf("bias stage :%s %d part \n", &psid_[0], part);
      ftrl_processor.Run(train_set, test_set, base_vec_, reg_offset); //sgd_processor.Run(train_set);
    }
    bias_vec_ = ftrl_processor.weight();
  }
  
}

void Worker::LangrangeUpdate(const AdmmConfig& admm_params) {
  for (auto i = 0u; i < langr_vec_.size(); ++i) {
    langr_vec_[i] += admm_params.step_size*(base_vec_[i] - admm_params.global_weights[i]);
  }
}

void Worker::GetWeights(AdmmConfig& admm_params, std::vector<Worker::real_t>& ptr) const {
  for (auto i = 0u; i < ptr.size(); ++i) {
    ptr[i] = base_vec_[i] * admm_params.step_size + langr_vec_[i];
  }
}

} // namespace admm
