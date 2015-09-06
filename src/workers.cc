
#include <dmlc/logging.h>
#include "workers.h"
#include "ftrl.h"
#include "config.h"

namespace admm{
    
Worker::Worker() {
}

Worker::Worker(const AdmmConfig& admm_params) {
    base_vec_.resize(admm_params.global_weights.size());
    bias_vec_.resize(admm_params.global_weights.size());
    langr_vec_.resize(admm_params.global_weights.size());
    
    for(auto i = 0u; i < base_vec_.size(); ++i){
        base_vec_[i] = 0;
        bias_vec_[i] = 0;
        langr_vec_[i] = 0;
    }
}

Worker::~Worker() {
}

void Worker::BaseUpdate(SampleSet& sample_set, const AdmmConfig& admm_params) {
    FtrlConfig ftrl_params(admm_params);
    ftrl_params.l_2 = admm_params.step_size;
    ftrl_params.l_1 = 0;

    ::ftrl::FtrlSolver ftrl_processor(ftrl_params);

    std::vector<::ftrl::FtrlSolver::real_t> offset(ftrl_params.dim, 0);
    //set the offset vector
    for (auto i = 0u; i < ftrl_params.dim; ++i) {
        offset[i] = admm_params.global_weights[i] - langr_vec_[i]/admm_params.step_size + bias_vec_[i]; 
    }
    ftrl_processor.Run(sample_set, offset);
    
    base_vec_ = ftrl_processor.weight();
    //recover the modified base weights 
    for(auto i = 0u; i < ftrl_params.dim; ++i) {
        base_vec_[i] += -admm_params.global_weights[i] + langr_vec_[i]/admm_params.step_size; 
    }
}

void Worker::BiasUpdate(SampleSet& sample_set, const AdmmConfig& admm_params) {
    FtrlConfig ftrl_params(admm_params);
    ftrl_params.l_2 = 0;
    ftrl_params.l_1 = admm_params.bias_var;

    ::ftrl::FtrlSolver ftrl_processor(ftrl_params);

    //set the offset vector
    std::vector<::ftrl::FtrlSolver::real_t> offset = base_vec_;
    ftrl_processor.Run(sample_set, offset);

    bias_vec_ = ftrl_processor.weight();
}

void Worker::LangrangeUpdate(const SampleSet& sample_set, const AdmmConfig& admm_params) {
    if (langr_vec_.size() != admm_params.global_weights.size()) {
        LOG(ERROR) << id_ <<"th worker: " << "langrange coefs are different with global weights in length";
        return;
    }
    for (auto i = 0u; i < langr_vec_.size(); ++i) {
        langr_vec_[i] += admm_params.step_size*(base_vec_[i] - admm_params.global_weights[i]);
    }
}

std::vector<Worker::real_t> Worker::GetWeights(AdmmConfig& admm_params) const{
    std::vector<real_t> result;
    result.resize(base_vec_.size());
    for (auto i = 0u; i < base_vec_.size(); ++i) {
        result[i] = base_vec_[i] + langr_vec_[i]/admm_params.step_size;
    }
    return result;
}

} // namespace admm
