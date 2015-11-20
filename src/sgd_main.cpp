#include <memory>
#include <cstdlib>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "sgd.h"
#include "config.h"
#include "metrics.h"

using namespace rabit;
using namespace sgd; 
using namespace dmlc;

float Predict(dmlc::Row<std::size_t>& x, std::vector<float>& weight_vec) {
  auto* ptr_weight = &weight_vec[0];
  auto inner_product = x.SDot(ptr_weight, weight_vec.size());

  return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35))));
}

class Model : public dmlc::Serializable {
  public:
    SgdSolver sgd_processor_;
    ::admm::FtrlConfig ftrl_params_;

    void Load(dmlc::Stream *fi) {
    }

    void Save(dmlc::Stream *fo) const {
      dmlc::ostream os(fo);
      std::vector<float> weights = sgd_processor_.GetWeight();
      for (size_t i = 0; i < weights.size(); ++i) {
        os << weights[i] << ' ';
      }
    }


    void InitModel(float l_1,
                   float l_2,
                   float alpha,
                   float beta,
                   std::size_t dim) {
      ftrl_params_.Init(l_1, l_2, alpha, beta, dim); 
      std::vector<float> offset(dim, 0);
      std::vector<float> reg_offset(dim, 0);
      sgd_processor_.Init(ftrl_params_.dim, 0.01, ftrl_params_.l_2, offset, reg_offset);
    }
}; 

int main(int argc, char* argv[]) {

  InitLogging(argv[0]);
  rabit::Init(argc, argv);

  int max_iter = atoi(argv[6]);
  std::string train_path(argv[7]);
  std::string test_path(argv[8]);
  std::string output_path(argv[9]);
  std::string train_name = train_path + std::string(argv[10 + rabit::GetRank()]); 
  std::string test_name = test_path + std::string(argv[10 + rabit::GetRank()]); 

  
  Metrics metrics;
  Model sgd_model;
  sgd_model.InitModel(atof(argv[1]),
                       atof(argv[2]),
                       atof(argv[3]),
                       atof(argv[4]),
                       atoi(argv[5]));

  rabit::TrackerPrintf("sgd execution \n");
  for (int i = 0; i < max_iter; ++i) {
    rabit::TrackerPrintf("%d th iteration: \n", i);
    ::admm::SampleSet* train_set = new ::admm::SampleSet;
    CHECK(train_set->Initialize(train_name, 0, 1));

    sgd_model.sgd_processor_.Run(*train_set);

    std::vector<std::vector<float>> group_w;
    group_w.push_back(sgd_model.sgd_processor_.GetWeight());
    metrics.LogLoss(*train_set, group_w, true);
    metrics.Auc(*train_set, group_w, true);
    delete train_set;

    ::admm::SampleSet* test_set = new ::admm::SampleSet;
    CHECK(test_set->Initialize(test_name, 0, 1)); 
    metrics.LogLoss(*test_set, group_w, false);
    metrics.Auc(*test_set, group_w, false);
    delete test_set;
  }

  std::string ftrl_weight = output_path + "sgd_weight_" + std::string(argv[10 + rabit::GetRank()]);
  auto *streamb(dmlc::Stream::Create(&ftrl_weight[0], "w"));
  sgd_model.Save(streamb);

  delete streamb;

  rabit::Finalize();
  return 0;
}
