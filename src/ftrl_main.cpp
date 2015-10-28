#include <memory>
#include <cstdlib>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "ftrl.h"
#include "config.h"
#include "metrics.h"

using namespace rabit;
using namespace ftrl; 
using namespace dmlc;

float Predict(dmlc::Row<std::size_t>& x, std::vector<float>& weight_vec) {
  auto* ptr_weight = &weight_vec[0];
  auto inner_product = x.SDot(ptr_weight, weight_vec.size());

  return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35))));
}

class Model : public dmlc::Serializable {
  public:
    FtrlSolver ftrl_processor_;
    ::admm::FtrlConfig ftrl_params_;

    void Load(dmlc::Stream *fi) {
    }

    void Save(dmlc::Stream *fo) const {
      dmlc::ostream os(fo);
      std::vector<float> weights = ftrl_processor_.weight();
      for (size_t i = 0; i < weights.size(); ++i) {
        os << weights[i] << ' ';
      }
    }

    void SaveAuc(dmlc::Stream *fo, ::admm::SampleSet &sample_set) {
      dmlc::ostream os(fo);
      sample_set.Rewind();
      while(sample_set.Next()) {
        dmlc::Row<std::size_t> x = sample_set.GetData();
        os << x.label << ' ';
      }
      os << '\n';

      std::vector<float> weights = ftrl_processor_.weight();
      sample_set.Rewind();
      while(sample_set.Next()) {
        dmlc::Row<std::size_t> x = sample_set.GetData();
        auto predict = Predict(x, weights);
        os << predict << ' ';
      }
    }

    void InitModel(float l_1,
                   float l_2,
                   float alpha,
                   float beta,
                   std::size_t dim) {
      ftrl_params_.Init(l_1, l_2, alpha, beta, dim); 
      ftrl_processor_.Init(ftrl_params_);
    }
}; 

int main(int argc, char* argv[]) {

  InitLogging(argv[0]);
  rabit::Init(argc, argv);

  rabit::TrackerPrintf("Initialization \n");
  int max_iter = atoi(argv[6]);
  std::string train_path(argv[7]);
  std::string test_path(argv[8]);
  std::string train_name = train_path + std::string(argv[9 + rabit::GetRank()]); 
  std::string test_name = test_path + std::string(argv[9 + rabit::GetRank()]); 

  ::admm::SampleSet train_set;
  CHECK(train_set.Initialize(train_name, 0, 1));
  ::admm::SampleSet test_set;
  CHECK(test_set.Initialize(test_name, 0, 1)); 
  
  Metrics metrics;
  Model ftrl_model;
  ftrl_model.InitModel(atof(argv[1]),
                       atof(argv[2]),
                       atof(argv[3]),
                       atof(argv[4]),
                       atoi(argv[5]));
  std::vector<float> offset;
  std::vector<float> reg_offset;

  rabit::TrackerPrintf("ftrl execution \n");
  for (int i = 0; i < max_iter; ++i) {
    ftrl_model.ftrl_processor_.Run(train_set, offset, reg_offset);
    rabit::TrackerPrintf("%d th iteration: \n", i);

    std::vector<std::vector<float>> group_w;
    group_w.push_back(ftrl_model.ftrl_processor_.weight_);
    metrics.LogLoss(train_set, group_w, true);
    metrics.Auc(train_set, group_w, true);
    metrics.LogLoss(test_set, group_w, false);
    metrics.Auc(test_set, group_w, false);
  }

  //rabit::TrackerPrintf("compute auc \n");
  //std::string auc_name = test_path + "ftrl_auc" + "_" + std::string(argv[9 + rabit::GetRank()]);
  //auto *stream(dmlc::Stream::Create(&auc_name[0], "w"));
  //ftrl_model.SaveAuc(stream, test_set);

  //std::string ftrl_weight = test_path + "ftrl_weight_" + std::string(argv[9 + rabit::GetRank()]);
  //auto *streamb(dmlc::Stream::Create(&ftrl_weight[0], "w"));
  //ftrl_model.Save(streamb);

  //delete streamb;

  rabit::Finalize();
  return 0;
}
