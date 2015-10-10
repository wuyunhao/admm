#include <memory>
#include <cstdlib>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "ftrl.h"
#include "config.h"

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
    void LogLoss(::admm::SampleSet &sample_set, bool T) {
      sample_set.Rewind();

      std::vector<float> weights = ftrl_processor_.weight();
      float sum = 0;
      int count = 0;

      while(sample_set.Next()) {
        dmlc::Row<std::size_t> x = sample_set.GetData();
        auto predict = Predict(x, weights);
        sum +=  (int)x.label == 1? -log(predict) : -log(1.0f - predict);
        count++;
      }
      sum = sum/count; 
      if (T) 
        LOG(INFO) << "train  LOGLOSS: " << sum;
      else
        LOG(INFO) << "test LOGLOSS: " << sum;
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
  int niter = atoi(argv[5]);
  std::string path = argv[7];
  std::string train_name = path + std::string(argv[8]) + ".train";
  std::string test_name = path + std::string(argv[8]) + ".test";
  ::admm::SampleSet train_set;
  CHECK(train_set.Initialize(train_name, rabit::GetRank(), 1));
  ::admm::SampleSet test_set;
  CHECK(test_set.Initialize(test_name, rabit::GetRank(),1)); 

  Model ftrl_model;
  ftrl_model.InitModel(atof(argv[1]),
                       atof(argv[2]),
                       atof(argv[3]),
                       atof(argv[4]),
                       atoi(argv[6]));
  std::vector<float> offset;
  std::vector<float> reg_offset;

  rabit::TrackerPrintf("ftrl execution \n");
  for (int i = 0; i < niter; ++i) {
    ftrl_model.ftrl_processor_.Run(train_set, offset, reg_offset);
    LOG(INFO) << "\n  the "<< i << "th iteration: "; 
    ftrl_model.LogLoss(train_set, true);
    ftrl_model.LogLoss(test_set, false);
  }

  rabit::TrackerPrintf("compute auc \n");
  std::string auc_name = path + "ftrl_auc" + "_" + std::string(argv[8]);
  auto *stream(dmlc::Stream::Create(&auc_name[0], "w"));
  
  rabit::TrackerPrintf("====================>  train Logloss:\n");
  ftrl_model.LogLoss(train_set, false);
  rabit::TrackerPrintf("====================>  test Logloss:\n");
  ftrl_model.LogLoss(test_set, false);
  ftrl_model.SaveAuc(stream, test_set);

  std::string ftrl_weight = path + "ftrl_weight_" + std::string(argv[8]);
  auto *streamb(dmlc::Stream::Create(&ftrl_weight[0], "w"));
  ftrl_model.Save(streamb);

  delete stream;
  delete streamb;

  rabit::Finalize();
  return 0;
}
