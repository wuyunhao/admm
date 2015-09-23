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
        os << Predict(x, weights) << ' ';
      }
    }
    void InitModel(float l_1,
                   float l_2,
                   float alpha,
                   float beta,
                   std::size_t niter,
                   std::size_t dim) {
      ftrl_params_.Init(l_1, l_2, alpha, beta, niter, dim); 
      ftrl_processor_.Init(ftrl_params_);
    }
}; 

int main(int argc, char* argv[]) {

  InitLogging(argv[0]);
  rabit::Init(argc, argv);

  rabit::TrackerPrintf("Initialization \n");
  ::admm::SampleSet sample_set;
  CHECK(sample_set.Initialize(argv[7], rabit::GetRank(), 1));
  Model ftrl_model;
  ftrl_model.InitModel(atof(argv[1]),
                       atof(argv[2]),
                       atof(argv[3]),
                       atof(argv[4]),
                       atoi(argv[5]),
                       atoi(argv[6]));
  std::vector<float> offset;

  rabit::TrackerPrintf("ftrl execution \n");
  ftrl_model.ftrl_processor_.Run(sample_set, offset);
  std::vector<float> weight = ftrl_model.ftrl_processor_.weight();

  rabit::TrackerPrintf("compute auc \n");
  std::string path =  "data/";// "hdfs://ns1/user/yunhao1/admm/";
  std::string auc = path + "ftrl_auc";
  auto *stream(dmlc::Stream::Create(&auc[0], "w"));
  //get the test set
  ::admm::SampleSet test_set;
  CHECK(test_set.Initialize(argv[8], rabit::GetRank(),1)); 
  ftrl_model.SaveAuc(stream, test_set);
  delete stream;

  rabit::Finalize();
  return 0;
}
