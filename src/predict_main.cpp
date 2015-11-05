#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "workers.h"
#include "master.h"
#include "config.h"
#include "metrics.h"

using namespace dmlc;
using namespace rabit;

class LocalModel : public dmlc::Serializable {
  public:
   std::vector<std::vector<float>> group_w;
   int dim;

   void Load(dmlc::Stream *fi) {
     dmlc::istream is(fi);
     std::vector<float> weight;
     weight.resize(dim);
     for (int i = 0; i < dim; ++i) {
       is >> weight[i];
     }
     group_w.push_back(weight);
     for (int i = 0; i < dim; ++i) {
       is >> weight[i];
     }
     group_w.push_back(weight);
   }

   void Save(dmlc::Stream *fo) const {
   }
};

int main(int argc, char* argv[]) {
  InitLogging(argv[0]);
  rabit::Init(argc, argv);
  
  Metrics metrics;
  LocalModel ftrl, admm;
  admm.dim = atoi(argv[1]);
  ftrl.dim = atoi(argv[1]);

  float ratio = atof(argv[2]);
  std::string train_path(argv[3]);
  std::string test_path(argv[4]);
  std::string input_path(argv[5]);
  std::string pid_location(argv[6 + rabit::GetRank()]);

  std::string test_name = test_path + pid_location;
  std::string train_name = train_path + pid_location;
  std::string admm_params = input_path + "admm_weight_" + pid_location;
  std::string ftrl_params = input_path + "ftrl_weight_" + pid_location;

  auto *stream_admm(dmlc::Stream::Create(&admm_params[0], "r"));
  admm.Load(stream_admm);
  delete stream_admm;
  auto *stream_ftrl(dmlc::Stream::Create(&ftrl_params[0], "r"));
  ftrl.Load(stream_ftrl);
  delete stream_ftrl;


  ::admm::SampleSet* train_set = new ::admm::SampleSet;
  CHECK(train_set->Initialize(train_name, 0, 1));
  metrics.Auc(*train_set, ftrl.group_w, admm.group_w, ratio, true);
  delete train_set;

  ::admm::SampleSet* test_set = new ::admm::SampleSet;
  CHECK(test_set->Initialize(test_name, 0, 1));
  metrics.Auc(*test_set, ftrl.group_w, admm.group_w, ratio, false);
  delete test_set;

  rabit::Finalize();
  return 0;
}
