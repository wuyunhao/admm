#include <memory>
#include <vector>
#include <rabit.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include "sample_set.h"
#include "metrics.h"
#include "arg_parser.h"

using namespace dmlc;
using namespace rabit;

class Model : public dmlc::Serializable {
  public:
   std::vector<float> weight_;
   void Load(dmlc::Stream *fi) {
   }

   void Save(dmlc::Stream *fo) const {
   }
   
   void InitModel(const char* ptr) {
     auto* stream = dmlc::Stream::Create(ptr, "r");
     dmlc::istream is(stream);
     
     while(!is.eof()) {
       float val;
       is >> val;
       if(!is.good())
         break;
       weight_.push_back(val);
     }
   }
};

int main(int argc, char* argv[]) {
  InitLogging(argv[0]);
  rabit::Init(argc, argv);

  rabit::TrackerPrintf("Initialization \n");
  std::string loaded_file(argv[1]);
  std::string train_path(argv[2]);
  std::string test_path(argv[3]);
  
  std::string train_name = train_path + std::string(argv[rabit::GetRank() + 4]);
  std::string test_name = test_path + std::string(argv[rabit::GetRank() + 4]);

  ::admm::SampleSet train_set;
  CHECK(train_set.Initialize(train_name, 0, 1));
  ::admm::SampleSet test_set;
  CHECK(test_set.Initialize(test_name, 0, 1));

  Metrics metrics;
  Model evaluation;
  evaluation.InitModel(&loaded_file[0]);

  rabit::TrackerPrintf("%d dim\n", evaluation.weight_.size());
  std::vector<std::vector<float>> group_w;
  group_w.push_back(evaluation.weight_);

  rabit::TrackerPrintf("%d task -- %s metrics: \n", rabit::GetRank(), argv[rabit::GetRank() + 4]);
  metrics.LogLoss(train_set, group_w, true);
  metrics.Auc(train_set, group_w, true);
  metrics.LogLoss(test_set, group_w, false);
  metrics.Auc(test_set, group_w, false);

  rabit::Finalize();
  return 0;
}
