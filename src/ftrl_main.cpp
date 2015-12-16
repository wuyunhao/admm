#include <memory>
#include <iomanip>
#include <cstdlib>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "ftrl.h"
#include "config.h"
#include "metrics.h"
#include "arg_parser.h"

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
      dmlc::istream is(fi);
      ftrl_processor_.weight_.clear();
      ftrl_processor_.mid_weight_.clear();
      ftrl_processor_.squared_sum_.clear();
      std::string value;
      while(!is.eof()) {
        // w recovery
        is >> value;
        if (is.fail()) break;
        float val = atof(value.c_str());
        ftrl_processor_.weight_.push_back(val);

        //z_t recovery
        is >> value;
        val = atof(&value[0]);
        ftrl_processor_.mid_weight_.push_back(val);

        //n_t recovery
        is >> value;
        val = atof(&value[0]);
        ftrl_processor_.squared_sum_.push_back(val);
      }

    }

    void Save(dmlc::Stream *fo) const {
      dmlc::ostream os(fo);
      std::vector<float> weights = ftrl_processor_.weight();
      for (size_t i = 0; i < weights.size(); ++i) {
        os << i+1 << ":" << weights[i] << '\n';
      }
    }

    void SaveState(dmlc::Stream *fo) {
      dmlc::ostream os(fo);
      for (size_t i = 0; i < ftrl_processor_.dim_; ++i) {
        os << std::left << std::setw(17) << ftrl_processor_.weight_[i]
           << std::left << std::setw(17) << ftrl_processor_.mid_weight_[i]
           << std::left << std::setw(17) << ftrl_processor_.squared_sum_[i] << '\n';
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

    void InitModel(const char* conf) {
      ::admm::ArgParser arg_parser;
      arg_parser.FTRLParse(conf, ftrl_params_);
      ftrl_processor_.Init(ftrl_params_);
      ftrl_processor_.psid_ = arg_parser.psid[rabit::GetRank()];
      ftrl_processor_.num_part_ = arg_parser.num_part[rabit::GetRank()];

      if (arg_parser.load_path.size() != 0) {
        std::string ftrl_weight = ftrl_params_.output_path + "ftrl_weight_" + ftrl_processor_.psid_;
        auto *stream(dmlc::Stream::Create(&ftrl_weight[0], "r"));
        Load(stream);
        delete stream;
      }
    }
}; 

int main(int argc, char* argv[]) {

  InitLogging(argv[0]);
  rabit::Init(argc, argv);

  Metrics metrics;
  Model ftrl_model;
  ftrl_model.InitModel(argv[1]);

  std::string train_name = ftrl_model.ftrl_params_.train_path + ftrl_model.ftrl_processor_.psid_ + "_aggregated/part-00000";
  std::string test_name = ftrl_model.ftrl_params_.test_path + ftrl_model.ftrl_processor_.psid_ + "_aggregated/part-00000";

  int num_part = ftrl_model.ftrl_processor_.num_part_;
  int max_iter = ftrl_model.ftrl_params_.passes;
  size_t dim = ftrl_model.ftrl_params_.dim;
  std::vector<float> offset(dim, 0);
  std::vector<float> reg_offset(dim, 0);

  rabit::TrackerPrintf("ftrl execution \n");
  for (int i = 0; i < max_iter; ++i) {
    ::admm::SampleSet* train_set = new ::admm::SampleSet;
    ::admm::SampleSet* test_set = new ::admm::SampleSet;
    for (int part = 0; part < num_part; part++) {
      CHECK(train_set->Initialize(train_name, part, num_part));
      ftrl_model.ftrl_processor_.Run(*train_set, *test_set, offset, reg_offset);
    }
    delete train_set;

    CHECK(test_set->Initialize(test_name, 0, 1)); 
    std::vector<std::vector<float>> group_w;
    group_w.push_back(ftrl_model.ftrl_processor_.weight_);
    float loss = metrics.LogLoss(*test_set, group_w, false);
    float auc = metrics.Auc(*test_set, group_w, false);
    rabit::TrackerPrintf("%s | %.7f | %.7f | %d\n", &ftrl_model.ftrl_processor_.psid_[0], 
                          loss, auc, test_set->Size());
    delete test_set;
  }

  std::string ftrl_weight = ftrl_model.ftrl_params_.output_path + "ftrl_weight_" + ftrl_model.ftrl_processor_.psid_;
  auto *streamb(dmlc::Stream::Create(&ftrl_weight[0], "w"));
  ftrl_model.SaveState(streamb);
  delete streamb;

  std::string upload_params = ftrl_model.ftrl_params_.output_path + ftrl_model.ftrl_processor_.psid_ + ".bin";
  streamb = dmlc::Stream::Create(&upload_params[0], "w");
  ftrl_model.Save(streamb);
  delete streamb;

  rabit::Finalize();
  return 0;
}
