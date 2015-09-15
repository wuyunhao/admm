#include <memory>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <rabit.h>
#include <vector>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include "sample_set.h"
#include "workers.h"

typedef std::size_t IndexType;
typedef ::dmlc::Row<IndexType> Row;
typedef ::dmlc::Worker Worker;

class Auc : public dmlc::Serializable {
 public:
  std::vector<size_t> item_count_;
  std::vector<size_t> positive_predict_;
  std::vector<size_t> negative_predict_;

  void Load(dmlc::Stream *fi) {
  }
  void Save(dmlc::Stream *fo) const {
  }
  void InitAuc() {
    item_count_.resize(2); 
    item_count_[0] = 0;
    item_count_[1] = 0;
  }
};

float Predict(Row& x, Worker& worker_processor) {
  std::vector<float> sum_weight = worker_processor.base_vec_;
  for (size_t i = 0; i < sum_weight.size(); ++i) {
    sum_weight[i] += worker_processor.bias_vec_[i];
  }
  auto* ptr_weight = &sum_weight[0];
  auto inner_product = x.SDot(ptr_weight, sum_weight.size());

  return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
}

int main(int argc, char* argv[]) {
  sample_set.Rewind();
  Auc auc;
  auc.InitAuc();

  auto auc_compute = [&]()
  {
    while(sample_set.Next()) {
      Row x = sample_set.GetData(); 
      if (x.label > 0) {
        ++auc.item_count_[0];
      } else {
        ++auc.item_count_[1];
      }
    }
  };

  size_t p_max = auc.item_count_[0];
  size_t n_max = auc.item_count_[1];

  rabit::Allreduce<op::Sum>(&auc.item_count_[0], 2, auc_compute);
  rabit::Allreduce<op::Max>(&p_max, 1);
  rabit::Allreduce<op::Max>(&n_max, 1);

  auc.positive_predict_.resize(p_max * rabit::GetWorldSize(), 0);
  auc.negative_predict_.resize(n_max * rabit::GetWorldSize(), 0);

  sample_set.Rewind();
  size_t pos = rabit::GetRank()*p_max;
  size_t neg = rabit::GetRank()*n_max;
  while(sample_set.Next()) {
    auto x = sample_set.GetData();
    if (x.label > 0) {
      auc.positive_predict_[pos++] = Predict(x, local_model.worker_processor_);
    } else {
      auc.negative_predict_[neg++] = Predict(x, local_model.worker_processor_);
    } 
  }

  rabit::Allreduce<op::Sum>(&auc.positive_predict_[0], auc.positive_predict_.size());
  rabit::Allreduce<op::Sum>(&auc.negative_predict_[0], auc.negative_predict_.size());
  
  std::sort(auc.positive_predict_.begin(), auc.positive_predict_.end());
  std::sort(auc.negative_predict_.begin(), auc.negative_predict_.end());
  
  long int sorted_sum = 0;
  for (auto i = auc.item_count_[0]-1; i >= 0; --i) {
    for (auto j = auc.item_count_[1]-1; j >= 0; --j) {
      if (auc.positive_predict_[i] > auc.negative_predict_[j]) {
        sorted_sum += (j+1)*(auc.item_count_[0] - i) ;
        break;
      } 
    }
  }
  
  if (rabit::GetRank() == 0)
    printf("AUC of the progress is: %f", sorted_sum / (auc.item_count_[0]*auc.item_count_[1]);

}


