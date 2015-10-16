#include "metrics.h"
#include <algorithm>
#include <utility>
#include <rabit.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>

Metrics::Metrics() {
}

Metrics::~Metrics() {
}

void Metrics::Sort(const std::vector<Metrics::real_t>& input, std::vector<Metrics::real_t>& output) {
  int dim = (int)input.size();
  std::vector<std::pair<real_t, int>> items;
  items.resize(dim);
  output.resize(dim);

  for (int i = 0; i < dim; ++i) {
    items[i].first = input[i];
    items[i].second = i;
  }

  std::stable_sort(items.begin(), items.end(), [](const std::pair<real_t,int> &a, const std::pair<real_t,int> &b){ return a.first < b.first;});
  
  int last_rank = 0;
  std::pair<real_t, int> cur_item = items[0];              
  for (int i = 0; i < dim; ++i) {
    if (cur_item.first != items[i].first) {
      cur_item = items[i];
      for (int j = last_rank; j < i; ++j) {
        output[items[j].second] = (last_rank + i + 1.0f)/2;
      }
      last_rank = i;
    }
    if (i == dim - 1) {
      for (int j = last_rank; j < i+1; ++j) {
        output[items[j].second] = (last_rank + i + 2.0f)/2;
      }
    }
  }
}

Metrics::real_t Metrics::Auc(::admm::SampleSet& sample_set, std::vector<std::vector<real_t>>& weights, bool T) {
  std::vector<real_t> ranks;
  std::vector<int> labels;

  sample_set.Rewind();
  while(sample_set.Next()) {
    auto x = sample_set.GetData();
    auto inner_product = x.SDot(&weights[0][0], weights[0].size());
    for (size_t i = 1; i < weights.size(); ++i) {
      inner_product += x.SDot(&weights[i][0], weights[i].size());
    }
    auto predict = 1.0f/(1 + exp(- std::max(std::min(inner_product, 35.0f), - 35.0f)));
    ranks.push_back(predict);
    labels.push_back((int)x.label);
  }

  std::vector<real_t> sorted_ranks;
  Sort(ranks, sorted_ranks);
  real_t positive_sum = 0;
  real_t total_p = 0;
  real_t total_n = 0;
  for (size_t i = 0; i < ranks.size(); ++i) {
    if (labels[i] == 1) {
      total_p++;
      positive_sum += sorted_ranks[i];
    } else {
      total_n++;
    }
  }

  auto auc = positive_sum/(total_p*total_n) - (total_p + 1)/(2*total_n);
  if (T)
    rabit::TrackerPrintf("The %d processor Train AUC is: %f \n", rabit::GetRank(), auc);
  else
    rabit::TrackerPrintf("The %d processor Test AUC is: %f \n", rabit::GetRank(), auc);
  return auc;
}

Metrics::real_t Metrics::LogLoss(::admm::SampleSet& sample_set, std::vector<std::vector<Metrics::real_t>>& weights, bool T) {
  real_t sum = 0;
  int count = 0;

  sample_set.Rewind();
  while(sample_set.Next()) {
    ::dmlc::Row<std::size_t> x = sample_set.GetData();
    auto inner_product = x.SDot(&weights[0][0], weights[0].size());
    for (size_t i = 1; i < weights.size(); ++i) {
      inner_product += x.SDot(&weights[i][0], weights[i].size());
    }
    auto predict = 1.0f/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35))));
    sum += (int)x.label == 1? -log(predict): -log(1 - predict);
    count++;
  }
  
  if (T) 
    rabit::TrackerPrintf("The %d processor Train LogLoss is %f \n", rabit::GetRank(), sum/count);
  else 
    rabit::TrackerPrintf("The %d processor Test LogLoss is %f \n", rabit::GetRank(), sum/count);
  return sum/count;
}
