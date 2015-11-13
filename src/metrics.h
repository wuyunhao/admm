#pragma once
#include <vector>
#include "sample_set.h"

/*!
 * \brief measurements for ML algorithms
 */
class Metrics {
 public:
  typedef float real_t;

  Metrics();
  ~Metrics();

  /*!
   * \brief tied rank function
   * \params input the vector would be ranked
   * \params output the ranked result
   */
  void Sort(const std::vector<real_t>& input, std::vector<real_t>& output);
  /*!
   * \brief compute auc 
   * \pramas ranks the log likelihood vector
   * \params labels the corresponding labels of ranks
   */
  real_t Auc(::admm::SampleSet& sample_set, std::vector<std::vector<real_t>>& weights, bool T);
  real_t Auc(::admm::SampleSet& sample_set,
             std::vector<std::vector<real_t>>& weight_a,
             std::vector<std::vector<real_t>>& weight_b,
             real_t ratio,
             bool T); 
  /*!
   * \brief calculate logloss 
   * \params sample_set the sample set used
   * \params weights the weight vectors
   */
  real_t LogLoss(::admm::SampleSet& sample_set, std::vector<std::vector<real_t>>& weights, bool T);
};
