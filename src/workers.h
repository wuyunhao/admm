#pragma once
#include <vector>
#include <dmlc/logging.h>
#include <dmlc/data.h>
#include "sample_set.h"
#include "config.h"

namespace admm {
/*!
 * \brief Worker executing single logistic regression update
 * 
 * \param w_vec the weight vector for the task
 * \param v_vec the offset vector for the task
 * \param alpha_vec the langrange efficient vector for the task
 */
class Worker {
 public:
  typedef float real_t;
  
  Worker();
  virtual ~Worker();
  void InitWorker(std::size_t fdim);
  /*!
   * \brief update the base weights of the single model.
   * \param batches the traning instances specified by the task 
   * \param params the configure parameters
   */
  void BaseUpdate(SampleSet& sample_set, const AdmmConfig& admm_params, SampleSet& test_set);
  /*!
   * \brief update the bias weights of the single model.
   */
  void BiasUpdate(SampleSet& sample_set, const AdmmConfig& admm_params, SampleSet& test_set);
  /*!
   * \brief update the langrange coefficients of the single model.
   */
  void LangrangeUpdate(const SampleSet& sample_set, const AdmmConfig& admm_params);
  /*!
   * \brief return the final base weights and langranges of the single model
   */
  void GetWeights(AdmmConfig& admm_params, std::vector<real_t>& ptr) const;
  /*!
   * \brief compute the logloss for the current weights solution;
   */
  void LogLoss(SampleSet& sample_set, const AdmmConfig& admm_params, bool T);
 //private:
  /*!
   * \brief worker's ID
   */
  std::size_t id_;
  /*!
   * \brief the base weights
   */
  std::vector<real_t> base_vec_;
  /*!
   * \brief the bias weights
   */
  std::vector<real_t> bias_vec_;
  /*!
   * \brief the langrange coefficients
   */
  std::vector<real_t> langr_vec_;
};

} // namespace admm

