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
  void BaseUpdate(SampleSet& train_set, SampleSet& test_set, const AdmmConfig& admm_params);
  /*!
   * \brief update the bias weights of the single model.
   */
  void BiasUpdate(SampleSet& train_set, SampleSet& test_set, const AdmmConfig& admm_params);
  /*!
   * \brief update the langrange coefficients of the single model.
   */
  void LangrangeUpdate(const AdmmConfig& admm_params);
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
   * \brief worker's PSID
   */
  std::string psid_;
  /*!
   * \brief spliting number of data 
   */
  int num_part_;
  /*!
   * \brief spliting id of data 
   */
  int partid_;
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
  /*!
   * \brief the hold ftrl nt coefficients
   */
  std::vector<real_t> nt_vec_;
  /*!
   * \brief the hold ftrl zt coefficients
   */
  std::vector<real_t> zt_vec_;
  /*!
   * \brief indicate whether loading occurs
   */
  bool load;
};

} // namespace admm

