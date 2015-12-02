#pragma once
#include <cmath>
#include <vector>
#include <dmlc/data.h>
#include "sample_set.h"
#include "config.h"

namespace ftrl{
	
/*!
 * \brief FTRL_Proximal solver.
 * 
 *  For training Logistic regression with L_1 and 
 *  L_2 regularization
 */

class FtrlSolver {
 public:
  typedef float real_t;
  typedef std::size_t IndexType;
  typedef ::dmlc::Row<IndexType> Row;
  typedef ::admm::SampleSet SampleSet;
  /*!
   * \brief Initiate with l_1, l_2, num_iter, alpha, beta
   */
  FtrlSolver(real_t lambda_1,
			 real_t lambda_2,
			 real_t alpha_init,
			 real_t beta_init,
			 std::size_t dim_init);
  FtrlSolver();
  ~FtrlSolver();
  
  void Init(::admm::FtrlConfig &params);
  /*!
   * \brief Compute the value returned by logistic function. 
   *
   * \param x the processed training instance.
   * \param offset the intercept weights.
   * \return prediction of instance x.
   */
  real_t Predict(Row& x, const std::vector<real_t>& offset, const std::vector<real_t>& reg_offset);

  /*!
   * \brief assignment for the weights
   *
   * \param x the target which the weights are assigned with.
   */
  void Assign(const std::vector<real_t>& x, const std::vector<real_t>& y);
  /*!
   * \brief update model using ftrl algorithm
   *
   * \param x the processed training instance
   * \param predict the prediction of x
   */
  void Update(real_t predict, const Row& x, const std::vector<real_t>& reg_offset);

  /*!
   * \brief get the weight solution
   */
  std::vector<real_t> weight() const;

  /*!
   * \brief process the update with specified passes
   *
   * \param the samples set corresponding the model
   */
  void Run(SampleSet& train_set, SampleSet& test_set, const std::vector<real_t>& offset, const std::vector<real_t>& reg_offset);
 //protected:
  /*!
   * \brief the weights 
   */
  std::vector<real_t> weight_;
  /*!
   * \brief the median weights
   */
  std::vector<real_t> mid_weight_;

  /*!
   * \brief the squared sum of past gradient
   */
  std::vector<real_t> squared_sum_;
  /*!
   * \brief l_1 regularized coefficient
   */
  real_t l_1_;
  /*!
   * \brief l_2 regularized coefficient
   */
  real_t l_2_;
  /*!
   * \brief alpha for Per-coordinate learning rate
   */
  real_t alpha_;
  /*!
   * \brief beta for Per-coordinate learning rate
   */
  real_t beta_;
  /*!
   * \brief the dimension of the weights
   */
  std::size_t dim_;
  /*!
   * \brief the splitting number of data
   */
  int num_part_;
  /*!
   * \brief the PSID
   */
  std::string psid_;

};
} // namespace ftrl
