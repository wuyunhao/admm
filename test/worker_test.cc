
#include <vector>
#include <dmlc/data.h>
#include "gtest.h"
#include "../src/workers.h"
#include "../src/master.h"
#include "../src/config.h"

namespace admm{
    
class WorkerTest: public ::testing::Test{
 public:
  virtual void SetUp();
  virtual void TearDown();

 protected:
};

void WorkerTest::SetUp(){
}

void WorkerTest::TearDown(){
}

TEST_F(WorkerTest, update){
   typedef float real_t;

   SampleSet sample_set;
   sample_set.Initialize("test/wtest.libsvm",0,1);

   AdmmConfig admm_params(1,1,1,5); 

   Worker worker_processor(admm_params);
   Master master_processor;

   //update of the local base and bias weights
   worker_processor.BiasUpdate(sample_set, admm_params);
   worker_processor.BaseUpdate(sample_set, admm_params);

   //test the local base weights
   std::vector<real_t> result = worker_processor.GetWeights(admm_params);
   //TODO
   for(auto i = 0u; i < admm_params.global_weights.size(); ++i)
        EXPECT_DOUBLE_EQ(0,result[i]);

   //update of the global weights
   std::vector<std::vector<real_t>> RESULT;
   RESULT.push_back(result);
   master_processor.global_update(RESULT, admm_params);
   //test the global weights
   //TODO
   for(auto i = 0u; i < admm_params.global_weights.size(); ++i)
        EXPECT_DOUBLE_EQ(0, admm_params.global_weights[i]);

   //update of the langranges
   worker_processor.LangrangeUpdate(sample_set, admm_params);
   //test the langranges
   result = worker_processor.GetWeights(admm_params);
   //TODO
   for(auto i = 0u; i < admm_params.global_weights.size(); ++i)
   EXPECT_DOUBLE_EQ(0,result[i]);
}
}
