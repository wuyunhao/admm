
#include <vector>
#include <dmlc/data.h>
#include "gtest.h"
#include "../src/ftrl.h"

namespace ftrl{
    
class FtrlTest : public ::testing::Test{
 public:
  virtual void SetUp();
  virtual void TearDown();

 protected:
};

void FtrlTest::SetUp(){
}

void FtrlTest::TearDown(){
}

TEST_F(FtrlTest, Predict){
   typedef float real_t;
   typedef std::size_t IndexType;
   typedef ::dmlc::Row<IndexType> Row;
   
   FtrlSolver fs(0.1f, 0.2f, 1.0f, 1.0f, 1u, 5u);
   real_t val[] = {1.0, 2.0, 1.4};
   IndexType index[] = {1, 3, 4};
   Row Inst;
   Inst.value = &val[0];
   Inst.index = &index[0];
   Inst.length = 3;
   auto prediction1 = fs.Predict(Inst);
   
   EXPECT_DOUBLE_EQ(0.5, prediction1);

   std::vector<real_t> x(5, 2.0);
   fs.Assign(x);
   std::vector<real_t> repeat = fs.weight();
   EXPECT_EQ(repeat.size(),x.size());	
   for(auto i = 0u; i < x.size(); i++){
		EXPECT_DOUBLE_EQ(repeat[i], x[i]);
   }

   auto prediction2 = fs.Predict(Inst);

   EXPECT_DOUBLE_EQ(0.5, prediction2);
}

}
