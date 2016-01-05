
#include "sample_set.h"
#include <dmlc/logging.h>

namespace admm {

SampleSet::SampleSet() {
}

SampleSet::~SampleSet() {
}

bool SampleSet::Initialize(const std::string& uri,
    uint32_t part_index, uint32_t num_parts) {
  auto *fs = dmlc::Stream::Create(uri.c_str(), "r", true);
  if (fs == NULL) {
    rabit::TrackerPrintf("Fail open %s\n", uri.c_str());
    LOG(ERROR) << "Failed open " << uri << " as libsvm formatted file.";
    return false;
  }
  delete fs;
  fs = NULL;
  rbiter_.reset();
  auto rbiter = ::dmlc::RowBlockIter<IndexType>::Create(
      uri.c_str(), part_index, num_parts, "libsvm");
  rbiter_.reset(rbiter);
  rb_size_ = -1;
  current_index_ = -1;
  return true;
}

bool SampleSet::Next() {
  ++current_index_;
  //LOG(INFO) << "current_index_:" << current_index_ << "\n";
  //LOG(INFO) << "rb_size_:" << rb_size_ << "\n";
  if (current_index_ >= rb_size_) {
    if (!rbiter_->Next()) {
      return false;
    }
    rb_size_ = rbiter_->Value().size;
    current_index_ = 0;
  //LOG(INFO) << "NEW RBSIZE " << rb_size_ << ", new index " << current_index_ << "\n";
  }
  return true;
}

::dmlc::Row<SampleSet::IndexType> SampleSet::GetData() {
  //LOG(INFO) << "Getting data, index: " << current_index_ << ", blocksize " << rbiter_->Value().size;
  auto x = rbiter_->Value()[current_index_];
  return x; 
}

void SampleSet::Rewind() {
  rbiter_->BeforeFirst();
  current_index_ = -1;
  rb_size_ = -1;
}

::dmlc::Row<SampleSet::IndexType> SampleSet::GetLastData() {
  if (current_index_ >= 0 && current_index_ == rb_size_) {
    return rbiter_->Value()[current_index_ - 1];
  }
  else {
    LOG(ERROR) << "failed to get the last data. ";
    ::dmlc::Row<IndexType> last;
    return last;
  }
}

::dmlc::Row<SampleSet::IndexType> SampleSet::TranslateData(const ::dmlc::Row<SampleSet::IndexType>& x) {
  ::dmlc::Row<IndexType> new_x;
  new_x.label = x.label;
  new_x.weight = x.weight;
  new_x.length = 1 + x.length;

  IndexType* new_x_index = new IndexType[new_x.length];
  float* new_x_value = new float[new_x.length];
  
  new_x_index[0] = 0;
  new_x_value[0] = 1;
  for (size_t i = 0; i < x.length; ++i) {
    new_x_index[i+1] = x.index[i] + 1;
    new_x_value[i+1] = x.value[i];
  }
  new_x.index = new_x_index;
  new_x.value = new_x_value;
  return new_x;
}

int SampleSet::Size() {
  return rb_size_;
}

}  // namespace admm
