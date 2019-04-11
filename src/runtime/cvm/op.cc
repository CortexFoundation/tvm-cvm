#include "op.h"

#include <mutex>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(::tvm::relay::OpRegistry);
}  // namespace dmlc

namespace tvm{
namespace runtime{
namespace cvm{

::dmlc::Registry<OpRegistry>* OpRegistry::Registry() {
  return ::dmlc::Registry<OpRegistry>::Get();
}

// single manager of operator information.
struct OpManager {
  // mutex to avoid registration from multiple threads.
  std::mutex mutex;
  // global operator counter
  std::atomic<int> op_counter{0};
  // storage of additional attribute table.
  std::unordered_map<std::string, std::unique_ptr<GenericOpMap>> attr;
  // frontend functions
  std::vector<PackedFunc*> frontend_funcs;
  // get singleton of the op manager
  static OpManager* Global() {
    static OpManager inst;
    return &inst;
  }
};

OpRegistry::OpRegistry() {
  OpManager* mgr = OpManager::Global();
  index_ = mgr->op_counter++;
}

void OpRegistry::UpdateAttr(const std::string& key,
                            TVMRetValue value,
                            int plevel) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  std::unique_ptr<GenericOpMap>& op_map = mgr->attr[key];
  if (op_map == nullptr) {
    op_map.reset(new GenericOpMap());
    op_map->attr_name_ = key;
  }
  uint32_t index = op_.index_;
  if (op_map->data_.size() <= index) {
    op_map->data_.resize(index + 1, std::make_pair(TVMRetValue(), 0));
  }
  std::pair<TVMRetValue, int>& p = op_map->data_[index];
  CHECK(p.second != plevel)
      << "Attribute " << key << " of operator " << this->name
      << " is already registered with same plevel=" << plevel;
  if (p.second < plevel) {
    op_map->data_[index] = std::make_pair(value, plevel);
  }
}

// Get attribute map by key
const GenericOpMap& Op::GetAttr(const std::string& key) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex> lock(mgr->mutex);
  auto it = mgr->attr.find(key);
  if (it == mgr->attr.end()) {
    LOG(FATAL) << "Operator attribute \'" << key << "\' is not registered";
  }
  return *it->second.get();
}

}
}
}
