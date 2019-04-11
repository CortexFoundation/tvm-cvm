#ifndef CVM_RUNTIME_OP_H_
#define CVM_RUNTIME_OP_H_

#include <tvm/runtime/c_runtime_api.h>
#include <dmlc/registry.h>
#include <tvm/runtime/packed_func.h>

#include <string>

namespace tvm {
namespace runtime {
namespace cvm{

class OpRegistry;
class GenericOpMap;

class Op {
 public:
	std::string name;
	/*!
	 * \brief detailed description of the operator
	 *  This can be used to generate docstring automatically for the operator.
	 */
	std::string description;
	/*!
	 * \brief support level of the operator,
	 *  The lower the more priority it contains.
	 *  This is in analogies to BLAS levels.
	 */
	int32_t support_level = 10;
	/*!
	 * \brief Get additional registered attribute about operators.
	 *  If nothing has been registered, an empty OpMap will be returned.
	 * \param attr_name The name of the attribute.
	 * \return An OpMap of specified attr_name.
	 * \tparam ValueType The type of the attribute.
	 */
	template <typename ValueType>
	inline static GenericOpMap GetAttr(const std::string& attr_name);

 private:
	// friend class
	friend class GenericOpMap;
	friend class OpRegistry;
	// Program internal unique index of operator.
	// Used to help index the program.
	uint32_t index_{0};
};

/*! \brief Helper structure to register operators */
class OpRegistry {
 public:
	/*! \return the operator */
	const Op& op() const { return op_; }
  /*!
   * \brief setter function during registration
   *  Set the description of operator
   * \param descr the description string.
   * \return reference to self.
   */
  inline OpRegistry& describe(const std::string& descr) {
	  op_.description = descr;
	  return *this;
  }
  /*!
   * \brief Set the support level of op.
   * \param level The support level.
   * \return reference to self.
   */
  inline OpRegistry& set_support_level(int32_t level) {
    op_.support_level = level;
    return *this;
  }
  // set the name of the op to be the same as registry
  inline OpRegistry& set_name() {  // NOLINT(*)
    if (op_.name.length() == 0) {
      op_.name = name;
    }
    return *this;
  }
  /*!
   * \brief Register additional attributes to operator.
   * \param attr_name The name of the attribute.
   * \param value The value to be set.
   * \param plevel The priority level of this set,
   *  an higher priority level attribute
   *  will replace lower priority level attribute.
   *  Must be bigger than 0.
   *
   *  Cannot set with same plevel twice in the code.
   *
   * \tparam ValueType The type of the value to be set.
   */
  template <typename ValueType>
  inline OpRegistry& set_attr(const std::string& attr_name,  // NOLINT(*)
                              const ValueType& value, int plevel = 10) {
    CHECK_GT(plevel, 0) << "plevel in set_attr must be greater than 0";
    TVMRetValue rv;
    rv = value;
    UpdateAttr(attr_name, rv, plevel);
    return *this;
  }

  /*! \return The global single registry */
  TVM_DLL static ::dmlc::Registry<OpRegistry>* Registry();

 private:
  friend class ::dmlc::Registry<OpRegistry>;
  // the name
  std::string name;
  /*! \brief The operator */
  Op op_;
  // private constructor
  OpRegistry();
  // update the attribute OpMap
  TVM_DLL void UpdateAttr(const std::string& key, TVMRetValue value,
                          int plevel);
};

/*!
 * \brief Generic map to store additional information of Op.
 */
class GenericOpMap {
 public:
  /*!
   * \brief Check if the map has op as key.
   * \param op The key to the map
   * \return 1 if op is contained in map, 0 otherwise.
   */
  inline int count(const Op& op) const {
		const uint32_t idx = op.index_;
		return idx < data_.size() ? (data_[idx].second != 0) : 0;
	}
  /*!
   * \brief get the corresponding value element at op
   * \param op The key to the map
   * \return the const reference to the content value.
   */
  inline const TVMRetValue& operator[](const Op& op) const {
		const uint32_t idx = op.index_;
    CHECK(idx < data_.size() && data_[idx].second != 0)
        << "Attribute " << attr_name_ << " has not been registered for Operator "
        << op.name;
    return data_[idx].first;
	}
	
  /*!
   * \brief get the corresponding value element at op with default value.
   * \param op The key to the map
   * \param def_value The default value when the key does not exist.
   * \return the const reference to the content value.
   * \tparam ValueType The content value type.
   */
  template <typename ValueType>
  inline ValueType get(const Op& op, ValueType value) const {
		const uint32_t idx = op.index_;
    if (idx < data_.size() && data_[idx].second != 0) {
      return data_[idx].first;
    } else {
      return value;
	  }
  }

 private:
  friend class OpRegistry;
  // the attribute field.
  std::string attr_name_;
  // internal data
  std::vector<std::pair<TVMRetValue, int> > data_;
  // The value
  GenericOpMap() = default;
};

// internal macros to make
#define CVM_REGISTER_VAR_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::runtime::cvm::OpRegistry& __make_##CVMOp

/*!
 * \def RELAY_REGISTER_OP
 * \brief Register a new operator, or set attribute of the corresponding op.
 *
 * \param OpName The name of registry
 *
 * \code
 *
 *  RELAY_REGISTER_OP("add")
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .set_attr<OpKernel>("gpu_kernel", AddKernel);
 *
 * \endcode
 */
#define CVM_REGISTER_OP(OpName)                        \
  DMLC_STR_CONCAT(RELAY_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::runtime::cvm::OpRegistry::Registry()               \
          ->__REGISTER_OR_GET__(OpName)                  \
          .set_name()

}
}
}
#endif // CVM_RUNTIME_OP_H_
