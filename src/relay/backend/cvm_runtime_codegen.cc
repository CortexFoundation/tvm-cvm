#include <dmlc/any.h>
#include <dmlc/json.h>
#include <tvm/expr_operator.h>
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>

#include <list>
#include <string>
#include <vector>

#include "utils.h"
#include "../../lang/attr_functor.h"

namespace tvm {
namespace relay {
namespace backend {

class CVMGraphNode;
class CVMGraphInputNode;
class CVMGraphOpNode;

using ShapeVector = std::vector<std::vector<int64_t> >;
using CVMGraphAttrs = std::unordered_map<std::string, dmlc::any>;
using CVMGraphNodePtr = std::shared_ptr<CVMGraphNode>;
using CVMGraphInputNodePtr = std::shared_ptr<CVMGraphInputNode>;
using CVMGraphOpNodePtr = std::shared_ptr<CVMGraphOpNode>;

/*!
 * \brief Attribute printer which prints the attributes in the call.
 */
class CVMAttrSerialization :
    public AttrVisitor,
    public AttrFunctor<void(const NodeRef&)> {
 public:
  CVMAttrSerialization(std::ostream& os,
                    std::string sep=", ")
      : os_(os), sep_(sep) {}

  void Initialize() { os_ << "{"; init_ = true; }
  void Finalize() { os_ << "}"; }
  void AppendSeparator() { 
    if (not init_) { os_ << sep_; }
    else { init_ = false; }
  }

  template<typename T>
  void PrintKV(const char* key, const T& value) {
    AppendSeparator();
    os_ << "\"" << key << "\":\"" << value << "\"";
  }

  std::string Bool2String(bool value) {
    return value ? "true" : "false";
  }

  void Visit(const char* key, double* value) final {
    PrintKV(key, value[0]);
  }
  void Visit(const char* key, int64_t* value) final {
    PrintKV(key, value[0]);
  }
  void Visit(const char* key, uint64_t* value) final {
    PrintKV(key, value[0]);
  }
  void Visit(const char* key, int* value) final {
    PrintKV(key, value[0]);
  }
  void Visit(const char* key, bool* value) final {
    PrintKV(key, Bool2String(value[0]));
  }
  void Visit(const char* key, std::string* value) final {
    PrintKV(key, value[0]);
  }
  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "do not allow void as argument";
  }
  void Visit(const char* key, NodeRef* value) final {
    AppendSeparator();
    os_ << "\"" << key << "\":\"";
    PrintAttr(value[0]);
    os_ << "\"";
  }
  void Visit(const char* key, DataType* value) final {
    PrintKV(key, runtime::TVMType2String(Type2TVMType(value[0])));
  }
  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "do not allow NDarray as argument";
  }
  void Visit(const char* key, runtime::Object* obj) final {
    LOG(FATAL) << "do not allow Object as argument";
  }

  void PrintAttr(const NodeRef& value, bool meta = false) {
    if (value.defined()) {
      VisitAttr(value);
    }
  }

  template<typename T>
  void PrintConstScalar(DataType dtype, const T* data) {
    if (dtype == Int(32)) {
      os_ << data[0];
    } else if (dtype == Float(32)) {
      os_ << data[0] << 'f';
    } else if (dtype == Bool()) {
      os_ << Bool2String(data[0] != 0);
    } else {
      os_ << dtype << "(" << data[0] << ")";
    }
  }

  void VisitAttrDefault_(const Node* op) final {
    PrintAttr(GetRef<NodeRef>(op), true);
  }

  void VisitAttr_(const ArrayNode* op) final {
    os_ << "[";
    bool first = true;
    for (NodePtr<Node> val : op->data) {
      if (not first) os_ << ",";
      else first = false;
      PrintAttr(NodeRef(val));
    }
    os_ << "]";
  }

  void VisitAttr_(const ir::IntImm* op) final {
    PrintConstScalar(op->type, &(op->value));
  }

  void VisitAttr_(const ir::UIntImm* op) final {
    PrintConstScalar(op->type, &(op->value));
  }

  void VisitAttr_(const ir::FloatImm* op) final {
    PrintConstScalar(op->type, &(op->value));
  }

  void VisitAttr_(const ir::StringImm* op) final {
    os_ << op->value;
  }

 private:
  std::ostream& os_;
  std::string sep_;
  bool init_;
};

struct CVMOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

/*! \brief Node types */
enum CVMGraphNodeType {
  kCVMGraphNop,
  kCVMGraphInputNode,
  kCVMGraphOpNode,
};

/*! \brief Base Node class */
class CVMGraphNode {
 public:
  CVMGraphNode() {}
  virtual void Save(dmlc::JSONWriter* writer) const {}
  virtual void Load(dmlc::JSONReader* reader) {}
  virtual CVMGraphNodeType Type() const { return kCVMGraphNop; }
  virtual ~CVMGraphNode() {}

 public:
  int num_outputs_{1};
  std::string name_;
  CVMGraphAttrs attrs_;
};

class CVMGraphNodeRef {
 public:
  CVMGraphNodeRef() {}
  CVMGraphNodeRef(int ident, int index, int version = 0)
    : ident_(ident), index_(index), version_(version) {}


  inline void Save(dmlc::JSONWriter* writer) const {
    std::list<int> ref_list{ ident_, index_, version_ };
    writer->Write(ref_list);
  }

  inline void Load(dmlc::JSONReader* reader) {
    LOG(FATAL) << "Not implemented.";
  }

 protected:
  int ident_;
  int index_{0};
  int version_{0};
};

/*! \brief Input Node */
class CVMGraphInputNode : public CVMGraphNode {
 public:
  CVMGraphInputNode() {}
  CVMGraphInputNode(const std::string& name, const CVMGraphAttrs& attrs) {
    name_ = name;
    attrs_ = attrs;
  }

  CVMGraphNodeType Type() const override { return kCVMGraphInputNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    const std::string op_name{"null"};
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_name);
    writer->WriteObjectKeyValue("name", this->name_);
    writer->WriteObjectKeyValue("inputs", std::list<int>());
    writer->EndObject();
  }
  static std::shared_ptr<CVMGraphNode> make_node_ptr(const std::string& name,
                                                  const CVMGraphAttrs& attrs) {
    auto ptr = std::make_shared<CVMGraphInputNode>(name, attrs);
    return std::dynamic_pointer_cast<CVMGraphNode>(ptr);
  }
};

/*! \brief Op Node */
class CVMGraphOpNode : public CVMGraphNode {
 public:
  CVMGraphOpNode() {}
  CVMGraphOpNode(const std::string& name,
              const CVMGraphAttrs& attrs,
              const std::string& op_name,
              const std::vector<CVMGraphNodeRef>& inputs,
              size_t num_outputs = 1) {
    name_ = name;
    op_name_ = op_name;
    inputs_ = inputs;
    attrs_ = attrs;
    num_outputs_ = num_outputs;
  }

  CVMGraphNodeType Type() const override { return kCVMGraphOpNode; }

  void Save(dmlc::JSONWriter* writer) const override {
    CVMGraphAttrs attrs;
    attrs["func_name"] = this->op_name_;
    attrs["flatten_data"] = std::string("0");
    attrs["num_inputs"] = std::to_string(this->inputs_.size());
    attrs["num_outputs"] = std::to_string(this->num_outputs_);
    writer->BeginObject();
    writer->WriteObjectKeyValue("op", op_type_name_);
    writer->WriteObjectKeyValue("name", name_);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("inputs", this->inputs_);
    writer->EndObject();
  }
  static std::shared_ptr<CVMGraphNode> make_node_ptr(const std::string& name,
                                                  const CVMGraphAttrs& nd_attrs,
                                                  const std::string& op_name,
                                                  const std::vector<CVMGraphNodeRef>& inputs,
                                                  size_t num_outputs = 1) {
    auto ptr = std::make_shared<CVMGraphOpNode>(name, nd_attrs, op_name, inputs, num_outputs);
    return std::dynamic_pointer_cast<CVMGraphNode>(ptr);
  }

 public:
  std::string op_name_;
  std::vector<CVMGraphNodeRef> inputs_;

 private:
  const std::string op_type_name_{"cvm_op"};
};

class CVMCodegen
  : public ExprFunctor<std::vector<CVMGraphNodeRef>(const Expr&)> {
 public:
  CVMOutput Codegen(relay::Function func) {
    for (auto param : func->params) {
      auto node_ptr = CVMGraphInputNode::make_node_ptr(param->name_hint(), CVMGraphAttrs());
      var_map_[param.get()] = AddNode(node_ptr, param);
    }

    heads_ = VisitExpr(func->body);
    std::ostringstream os;
    dmlc::JSONWriter writer(&os);
    GetJSON(&writer);
    CVMOutput ret;
    ret.graph_json = os.str();
    ret.params = params_;
    return ret;
  }

 protected:
  /*!
   * \brief Extract shape from expr to vector<int64_t>
   *
   * \param shape
   * \return std::vector<int64_t>
   */
  std::vector<int64_t> _ShapeToJSON(tvm::Array<HalideIR::Expr> shape) {
    std::vector<int64_t> ret;
    for (IndexExpr dim : shape) {
      const int64_t* pval = as_const_int(dim);
      ret.push_back(*pval);
    }
    return ret;
  }

  /*!
   * \brief Add node to graph
   *
   * \param node
   * \param expr
   * \return std::vector<_NodeRef>
   */
  std::vector<CVMGraphNodeRef> AddNode(CVMGraphNodePtr node, Expr expr) {
    auto checked_type = expr->checked_type();
    auto node_id = nodes_.size();
    nodes_.push_back(node);

    // Tuple return value, flatten as tuple
    if (const auto* tuple_type = checked_type.as<TupleTypeNode>()) {
      std::vector<CVMGraphNodeRef> ret;
      ShapeVector shape;
      std::vector<std::string> dtype;
      std::vector<int> precision;
      for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        if (const auto* typ = tuple_type->fields[i].as<TensorTypeNode>()) {
          ret.push_back(CVMGraphNodeRef(node_id, i));
          shape.emplace_back(_ShapeToJSON(typ->shape));
          dtype.emplace_back("int32");
          // dtype.emplace_back(DType2String(typ->dtype));
          precision.emplace_back(typ->precision);
        } else {
          LOG(FATAL) << "type " << checked_type->type_key() << " not supported";
        }
      }
      CHECK_EQ(node->Type(), kCVMGraphOpNode);
      auto op_nd = std::dynamic_pointer_cast<CVMGraphOpNode>(node);
      op_nd->attrs_["shape"] = shape;
      op_nd->attrs_["dtype"] = dtype;
      op_nd->attrs_["precision"] = precision;
      op_nd->num_outputs_ = tuple_type->fields.size();
      return ret;
    }
    // Normal tensor return type
    if (const auto* tensor_type = checked_type.as<TensorTypeNode>()) {
      ShapeVector shape;
      std::vector<std::string> dtype;
      std::vector<int> precision;
      shape.emplace_back(_ShapeToJSON(tensor_type->shape));
      dtype.emplace_back("int32");
      // dtype.emplace_back(DType2String(tensor_type->dtype));
      precision.emplace_back(tensor_type->precision);
      node->attrs_["shape"] = shape;
      node->attrs_["dtype"] = dtype;
      node->attrs_["precision"] = precision;
    } else {
      LOG(FATAL) << "type " << checked_type->type_key() << " not supported";
    }
    return {CVMGraphNodeRef(node_id, 0)};
  }

  /*! \brief Visitors */
  std::unordered_map<Expr, std::vector<CVMGraphNodeRef>, NodeHash, NodeEqual> visitor_cache_;
  std::vector<CVMGraphNodeRef> VisitExpr(const Expr& expr) final {
    if (visitor_cache_.count(expr)) return visitor_cache_.at(expr);
    std::vector<CVMGraphNodeRef> res;
    if (expr.as<ConstantNode>()) {
      res = VisitExpr_(expr.as<ConstantNode>());
    } else if (expr.as<TupleNode>()) {
      res = VisitExpr_(expr.as<TupleNode>());
    } else if (expr.as<VarNode>()) {
      res = VisitExpr_(expr.as<VarNode>());
    } else if (expr.as<GlobalVarNode>()) {
      res = VisitExpr_(expr.as<GlobalVarNode>());
    } else if (expr.as<FunctionNode>()) {
      res = VisitExpr_(expr.as<FunctionNode>());
    } else if (expr.as<CallNode>()) {
      res = VisitExpr_(expr.as<CallNode>());
    } else if (expr.as<LetNode>()) {
      res = VisitExpr_(expr.as<LetNode>());
    } else if (expr.as<IfNode>()) {
      res = VisitExpr_(expr.as<IfNode>());
    } else if (expr.as<OpNode>()) {
      res = VisitExpr_(expr.as<OpNode>());
    } else if (expr.as<TupleGetItemNode>()) {
      res = VisitExpr_(expr.as<TupleGetItemNode>());
    } else if (expr.as<RefCreateNode>()) {
      res = VisitExpr_(expr.as<RefCreateNode>());
    } else if (expr.as<RefReadNode>()) {
      res = VisitExpr_(expr.as<RefReadNode>());
    } else if (expr.as<RefWriteNode>()) {
      res = VisitExpr_(expr.as<RefWriteNode>());
    } else if (expr.as<ConstructorNode>()) {
      res = VisitExpr_(expr.as<ConstructorNode>());
    } else if (expr.as<MatchNode>()) {
      res = VisitExpr_(expr.as<MatchNode>());
    } else {
      LOG(FATAL) << "Not recognized expr: " << expr;
    }
    visitor_cache_[expr] = res;
    return res;
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const VarNode* op) final {
    Expr expr = GetRef<Expr>(op);
    return var_map_[expr.get()];
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const ConstantNode* op) final {
    Expr expr = GetRef<Expr>(op);
    size_t index = params_.size();
    LOG(WARNING) << "CVM Dump with constant node";
    std::string name = "p" + std::to_string(index);
    params_[name] = op->data;
    auto node = CVMGraphInputNode::make_node_ptr(name, CVMGraphAttrs());
    return AddNode(node, expr);
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const TupleNode* op) final {
    std::vector<CVMGraphNodeRef> fields;
    for (auto field : op->fields) {
      auto ref_vec = VisitExpr(field);
      for (auto ref : ref_vec) {
        fields.push_back(ref);
      }
    }
    return fields;
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const CallNode* op) final {
    Expr expr = GetRef<Expr>(op);
    Function func;
    if (op->op.as<OpNode>()) {
      LOG(FATAL) << "Operators should be transformed away; try applying"
                 << "the fuse_ops transformation to the expression.";
    } else if (op->op.as<GlobalVarNode>()) {
      LOG(FATAL) << "Not implemented";
    } else if (op->op.as<FunctionNode>()) {
      func = GetRef<Function>(op->op.as<FunctionNode>());
    } else {
      LOG(FATAL) << "TVM runtime does not support calls to " << op->op->type_key();
    }
    if (!func->IsPrimitive()) {
      LOG(FATAL) << "TVM only support calls to primitive functions "
                 << "(i.e functions composed of fusable operator invocations)";
    }

    std::vector<CVMGraphNodeRef> inputs;
    for (auto& arg : op->args) {
      auto res = this->VisitExpr(arg);
      for (auto nr : res) {
        inputs.push_back(nr);
      }
    }

    if (func->body.as<CallNode>()) {
      const auto& call_node = func->body.as<CallNode>();
      CHECK(call_node->op.as<OpNode>())
          << "Primitive function only allows call into primitive ops";
      std::string op_name = call_node->op.as<OpNode>()->name;
      std::stringstream ss;
      CVMAttrSerialization as(ss);
      as.Initialize();
      if (call_node->attrs.defined()) {
        // call_node->attrs->GetNodePtr()->VisitAttrs(&as);
        const_cast<BaseAttrsNode*>(call_node->attrs.operator->())->VisitNonDefaultAttrs(&as);
      }
      as.Finalize();
      std::string attrs = ss.str();
      auto node = CVMGraphOpNode::make_node_ptr(op_name,
                                             CVMGraphAttrs(),
                                             op_name,
                                             inputs);
      node->attrs_["op_attrs"] = attrs;
      return AddNode(node, expr);
    }

    LOG(FATAL) << "CallNode is not OpNode " << expr << std::endl;
    return this->VisitExpr(op->op);
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const LetNode* op) final {
    CHECK_EQ(var_map_.count(op->var.get()), 0);
    var_map_[op->var.get()] = VisitExpr(op->value);
    return VisitExpr(op->body);
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const TupleGetItemNode* op) final {
    auto vtuple = VisitExpr(op->tuple);
    return {vtuple[op->index]};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "FunctionNode not implemented " << GetRef<Expr>(op);
    return {};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const GlobalVarNode* op) final {
    LOG(FATAL) << "GlobalVarNode not implemented " << GetRef<Expr>(op);
    return {};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const IfNode* op) final {
    LOG(FATAL) << "IfNode not implemented " << GetRef<Expr>(op);
    return {};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const OpNode* op) final {
    LOG(FATAL) << "OpNode not implemented " << GetRef<Expr>(op);
    return {};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const RefCreateNode* op) final {
    LOG(FATAL) << "RefCreateNode not implemented " << GetRef<Expr>(op);
    return {};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const RefReadNode* op) final {
    LOG(FATAL) << "RefReadNode not implemented " << GetRef<Expr>(op);
    return {};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const RefWriteNode* op) final {
    LOG(FATAL) << "RefWriteNode not implemented " << GetRef<Expr>(op);
    return {};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const ConstructorNode* op) final {
    LOG(FATAL) << "ConstructorNode not implemented " << GetRef<Expr>(op);
    return {};
  }
  std::vector<CVMGraphNodeRef> VisitExpr_(const MatchNode* op) final {
    LOG(FATAL) << "MatchNode not implemented " << GetRef<Expr>(op);
    return {};
  }

  /*!
   * \brief Generate CVMGraph JSON
   *
   * \param writer json writer
   */
  void GetJSON(dmlc::JSONWriter* writer) {
    std::vector<size_t> arg_nodes;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      auto node = nodes_[i];
      if (node->Type() == kCVMGraphInputNode) {
        arg_nodes.push_back(i);
      }
    }
    size_t num_entry = 0;
    ShapeVector shapes;
    std::vector<std::string> dltypes;
    std::vector<std::string> op_attrs;
    std::vector<int> precisions;
    std::vector<size_t> node_row_ptr{0};
    for (auto node : nodes_) {
      const auto& shape_vec = dmlc::get<ShapeVector>(node->attrs_["shape"]);
      const auto& dtype_vec = dmlc::get<std::vector<std::string>>(node->attrs_["dtype"]);
      const auto& prec_vec = dmlc::get<std::vector<int>>(node->attrs_["precision"]);

      CHECK_EQ(node->num_outputs_, shape_vec.size());
      num_entry += node->num_outputs_;

      if (node->attrs_.count("op_attrs")) {
        const auto& op_attr = dmlc::get<std::string>(node->attrs_["op_attrs"]);
        op_attrs.push_back(op_attr);
      } else {
        op_attrs.push_back("{}");
      }
      precisions.insert(precisions.end(), prec_vec.begin(), prec_vec.end());
      shapes.insert(shapes.end(), shape_vec.begin(), shape_vec.end());
      dltypes.insert(dltypes.end(), dtype_vec.begin(), dtype_vec.end());
      node_row_ptr.push_back(num_entry);
    }
    writer->BeginObject();
    writer->WriteObjectKeyValue("nodes", nodes_);
    writer->WriteObjectKeyValue("arg_nodes", arg_nodes);
    writer->WriteObjectKeyValue("heads", heads_);
    std::unordered_map<std::string, std::vector<dmlc::any>> attrs;
    attrs["precision"].emplace_back(std::string("list_int"));
    attrs["precision"].emplace_back(precisions);
    attrs["dltype"].emplace_back(std::string("list_str"));
    attrs["dltype"].emplace_back(dltypes);
    attrs["shape"].emplace_back(std::string("list_shape"));
    attrs["shape"].emplace_back(shapes);
    attrs["op_attrs"].emplace_back(std::string("list_str"));
    attrs["op_attrs"].emplace_back(op_attrs);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("node_row_ptr", node_row_ptr);
    writer->EndObject();
  }

 protected:
  /*! \brief nodes */
  std::vector<CVMGraphNodePtr> nodes_;
  /*! \brief output of graph */
  std::vector<CVMGraphNodeRef> heads_;
  /*! \brief variable map */
  std::unordered_map<const Node*, std::vector<CVMGraphNodeRef>> var_map_;
  /*! \brief params */
  std::unordered_map<std::string, runtime::NDArray> params_;
};

class CVMRuntimeCodegenModule : public runtime::ModuleNode {
 public:
  CVMRuntimeCodegenModule() {}
  virtual PackedFunc GetFunction(const std::string& name,
                                 const std::shared_ptr<ModuleNode>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        codegen_ = std::make_shared<CVMCodegen>();
      });
    } else if (name == "codegen") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Function func = args[0];
        this->output_ = this->codegen_->Codegen(func);
      });
    } else if (name == "get_graph_json") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.graph_json;
      });
    } else if (name == "list_params_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Array<HalideIR::Expr> ret;
        for (const auto &kv : this->output_.params) {
          HalideIR::Expr name = ir::StringImm::make(kv.first);
          ret.push_back(name);
        }
        *rv = ret;
      });
    } else if (name == "get_param_by_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        std::string key = args[0];
        CHECK_GT(this->output_.params.count(key), 0);
        *rv = this->output_.params[key];
      });
    } else {
      return PackedFunc([](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  const char* type_key() const final {
    return "CVMRuntimeCodegenModule";
  }

 private:
  std::shared_ptr<CVMCodegen> codegen_;
  CVMOutput output_;
};

runtime::Module CreateCVMCodegenMod() {
  std::shared_ptr<CVMRuntimeCodegenModule> ptr =
    std::make_shared<CVMRuntimeCodegenModule>();
  return runtime::Module(ptr);
}

TVM_REGISTER_GLOBAL("relay.build_module._CVMRuntimeCodegen")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CreateCVMCodegenMod();
});

}  // namespace backend
}  // namespace relay
}  // namespace tvm

namespace dmlc {
namespace json {
// JSON utils
template <typename T>
inline bool SameType(const dmlc::any& data) {
  return std::type_index(data.type()) == std::type_index(typeid(T));
}

template <>
struct Handler<std::shared_ptr<tvm::relay::backend::CVMGraphNode>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::shared_ptr<tvm::relay::backend::CVMGraphNode>& data) {
    data->Save(writer);
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::shared_ptr<tvm::relay::backend::CVMGraphNode>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};

template <>
struct Handler<std::unordered_map<std::string, dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const std::unordered_map<std::string, dmlc::any>& data) {
    writer->BeginObject();
    for (const auto& kv : data) {
      auto k = kv.first;
      const dmlc::any& v = kv.second;
      if (SameType<std::string>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<int>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<int>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteObjectKeyValue(k, dmlc::get<std::vector<std::string>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndObject();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          std::unordered_map<std::string, dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};

template <>
struct Handler<std::vector<dmlc::any>> {
  inline static void Write(dmlc::JSONWriter* writer, const std::vector<dmlc::any>& data) {
    writer->BeginArray();
    for (const auto& v : data) {
      if (SameType<std::string>(v)) {
        writer->WriteArrayItem(dmlc::get<std::string>(v));
      } else if (SameType<int>(v)) {
        writer->WriteArrayItem(dmlc::get<int>(v));
      } else if (SameType<std::vector<size_t>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<size_t>>(v));
      } else if (SameType<std::vector<int>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<int>>(v));
      } else if (SameType<std::vector<std::vector<int64_t>>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::vector<int64_t>>>(v));
      } else if (SameType<std::vector<std::string>>(v)) {
        writer->WriteArrayItem(dmlc::get<std::vector<std::string>>(v));
      } else {
        LOG(FATAL) << "Not supported";
      }
    }
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, std::vector<dmlc::any>* data) {
    LOG(FATAL) << "Not implemented.";
  }
};
}  // namespace json
}  // namespace dmlc
