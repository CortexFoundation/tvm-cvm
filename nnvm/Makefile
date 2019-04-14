ROOTDIR = $(CURDIR)

ifndef config
ifneq ("$(wildcard ./config.mk)", "")
	config = config.mk
else
	config = make/config.mk
endif
endif
include $(config)

TVMPATH = ..

export LDFLAGS = -pthread -lm
export CFLAGS = -std=c++11 -Wall -O2 -Iinclude -fPIC
CFLAGS += -I$(TVMPATH)/include -I$(TVMPATH)/3rdparty/dlpack/include -I$(TVMPATH)/3rdparty/HalideIR/src -I$(TVMPATH)/topi/include

ifdef DMLC_CORE_PATH
  CFLAGS += -I$(DMLC_CORE_PATH)/include
else
  CFLAGS += -I$(TVMPATH)/3rdparty/dmlc-core/include
endif

ifneq ($(ADD_CFLAGS), NONE)
	CFLAGS += $(ADD_CFLAGS)
endif

ifneq ($(ADD_LDFLAGS), NONE)
	LDFLAGS += $(ADD_LDFLAGS)
endif

# plugin
PLUGIN_OBJ =
include $(NNVM_PLUGINS)

# specify tensor path
.PHONY: clean all test lint cpplint pylint doc cython cython3 cyclean

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
	SHARED_LIBRARY_SUFFIX := dylib
	WHOLE_ARCH= -all_load
	NO_WHOLE_ARCH= -noall_load
	LDFLAGS += -undefined dynamic_lookup
else
	SHARED_LIBRARY_SUFFIX := so
	WHOLE_ARCH= --whole-archive
	NO_WHOLE_ARCH= --no-whole-archive
endif

all: lib/libnnvm.a lib/libnnvm_compiler.$(SHARED_LIBRARY_SUFFIX) lib/libnnvm_rt.a

SRC = $(wildcard src/*.cc src/c_api/*.cc src/core/*.cc src/pass/*.cc)
SRC_COMPILER = $(wildcard src/top/*/*.cc wildcard src/top/vision/*/*.cc src/compiler/*.cc src/compiler/*/*.cc)
ALL_OBJ = $(patsubst %.cc, build/%.o, $(SRC))
TOP_OBJ = $(patsubst %.cc, build/%.o, $(SRC_COMPILER))
ALL_DEP = $(ALL_OBJ)
RT_DEP = build/src/top/nn/pooling.o build/src/top/nn/convolution.o build/src/top/nn/upsampling.o\
		 build/src/top/vision/nms.o build/src/top/image/resize.o build/src/top/tensor/broadcast.o build/src/top/tensor/matrix_op.o build/src/top/tensor/elemwise.o\
		 build/src/top/tensor/transform.o build/src/top/tensor/reduce.o build/src/top/tensor/state_op.o build/src/top/vision/yolo/reorg.o build/src/top/vision/ssd/mutibox_op.o\
		 build/src/compiler/simplify_inference.o build/src/compiler/compile_engine.o build/src/compiler/precompute_prune.o build/src/compiler/graph_fuse.o build/src/compiler/packed_func_ext.o\
		 build/src/compiler/graph_compile.o build/src/compiler/alter_op_layout.o build/src/compiler/graph_hash.o build/src/compiler/graph_runtime.o build/src/compiler/fold_scale_axis.o\
		 build/src/c_api/c_api_symbolic.o build/src/c_api/c_api_graph.o build/src/c_api/c_api_error.o build/src/core/pass.o build/src/core/symbolic.o build/src/core/node.o build/src/core/op.o\
		 build/src/core/graph.o build/src/pass/gradient.o build/src/pass/infer_shape_type.o build/src/pass/correct_layout.o\
		 build/src/pass/order_mutation.o build/src/pass/plan_memory.o build/src/pass/print_graph_ir.o build/src/pass/saveload_json.o build/src/pass/place_device.o build/src/top/nn/nn.o 

include tests/cpp/unittest.mk

test: $(TEST)

build/src/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/src/$*.o $< >build/src/$*.d
	$(CXX) -c $(CFLAGS) -c $< -o $@

lib/libnnvm.a: $(ALL_DEP)
	@mkdir -p $(@D)
	$(AR) crv $@ $(filter %.o, $?)

lib/libnnvm_rt.a: $(RT_DEP)
	@mkdir -p $(@D)
	$(AR) crv $@ $(filter %.o, $?)

lib/libnnvm_compiler.$(SHARED_LIBRARY_SUFFIX): lib/libnnvm.a ${TOP_OBJ}
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -shared -o $@ $(filter %.o, $^) $(LDFLAGS) -Wl,${WHOLE_ARCH} lib/libnnvm.a -Wl,${NO_WHOLE_ARCH}

cython:
	cd python; python setup.py build_ext --inplace

cython3:
	cd python; python3 setup.py build_ext --inplace

cyclean:
	rm -rf python/nnvm/*/*.so python/nnvm/*/*.dylib python/nnvm/*/*.cpp

lint: pylint cpplint

doc:
	doxygen docs/Doxyfile

cpplint:
	python ../dmlc-core/scripts/lint.py nnvm cpp include src

pylint:
	pylint python/nnvm --rcfile=$(ROOTDIR)/tests/lint/pylintrc

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o cli_test

-include build/*.d
-include build/*/*.d
-include build/*/*/*.d
-include build/*/*/*/*.d
