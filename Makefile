default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -pedantic -Werror -ftemplate-backtrace-limit=0

RELEASE_FLAGS += -fno-rtti

CXX_FLAGS +=  -Inice_svm/include -Idll/include -Idll/etl/lib/include -Idll/etl/include/ -Idll/mnist/include/
LD_FLAGS += -lpthread -lsvm

CXX_FLAGS += -DETL_PARALLEL -DETL_VECTORIZE_FULL

DLL_BLAS_PKG ?= mkl

# Activate BLAS mode on demand
ifneq (,$(ETL_MKL))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags $(DLL_BLAS_PKG))
LD_FLAGS += $(shell pkg-config --libs $(DLL_BLAS_PKG))

# Disable warning for MKL
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif

else
ifneq (,$(ETL_BLAS))
CXX_FLAGS += -DETL_BLAS_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)

# Disable warning for MKL
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif

endif
endif

# On demand activation of cublas support
ifneq (,$(ETL_CUBLAS))
CXX_FLAGS += -DETL_CUBLAS_MODE $(shell pkg-config --cflags cublas)
LD_FLAGS += $(shell pkg-config --libs cublas)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

# On demand activation of cufft support
ifneq (,$(ETL_CUFFT))
CXX_FLAGS += -DETL_CUFFT_MODE $(shell pkg-config --cflags cufft)
LD_FLAGS += $(shell pkg-config --libs cufft)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

# On demand activation of cudnn support
ifneq (,$(ETL_CUDNN))
CXX_FLAGS += -DETL_CUDNN_MODE $(shell pkg-config --cflags cudnn)
LD_FLAGS += $(shell pkg-config --libs cudnn)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

CPP_FILES=$(wildcard src/*.cpp)

# Compile all the sources
$(eval $(call auto_folder_compile,src))

# Generate executables for experiments
$(eval $(call add_executable,dll_ae,src/main.cpp))
$(eval $(call add_executable_set,dll_ae,dll_ae))

release: release_dll_ae
release_debug: release_debug_dll_ae
debug: debug_dll_ae

all: release debug release_debug

clean: base_clean

-include tests.mk

include make-utils/cpp-utils-finalize.mk
