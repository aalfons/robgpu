include config.mk

EXT := cu 
CxxEXT := cpp

OBJS := cuseful.o rgcor.o colwisesd.o colwisemean.o colwisesum.o colwisesum_kernel.o colwisesd_kernel.o tool_kernels.o
CxxOBJS := rcor.o rcolwisesd.o rcolwisemean.o rcolwisesum.o reduction_tools.o

R_HOME := $(shell R RHOME)

#compiler/preprocessor options
INCS := -I. -I"$(CULA_INC)" -I"$(CUDA_INC)" -I"$(R_INC)" $(shell Rscript -e "Rcpp:::CxxFlags()")
PARAMS := $(DEVICEOPTS) -Xcompiler $(CPICFLAGS)

CxxINCS := -I. -I"$(R_INC)" $(shell Rscript -e "Rcpp:::CxxFlags()") $(shell Rscript -e "RcppArmadillo:::CxxFlags()")

CxxPARAMS := $(CPICFLAGS)

#linker options
LD_PARAMS := $(DEVICEOPTS) -Xlinker '$(RPATH) $(R_FRAMEWORK)'
LIBS :=  -L"$(R_LIB)" -L"$(CUDA_LIB)" -lcublas $(shell R CMD config BLAS_LIBS) $(shell R CMD config LAPACK_LIBS) $(shell R CMD config FLIBS) -L/usr/local/lib/R/site-library/Rcpp/lib -lRcpp -Xlinker=-rpath,/usr/local/lib/R/site-library/Rcpp/lib

TARGETS := robgpu.so

NVCC := $(CUDA_HOME)/bin/nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30

all: $(TARGETS) 

$(TARGETS): $(OBJS) $(CxxOBJS)
	$(NVCC) -shared $(LD_PARAMS) $(LIBS) $(OBJS) $(CxxOBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(NVCC) -c $(INCS) $(PARAMS) $^ -o $@

$(CxxOBJS): %.o: %.$(CxxEXT)
	g++ -c $(CxxINCS) $(CxxPARAMS) $^ -o $@


clean:
	rm -rf *o

.PHONY: all clean
