include config.mk

EXT := cu 
CxxEXT := cpp

OBJS := colwisemean.o colwisesd.o cuseful.o gpuMCD.o reduction_kernel.o tools.o 
#rinterface.o mi.o sort.o granger.o qrdecomp.o correlation.o hcluster.o distance.o matmult.o lsfit.o kendall.o cuseful.o

CxxOBJS := compute_det.o rcppfun.o rgpuMCD.o 

R_HOME := $(shell R RHOME)

#compiler/preprocessor options
INCS := -I. -I"$(CULA_INC)" -I"$(CUDA_INC)" -I"$(R_INC)" $(shell Rscript -e "Rcpp:::CxxFlags()")
PARAMS := $(DEVICEOPTS) -Xcompiler $(CPICFLAGS)

CxxINCS := -I. -I"$(R_INC)" $(shell Rscript -e "Rcpp:::CxxFlags()")
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
