CC=gcc

OPTFLAGS = -O3

USE_MKL=0

BINFOLDER := bin
BUILDFOLDER := build
SRCFOLDER := src

SRCEXT := c
SOURCES := $(shell find $(SRCFOLDER) -type f -name "*.$(SRCEXT)")
OBJECTS := $(patsubst $(SRCFOLDER)/%,$(BUILDFOLDER)/%,$(SOURCES:.$(SRCEXT)=.o))

# If your linear algebra library (mkl or blas and lapack) can not be found, uncomment these lines and write the respective folder.
#INCLUDEPATH = -I/usr/...
#LIBRARYPATH = -L/usr/...

ifeq ($(USE_MKL),1)
	CFLAGS= -DUSE_MKL
	LIBS = -lmkl_core -fopenmp -lmkl_sequential -lmkl_intel_lp64 -lm
else
	LIBS = -fopenmp -lm -lblas -llapack
endif

COMMONOBJ := $(BUILDFOLDER)/ParallelAlgorithms.o $(BUILDFOLDER)/IOStructures.o $(BUILDFOLDER)/kernels.o

all: LIBIRWLS-predict PIRWLS-train PSIRWLS-train

PIRWLS-train: $(BUILDFOLDER)/PIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/$@ $(LIBS)

PSIRWLS-train: $(BUILDFOLDER)/PSIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PSIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/$@ $(LIBS)

LIBIRWLS-predict: $(BUILDFOLDER)/LIBIRWLS-predict.o $(COMMONOBJ)
	@echo " Linking LIBIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/$@ $(LIBS)

$(BUILDFOLDER)/%.o: $(SRCFOLDER)/%.$(SRCEXT)
	@echo " mkdir -p $(BUILDFOLDER)"; mkdir -p $(BUILDFOLDER)
	@echo " $(CC) $(LIBS) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<"; $(CC) $(LIBS) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " rm -rf $(BINFOLDER) $(BUILDFOLDER)"; rm -rf $(BINFOLDER) $(BUILDFOLDER)

.PHONY: clean
