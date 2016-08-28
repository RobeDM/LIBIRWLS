CC=gcc

OPTFLAGS = -O3

USE_MKL=0

BINFOLDER := bin
BUILDFOLDER := build
SRCFOLDER := src

SRCEXT := c
SOURCES := $(shell find $(SRCFOLDER) -type f -name "*.$(SRCEXT)")
OBJECTS := $(patsubst $(SRCFOLDER)/%,$(BUILDFOLDER)/%,$(SOURCES:.$(SRCEXT)=.o))


ifeq ($(USE_MKL),1)
	CFLAGS= -DUSE_MKL
	LIBS = -lmkl_core -fopenmp -lmkl_sequential -lmkl_intel_lp64 -lm
	#INCLUDEPATH = -I/opt/intel/composerxe-2013.1.106/mkl/include/
	#LIBRARYPATH = -L/opt/intel/composerxe-2013.1.106/mkl/lib/intel64/
else
	LIBS = -fopenmp -lblas -lm -llapack
	INCLUDEPATH = -I/usr/local/atlas/include/
	#LIBRARYPATH = -L...
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
	@echo " $(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<"; $(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " rm -rf $(BINFOLDER) $(BUILDFOLDER)"; rm -rf $(BINFOLDER) $(BUILDFOLDER)

.PHONY: clean
