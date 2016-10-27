CC=gcc
OSX=0

OPTFLAGS = -fPIC -O3 -fopenmp

BINFOLDER := bin
BUILDFOLDER := build
SRCFOLDER := src

SRCEXT := c
SOURCES := $(shell find $(SRCFOLDER) -type f -name "*.$(SRCEXT)")
OBJECTS := $(patsubst $(SRCFOLDER)/%,$(BUILDFOLDER)/%,$(SOURCES:.$(SRCEXT)=.o))

INCLUDE  = -Iinclude

LIBS = -lm -llapack -lf77blas -lcblas -latlas -lgfortran -fopenmp

CCOPTION = 

ifeq ($(OSX),1)
    LIBRARYPATH = -L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/
    LIBS = -lLAPACK -lBLAS -fopenmp
    CCOPTION = -DOSX
endif

ifdef ALGEBRADIR
    LIBRARYPATH = -L$(ALGEBRADIR)/lib/
endif


COMMONOBJ := $(BUILDFOLDER)/ParallelAlgorithms.o $(BUILDFOLDER)/IOStructures.o $(BUILDFOLDER)/kernels.o $(BUILDFOLDER)/LIBIRWLS-predict.o $(BUILDFOLDER)/PSIRWLS-train.o $(BUILDFOLDER)/PIRWLS-train.o

all: LIBIRWLS-predict PIRWLS-train PSIRWLS-train

PIRWLS-train: $(BUILDFOLDER)/ExecPIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS) 

PSIRWLS-train: $(BUILDFOLDER)/ExecPSIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PSIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

LIBIRWLS-predict: $(BUILDFOLDER)/ExecLIBIRWLS-predict.o $(COMMONOBJ)
	@echo " Linking LIBIRWLS-predict"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/LIBIRWLS-predict $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/LIBIRWLS-predict $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

$(BUILDFOLDER)/%.o: $(SRCFOLDER)/%.$(SRCEXT)
	@echo " mkdir -p $(BUILDFOLDER)"; mkdir -p $(BUILDFOLDER)
	@echo " $(CC) $(CCOPTION) $(OPTFLAGS) $(CFLAGS) $(INCLUDE) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<"; $(CC) $(CCOPTION) $(OPTFLAGS) $(CFLAGS) $(INCLUDE) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " rm -rf $(BINFOLDER) $(BUILDFOLDER)"; rm -rf $(BINFOLDER) $(BUILDFOLDER)

.PHONY: clean
