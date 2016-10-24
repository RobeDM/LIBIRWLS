CC=gcc

OPTFLAGS = -fPIC -O3 -fopenmp

BINFOLDER := bin
BUILDFOLDER := build
SRCFOLDER := src

SRCEXT := c
SOURCES := $(shell find $(SRCFOLDER) -type f -name "*.$(SRCEXT)")
OBJECTS := $(patsubst $(SRCFOLDER)/%,$(BUILDFOLDER)/%,$(SOURCES:.$(SRCEXT)=.o))

INCLUDE  = -Iinclude

ifdef ATLASDIR
    INCLUDEPATH = -I$(ATLASDIR)/include/
    LIBRARYPATH = -L$(ATLASDIR)/lib/
endif

LIBS = -lm -llapack -lf77blas -lcblas -latlas -lgfortran -fopenmp


COMMONOBJ := $(BUILDFOLDER)/ParallelAlgorithms.o $(BUILDFOLDER)/IOStructures.o $(BUILDFOLDER)/kernels.o $(BUILDFOLDER)/LIBIRWLS-predict.o $(BUILDFOLDER)/PSIRWLS-train.o $(BUILDFOLDER)/PIRWLS-train.o

all: LIBIRWLS-predict PIRWLS-train PSIRWLS-train

PIRWLS-train: $(BUILDFOLDER)/ExecPIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS) 

PSIRWLS-train: $(BUILDFOLDER)/ExecPSIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PSIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

LIBIRWLS-predict: $(BUILDFOLDER)/ExecLIBIRWLS-predict.o $(COMMONOBJ)
	@echo " Linking LIBIRWLS-predict"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/LIBIRWLS-predict $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

$(BUILDFOLDER)/%.o: $(SRCFOLDER)/%.$(SRCEXT)
	@echo " mkdir -p $(BUILDFOLDER)"; mkdir -p $(BUILDFOLDER)
	@echo " $(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDE) -c -o $@ $<"; $(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDE) $(INCLUDEPATH) $(LIBRARYPATH) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " rm -rf $(BINFOLDER) $(BUILDFOLDER)"; rm -rf $(BINFOLDER) $(BUILDFOLDER)

.PHONY: clean
