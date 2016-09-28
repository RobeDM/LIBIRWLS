CC=gcc

OPTFLAGS = -O3 -fopenmp

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


COMMONOBJ := $(BUILDFOLDER)/ParallelAlgorithms.o $(BUILDFOLDER)/IOStructures.o $(BUILDFOLDER)/kernels.o

all: LIBIRWLS-predict PIRWLS-train PSIRWLS-train

PIRWLS-train: $(BUILDFOLDER)/PIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS) 

PSIRWLS-train: $(BUILDFOLDER)/PSIRWLS-train.o $(COMMONOBJ)
	@echo " Linking PSIRWLS-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

LIBIRWLS-predict: $(BUILDFOLDER)/LIBIRWLS-predict.o $(COMMONOBJ)
	@echo " Linking LIBIRWLS-predict"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

$(BUILDFOLDER)/%.o: $(SRCFOLDER)/%.$(SRCEXT)
	@echo " mkdir -p $(BUILDFOLDER)"; mkdir -p $(BUILDFOLDER)
	@echo " $(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDE) -c -o $@ $<"; $(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDE) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " rm -rf $(BINFOLDER) $(BUILDFOLDER)"; rm -rf $(BINFOLDER) $(BUILDFOLDER)

.PHONY: clean
