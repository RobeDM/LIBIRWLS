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

ifdef VECLIBDIR
    LIBRARYPATH = -L$(VECLIBDIR)/
endif

ifdef ATLASDIR
    LIBRARYPATH = -L$(ATLASDIR)/lib/
endif

COMMONOBJ := $(BUILDFOLDER)/ParallelAlgorithms.o $(BUILDFOLDER)/IOStructures.o $(BUILDFOLDER)/kernels.o $(BUILDFOLDER)/LIBIRWLS-predict.o $(BUILDFOLDER)/budgeted-train.o $(BUILDFOLDER)/full-train.o

all: LIBIRWLS-predict full-train budgeted-train

full-train: $(BUILDFOLDER)/Exec-full-train.o $(COMMONOBJ)
	@echo " Linking full-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS) 

budgeted-train: $(BUILDFOLDER)/Exec-budgeted-train.o $(COMMONOBJ)
	@echo " Linking budgeted-train"
	mkdir -p $(BINFOLDER)
	@echo " $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)"; $(CC) $(CCOPTION) $^ -o $(BINFOLDER)/$@ $(INCLUDEPATH) $(LIBRARYPATH) $(LIBS)

LIBIRWLS-predict: $(BUILDFOLDER)/Exec-LIBIRWLS-predict.o $(COMMONOBJ)
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
