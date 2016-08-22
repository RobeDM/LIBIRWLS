CC=gcc

OPTFLAGS = -O3

USE_MKL=0

ifeq ($(USE_MKL),1)
	CFLAGS= -DUSE_MKL
	LIBS = -lmkl_core -fopenmp -lmkl_sequential -lmkl_intel_lp64 -lm
	#INCLUDEPATH = -I/opt/intel/composerxe-2013.1.106/mkl/include/
	#LIBRARYPATH = -L/opt/intel/composerxe-2013.1.106/mkl/lib/intel64/
else
	LIBS = -fopenmp -lblas -lm -llapack
	#INCLUDEPATH = -I...
	#LIBRARYPATH = -L...
endif

all: PIRWLS-predict PIRWLS-train PSIRWLS-predict PSIRWLS-train

PIRWLS-predict: PIRWLS-predict.c
	$(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -o PIRWLS-predict PIRWLS-predict.c $(LIBS)

PIRWLS-train: PIRWLS-train.c
	$(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -o PIRWLS-train PIRWLS-train.c $(LIBS)

PSIRWLS-predict: PSIRWLS-predict.c
	$(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -o PSIRWLS-predict PSIRWLS-predict.c $(LIBS)

PSIRWLS-train: PSIRWLS-train.c
	$(CC) $(OPTFLAGS) $(CFLAGS) $(INCLUDEPATH) $(LIBRARYPATH) -o PSIRWLS-train PSIRWLS-train.c $(LIBS)

clean:
	rm -f PIRWLS-train PIRWLS-predict PSIRWLS-train PSIRWLS-predict
