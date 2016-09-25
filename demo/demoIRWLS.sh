#!/bin/bash

echo "***************************************************"
echo "* DOWNLOADING DATASETS FROM THE LIBSVM REPOSITORY *"
echo "***************************************************"

mkdir ../data
echo "Downloading ADULT dataset for training: a9a"
wget  -O ../data/a9a 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a'
echo "Downloading ADULT dataset for testing: a9a.t"
wget  -O ../data/a9a.t 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t'



file="../bin/PIRWLS-train"
if [ -f "$file" ]
then

file="../bin/LIBIRWLS-predict"
if [ -f "$file" ]
then

echo "*************************************************"
echo "* RUNNING SVM (PIRWLS algorithm) USING 1 THREAD *"
echo "*************************************************"
echo " "
../bin/PIRWLS-train -c 1000 -g 0.001 -t 1 ../data/a9a ../data/a9a.model


echo "**************************************************"
echo "* RUNNING SVM (PIRWLS algorithm) USING 2 THREADS *"
echo "**************************************************"
echo " "
../bin/PIRWLS-train -c 1000 -g 0.001 -t 2 ../data/a9a ../data/a9a.model

echo "*****************************************************"
echo "* USING THE MODEL CREATED TO CLASSIFY A NEW DATASET *"
echo "*****************************************************"
echo " "
../bin/LIBIRWLS-predict -l 1 -t 1 ../data/a9a.t ../data/a9a.model ../data/a9a.output

else
echo "$file not found. Please, build this code using the make command."
fi

else
echo "$file not found. Please, build this code using the make command."
fi



file="../bin/PIRWLS-train"
if [ -f "$file" ]
then

file="../bin/LIBIRWLS-predict"
if [ -f "$file" ]
then

	echo "*******************************************************"
	echo "* RUNNING SEMIPARAMETRIC SVM (PSIRWLS) USING 1 THREAD *"
	echo "*******************************************************"
	echo " "
	../bin/PSIRWLS-train -c 1000 -g 0.0001 -s 75 -t 1 ../data/a9a ../data/a9a.model

	echo "********************************************************"
	echo "* RUNNING SEMIPARAMETRIC SVM (PSIRWLS) USING 2 THREADS *"
	echo "********************************************************"
	echo " "
	../bin/PSIRWLS-train -c 1000 -g 0.0001 -s 75 -t 2 ../data/a9a ../data/a9a.model

	echo "*****************************************************"
	echo "* USING THE MODEL CREATED TO CLASSIFY A NEW DATASET *"
	echo "*****************************************************"
	echo " "
	../bin/LIBIRWLS-predict -l 1 -t 1 ../data/a9a.t ../data/a9a.model ../data/a9a.output


else
	echo "$file not found. Please, build this code using the make command."
fi

else
	echo "$file not found. Please, build this code using the make command."
fi

