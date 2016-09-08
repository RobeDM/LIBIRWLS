#!/bin/bash

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
	../bin/PIRWLS-train -c 100 -g 0.1 -t 1 ../data/w7a ../data/w7a.model


	echo "**************************************************"
	echo "* RUNNING SVM (PIRWLS algorithm) USING 2 THREADS *"
	echo "**************************************************"
	echo " "
	../bin/PIRWLS-train -c 100 -g 0.1 -t 2 ../data/w7a ../data/w7a.model

	echo "*****************************************************"
	echo "* USING THE MODEL CREATED TO CLASSIFY A NEW DATASET *"
	echo "*****************************************************"
	echo " "
	../bin/LIBIRWLS-predict -l 1 -t 1 ../data/w7a.t ../data/w7a.model ../data/w7a.output

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
	../bin/PSIRWLS-train -c 100 -g 0.1 -s 100 -t 1 ../data/w7a ../data/w7a.model

	echo "********************************************************"
	echo "* RUNNING SEMIPARAMETRIC SVM (PSIRWLS) USING 2 THREADS *"
	echo "********************************************************"
	echo " "
	../bin/PSIRWLS-train -c 100 -g 0.1 -s 100 -t 2 ../data/w7a ../data/w7a.model

	echo "*****************************************************"
	echo "* USING THE MODEL CREATED TO CLASSIFY A NEW DATASET *"
	echo "*****************************************************"
	echo " "
	../bin/LIBIRWLS-predict -l 1 -t 1 ../data/w7a.t ../data/w7a.model ../data/w7a.output


else
	echo "$file not found. Please, build this code using the make command."
fi

else
	echo "$file not found. Please, build this code using the make command."
fi

