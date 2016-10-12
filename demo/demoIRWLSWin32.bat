:: Test de la libreria
 
@echo off 

ECHO ***************************************************
ECHO * DOWNLOADING DATASETS FROM THE LIBSVM REPOSITORY *
ECHO ***************************************************
ECHO 
 
mkdir ..\data 
 
ECHO Downloading ADULT dataset for training: a9a

powershell -Command "(New-Object Net.WebClient).DownloadFile('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a', '..\data\a9a')" 

ECHO Downloading ADULT dataset for testing: a9a.t
 
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t', '..\data\a9a.t')" 
 
If exist "..\windows\win32\PIRWLS-train.exe" (
	
ECHO *************************************************
ECHO * RUNNING SVM ^(PIRWLS algorithm^) USING 1 THREAD *
ECHO *************************************************
ECHO  

..\windows\win32\PIRWLS-train.exe -c 1000 -g 0.001 -t 1 ..\data\a9a ..\data\a9a.model		

ECHO **************************************************
ECHO * RUNNING SVM ^(PIRWLS algorithm^) USING 2 THREADS *
ECHO **************************************************
ECHO  

..\windows\win32\PIRWLS-train.exe -c 1000 -g 0.001 -t 2 ..\data\a9a ..\data\a9a.model			
	

ECHO *****************************************************
ECHO * USING THE MODEL CREATED TO CLASSIFY A NEW DATASET *
ECHO *****************************************************
ECHO  

..\windows\win32\LIBIRWLS-predict.exe -l 1 -t 1 ..\data\a9a.t ..\data\a9a.model ..\data\a9a.output	
	
) ELSE (
ECHO ..\windows\win32\PIRWLS-train.exe not found.
)



If exist "..\windows\win32\PSIRWLS-train.exe" (
	
ECHO ******************************************************
ECHO * RUNNING SEMIPARAMETRIC SVM ^(PSIRWLS^) USING 1 THREAD *
ECHO *******************************************************
ECHO  
..\windows\win32\PSIRWLS-train.exe -c 1000 -s 75 -g 0.0001 -t 1 ..\data\a9a ..\data\a9a.model		

ECHO ********************************************************
ECHO * RUNNING SEMIPARAMETRIC SVM ^(PSIRWLS^) USING 2 THREADS *
ECHO ********************************************************

ECHO    
..\windows\win32\PSIRWLS-train.exe -c 1000 -s 75 -g 0.0001 -t 2 ..\data\a9a ..\data\a9a.model			
	
ECHO *****************************************************
ECHO * USING THE MODEL CREATED TO CLASSIFY A NEW DATASET *
ECHO *****************************************************
ECHO  

..\windows\win32\LIBIRWLS-predict.exe -l 1 -t 1 ..\data\a9a.t ..\data\a9a.model ..\data\a9a.output	
	
	
) ELSE (
ECHO ..\windows\win32\PSIRWLS-train.exe not found.
)
