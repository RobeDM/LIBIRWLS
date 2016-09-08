:: Test de la libreria
 
@echo off 
 
If exist "..\windows\PIRWLS-train.exe" (
	
ECHO *************************************************
ECHO * RUNNING SVM ^(PIRWLS algorithm^) USING 1 THREAD *
ECHO *************************************************
ECHO  

..\windows\PIRWLS-train.exe -c 100 -g 0.1 -t 1 ..\data\w7a ..\data\w7a.model		

ECHO **************************************************
ECHO * RUNNING SVM ^(PIRWLS algorithm^) USING 2 THREADS *
ECHO **************************************************
ECHO  

..\windows\PIRWLS-train.exe -c 100 -g 0.1 -t 2 ..\data\w7a ..\data\w7a.model			
	

ECHO *****************************************************
ECHO * USING THE MODEL CREATED TO CLASSIFY A NEW DATASET *
ECHO *****************************************************
ECHO  

..\windows\LIBIRWLS-predict.exe -l 1 -t 1 ..\data\w7a.t ..\data\w7a.model ..\data\w7a.output	
	
) ELSE (
ECHO ..\bin\PIRWLS-train.exe  not found. Please, build this code using the make command.
)



If exist "..\windows\PSIRWLS-train.exe" (
	
ECHO ******************************************************
ECHO * RUNNING SEMIPARAMETRIC SVM ^(PSIRWLS^) USING 1 THREAD *
ECHO *******************************************************
ECHO  
..\windows\PSIRWLS-train.exe -c 100 -s 100 -g 0.1 -t 1 ..\data\w7a ..\data\w7a.model		

ECHO ********************************************************
ECHO * RUNNING SEMIPARAMETRIC SVM ^(PSIRWLS^) USING 2 THREADS *
ECHO ********************************************************

ECHO    
..\windows\PSIRWLS-train.exe -c 100 -s 100 -g 0.1 -t 2 ..\data\w7a ..\data\w7a.model			
	
ECHO *****************************************************
ECHO * USING THE MODEL CREATED TO CLASSIFY A NEW DATASET *
ECHO *****************************************************
ECHO  

..\windows\LIBIRWLS-predict.exe -l 1 -t 1 ..\data\w7a.t ..\data\w7a.model ..\data\w7a.output	
	
	
) ELSE (
ECHO ..\bin\PIRWLS-train.exe  not found. Please, build this code using the make command.
)
