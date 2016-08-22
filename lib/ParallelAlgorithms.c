/*
 ============================================================================
 Author      : Roberto Diaz Morales
 ============================================================================
 
 Copyright (c) 2016 Roberto Díaz Morales

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
 (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
 publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 ============================================================================
 */


#include <sys/time.h>

double **auxmemory1;
double **auxmemory2;
double **auxmemory3;

void initMemory(int Threads, int size){

    auxmemory1=(double **) calloc(Threads,sizeof(double*));
    auxmemory2=(double **) calloc(Threads,sizeof(double*));
    auxmemory3=(double **) calloc(Threads,sizeof(double*));

    int i;
    for(i=0;i<Threads;i++){
        auxmemory1[i]=(double *) calloc(pow(ceil(1.0*(size)),2),sizeof(double));
        auxmemory2[i]=(double *) calloc(pow(ceil(1.0*(size)),2),sizeof(double));
        auxmemory3[i]=(double *) calloc(pow(ceil(1.0*(size)),2),sizeof(double));
    }			

}


void updateMemory(int Threads, int size){

    int i;
    for(i=0;i<Threads;i++){
        auxmemory1[i]=realloc(auxmemory1[i],pow(ceil(1.0*(size)),2)*sizeof(double));
        auxmemory2[i]=realloc(auxmemory2[i],pow(ceil(1.0*(size)),2)*sizeof(double));
        auxmemory3[i]=realloc(auxmemory3[i],pow(ceil(1.0*(size)),2)*sizeof(double));
    }

}



void ParallelChol(double *matrix,int r,int c, int ro, int co, int n,int nCores, int deep){	
		double *memaux = (double *)calloc(2*pow(ceil(0.5*n),2),sizeof(double));
		int blockSize = pow(ceil(0.5*n),2)/nCores;	
	
		int i;
		#pragma omp parallel default(shared) private(i)
		{	
		#pragma omp for schedule(static)	
		for (i=0;i<nCores;i++){
			Chol(matrix,r,c,ro,co,n,nCores,i,deep,0,memaux,blockSize);
		}
		}		
}

void Chol(double *matrix,int r,int c, int ro, int co, int n,int nCores,int numTh, int deep,int posIni,double *memaux, int blockSize){	
	if(deep<=1){	
		if(numTh==posIni){
            
		    double *m=auxmemory1[numTh];		
		    getSubMatrix(matrix,r,c,ro,co,m, n,n,1);
		    int info;
		    char s='L';
		    dpotrf_(&s,&n, m, &n,&info);
            
		    if(info != 0 & posIni==2){
                printf("Error en dpotrf %d ------------------------------\n",info);
                exit(0);
            }
		    int i,j;
    		    for(i=1;i<n;i++){
			for(j=0;j<i;j++){
				m[i*n+j]=0.0;
			}
		    }
			
		    putSubMatrix(matrix,r,c,ro,co,m, n,n,1);

		}
	}else{

		int size1=ceil(0.5*n);
		int size2=n-size1;

		Chol(matrix,r,c,ro,co,size1,nCores,numTh,deep-1,posIni,memaux,blockSize);

		#pragma omp barrier			
		
		MoveMatrix(matrix,r,ro,c,co,&memaux[posIni*blockSize],size1,0,size1,0,size1,size1, nCores,posIni,numTh);

		#pragma omp barrier			
		
		TriangleInversion(&memaux[posIni*blockSize],size1, size1, 0, 0, size1, nCores,posIni,numTh,&memaux[size1*size1],blockSize);

		#pragma omp barrier			
		
		MoveMatrix(matrix,r,ro+size1,c,co,&memaux[posIni*blockSize+size1*size1],size2,0,size1,0,size2,size1, nCores,posIni,numTh);

		#pragma omp barrier			

		NLTProduct(&memaux[posIni*blockSize],size1,0,size1,0,&memaux[posIni*blockSize+size1*size1],size2,0,size1,0,size1,size2,1.0,matrix,r,ro+size1,c, co, nCores,posIni,numTh);

		#pragma omp barrier			

		AATProduct(matrix,r,ro+size1,c,co,size2,size1,-1.0,1.0,matrix,r,ro+size1,c, co+size1, nCores,posIni,numTh);

		#pragma omp barrier			

		if(numTh==posIni){
			double *Zeroes = (double *) malloc(size1*size2*sizeof(double));
			putSubMatrix(matrix,r,c,ro,co+size1,Zeroes, size1,size2,1);
			free(Zeroes);
		}

		#pragma omp barrier			

		Chol(matrix,r,c,ro+size1,co+size1,size2,nCores,numTh,deep-1,posIni,memaux,blockSize);		
		
		#pragma omp barrier							
		
	}
}

void ParallelLinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores){
	
    if(n<nCores){
    
		double *memaux = (double *)calloc(2*pow(ceil(0.5*n),2),sizeof(double));
		int blockSize = pow(ceil(0.5*n),2)/nCores;	
		
		int i;
    
		#pragma omp parallel default(shared) private(i)
		{	
		#pragma omp for schedule(static)	
		for (i=0;i<nCores;i++){
			LinearSystem(matrix1,r1,c1,ro1,co1,matrix2,r2,c2,ro2,co2,n,m,result,rr,cr,ror,cor,nCores,i,0,memaux,blockSize);
		}
		}
        free(memaux);
    }else{

        int info;
        char s='L';
        int cols=1;
        dpotrf_(&s,&n, matrix1, &n,&info);
        memcpy(result,matrix2,n*sizeof(double));
        dpotrs_(&s,&n,&cols, matrix1, &n, result,&n,&info);
        
    }
}


void LinearSystem(double *matrix1,int r1,int c1, int ro1, int co1,double *matrix2,int r2,int c2, int ro2, int co2,int n, int m,double *result,int rr,int cr, int ror, int cor, int nCores, int numTh,int posIni,double *memaux, int blockSize){	
         

	int deep=log(nCores)/log(2)+1;
 
	Chol(matrix1,r1,c1,ro1,co1,n,nCores,numTh,deep+2,posIni,memaux,blockSize);

	#pragma omp barrier	
        
	int info;
	char s='L';
    int ncols = 1;
	int j,k;
    for(j=0;j<n;j++){
	    for(k=0;k<j;k++){
	        matrix1[j*n+k]=0.0;
	    }
    }
    
    if(numTh==0){
        memcpy(&result[0],&matrix2[0],n*sizeof(double));
        dpotrs_(&s,&n,&ncols, &matrix1[0], &n, &result[0],&n,&info);
    }
       
		
        /*        
	TriangleInversion(matrix1,r1, c1, ro1, co1, n, nCores,posIni,numTh,memaux,blockSize);		
	#pragma omp barrier			
        if(numTh==0){
        gettimeofday(&stop, NULL);
        printf("Total %ld\n",((stop.tv_sec-start.tv_sec)*1000+(stop.tv_usec-start.tv_usec)/1000));	
        gettimeofday(&start, NULL);
	}
	LNProduct(matrix1 ,r1,ro1,c1,co1,matrix2,r2,ro2,c2,co2,n,m,1.0,result,rr,ror,cr,cor,nCores,posIni,numTh);			
	#pragma omp barrier			
        if(numTh==0){
        gettimeofday(&stop, NULL);
        printf("Total %ld\n",((stop.tv_sec-start.tv_sec)*1000+(stop.tv_usec-start.tv_usec)/1000));	
        gettimeofday(&start, NULL);
	}

	//MoveMatrix(result,rr,ror,cr,cor,matrix2,r2,ro2,c2,co2,n,m, nCores,posIni,numTh);
	MoveMatrix(result,rr,ror,cr,cor,&memaux[posIni*blockSize],n,0,m,0,n,m, nCores,posIni,numTh);
	#pragma omp barrier			
        if(numTh==0){
        gettimeofday(&stop, NULL);
        printf("Total %ld\n",((stop.tv_sec-start.tv_sec)*1000+(stop.tv_usec-start.tv_usec)/1000));	
        gettimeofday(&start, NULL);
	}
	//LTNProduct(matrix1,r1,ro1,c1,co1,matrix2,r2,ro2,c2,co2,n,m,1.0,result,rr,ror,cr,cor,nCores,posIni,numTh);				
	LTNProduct(matrix1,r1,ro1,c1,co1,&memaux[posIni*blockSize],n,0,m,0,n,m,1.0,result,rr,ror,cr,cor,nCores,posIni,numTh);				
	#pragma omp barrier			
        if(numTh==0){
        gettimeofday(&stop, NULL);
        printf("Total %ld\n",((stop.tv_sec-start.tv_sec)*1000+(stop.tv_usec-start.tv_usec)/1000));	
        gettimeofday(&totalstop, NULL);
        printf("Total %ld\n",((totalstop.tv_sec-totalstart.tv_sec)*1000+(totalstop.tv_usec-totalstart.tv_usec)/1000));	
        
        }
        printMatCol(result,n,1);
        */
}

void DiagInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int numTh){
	if(nCores <= 1){		
		if(n>0){
			double *m=auxmemory1[numTh];
			getSubMatrix(matrix,r,c,ro,co,m, n,n,1);
			int info;
			char s1='L';
			char s2='N';			
			dtrtri_(&s1,&s2,&n, m, &n,&info);
			if(info != 0){printf("Error en dtrtri %d ------------------------------\n",info);}			
			putSubMatrix(matrix,r,c,ro,co,m, n,n,1);			
		}
	}else{
		int size1=ceil(0.5*n);
		int size2=n-size1;

		if(numTh<posIni+nCores/2)
			DiagInversion(matrix,r,c,ro,co,size1,nCores/2,posIni,numTh);
		else
			DiagInversion(matrix,r,c,ro+size1,co+size1, size2,nCores/2,posIni+nCores/2,numTh);
	}	
	
}

void InversionNLProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize){

		int size1=ceil(0.5*n);
		int size2=n-size1;
		
		if(deep==1){
			double *C1=&memaux[posIni*blockSize];
			NLProduct(matrix,r,ro,c,co,matrix,r,ro+size1,c,co,size1,size2,-1.0,C1,size2,0,size1,0, nCores,posIni,numTh);
		}else{
			if(numTh<posIni+nCores/2)
				InversionNLProducts(matrix,r,c,ro,co,size1,nCores/2,posIni,deep-1,numTh,memaux,blockSize);
			else
				InversionNLProducts(matrix,r,c,ro+size1,co+size1,size2,nCores/2,posIni+nCores/2,deep-1,numTh,memaux,blockSize);
		}	

}

void InversionLNProducts(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni,int deep,int numTh, double *memaux, int blockSize){

		int size1=ceil(0.5*n);
		int size2=n-size1;
		
		if(deep==1){
			double *C1=&memaux[posIni*blockSize];
			LNProduct(matrix,r,ro+size1,c,co+size1,C1,size2,0,size1,0,size2,size1,1.0,matrix,r,ro+size1,c,co, nCores,posIni,numTh);
		}else{
			if(numTh<posIni+nCores/2)
				InversionLNProducts(matrix,r,c,ro,co,size1,nCores/2,posIni,deep-1,numTh,memaux,blockSize);
			else
				InversionLNProducts(matrix,r,c,ro+size1,co+size1,size2,nCores/2,posIni+nCores/2,deep-1,numTh,memaux,blockSize);
		}	

}

void TriangleInversion(double *matrix,int r, int c, int ro, int co, int n,int nCores,int posIni, int numTh,double *memaux,int blockSize){

	#pragma omp barrier			
		
	DiagInversion(matrix,r,c,ro,co,n,nCores,posIni,numTh);
		
	#pragma omp barrier			
	
	int deep=log(nCores)/log(2);
	int o;	
	
	#pragma omp barrier			
	
	for (o=deep;o>=1;o--){
			
		InversionNLProducts(matrix,r,c,ro,co,n,nCores,posIni,o,numTh,memaux,blockSize);
			
		#pragma omp barrier			
			
		InversionLNProducts(matrix,r,c,ro,co,n,nCores,posIni,o,numTh,memaux,blockSize);
			
		#pragma omp barrier			
	}	

	
}



void NNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh,int orientation){
	if(nCores <= 1){
		if(n1 >0 & n3>0){			
			double *mresultT=auxmemory2[numTh];			
			
			//double *mresultT = (double *) malloc(n1*n3*sizeof(double));							
			getSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n3,nCores);
			if(n2 > 0){
				//double *m1T = (double *) malloc(n1*n2*sizeof(double));							
				//double *m2T = (double *) malloc(n2*n3*sizeof(double));											
				double *m1T=auxmemory1[numTh];
				double *m2T=auxmemory3[numTh];

				getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n2,nCores);				
				getSubMatrix(m2,r2,c2,ro2,co2,m2T, n2,n3,nCores);						
				cblas_dgemm (CblasColMajor, CblasNoTrans, CblasNoTrans, n1,n3,n2,K1, m1T, n1, m2T, n2, K2, mresultT, n1);				

				//free(m1T);
				//free(m2T);				
			}else{
				int i;
				for(i=0;i<n2*n3;i++) mresultT[i]=K2*mresultT[i];				
			}		
			putSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n3,nCores);
			//free(mresultT);
		}
		
	}else{
		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n2);
		int cols1B=n2-cols1A;
		int rows2A=ceil(0.5*n2);
		int rows2B=n2-rows2A;
		int cols2A=ceil(0.5*n3);
		int cols2B=n3-cols2A;				



		if(numTh<posIni+nCores/2){
				NNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols1A,cols2A,K1,K2,result,rr,ror,cr,cor, nCores/2,posIni,numTh,orientation);			
				NNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1A,cols1B,cols2A,K1,1.0,result,rr,ror,cr,cor, nCores/2,posIni,numTh,orientation);

				if(orientation==1){
					NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2,rows1B,cols1A,cols2A,K1,K2,result,rr,ror+rows1A,cr,cor,nCores/2,posIni,numTh,orientation);
					NNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1B,cols2A,K1,1.0,result,rr,ror+rows1A,cr,cor, nCores/2,posIni,numTh,orientation);				
				}else{
					NNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1A,cols2B,K1,K2,result,rr,ror,cr,cor+cols2A, nCores/2,posIni,numTh,orientation);
					NNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1A,cols1B,cols2B,K1,1.0,result,rr,ror,cr,cor+cols2A, nCores/2,posIni,numTh,orientation);
				}
								
		}else{
				if(orientation==2){
					NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2,rows1B,cols1A,cols2A,K1,K2,result,rr,ror+rows1A,cr,cor,nCores/2,posIni+nCores/2,numTh,orientation);
					NNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1B,cols2A,K1,1.0,result,rr,ror+rows1A,cr,cor, nCores/2,posIni+nCores/2,numTh,orientation);				
				}else{
					NNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1A,cols2B,K1,K2,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,orientation);
					NNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1A,cols1B,cols2B,K1,1.0,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,orientation);
				}
				
				NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1B,cols1A,cols2B,K1,K2,result,rr,ror+rows1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,orientation);
				NNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1B,cols2B,K1,1.0,result,rr,ror+rows1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,orientation);
			
		}
	}
}



void MoveMatrix(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2, int n1, int n2, int nCores,int posIni,int numTh){
	if(nCores <= 1){
		if(n1 >0 & n2>0){			
			double *mmv=auxmemory2[numTh];
			getSubMatrix(m1,r1,c1,ro1,co1,mmv, n1,n2,nCores);
			putSubMatrix(m2,r2,c2,ro2,co2,mmv, n1,n2,nCores);
		}		
	}else{
		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n2);
		int cols1B=n2-cols1A;

		if(numTh<posIni+nCores/2){
				MoveMatrix(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols1A,nCores/2,posIni,numTh);			
				MoveMatrix(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows1A,c2,co2,rows1B,cols1A,nCores/2,posIni,numTh);											
		}else{
				MoveMatrix(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols1A,rows1A,cols1B,nCores/2,posIni+nCores/2,numTh);			
				MoveMatrix(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows1A,c2,co2+cols1A,rows1B,cols1B,nCores/2,posIni+nCores/2,numTh);											
		}
	}
}



void TNNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
	if(nCores <= 1){
		if(n2 >0 & n3>0){	
			//double *mresultT = (double *) malloc(n2*n3*sizeof(double));											
			double *mresultT=auxmemory2[numTh];		
			getSubMatrix(result,rr,cr,ror,cor,mresultT, n2,n3,nCores);
			if(n1 >0){
				//double *m1T = (double *) malloc(n1*n2*sizeof(double));							
				//double *m2T = (double *) malloc(n1*n3*sizeof(double));											
				double *m1T=auxmemory1[numTh];
				double *m2T=auxmemory3[numTh];		
				getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n2,nCores);			
				getSubMatrix(m2,r2,c2,ro2,co2,m2T, n1,n3,nCores);								
				cblas_dgemm (CblasColMajor, CblasTrans, CblasNoTrans, n2,n3,n1,K1, m1T, n1, m2T, n1, K2, mresultT, n2);				
				//free(m1T);
				//free(m2T);				
			}else{
				int i;
				for(i=0;i<n2*n3;i++) mresultT[i]=K2*mresultT[i];				
			}			
			putSubMatrix(result,rr,cr,ror,cor,mresultT, n2,n3,nCores);			
			//free(mresultT);
		}
	}else{
		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n2);
		int cols1B=n2-cols1A;
		int rows2A=ceil(0.5*n1);
		int rows2B=n1-rows1A;
		int cols2A=ceil(0.5*n3);
		int cols2B=n3-cols2A;
		
		if(numTh<posIni+nCores/2){
				TNNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols1A,cols2A,K1,K2,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
				TNNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1A,cols2A,K1,1.0,result,rr,ror,cr,cor, nCores/2,posIni,numTh);

				TNNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2,c2,co2,rows1A,cols1B,cols2A,K1,K2,result,rr,ror+cols1A,cr,cor,nCores/2,posIni,numTh);
				TNNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1B,cols2A,K1,1.0,result,rr,ror+cols1A,cr,cor, nCores/2,posIni,numTh);
		}else{
				TNNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1A,cols2B,K1,K2,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
				TNNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1A,cols2B,K1,1.0,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);

				TNNProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1B,cols2B,K1,K2,result,rr,ror+cols1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
				TNNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1B,cols2B,K1,1.0,result,rr,ror+cols1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
			
		}				

	}
}


void NNTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,int n3,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
	if(nCores <= 1){
		if(n1 >0 & n3>0){								
			//double *mresultT = (double *) malloc(n1*n3*sizeof(double));				
			double *mresultT=auxmemory2[numTh];		
			getSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n3,nCores);
			if(n2 > 0){
				//double *m1T = (double *) malloc(n1*n2*sizeof(double));							
				//double *m2T = (double *) malloc(n3*n2*sizeof(double));											
				double *m1T=auxmemory1[numTh];
				double *m2T=auxmemory3[numTh];	
				getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n2,nCores);
				getSubMatrix(m2,r2,c2,ro2,co2,m2T, n3,n2,nCores);
				
				cblas_dgemm (CblasColMajor, CblasNoTrans, CblasTrans, n1,n3,n2,K1, m1T, n1, m2T, n3, K2, mresultT, n1);
				//free(m1T);
				//free(m2T);
			}else{
				int i;
				for(i=0;i<n1*n3;i++) mresultT[i]=K2*mresultT[i];
			}						
			putSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n3,nCores);
			//free(mresultT);
		}
	}else{

		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n2);
		int cols1B=n2-cols1A;
		int rows2A=ceil(0.5*n3);
		int rows2B=n3-rows2A;
		int cols2A=ceil(0.5*n2);
		int cols2B=n2-cols2A;
	
		if(numTh<posIni+nCores/2){
				NNTProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols1A,rows2A,K1,K2,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
				NNTProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1A,cols1B,rows2A,K1,1.0,result,rr,ror,cr,cor, nCores/2,posIni,numTh);

				NNTProduct(m1,r1,ro1,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1A,cols1A,rows2B,K1,K2,result,rr,ror,cr,cor+rows2A, nCores/2,posIni,numTh);
				NNTProduct(m1,r1,ro1,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1A,cols1B,rows2B,K1,1.0,result,rr,ror,cr,cor+rows2A, nCores/2,posIni,numTh);
				
		}else{				

				NNTProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1A,rows2B,K1,K2,result,rr,ror+rows1A,cr,cor+rows2A, nCores/2,posIni+nCores/2,numTh);
				NNTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1B,rows2B,K1,1.0,result,rr,ror+rows1A,cr,cor+rows2A, nCores/2,posIni+nCores/2,numTh);
				
				NNTProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2,rows1B,cols1A,rows2A,K1,K2,result,rr,ror+rows1A,cr,cor,nCores/2,posIni+nCores/2,numTh);
				NNTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1B,cols1B,rows2A,K1,1.0,result,rr,ror+rows1A,cr,cor, nCores/2,posIni+nCores/2,numTh);				
			
		}			
	}
}


void AATProduct(double *m1,int r1,int ro1,int c1, int co1,int n1,int n2,double K1,double K2,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
	if(nCores <= 1){		
		if(n1 >0 & n2>0){
		double *m1T = (double *) malloc(n1*n2*sizeof(double));			
		double *mresultT = (double *) malloc(n1*n1*sizeof(double));				
		//double *m1T=auxmemory1[numTh];
		getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n2,nCores);
		//double *mresultT=auxmemory2[numTh];		
		getSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n1,nCores);
		cblas_dsyrk (CblasColMajor, CblasLower, CblasNoTrans, n1,n2,K1, m1T, n1, K2, mresultT, n1);
		int i,j;
		for(i=0;i<n1;i++){
			for(j=i;j<n1;j++){
				mresultT[j*n1+i]=mresultT[i*n1+j];
			}
		}				
		putSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n1,nCores);
		free(m1T);
		free(mresultT);
		}
	}else{
		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n2);
		int cols1B=n2-cols1A;
		
		if(numTh<posIni+nCores/2){
				AATProduct(m1,r1,ro1,c1,co1,rows1A,cols1A,K1,K2,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
				AATProduct(m1,r1,ro1,c1,co1+cols1A,rows1A,cols1B,K1,1.0,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
		}else{
				AATProduct(m1,r1,ro1+rows1A,c1,co1,rows1B,cols1A,K1,K2,result,rr,ror+rows1A,cr,cor+rows1A, nCores/2,posIni+nCores/2,numTh);
				AATProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,rows1B,cols1B,K1,1.0,result,rr,ror+rows1A,cr,cor+rows1A, nCores/2,posIni+nCores/2,numTh);
			
		}				
		NNTProduct(m1,r1,ro1+rows1A,c1,co1,m1,r1,ro1,c1,co1,rows1B,cols1A,rows1A,K1,K2,result,rr,ror+rows1A,cr,cor,nCores,posIni,numTh);
		NNTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m1,r1,ro1,c1,co1+cols1A,rows1B,cols1B,rows1A,K1,1.0,result,rr,ror+rows1A,cr,cor, nCores,posIni,numTh);
		
	}
}



void LNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
	if(nCores <= 1){
		if(n1 >0 & n2>0){
		//double *m1T = (double *) malloc(n1*n1*sizeof(double));			
		//double *mresultT = (double *) malloc(n2*n1*sizeof(double));				
		double *m1T=auxmemory2[numTh];
		double *mresultT=auxmemory1[numTh];
		getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n1,nCores);		
		getSubMatrix(m2,r2,c2,ro2,co2,mresultT, n1,n2,nCores);		
		cblas_dtrmm (CblasColMajor, CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,n1,n2,K1, m1T, n1, mresultT, n1);
		putSubMatrix(result,rr,cr,ror,cor,mresultT, n1,n2,nCores);
		//free(m1T);
		//free(mresultT);
	 }
	}else{
		
		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n1);
		int cols1B=n1-cols1A;
		int rows2A=ceil(0.5*n1);
		int rows2B=n1-cols1A;		
		int cols2A=ceil(0.5*n2);
		int cols2B=n2-cols2A;
		
		
		if(numTh<posIni+nCores/2){			
				LNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols2A,K1,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
				
				LNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols2A,K1,result,rr,ror+rows1A,cr,cor, nCores/2,posIni,numTh);
				NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2,rows1B,cols1A,cols2A,K1,1.0,result,rr,ror+rows1A,cr,cor,nCores/2,posIni,numTh,1);
		}else{
				LNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols2B,K1,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
				
				LNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols2B,K1,result,rr,ror+rows1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
				NNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1B,cols1A,cols2B,K1,1,result,rr,ror+rows1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh,1);
			
		}		
			
	}
}

void LTNProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){	
	if(nCores <= 1){
		if(n1 >0 & n2>0){
		//double *m1T = (double *) malloc(n1*n1*sizeof(double));			
		//double *m2T = (double *) malloc(n1*n2*sizeof(double));						int in;		
		int in;
		double *m1T=auxmemory1[numTh];
		double *m2T=auxmemory2[numTh];
		//double *mresultT=auxmemory3[numTh];
		getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n1,1);
		//getSubMatrix(m2,n1,n2,0,0,m2T, n1,n2,1);
		//getSubMatrix(m2,r2,c2,ro2,co2,m2T, n1,n2,nCores);		
		getSubMatrix(result,rr,cr,ror,cor,m2T, n1,n2,1);
		//for(in=0;in<n1*n2;in++){if(mresultT[in]!= m2T[in])printf("BO\n");}				
		//getSubMatrix(m2,r2,c2,ro2,co2,m2T, n1,n2,nCores);		
		//printf("Valores %d %d %d %d %d %f %f %f %f\n",n1,n2,K1,n1,n1,m1T[0],m1T[n1*n1-1],mresultT[0],mresultT[n1*n2-1]);
		cblas_dtrmm (CblasColMajor, CblasLeft,CblasLower,CblasTrans,CblasNonUnit,n1,n2,K1, m1T, n1, m2T, n1);
		//printf("Valores %d %d %d %d %d %f %f %f %f\n",n1,n2,K1,n1,n1,m1T[0],m1T[n1*n1-1],mresultT[0],mresultT[n1*n2-1]);
		putSubMatrix(result,rr,cr,ror,cor,m2T, n1,n2,1);
		//free(m1T);
		//free(m2T);
		}
	}else{
		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n1);
		int cols1B=n1-cols1A;
		int rows2A=ceil(0.5*n1);
		int rows2B=n1-cols1A;		
		int cols2A=ceil(0.5*n2);
		int cols2B=n2-cols2A;
		
						
		if(numTh<posIni+nCores/2){							
				LTNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2,rows1B,cols2A,K1,result,rr,ror+cols1A,cr,cor, nCores/2,posIni,numTh);				
				
				LTNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,cols2A,K1,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
				TNNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1B,cols1A,cols2A,K1,1.0,result,rr,ror,cr,cor,nCores/2,posIni,numTh);			

		}else{	
				LTNProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols2B,K1,result,rr,ror+cols1A,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);
				
				LTNProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2+cols2A,rows1A,cols2B,K1,result,rr,ror,cr,cor+cols2A, nCores/2,posIni+nCores/2,numTh);			
				TNNProduct(m1,r1,ro1+rows1A,c1,co1,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,cols1A,cols2B,K1,1.0,result,rr,ror,cr,cor+cols2A,nCores/2,posIni+nCores/2,numTh);									
				

		}				
	}
}


void NLProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
	if(nCores <= 1){
		if(n1 >0 & n2>0){
		//double *m1T = (double *) malloc(n1*n1*sizeof(double));			
		//double *mresultT = (double *) malloc(n2*n1*sizeof(double));	
		double *m1T=auxmemory1[numTh];
		double *mresultT=auxmemory2[numTh];
		getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n1,nCores);		
		getSubMatrix(m2,r2,c2,ro2,co2,mresultT, n2,n1,nCores);		
		cblas_dtrmm (CblasColMajor, CblasRight,CblasLower,CblasNoTrans,CblasNonUnit,n2,n1,K1, m1T, n1, mresultT, n2);
		putSubMatrix(result,rr,cr,ror,cor,mresultT, n2,n1,nCores);
		//free(m1T);
		//free(mresultT);
		}
	}else{

		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n1);
		int cols1B=n1-cols1A;
		int rows2A=ceil(0.5*n2);
		int rows2B=n2-rows2A;		
		int cols2A=ceil(0.5*n1);
		int cols2B=n1-cols2A;
		
		
		if(numTh<posIni+nCores/2){
				NLProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,rows2A,K1,result,rr,ror,cr,cor, nCores/2,posIni,numTh);
				NNProduct(m2,r2,ro2,c2,co2+cols2A,m1,r1,ro1+rows1A,c1,co1,rows2A,cols2B,cols1A,K1,1,result,rr,ror,cr,cor, nCores/2,posIni,numTh,2);
				
				NLProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1B,rows2A,K1,result,rr,ror,cr,cor+cols1A, nCores/2,posIni,numTh);				
		}else{
				NLProduct(m1,r1,ro1,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1A,rows2B,K1,result,rr,ror+rows2A,cr,cor,nCores/2,posIni+nCores/2,numTh);
			  NNProduct(m2,r2,ro2+rows2A,c2,co2+cols2A,m1,r1,ro1+rows1A,c1,co1,rows2B,cols2B,cols1A,K1,1,result,rr,ror+rows2A,cr,cor, nCores/2,posIni+nCores/2,numTh,2);						
				
				NLProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,rows2B,K1,result,rr,ror+rows2A,cr,cor+cols1A, nCores/2,posIni+nCores/2,numTh);			
		}						

	}
}


void NLTProduct(double *m1,int r1,int ro1,int c1, int co1,double *m2,int r2,int ro2,int c2, int co2,int n1,int n2,double K1,double *result,int rr,int ror,int cr, int cor, int nCores,int posIni,int numTh){
	if(nCores <= 1){
		if(n1 >0 & n2>0){
		//double *m1T = (double *) malloc(n1*n1*sizeof(double));			
		//double *m2T = (double *) malloc(n2*n1*sizeof(double));			
		double *m1T=auxmemory1[numTh];
		double *m2T=auxmemory2[numTh];
		getSubMatrix(m1,r1,c1,ro1,co1,m1T, n1,n1,nCores);		
		getSubMatrix(m2,r2,c2,ro2,co2,m2T, n2,n1,nCores);	
		cblas_dtrmm (CblasColMajor, CblasRight,CblasLower,CblasTrans,CblasNonUnit,n2,n1,K1, m1T, n1, m2T, n2);
		//double *mresultT = (double *) malloc(n2*n1*sizeof(double));					
		//getSubMatrix(result,rr,cr,ror,cor,mresultT, n2,n1,nCores);
		//int i;
		//for(i=0;i<n2*n1;i++)mresultT[i]=mresultT[i]+m2T[i];		
		putSubMatrix(result,rr,cr,ror,cor,m2T, n2,n1,nCores);

		//free(m1T);
		//free(m2T);
		//free(mresultT);		
		}
	}else{
		int rows1A=ceil(0.5*n1);
		int rows1B=n1-rows1A;
		int cols1A=ceil(0.5*n1);
		int cols1B=n1-cols1A;
		int rows2A=ceil(0.5*n2);
		int rows2B=n2-rows2A;
		int cols2A=ceil(0.5*n1);
		int cols2B=n1-cols2A;

		if(numTh<posIni+nCores/2){
				NLTProduct(m1,r1,ro1,c1,co1,m2,r2,ro2,c2,co2,rows1A,rows2A,K1,result,rr,ror,cr,cor,nCores/2,posIni,numTh);
				
				NLTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2,c2,co2+cols2A,rows1B,rows2A,K1,result,rr,ror,cr,cor+rows1A,nCores/2,posIni,numTh);
				NNTProduct(m2,r2,ro2,c2,co2,m1,r1,ro1+rows1A,c1,co1,rows2A,cols2A,rows1B,K1,1.0,result,rr,ror,cr,cor+rows1A,nCores/2,posIni,numTh);


		}else{			

				NLTProduct(m1,r1,ro1,c1,co1,m2,r2,ro2+rows2A,c2,co2,rows1A,rows2B,K1,result,rr,ror+rows2A,cr,cor, nCores/2,posIni+nCores/2,numTh);		

				NLTProduct(m1,r1,ro1+rows1A,c1,co1+cols1A,m2,r2,ro2+rows2A,c2,co2+cols2A,rows1B,rows2B,K1,result,rr,ror+rows2A,cr,cor+rows1A, nCores/2,posIni+nCores/2,numTh);				
				NNTProduct(m2,r2,ro2+rows2A,c2,co2,m1,r1,ro1+rows1A,c1,co1,rows2B,cols2A,rows1B,K1,1.0,result,rr,ror+rows2A,cr,cor+rows1A, nCores/2,posIni+nCores/2,numTh);										
									
				
			
		}		
		
		
		
	}
}



void getSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores){
	
	int j;	
	for(j=0;j<size4;j++){
		//printf("En A posicion %d guardo lo de matrix posicion %d de tamaÃ’o %d\n",j*size3,(j+O2)*size1+O1,size3);
			memcpy(&A[j*size3],&matrix[(j+O2)*size1+O1],size3*sizeof(double));
	}	
}

void putSubMatrix(double *matrix,int size1,int size2,int O1,int O2,double *A, int size3,int size4,int nCores){
	
	int j;	

	for(j=0;j<size4;j++){
			memcpy(&matrix[(j+O2)*size1+O1],&A[j*size3],size3*sizeof(double));
	}
	
}
