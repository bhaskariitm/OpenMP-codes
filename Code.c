#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include<omp.h>
#include<time.h>

/*normal implementation*/

void MatMul1( int m, int n, int p, int b, double alpha, double beta, double *A, double *B, double *D )
{   int i,j,k;

    
    for( i = 0 ; i < m ; i++ ){
	
      for( k = 0 ; k < p ; k++ )
      {
        D[i*p+k] *= beta ;
        for( j = 0 ; j < n ; j++ )
          D[i*p+k] += alpha*A[i*n+j]*B[j*p+k] ;
      }
    }


}

// Cache efficient implementation
void MatMul2( int m, int n, int p, int b, double alpha, double beta, double *A, double *B, double *C )
{  																				 
		omp_set_num_threads(4);
	#pragma omp parallel shared(m,n,p,b,alpha,beta,A,B,C) num_threads(4)
	{
		double sum=0;
		int x,y,z,k,i,j;
		
		 #pragma omp for schedule(dynamic) collapse(3)
			for (  i=0; i<n; i+=b ){
			
		        for ( j=0; j<p; j+=b ){
				
		            for ( k=0; k<m; k+=b ){
					 
		            /* normal multiply inside the blocks */
		                for (  x=k; x<k+b; x++ ){
						
		                    for ( y=j; y<j+b; y++ ){
								sum=C[x*p+y];
								if(i==0)
									sum=beta*C[x*p+y];
		                        for ( z=i; z<i+b; z++ ){
					
									sum += alpha*A[x*n+z]*B[z*p+y];
								}
						
							#pragma omp critical
								C[x*p+y]=sum;
						
							}
							
						}
					}
				}
			}
    }
}

// cache-inefficient implementation
void MatMul3( int m, int n, int p, int b, double alpha, double beta, double *A, double *B, double *C )
{   omp_set_num_threads(4);
	#pragma omp parallel shared(m,n,p,b,alpha,beta,A,B,C) 
	{
		double sum=0;
		int x,y,z,k,i,j;
		 #pragma omp for schedule(static) collapse(3)
		 			for (  i=0; i<m; i+=b ){
			
		        		for ( j=0; j<p; j+=b ){
				
		            		for ( k=0; k<n; k+=b ){
					
		             //normal multiply inside the blocks 
		                		for (  y=i; y<i+b; y++ ){
						
		                    		for ( x=j; x<j+b; x++ ){
										sum=C[y*p+x];
										if(k==0)
											sum=beta*C[y*p+x];
		                        		for ( z=k; z<k+b; z++ ){
					
											sum += alpha*A[y*n+z]*B[z*p+x];
										}
										if(k==0){
											#pragma omp critical
											C[y*p+x]=sum;
										}
										else
											C[y*p+x]=sum;
									}
								}
							}
						}
					}	
    
	}
}

int main(){
    double *A, *B, *C,*D,*T ,alpha, beta ;int count=0;
    int m, n, p, b,row,col;
    printf("\nenter m,n,p,b\n");
    scanf("%d %d %d %d",&m,&n,&p,&b);
    
    A = (double *) malloc( sizeof(double) * m * n ) ;
    B = (double *) malloc( sizeof(double) * n * p ) ;
    C = (double *) malloc( sizeof(double) * m * p ) ;
    D = (double *) malloc( sizeof(double) * m * p ) ;
    T = (double *) malloc( sizeof(double) * m * p ) ;
    
    if ( ( A == NULL ) || ( B == NULL ) || ( C == NULL ) )
    {
        printf( "Out of Memory\n" ) ;
        exit(1) ;
    }
    for( row = 0 ; row < m ; row++ )
        for( col = 0 ; col < p ; col++ )
        	C[row*p+col]=1;
        	
     for( row = 0 ; row < m ; row++ )
        for( col = 0 ; col < p ; col++ )
        	T[row*p+col]=1;
    
    for( row = 0 ; row < m ; row++ )
        for( col = 0 ; col < p ; col++ )
        	D[row*p+col]=1;

    // m = n = p = 64 ;
    // b = 16 ;
   // printf("\nenter 1 mat\n");
	for( row = 0 ; row < m ; row++ )
            for( col = 0 ; col < n ; col++ )
                A[row*n+col]=5.0 ;
    //printf("\n=============================================================\n");
	
   // printf("\nenter 2 mat\n");
	for( row = 0 ; row < n ; row++ )
            for( col = 0 ; col < p ; col++ )
                B[row*p+col]=2.0 ; 

	
	
	double t1 = omp_get_wtime();
    MatMul1(m,n,p,b,7,5,A,B,D);
    double t2 = omp_get_wtime();
    MatMul2(m,n,p,b,7,5,A,B,C);
    double t3 = omp_get_wtime();
    MatMul3(m,n,p,b,7,5,A,B,T);
	double t4 = omp_get_wtime();
 
	printf("\nmm1: %0.20f \t  mm2: %0.20f \t mm3: %0.20f",t2-t1,t3-t2,t4-t3);
	printf("\n");
	 
	 
	 printf("\n========  D   =============\n");
	for( row = 0 ; row < m ; row++ )
            for( col = 0 ; col < p ; col++ )
                printf("%f : ",D[row*p+col]); 
            printf("\n");


     printf("\n=======    C    =============\n");
	for( row = 0 ; row < m ; row++ )
            for( col = 0 ; col < p ; col++ )
                printf("%f : ",C[row*p+col]); 
            printf("\n");

 printf("\n========    T   ===============\n");
	for( row = 0 ; row < m ; row++ )
            for( col = 0 ; col < p ; col++ )
                printf("%f : ",T[row*p+col]); 
            printf("\n");


	
}

