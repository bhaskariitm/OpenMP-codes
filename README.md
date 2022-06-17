# OpenMP-codes
This repository has openMP codes for NPTEL course : introduction to parallel programming in openMP

## Problem statement: 
Given two matrices `A` and `B`, create a matrix `C = alpha*A*B + beta*C`, where `alpha` and `beta` are constant numbers taken as user input.
Assume `C` has been initialized with the value of 1 in each cell.
 
### Usual implementation (serial code)

```C
void MatMul1( int m, int n, int p, int b, double alpha, double beta, double *A, double *B, double *C ){
  
  int i,j,k;
  
  for( i = 0 ; i < m ; i++ ){
	    for( k = 0 ; k < p ; k++ ){
          C[i*p+k] *= beta ;
          for( j = 0 ; j < n ; j++ )
              C[i*p+k] += alpha*A[i*n+j]*B[j*p+k] ;
      }
  }
}
```


### Parallel Program: 
In my system i have 4 cores in the processor, and i have used 4 threads and set have `OMP_PLACES`, `OMP_PROC_BIND` environment variables accordingly

Here we are trying to create the `C` matrix by dividing the larger matrices into smaller blocks, these blocks are like submatrices and the overall task of multiplication is distributed among the spawned threads. We have to note that, only the block multiplications must be parallelized among threads, but inside the block which ahs been allocated already a single thread should work inorder to maintain consistency. In order to have this we use `collapse(3)` so that the problem space is collapsed upto 3 levels which means the first 3 for loops would be unwounded. 
   
```C
void MatMul( int m, int n, int p, int b, double alpha, double beta, double *A, double *B, double *C )
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
								#pragma omp critical
									C[y*p+x]=sum;
								
							}
						 }
					   }
				    }
			   }
	
  	}
 }

```
**Observation**
1.  This algorithm gives good performance only when the input sizes are large, parallelization does not mean always we will get better result.
    Reason: For a small data set the overhead of spawning multiple threads, critical section at line 57 also stops multiple threads to write parallely.Also when a         thread writes to the matrix `C` it also invalidates the entry in local cache of other or update it. All these overheads are overpowered by the parallel execution       when the problem space is relatively large and hence the serial code takes a comparitively larger time. 
  2.  The scheduling of work can be made dynamic or static i.e. we can use `#pragma omp for schedule(dynamic) collapse(3)`, but in this case what we observe is it does not give that much improvement, most probably the reason for that is, in this code there is not a lot of conditional execution and hence most iterations would have same amount of work for the threads i.e. there program flow path would have similar workload and hence dynamic scheduling ends up equipartitioning the work at the end, as threads take similar amount of times to complete.
  3.  We must also try to reduce the critical section as much as possible, this increase the performance of the code as there is less restriction on thread execution, that's why we use a separate private variable `sum` for each thread which will store the sum calculated by that thread, and then while updating the value in the shared space of matrix `C` we do that in a critical section.
  4.  This approach of dividing the matrix into blocks and then allocating these block multiplication to threads incorporates a lot of memory accesses because the ususal way of matrix multiplication does not take into account the locality of reference in cache. 
  5.  Even the choice of block size greatly determines our code's performance, reason being the reusability of cache, if block size is very less the execution boils down to element by element multiplication and totally ignores the block approach of parallelization. On the other hand very large block size means it may go out of cache and reusability of the data could not be exploited.
  
*In general cache can be advantageous when either of these two things can happen:*
 1) *Reusability of the data stored (temporal locality)*
 2) *Cache line stores data contiguously located in the memory, this data may be used by the algo in subsequent iteration (spatial Locality)*


 ### Cache Efficient Block Multiplication approach:
 In usual matrix multiplication we multiply row of one matrix with column of other, but in this approach everytime data corresponding to new row and column is fetched from the memory in cache which means the cache hit ratio is less and hence the speed of execution falls. In a block approach if we can calculate all the partial product terms corresponding to a given block first then we can exploit the reusability of cache data in the process and hence speed of execution increases.
So in this approach we just swap the loops in `Line 39` and `Line 43` of previous code, and this gives us a cache efficient multiplication block code. 

```C
void MatMul( int m, int n, int p, int b, double alpha, double beta, double *A, double *B, double *C )
{   omp_set_num_threads(4);
	#pragma omp parallel shared(m,n,p,b,alpha,beta,A,B,C) 
	{
		double sum=0;
		int x,y,z,k,i,j;
		 #pragma omp for schedule(static) collapse(3)
		 	    for ( k=0; k<n; k+=b ){
			
		        	for ( j=0; j<p; j+=b ){
				
		            		for (  i=0; i<m; i+=b ){
					
		             		//normal multiply inside the blocks 
		                		for (  y=i; y<i+b; y++ ){
						
		                    			for ( x=j; x<j+b; x++ ){
								sum=C[y*p+x];
								if(k==0)
								        sum=beta*C[y*p+x];
		                        		
                                				for ( z=k; z<k+b; z++ ){
					                        	sum += alpha*A[y*n+z]*B[z*p+x];
								}
								#pragma omp critical
									C[y*p+x]=sum;
								
							}
						 }
					   }
				    }
			   }
	
  	}
 }
```

This is the final code which I had submitted.
