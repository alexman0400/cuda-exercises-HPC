/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define TILE_WIDTH	32
#define FILTER_LENGTH	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005

__constant__ float c_Filter[33];
 
void cudaCheckErrors() {
	cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
		// something's gone wrong
		// print out the CUDA error as a string
		printf("CUDA Error: %s\n", cudaGetErrorString(error));
		// we can't recover from the error -- exit the program
		exit(0);
	}
}
////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
}

__global__ void convolutionRowGPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {
	int k;
	float sum;
	
	// create shorthand names for threadIdx & blockIdx
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x,  by = blockIdx.y;

	// allocate 2D tiles in __shared__ memory
	__shared__ float s_Src[TILE_WIDTH][TILE_WIDTH];

	// calculate the row & column index of the element
	int y = by*blockDim.y + ty;
	int x = bx*blockDim.x + tx;
		
	for (int m = 0; m < (2 * filterR -1)/TILE_WIDTH; m++) {
		s_Src[tx][ty] = h_Src[(y * TILE_WIDTH) + ( m*TILE_WIDTH + x)];
		__syncthreads();
		sum = 0;
		for (k = -filterR; k <= filterR-1; k++) {
			int d = tx + k;
			if (d >= 0 && d < TILE_WIDTH) {
			  sum += s_Src[d%TILE_WIDTH][ty + d/TILE_WIDTH] * c_Filter[filterR - k];
			}
			h_Dst[y * imageW + x] = sum;
		}
	
		__syncthreads();
	}
}
__global__ void convolutionColumnGPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {
	int k;
	float sum = 0;
	
	// create shorthand names for threadIdx & blockIdx
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x,  by = blockIdx.y;

	// allocate 2D tiles in __shared__ memory
	__shared__ float s_Src[TILE_WIDTH][TILE_WIDTH];

	// calculate the row & column index of the element
	int y = by*blockDim.y + ty;
	int x = bx*blockDim.x + tx;
	
	for (int m = 0; m < imageH/TILE_WIDTH; m++) {
		s_Src[tx][ty] = h_Src[(m*TILE_WIDTH + ty)* imageW + x];
		__syncthreads();
	
		for (k = -filterR; k <= filterR; k++) {
			int d = y + k;

			if (d >= 0 && d < TILE_WIDTH) {
			  sum += s_Src[tx + d/TILE_WIDTH][d%TILE_WIDTH] * c_Filter[filterR - k];
			}   

			h_Dst[y * imageW + x] = sum;
		  }
	
		__syncthreads();
	}
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
	struct timespec  tv1, tv2;
	
	float error/* maxerror=0 */;
	float elapsedTime;
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
	*h_Output,
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;

	int imageW;
    int imageH;
    unsigned int i;
	
	dim3 threadsPerBlock, blocks;
	
	cudaEvent_t S,E;
	cudaEventCreate(&S);
	cudaCheckErrors();
	cudaEventCreate(&E);
	cudaCheckErrors();
	
	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
	h_Output = (float *)malloc(imageW * imageH * sizeof(float));
    if (h_Filter==NULL || h_Input==NULL || h_Buffer==NULL || h_OutputCPU==NULL) {
      printf("error with malloc");
      return 0;
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }


    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
	
	
			
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    
	printf("Allocating and initializing device arrays...\n");
    cudaMalloc( (void**) &d_Filter, FILTER_LENGTH * sizeof(float));
	cudaCheckErrors();
    cudaMalloc( (void**) &d_Input, imageW * imageH * sizeof(float));
	cudaCheckErrors();
    cudaMalloc( (void**) &d_Buffer, imageW * imageH * sizeof(float));
	cudaCheckErrors();
    cudaMalloc( (void**) &d_OutputGPU, imageW * imageH * sizeof(float));
	cudaCheckErrors();
	
    
	cudaMemset(d_Buffer, 0, imageW * imageH);
	cudaCheckErrors();
	cudaMemset(d_OutputGPU, 0, imageW * imageH);
    cudaCheckErrors();
	
	cudaEventRecord(S, NULL);
	cudaCheckErrors();
	
    printf("Copying the arrays from host to device...\n");
	cudaMemcpyToSymbol(c_Filter, h_Filter, FILTER_LENGTH * sizeof(float));
	cudaDeviceSynchronize();
	cudaCheckErrors();
	
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudaCheckErrors();
	
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudaCheckErrors();
    
    printf("GPU computation...\n");
	
	threadsPerBlock.x = imageH;
	blocks.x = 1;
	for(i=threadsPerBlock.x; i>32; i=i/2) {
		threadsPerBlock.x = threadsPerBlock.x/2;
		blocks.x = blocks.x*2;
	}
	threadsPerBlock.y = threadsPerBlock.x;
	blocks.y = blocks.x;
	
	printf("Number of blocks = %d * %d\nNumber of threads per block = %d * %d\n",blocks.x, blocks.x, threadsPerBlock.x, threadsPerBlock.x);
	convolutionRowGPU<<<blocks,threadsPerBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius); // convolution kata grammes
	cudaDeviceSynchronize();
	cudaCheckErrors();
	
	convolutionColumnGPU<<<blocks,threadsPerBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius); // convolution kata sthles
	cudaDeviceSynchronize();
	cudaCheckErrors();
	
	cudaMemcpy(h_Output, d_OutputGPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheckErrors();
    
	cudaEventRecord(E, NULL);
	cudaCheckErrors();
	cudaEventSynchronize(E);
	cudaCheckErrors();
	
	cudaEventElapsedTime(&elapsedTime, S, E); 
	cudaCheckErrors();
	
	printf ("CPU execution time = %10g seconds\n",
			(float) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(float) (tv2.tv_sec - tv1.tv_sec));
	printf("GPU execution Time: %f\n", elapsedTime); // Print Elapsed time
	
	// Destroy CUDA Event API Events
	cudaEventDestroy(S);
	cudaCheckErrors();
	cudaEventDestroy(E);
	cudaCheckErrors();
	
	
	
	
	
	
	////error calculation
    // for (i = 0; i < imageW * imageH; i++) {
		// error = ABS(h_Output[i] - h_OutputCPU[i]);
		// printf("%d: %f - %f\n", i, h_Buffer[i], h_Output[i]);
		// if(error>accuracy){
			// maxerror = error;
			// printf("maxerror is: %d\n", maxerror);
			// printf("error spotted\n");
			// free all the allocated memory
			// free(h_OutputCPU);
			// free(h_Buffer);
			// free(h_Input);
			// free(h_Filter);
			// free(h_Output);
			// cudaFree(d_OutputGPU);
			// cudaFree(d_Buffer);
			// cudaFree(d_Input);
			// cudaFree(d_Filter);
			
			//// Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
			// cudaDeviceReset();
			// return 0;
		// }
	// }
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    free(h_Output);
	cudaFree(d_OutputGPU);
	cudaFree(d_Buffer);
    cudaFree(d_Input);
    cudaFree(d_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
