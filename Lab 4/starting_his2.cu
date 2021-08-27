#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);
void run_gpu_gray_test(PGM_IMG img_in, char *out_filename);
__global__ void histogram_kernel(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin, int min, int d);
void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);
void free_pgm(PGM_IMG img);
void write_pgm(PGM_IMG img, const char * path);
PGM_IMG read_pgm(const char * path);

// Prints out the error made while a CUDA kernel is running, if there is one
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

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;
	
	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
    run_cpu_gray_test(img_ibuf_g, argv[2]);
	run_gpu_gray_test(img_ibuf_g, argv[2]);
    free_pgm(img_ibuf_g);

	return 0;
}

// Function that prints out the CUDA runtime of the grayscale image enhancement
void run_gpu_gray_test(PGM_IMG img_in, char *out_filename)
{
	cudaEvent_t S,E;
	cudaEventCreate(&S);
	cudaCheckErrors();
	cudaEventCreate(&E);
	cudaCheckErrors();
	PGM_IMG	h_Result;
    int hist[256], i = 0, min = 0, d, *d_Hist;
	float elapsedTime;
	unsigned char *d_Result, *d_Img;
	
	h_Result.img = (unsigned char *)malloc(sizeof(unsigned char)*img_in.h * img_in.w);
	
	h_Result.w = img_in.w;
	h_Result.h = img_in.h;
	
	cudaEventRecord(S, NULL);
	cudaCheckErrors();
	
	histogram(hist, img_in.img, img_in.h * img_in.w, 256);
	
	//Memory allocation on GPU
	printf("Allocating GPU memory...\n");
	cudaMalloc( (void**) &d_Hist, 256 * sizeof(int));
	cudaCheckErrors();
	cudaMalloc( (void**) &d_Img, img_in.w * img_in.h * sizeof(unsigned char));
	cudaCheckErrors();
	cudaMalloc( (void**) &d_Result, img_in.w * img_in.h * sizeof(unsigned char));
	cudaCheckErrors();
	cudaMemcpy(d_Hist, hist, 256 * sizeof(256), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudaCheckErrors();
	cudaMemcpy(d_Img, img_in.img, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudaCheckErrors();
	cudaMemset(d_Result, 0, img_in.w * img_in.h * sizeof(unsigned char));
	cudaDeviceSynchronize();
	cudaCheckErrors();
	
	printf("Starting GPU processing...\n");
	
	//Kernel
	while(min == 0){
        min = hist[i++];
    }
	d = img_in.w * img_in.h - min;
	histogram_kernel<<<img_in.h*img_in.w/1024,1024>>>(d_Result,d_Img,d_Hist, img_in.w * img_in.h, 256, min, d);
    
	cudaMemcpy(h_Result.img, d_Result, img_in.w * img_in.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaCheckErrors();
	
	cudaEventRecord(E, NULL);
	cudaCheckErrors();
	cudaEventSynchronize(E);
	cudaCheckErrors();
	
	cudaEventElapsedTime(&elapsedTime, S, E); 
	cudaCheckErrors();
	
	printf("GPU execution Time: %f\n", elapsedTime); // Print Elapsed time
	
    write_pgm(h_Result, out_filename);
	
	// Destroy CUDA Event API Events
	cudaEventDestroy(S);
	cudaCheckErrors();
	cudaEventDestroy(E);
	cudaCheckErrors();
	
	//free memory everywhere
	cudaFree(d_Result);
	cudaCheckErrors();
	cudaFree(d_Img);
	cudaCheckErrors();
	cudaFree(d_Hist);
	cudaCheckErrors();
    free_pgm(h_Result);
	
	cudaDeviceReset();
}

// Main kernel which runs image enhancerment using histogram equalization
__global__ void histogram_kernel(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin, int min, int d) {
	int pos = threadIdx.x + blockIdx.x*blockDim.x;
	int cdf;
	int i = -1;
	__shared__ int lut[256];
	
	if (threadIdx.x < 256) {
		do {
			i++;
			cdf += hist_in[i];
		}
		while(i < threadIdx.x);
		lut[threadIdx.x] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[threadIdx.x] < 0){
            lut[threadIdx.x] = 0;
        }
	}
	__syncthreads();
	if(lut[img_in[pos]] > 255){
		img_out[pos] = 255;
	}
	else{
		img_out[pos] = (unsigned char)lut[img_in[pos]];
	}
}

// Run the CPU version of the Algorithm
void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    PGM_IMG img_obuf;
    
    
    printf("Starting CPU processing...\n");
    img_obuf = contrast_enhancement_g(img_in);
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
}

// Read the input image
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

// Write the new image created after the enehancement was implemented
void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

// Free memory of the struct that the image saved in
void free_pgm(PGM_IMG img)
{
    free(img.img);
}


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

// Perform image sharpening with CPU
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
        
    }
    
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}


PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
    return result;
}
