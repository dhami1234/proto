/*
  Copyright Marcin Krotkiewski, University of Oslo, 2012


  updated by Brian Van Straalen 2018
   updated for the loss of cutil.h
   updated to use 6 streams of execution
   updated to use std::chrono

  compile command
  >nvcc -std=c++11 -O3 vector_main.cu -o vector_main.exe

  options:   -nx
             -ny
             -nz
             -nstream
             -nbox
             -niter
             -pitch
             -pitchy
             -routine


*/
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <vector>
//#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cooperative_groups.h>
#include <cuda_profiler_api.h>

#define HERE fprintf(stderr, "HERE %d\n", __LINE__);
#define MSINGLE
//#undef MSINGLE
#ifdef MSINGLE
typedef float mfloat;
#else
typedef double mfloat;
#endif


#define cutilSafeCall(err)     __cudaSafeCall   (err, __FILE__, __LINE__)
#define cutilCheckError(err)   __cutilCheckError(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line){
  if(cudaSuccess != err) {
    printf("%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}
inline void __cutilCheckError( bool err, const char *file, const int line )
{
    if( true != err) {
        fprintf(stderr, "CUTIL CUDA error in file <%s>, line %i.\n",
                file, line);
        exit(-1);
    }
}
mfloat h_kernel_3c_all[3*3*3] = {-1./12, -1./6, -1./12,
				 -1./6 ,  0. , -1./6,
				 -1./12, -1./6, -1./12,
			 
				 -1./6 , 0., -1./6,
				 0., 2.+2.0/3.0, 0.,
				 -1./6, 0., -1./6,
			 
				 -1./12, -1./6, -1./12,
				 -1./6 ,  0. , -1./6,
				 -1./12, -1./6, -1./12};

__device__ __constant__ mfloat d_kernel_3c[3*3*3];


//cudaExtent gridExtent;

cudaArray *cu_array;
//cudaPitchedPtr p_T1, p_T2;
//mfloat *d_T1, *d_T2;
//mfloat *h_T1, *h_T2;


extern "C"{
#include "kernels2.cu"
}

__inline__ mfloat host_convolution_3x3(const mfloat *kernel, const mfloat *data,
				       const int tx, const int ty, const int bx,
				       int nx, int ny, int nz)
{
  // periodic boundaries in X-Y
  int txm = (tx-1+nx)%(nx);
  int txp = (tx+1)%(nx);
  int tym = (ty-1+ny)%(ny);
  int typ = (ty+1)%(ny);

  return 
    kernel[0]*data[txm + tym*bx] +
    kernel[1]*data[tx  + tym*bx] +
    kernel[2]*data[txp + tym*bx] +

    kernel[3]*data[txm + ty*bx] +
    kernel[4]*data[tx  + ty*bx] +
    kernel[5]*data[txp + ty*bx] +

    kernel[6]*data[txm + typ*bx] +
    kernel[7]*data[tx  + typ*bx] +
    kernel[8]*data[txp + typ*bx] ;
}


void host_convolution(mfloat *out, const mfloat *in, int nx, int ny, int nz, int pitch, int pitchy, mfloat *kernel)
{
  mfloat temp;
  
  for(int k=1; k<nz-1; k++){
    for(int j=0; j<ny; j++){
      for(int i=0; i<nx; i++){
	temp  =  host_convolution_3x3(kernel   , in + (k-1)*pitch*pitchy, i, j, pitch, nx, ny, nz);
	temp +=  host_convolution_3x3(kernel+9 , in + (k-0)*pitch*pitchy, i, j, pitch, nx, ny, nz);
	temp +=  host_convolution_3x3(kernel+18, in + (k+1)*pitch*pitchy, i, j, pitch, nx, ny, nz);
	out[k*pitch*pitchy + j*pitch + i] = temp;
      }
    }
  }
}


void copy_cube_simple(void *d, void *s, int nx, int ny, int nz, int kind, cudaStream_t stream=0)
{
  switch(kind){
  case cudaMemcpyHostToDevice:
    cutilSafeCall(cudaMemcpyAsync(d, s, nx*ny*nz*sizeof(mfloat), cudaMemcpyHostToDevice, stream));
    break;
  case cudaMemcpyDeviceToDevice:
    cutilSafeCall(cudaMemcpyAsync(d, s, nx*ny*nz*sizeof(mfloat), cudaMemcpyDeviceToDevice, stream));
    break;
  case cudaMemcpyDeviceToHost:
    cutilSafeCall(cudaMemcpyAsync(d, s, nx*ny*nz*sizeof(mfloat), cudaMemcpyDeviceToHost, stream));
    break;
  }
}

 using namespace std::chrono;


void ctoc(high_resolution_clock::time_point timer, uint niter, float unit_mem, int nrw_center, int nro_halo, int thrdim_x, int thrdim_y, int nx, int ny, int nz)
{  
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2-timer);
  double fps = ((double)niter) / (time_span.count());
  double halo_overhead = (double)(2*thrdim_x + 2*(thrdim_y+2)*32/sizeof(mfloat))/(double)(thrdim_x*thrdim_y);
  double effmembwd = (nrw_center+nro_halo)*unit_mem / (time_span.count()/ (double)niter);
  double hwmembwd  = (nrw_center+nro_halo+halo_overhead)*unit_mem / (time_span.count()/ (double)niter);
  double ptsthrough = (double)nx*ny*nz*niter/(double)(time_span.count());
  fprintf(stderr, "(%d, %d, %d): (TX, TY) = (%d, %d), fps %e, time %e, pts/s %e, effmembwd GB/s %3.1e, overhead %3.3e hwmembwd (GB/s) %3.3e)\n", 
	  nx, ny, nz, thrdim_x, thrdim_y, fps, time_span.count(), ptsthrough, effmembwd/1e9, halo_overhead, hwmembwd/1e9);
}


void compute_difference(void *ptr, mfloat *h_T1, mfloat *h_T2, int nx, int ny, int nz, int pitch, int pitchy, int thrdim_x, int thrdim_y, float niter)
{
  double temp = 0;

  bzero(h_T1, sizeof(mfloat)*pitch*pitchy*nz);
  if(ptr){
    copy_cube_simple(h_T1, ptr, pitch, pitchy, nz, cudaMemcpyDeviceToHost);
  }

  for(int k=0; k<nz; k++){
    for(int j=0; j<ny; j++){
      for(int i=0; i<nx; i++){
	temp  =  std::max(temp, (double)fabs(h_T1[k*pitch*pitchy + j*pitch + i] - h_T2[k*pitch*pitchy + j*pitch + i]));
      }
    }
  }

  printf("validation, CPU vs GPU: %e\n", temp);
} 


dim3 get_grid(dim3 block, int nx, int ny, int nz, int thrdim_x, int thrdim_y)
{
  int modx = nx%thrdim_x;
  int mody = ny%thrdim_y;
  
  dim3 grid(nx/block.x, ny/block.y, 1);
  if(modx){
    grid.x++;
  }
  if(mody){
    grid.y++;
  }
  return grid;
}

void GetCmdLineArgumenti(int argc, const char** argv, const char* name, int* rtn)
{
  size_t len = strlen(name);
  for(int i=1; i<argc; i+=2)
    {
      if(strcmp(argv[i]+1,name) ==0)
        {
         *rtn = atoi(argv[i+1]);
         std::cout<<name<<"="<<" "<<*rtn<<std::endl;
          break;
        }
    }
}

int bigTest(int argc, char*argv[])
{

  using std::vector;
  int device = 0;
  int nx = 128;
  int ny = 128;
  int nz = 128;

  // thrdim_y = 6 makes it so that the number of interior values that a block uses is
  // a multiple of the number of halo values used. See Krotiewski p. 539
  // M * (BX+2 + 2*(BY+2)*8) = BX*BY --> if M = 1 and BX = 32, BY = 6
  int routine = 2, thrdim_x = 32, thrdim_y = 6;
  int nstream = 8;
  int nbox = 64;
  /* -------------------- */
  /* command-line parameters */
  /* -------------------- */
  GetCmdLineArgumenti(argc, (const char**)argv, "nx", &nx);
  ny = nx;
  nz = nx;
  GetCmdLineArgumenti(argc, (const char**)argv, "ny", &ny);
  GetCmdLineArgumenti(argc, (const char**)argv, "nz", &nz);
  GetCmdLineArgumenti(argc, (const char**)argv, "nbox", &nbox);
  GetCmdLineArgumenti(argc, (const char**)argv, "nstream", &nstream);
  GetCmdLineArgumenti(argc, (const char**)argv, "routine", &routine);
  GetCmdLineArgumenti(argc, (const char**)argv, "device", &device);
  int niter = 10*nstream;
  GetCmdLineArgumenti(argc, (const char**)argv, "niter", &niter);
  void* junk;
  cudaMalloc(&junk, 10); // force a cuda runtime initialization;
  
  omp_set_dynamic(0);
  omp_set_num_threads(nstream);
  vector<cudaStream_t> streams(nstream);
  for(int istream = 0; istream < nstream; istream++)
  {
    cudaStreamCreate(&streams[istream]);
  }

  /* choose device */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  cudaSetDevice(device);

  int pitch  = nx;
  int pitchy = ny;

  /* special case - much worse performance with default pitch on tesla1060 */
  if(nx==352) pitch = nx + 64;

  GetCmdLineArgumenti(argc, (const char**)argv, "pitch", &pitch);
  GetCmdLineArgumenti(argc, (const char**)argv, "pitchy", &pitchy);


  /* -------------------- */
  /* Initialization */
  /* -------------------- */

  /* allocate alligned 3D data on the GPU */
  //gridExtent = make_cudaExtent(pitch*sizeof(mfloat), pitchy, nz);

  //std::cout<< "grid extent depth = " << gridExtent.depth << ", height = "<< gridExtent.height << ", width = " << gridExtent.width << std::endl;

  //vector<cudaPitchedPtr> vec_p_T1(nbox);
  //vector<cudaPitchedPtr> vec_p_T2(nbox);
  vector<mfloat*> vec_h_T1(nbox);
  vector<mfloat*> vec_h_T2(nbox);
  vector<mfloat*> vec_d_T1(nbox);
  vector<mfloat*> vec_d_T2(nbox);

  mfloat *workspace1, *workspace2;
  int patchSize=nx*ny*nz*sizeof(mfloat);
  patchSize+=(sizeof(mfloat)*2*nx + 2*(ny+2)*32)*nz;
  int pad = 32/sizeof(mfloat);
  cudaMalloc(&workspace1, nbox*patchSize);
  cudaMalloc(&workspace2, nbox*patchSize);

  for(int ibox = 0; ibox < nbox; ibox++)
  {
 
    //cutilSafeCall(cudaMalloc3D(&(vec_p_T1[ibox]), gridExtent));
    //cutilSafeCall(cudaMalloc3D(&(vec_p_T2[ibox]), gridExtent));

    //vec_d_T1[ibox]  = (mfloat*)(vec_p_T1[ibox].ptr);
    //vec_d_T2[ibox]  = (mfloat*)(vec_p_T2[ibox].ptr);
    vec_d_T1[ibox] = workspace1+ibox*nz*(ny*nx +2*nx+2*(ny+2)*pad);
    vec_d_T2[ibox] = workspace2+ibox*nz*(ny*nx +2*nx+2*(ny+2)*pad);
  }

  //set memory and allocate host data
  mfloat* h_T1;
  cutilSafeCall(cudaMallocHost(&h_T1, patchSize));
  srand(1);
  for(int k=0; k<nz; k++)
      for(int j=0; j<ny; j++)
          for(int i=0; i<nx; i++) {
              int index = 2*pad*(k*(ny+2)+j+1.5) + nx*(k*(ny+2)+j+1) + i;
              h_T1[index] = 1.0 - 2.0*(mfloat)(rand()/RAND_MAX);
          }
  for(int k=0; k<nz; k++) {
    long offset = k*(nx+2*pad)*(ny+2);
    memcpy(&h_T1[offset+pad], &h_T1[offset+ny*(nx+2*pad)+pad], nx*sizeof(mfloat));
    memcpy(&h_T1[offset+(ny+1)*(nx+2*pad)+pad], &h_T1[offset+nx+3*pad], nx*sizeof(mfloat));
    for(int j=0; j<ny+2; j++) {
        memcpy(&h_T1[offset+j*(nx+2*pad)], &h_T1[offset+j*(nx+2*pad)+nx], 32);
        memcpy(&h_T1[offset+j*(nx+2*pad)+nx+pad], &h_T1[offset+j*(nx+2*pad)+pad], 32);
    }
  }
  /*cudaMemcpyAsync(h_T1[pad], h_T1[ny*(nx+2*pad)+pad], nx*sizeof(mfloat), cudaMemcpyDeviceToDevice, 0);
  cudaMemcpyAsync(h_T1[(ny+1)*(nx+2*pad)+32*sizeof(mfloat)], nx+3*pad, nx*sizeof(mfloat), cudaMemcpyDeviceToDevice, 1);
  cudaMemcpy3DParms parms = {0};
  parms.extent = make_cudaExtent(32, ny+2, nz);
  parms.dstPos = make_cudaPos(0, 0, 0);
  parms.srcPos = make_cudaPos(nx, 0, 0);
  parms.kind = cudaMemcpyDeviceToDevice;
  parms.srcPtr = vec_d_T1[0];
  parms.dstPtr = vec_d_T1[0];
  cudaPos right = make_cudaPos(nx+pad, 0, 0);*/
  copy_cube_simple(vec_d_T1[0], h_T1, nx+2*pad, ny+2, nz, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  for(int ibox = 1; ibox < nbox; ibox++)
  {
 
    int istream=ibox%nstream;
    
    //mfloat* h_T1 = vec_h_T1[ibox];
    //mfloat* h_T2 = vec_h_T2[ibox];
    mfloat* d_T1 = vec_d_T1[ibox];
    //mfloat* d_T2 = vec_d_T2[ibox];

    //pitch = vec_p_T1[ibox].pitch/sizeof(mfloat);

    // cutilSafeCall(cudaMemset(vec_d_T1[ibox], 0, pitch*pitchy*nz*sizeof(mfloat)));
    // cutilSafeCall(cudaMemset(vec_d_T2[ibox], 0, pitch*pitchy*nz*sizeof(mfloat)));

      /* allocate and initialize host data */
    // h_T1 = (mfloat*)calloc(pitch*pitchy*nz, sizeof(mfloat));
    //h_T2 = (mfloat*)calloc(pitch*pitchy*nz, sizeof(mfloat)); 



    /* copy data to the GPU */
    copy_cube_simple(d_T1, vec_d_T1[0], nx+2*pad, ny+2, nz, cudaMemcpyDeviceToDevice, streams[istream]);

  }
  /* copy stencil to the GPU */
  cutilSafeCall(cudaMemcpyToSymbol(d_kernel_3c, h_kernel_3c_all,
                                   sizeof(mfloat)*27, 0, cudaMemcpyHostToDevice));

  /* -------------------- */
  /* performance tests    */
  /* -------------------- */
  
  
  dim3 block(thrdim_x, thrdim_y, 1);
  dim3 grid = get_grid(block, nx, ny, nz, thrdim_x, thrdim_y);
  int shmem = 2*block.x*block.y*sizeof(mfloat);    

  int kstart = 1;
  int kstop = nz-1;
 
  cudaError_t sync, async;

  high_resolution_clock::time_point time_start, time_end;
  cudaProfilerStart();
#pragma omp parallel
{
  int tid = omp_get_thread_num();
  if (tid == 0)
    time_start = high_resolution_clock::now(); 

  for(int ibox = 0; ibox < nbox; ibox++)
  {
 
    //int istream = ibox%nstream;

    mfloat* h_T1 = vec_h_T1[ibox];
    mfloat* h_T2 = vec_h_T2[ibox];
    mfloat* d_T1 = vec_d_T1[ibox];
    mfloat* d_T2 = vec_d_T2[ibox];

    for(int it=tid; it<niter; it+=nstream)
    {

  //  int istream = (ibox*niter + it) % nstream; 
  /*      if(routine==1)
          stencil27_symm_exp<<<grid, block, 2*(block.x)*(block.y)*sizeof(mfloat),streams[istream]>>>
            (d_T1, d_T2, nx, ny, nz, kstart, kstop);
        else if(routine==2)
    */      stencil27_symm_exp_prefetch<<<grid, block, 2*shmem, streams[tid]>>>
            (d_T1, d_T2, nx+2*pad, ny+2, kstart, kstop);
  /*      else if(routine==3)
          stencil27_symm_exp_new<<<grid, block, 2*(block.x)*(block.y)*sizeof(mfloat),streams[istream]>>>
            (d_T1, d_T2, nx, ny, nz, kstart, kstop);
        else
          stencil27_symm_exp_prefetch_new<<<grid, block, 2*(block.x)*(block.y)*sizeof(mfloat),streams[istream]>>>
            (d_T1, d_T2, nx, ny, nz, kstart, kstop);
    */  
    }
    /* finalize */
    //unsigned long long int numflops = 2*niter*27*nx*ny*nz;
 }
  if (tid == 0) {
    sync = cudaGetLastError();  
    if (sync != cudaSuccess)
      std::cout << "SYNCHRONOUS ERROR: " << cudaGetErrorString(sync) << std::endl;
    async = cudaDeviceSynchronize();
    if (async != cudaSuccess)
      std::cout << "ASYNCHRONOUS ERROR: " << cudaGetErrorString(async) << std::endl;
    time_end = high_resolution_clock::now(); 
  }
}
  cudaProfilerStop();
  duration<double> time_span = duration_cast<duration<double>>(time_end-  time_start);
  double microseconds = 1.0e6*(time_span.count());
  long long nptsperbox = nx*ny*nz;
  long long flops =  2*niter*27*(nptsperbox)*nbox;
  double tera_flop_rate = flops/microseconds/1.0e6;
  std::cout << "nx = " << nx << ", ny = " << ny << ", nz = " << nz << ", nbox = " 
            << nbox << ", nstream = " << nstream << ", niter = " << niter << std::endl;
  std::cout << "time = " << microseconds << "mu s, num ops = " << flops 
            << ", flop rate = " << std::fixed << tera_flop_rate << "TFlops"  << std::endl;
//  ctoc(timer, niter, nbox*nx*ny*nz*sizeof(mfloat), 1, 1, thrdim_x, thrdim_y, nx, ny, nz);   
  
  /* perform computations on host */
//
//  bzero(h_T2, sizeof(mfloat)*pitch*pitchy*nz); 
//  host_convolution(h_T2, h_T1, nx, ny, nz, pitch, pitchy, h_kernel_3c_all);
//
//  /* compute difference in the results */
//  compute_difference(d_T2, h_T1, h_T2, nx, ny, nz, pitch, pitchy, thrdim_x, thrdim_y, niter);


  for(int istream = 0; istream < nstream; istream++)
  {
    cudaStreamDestroy(streams[istream]);
  }
  return 0;
}


int main(int argc, char*argv[])
{
  
  int retval = bigTest(argc, argv);

cudaDeviceReset();
  return retval;
}
