/*
  Copyright Marcin Krotkiewski, University of Oslo, 2012
*/

#define stencil_3x3(c0, c1, c2, shm, tx, ty, bx)			\
  c0*((shm)[tx+0+(ty+0)*bx]) +						\
  c1*((shm)[tx-1+(ty+0)*bx]  + (shm)[tx+1+(ty+0)*bx] + (shm)[tx+0+(ty-1)*bx] + (shm)[tx+0+(ty+1)*bx]) + \
  c2*((shm)[tx-1+(ty-1)*bx]  + (shm)[tx+1+(ty-1)*bx] + (shm)[tx-1+(ty+1)*bx] + (shm)[tx+1+(ty+1)*bx])
  

__device__ inline mfloat stencil_3x3_function(mfloat c0, mfloat c1, mfloat c2, mfloat* shm,
                                uint tx, uint ty, uint bx)
{
  mfloat rtn = 0;
  rtn+=  c0*((shm)[tx+0+(ty+0)*bx]);
  rtn+=  c1*((shm)[tx-1+(ty+0)*bx]  + (shm)[tx+1+(ty+0)*bx] + (shm)[tx+0+(ty-1)*bx] + (shm)[tx+0+(ty+1)*bx]);
  rtn += c2*((shm)[tx-1+(ty-1)*bx]  + (shm)[tx+1+(ty-1)*bx] + (shm)[tx-1+(ty+1)*bx] + (shm)[tx+1+(ty+1)*bx]);
  return rtn;
}
  
  
#define stencil_3x3_reg(c0, c1, c2)					\
  c0*r5 +								\
  c1*(r2+r4+r6+r8) +							\
  c2*(r1+r3+r7+r9)

__device__ inline void push_regs_3x3(mfloat *shm, mfloat *r, uint tx, uint ty, uint bx)
{
    r[0] = shm[tx+ty*bx];
    if (tx<16) // ensures all reads are from same bank
        r[3] = shm[tx+16+(ty+1)*bx];
    else
        r[3] = shm[tx-16+(ty+1)*bx];
    r[6] = shm[tx+(ty+2)*bx];
}

__device__ inline mfloat calc_regs_3x3(mfloat *r, mfloat c0, mfloat c1, mfloat c2) {
    return c0*r[4] + c1*(r[1]+r[3]+r[5]+r[7]) + c2*(r[0]+r[2]+r[6]+r[8]);
}

#define push_regs_exp(shm, bx)			\
  {						\
    r1=(shm)[tx-1+(ty-1)*bx];			\
    r2=(shm)[tx+0+(ty-1)*bx];			\
    r3=(shm)[tx+1+(ty-1)*bx];			\
						\
    r4=(shm)[tx-1+(ty+0)*bx];			\
    r5=(shm)[tx+0+(ty+0)*bx];			\
    r6=(shm)[tx+1+(ty+0)*bx];			\
	    					\
    r7=(shm)[tx-1+(ty+1)*bx];			\
    r8=(shm)[tx+0+(ty+1)*bx];			\
    r9=(shm)[tx+1+(ty+1)*bx];			\
  }						\

namespace cg = cooperative_groups;

__global__ void stencil27_symm_exp(mfloat *in, mfloat *out, uint dimx, uint dimy, uint dimz, uint kstart, uint kend)
{
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;
  const  int ix = blockIdx.x*blockDim.x + threadIdx.x;	
  const  int iy = blockIdx.y*blockDim.y + threadIdx.y;

  const uint ti = threadIdx.y*blockDim.x + threadIdx.x;
  const uint pad = 32/sizeof(mfloat); // halos to left & right of interior require 32 byte memory transaction
  const uint bx= blockDim.x+2*pad;
  const uint txe= ti%bx; // this thread's block-relative x-axis index for first read
  const uint tye= ti/bx; // this thread's block-relative y-axis index for first read
  const uint txe2= (ti+blockDim.x*blockDim.y)%bx; // because of halos, each thread reads two values
  const uint tye2= (ti+blockDim.x*blockDim.y)/bx;
  int  ixe= blockIdx.x*blockDim.x + txe - pad; // this thread's global x-axis index for first read
  int  iye= blockIdx.y*blockDim.y + tye - 1;
  int  ixe2= blockIdx.x*blockDim.x + txe2 - pad;
  int  iye2= blockIdx.y*blockDim.y + tye2 - 1;

  // periodicity
  if(ixe<0)       ixe  += dimx;
  if(ixe>dimx-1)  ixe  -= dimx;
  if(ixe2<0)      ixe2 += dimx;
  if(ixe2>dimx-1) ixe2 -= dimx;

  if(iye<0)       iye  += dimy;
  if(iye>dimy-1)  iye  -= dimy;
  if(iye2<0)      iye2 += dimy;
  if(iye2>dimy-1) iye2 -= dimy;

  mfloat t1 = 0;
  mfloat t2 = 0;
  mfloat t3 = 0;
  mfloat *kernel = d_kernel_3c;
  mfloat C0, C1, C2, C3;
  C0 = kernel[9+4];
  C1 = kernel[4];
  C2 = kernel[1];
  C3 = kernel[0];
  uint i1, i2;

  cg::thread_block block = cg::this_thread_block();
  extern __shared__ mfloat shm[];			

  i1 = ixe+iye*dimx;
  i2 = ixe2+iye2*dimy;

  shm[txe +tye *bx] = in[i1];
  shm[txe2+tye2*bx] = in[i2];

  block.sync();
   t1 = stencil_3x3(C1, C2, C3, shm, tx+pad, ty+1, bx);
  block.sync();

  i1 += dimx*dimy;
  i2 += dimx*dimy;

  shm[txe +tye *bx] = in[i1];
  shm[txe2+tye2*bx] = in[i2];

  block.sync();
  t2 = stencil_3x3(C1, C2, C3, shm, tx+pad, ty+1, bx);
  t1+= stencil_3x3(C0, C1, C2, shm, tx+pad, ty+1, bx);
  block.sync();

  for(uint kk=kstart; kk<kend; kk++){

    block.sync();

    i1 += dimx*dimy;
    i2 += dimx*dimy;

    shm[txe +tye *bx] = in[i1];
    shm[txe2+tye2*bx] = in[i2];

    block.sync();
    t3 = stencil_3x3(C1, C2, C3, shm, tx+pad, ty+1, bx);

    out[ix + iy*dimx + kk*dimx*dimy] = t1 + t3;
    t1 = t2 + stencil_3x3(C0, C1, C2, shm, tx+pad, ty+1, bx);
    t2 = t3;
  }
}


__global__ void stencil27_symm_exp_prefetch(mfloat *in, mfloat *out, uint dimx, uint dimy, uint dimz, uint kstart, uint kend)
{
  mfloat r[9];

  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;
  const  int ix = blockIdx.x*blockDim.x + threadIdx.x; // global x index
  const  int iy = blockIdx.y*blockDim.y + threadIdx.y; // global y index

  const uint ti = threadIdx.y*blockDim.x + threadIdx.x; // local 2D index
  const uint pad = 32/sizeof(mfloat);
  const uint bx= blockDim.x+2*pad;
  const uint txe= (2*ti)%bx; // local x-coordinate of float2 read in padded space
  const uint tye= (2*ti)/bx; // local y-coordinate of float2 read in padded space
  const uint wi = ty*2*32 + tx; // 2D index of first shmem write
  const uint wxe= wi%bx; // x-coordinate of 1st shmem write
  const uint wye= wi/bx; // y-coordinate of 1st shmem write
  const uint wxe2= (wi+32)%bx; // x-coordinate of 2nd shmem write
  const uint wye2= (wi+32)/bx; // y-coordinate of 2nd shmem write
//  const uint txe2= (ti+blockDim.x*blockDim.y)%bx;
//  const uint tye2= (ti+blockDim.x*blockDim.y)/bx;
  int  ixe= blockIdx.x*blockDim.x + txe - pad; // global x-coordinate of float2 read in unpadded space
  int  iye= blockIdx.y*blockDim.y + tye - 1; // global y-coordinate of float2 read in unpadded space
//  int  ixe2= blockIdx.x*blockDim.x + txe2 - pad;
//  int  iye2= blockIdx.y*blockDim.y + tye2 - 1;

  // periodicity
  if(ixe<0)       ixe  += dimx;
  if(ixe>dimx-1)  ixe  -= dimx;
//  if(ixe2<0)      ixe2 += dimx;
//  if(ixe2>dimx-1) ixe2 -= dimx;

  if(iye<0)       iye  += dimy;
  if(iye>dimy-1)  iye  -= dimy;
//  if(iye2<0)      iye2 += dimy;
//  if(iye2>dimy-1) iye2 -= dimy;

  mfloat t1 = 0;
  mfloat t2 = 0;
  mfloat t3 = 0;
  mfloat *kernel = d_kernel_3c;
  mfloat C0, C1, C2, C3;
  C0 = kernel[9+4];
  C1 = kernel[4];
  C2 = kernel[1];
  C3 = kernel[0];

  uint i1; //, i2;
//  mfloat s[2];
  float2 read;

  cg::thread_block block = cg::this_thread_block();						
  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  extern __shared__ mfloat sh1[];
  mfloat *sh2 = sh1 + 2*blockDim.x*blockDim.y; // offset by one slice of padded data (same as bx*8)
  mfloat *shm[2] = { sh1, sh2 };

  i1 = ixe+iye*dimx;
//  i2 = ixe2+iye2*dimx;
  read = reinterpret_cast<float2*>(in)[i1/2]; // reads 8 contiguous bytes of data
//  shm[0][txe +tye *bx] = in[i1]; // load and store first slice
//  shm[0][txe2+tye2*bx] = in[i2];
  if (warp.thread_rank() < 16) // get first value from higher-ranked thread
    read.y = warp.shfl_xor(read.x, 16);
  else // get second value from lower-ranked thread
    read.x = warp.shfl_xor(read.y, 16);
  shm[0][wxe +wye *bx] = read.x; // read's two values should be in same lane
  shm[0][wxe2+wye2*bx] = read.y;

  i1 += dimx*dimy;
//  i2 += dimx*dimy;
  read = reinterpret_cast<float2*>(in)[i1/2]; // load second slice
//  s[0] = in[i1];
//  s[1] = in[i2];
  block.sync();
  push_regs_3x3(shm[0]+pad, &r[1], tx, ty, bx); // push middle of first slice 
  r[4] = warp.shfl_xor(r[4], 16); // thread gets its correct center value
  if (tx==0)
      push_regs_3x3(shm[0]+pad, &r[0], tx-1, ty, bx); // push block's left slice
  if (tx==31)
      push_regs_3x3(shm[0]+pad, &r[2], tx+1, ty, bx); // push block's right slice
  if (warp.thread_rank() < 16)
    read.y = warp.shfl_xor(read.x, 16); // threads now have data 32 bytes apart
  else
    read.x = warp.shfl_xor(read.y, 16);
  shm[1][wxe +wye *bx] = read.x; // store second slice
  shm[1][wxe2+wye2*bx] = read.y;
  for (int i = 0; i < 3; i++) {
      r[3*i+0] = warp.shfl_up(r[3*i+1], 1); // fill left halo slice
      r[3*i+2] = warp.shfl_down(r[3*i+1], 1); // fill right halo slice
  }
  t1 = calc_regs_3x3(r, C1, C2, C3); // calculate fist slice

  i1 += dimx*dimy;
//  i2 += dimx*dimy;
  read = reinterpret_cast<float2*>(in)[i1/2];
//  s[0] = in[i1];
//  s[1] = in[i2];
  block.sync();
  push_regs_3x3(shm[1]+pad, &r[1], tx, ty, bx);
  r[4] = warp.shfl_xor(r[4], 16);
  if (tx==0)
      push_regs_3x3(shm[1]+pad, &r[0], tx-1, ty, bx);
  if (tx==31)
      push_regs_3x3(shm[1]+pad, &r[2], tx+1, ty, bx);
  if (warp.thread_rank() < 16)
    read.y = warp.shfl_xor(read.x, 16); // threads now have data 32 bytes apart
  else
    read.x = warp.shfl_xor(read.y, 16);
  shm[0][wxe +wye *bx] = read.x; // store second slice
  shm[0][wxe2+wye2*bx] = read.y;
  for (int i = 0; i < 3; i++) {
      r[3*i+0] = warp.shfl_up(r[3*i+1], 1); // fill left halo slice
      r[3*i+2] = warp.shfl_down(r[3*i+1], 1); // fill right halo slice
  }
  t2 = calc_regs_3x3(r, C1, C2, C3);
  t1+= calc_regs_3x3(r, C0, C1, C2);

  uint j=0, k=1, kk;
  for(kk=kstart; kk<kend-1; kk++){

    i1 += dimx*dimy;
//    i2 += dimx*dimy;
    read = reinterpret_cast<float2*>(in)[i1/2];
//    s[0] = in[i1];
//    s[1] = in[i2];

    block.sync();
    push_regs_3x3(shm[j]+pad, &r[1], tx, ty, bx);
    r[4] = warp.shfl_xor(r[4], 16);
    if (tx==0)
      push_regs_3x3(shm[j]+pad, &r[0], tx-1, ty, bx);
    if (tx==31)
      push_regs_3x3(shm[j]+pad, &r[2], tx+1, ty, bx);
   
    if (warp.thread_rank() < 16)
      read.y = warp.shfl_xor(read.x, 16); // threads now have data 32 bytes apart
    else
      read.x = warp.shfl_xor(read.y, 16);

    shm[k][wxe +wye *bx] = read.x; // store second slice
    shm[k][wxe2+wye2*bx] = read.y;

    for (int i = 0; i < 3; i++) {
      r[3*i+0] = warp.shfl_up(r[3*i+1], 1); // fill left halo slice
      r[3*i+2] = warp.shfl_down(r[3*i+1], 1); // fill right halo slice
    }

    t3 = calc_regs_3x3(r, C1, C2, C3);
    out[ix + iy*dimx + kk*dimx*dimy] = t1 + t3;
    t1 = t2 + calc_regs_3x3(r, C0, C1, C2);
    t2 = t3;

    j=1-j; k=1-k;
  }

  block.sync();  
  push_regs_3x3(shm[j]+pad, &r[1], tx, ty, bx);
  r[4] = warp.shfl_xor(r[4], 16);
  if (tx==0)
    push_regs_3x3(shm[j]+pad, &r[0], tx-1, ty, bx);
  if (tx==31)
    push_regs_3x3(shm[j]+pad, &r[2], tx+1, ty, bx);
   
  for (int i = 0; i < 3; i++) {
    r[3*i+0] = warp.shfl_up(r[3*i+1], 1); // fill left halo slice
    r[3*i+1] = warp.shfl_down(r[3*i+1], 1); // fill right halo slice
  }

  t3 = calc_regs_3x3(r, C1, C2, C3);
  out[ix + iy*dimx + kk*dimx*dimy] = t1 + t3;
}


__global__ void stencil27_symm_exp_new(mfloat *in, mfloat *out, uint dimx, uint dimy, uint dimz, uint kstart, uint kend)
{
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;
  const uint ix = blockIdx.x*32 + threadIdx.x; // 32 = blockDim.x
  const uint iy = blockIdx.y*6 + threadIdx.y; // 6 = blockDim.y
  const uint ti = threadIdx.y*32 + threadIdx.x;
  const uint pad = 32/sizeof(mfloat);
  const uint width = 32+2*pad; // width of slice, including halos
  const uint tye= ti/width;
  const uint txe= ti-tye*width;
  const uint tye2=tye+4; // including halos, slice has 8 rows, so tye2 is 4 rows below tye

  int  ixe = blockIdx.x*32 + txe  - pad;
  int  iye = blockIdx.y*6  + tye  - 1;
  int  iye2= blockIdx.y*6  + tye2 - 1;

  // periodicity
  if(ixe<0)       ixe  += dimx;
  if(ixe>dimx-1)  ixe  -= dimx;
  if(iye<0)       iye  += dimy;
  if(iye>dimy-1)  iye  -= dimy;
  if(iye2<0)      iye2 += dimy;
  if(iye2>dimy-1) iye2 -= dimy;

  uint i1, i2;
  
  i1 = ixe+iye*dimx ;
  i2 = ixe+iye2*dimx ;

  mfloat t1 = 0;
  mfloat t2 = 0;
  mfloat t3 = 0;
  mfloat *kernel = d_kernel_3c;
  mfloat C0, C1, C2, C3;
  C0 = kernel[9+4];
  C1 = kernel[4];
  C2 = kernel[1];
  C3 = kernel[0];

  cg::thread_block block = cg::this_thread_block();
  extern __shared__ mfloat shm[];			

  shm[txe +tye *width] = in[i1];
  shm[txe+tye2*width] = in[i2];

  block.sync();
  t1 = stencil_3x3(C1, C2, C3, shm, tx+pad, ty+1, width);
  block.sync();

  i1 += dimx*dimy;
  i2 += dimx*dimy;

  shm[txe +tye *width] = in[i1];
  shm[txe+tye2*width] = in[i2];

  block.sync();
  t2 = stencil_3x3(C1, C2, C3, shm, tx+pad, ty+1, width);
  t1+= stencil_3x3(C0, C1, C2, shm, tx+pad, ty+1, width);
  block.sync();

  for(uint kk=kstart; kk<kend; kk++){

    block.sync();

    i1 += dimx*dimy;
    i2 += dimx*dimy;

    shm[txe +tye *width] = in[i1];
    shm[txe+tye2*width] = in[i2];

    block.sync();
    t3 = stencil_3x3(C1, C2, C3, shm, tx+pad, ty+1, width);

    out[ix + iy*dimx + kk*dimx*dimy] = t1 + t3;
    t1 = t2 + stencil_3x3(C0, C1, C2, shm, tx+pad, ty+1, width);
    t2 = t3;
  }
}


__global__ void stencil27_symm_exp_prefetch_new(mfloat *in, mfloat *out, uint dimx, uint dimy, uint dimz, uint kstart, uint kend)
{
  mfloat r1, r2, r3, r4, r5, r6, r7, r8, r9;

  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;
  const uint ix = blockIdx.x*32 + threadIdx.x;
  const uint iy = blockIdx.y*6  + threadIdx.y;
  const uint ti = threadIdx.y*32 + threadIdx.x;
  const uint pad= 32/sizeof(mfloat);
  const uint width = 32+2*pad;
  const uint tye= ti/width;
  const uint txe= ti-tye*width;
  const uint tye2=tye+4;

  int  ixe = blockIdx.x*32 + txe  - pad;
  int  iye = blockIdx.y*6  + tye  - 1;
  int  iye2= blockIdx.y*6  + tye2 - 1;

  // periodicity
  if(ixe<0)       ixe  += dimx;
  if(ixe>dimx-1)  ixe  -= dimx;
  if(iye<0)       iye  += dimy;
  if(iye>dimy-1)  iye  -= dimy;
  if(iye2<0)      iye2 += dimy;
  if(iye2>dimy-1) iye2 -= dimy;

  uint i1, i2;
  
  i1 = ixe+iye*dimx ;
  i2 = ixe+iye2*dimx ;

  mfloat t1 = 0;
  mfloat t2 = 0;
  mfloat t3 = 0;
  mfloat *kernel = d_kernel_3c;
  mfloat C0, C1, C2, C3;
  C0 = kernel[9+4];
  C1 = kernel[4];
  C2 = kernel[1];
  C3 = kernel[0];

  cg::thread_block block = cg::this_thread_block();
  extern __shared__ mfloat shm[];

  shm[txe +tye *width] = in[i1];
  shm[txe+tye2*width] = in[i2];

  block.sync();  
  push_regs_exp(shm+pad+width, width);  
  block.sync();

  i1 += dimx*dimy;
  i2 += dimx*dimy;

  shm[txe +tye *width] = in[i1];
  shm[txe+tye2*width] = in[i2];

  t1 = stencil_3x3_reg(C1, C2, C3);

  block.sync();  
  push_regs_exp(shm+pad+width, width);  
  block.sync();

  i1 += dimx*dimy;
  i2 += dimx*dimy;

  shm[txe +tye*width] = in[i1];
  shm[txe+tye2*width] = in[i2];

  t2 = stencil_3x3_reg(C1, C2, C3);
  t1+= stencil_3x3_reg(C0, C1, C2);

  for(uint kk=kstart; kk<kend-1; kk++){

    block.sync();  
    push_regs_exp(shm+pad+width, width);  
    block.sync();

    i1 += dimx*dimy;
    i2 += dimx*dimy;

    shm[txe +tye *width] = in[i1];
    shm[txe+tye2*width] = in[i2];

    t3 = stencil_3x3_reg(C1, C2, C3);

    out[ix + iy*dimx + kk*dimx*dimy] = t1 + t3;
    t1 = t2 + stencil_3x3_reg(C0, C1, C2);
    t2 = t3;
  }

  block.sync();  
  push_regs_exp(shm+pad+width, width);  
  block.sync();

  out[ix + iy*dimx + (kend-1)*dimx*dimy] = t1 + stencil_3x3_reg(C1, C2, C3);
}
