#include "Proto.H"
#include "EBProto.H"

using namespace Proto;



template<typename T>
inline T&
getIrregData(T& a_s)
{
  return a_s;
}

template <CENTERING cent, typename  data_t, unsigned int ncomp>
inline IrregData<cent, data_t, ncomp>&
getIrregData(EBBoxData<cent, data_t, ncomp>& a_s)
{
  return a_s.getIrregData();
}


template<CENTERING cent,  typename data_t, unsigned int ncomp>
struct
uglyStruct
{
  data_t*        m_startPtr;
  unsigned int   m_varsize;
  unsigned int   m_offset;
  Point          m_index;
};

template <CENTERING cent, typename T>
inline T
getUglyStruct(const vector<EBIndex<cent> >& a_indices,
              T& a_T)
{
  return a_T;
}

template <CENTERING cent, typename  data_t, unsigned int ncomp>
inline vector< uglyStruct<cent, data_t, ncomp> >
getUglyStruct(const vector<EBIndex<cent> >& a_indices,
              IrregData<cent, data_t, ncomp>& a_s )
{
  data_t*      debPtr  = a_s.data();
  printf("debptr = %p\n", debPtr);
  vector< uglyStruct<cent, data_t, ncomp> > retval;
  for(int ivec = 0; ivec < a_indices.size(); ivec++)
  {
    uglyStruct<cent, data_t, ncomp>  vecval;
    vecval.m_startPtr = a_s.data();
    vecval.m_varsize  = a_s.vecsize();
    vecval.m_offset   = a_s.index(a_indices[ivec], 0);
    vecval.m_index    = a_indices[ivec].m_pt;
    retval.push_back(vecval);
  }
  return retval;
}




template<typename T>
inline T&
getBoxData(T& a_s)
{
  return a_s;
}

template <CENTERING cent, typename  data_t, unsigned int ncomp>
inline BoxData<data_t, ncomp>&
getBoxData(EBBoxData<cent, data_t, ncomp>& a_s)
{
  return a_s.getRegData();
}

#ifdef PROTO_CUDA
///cuda specific functions
///
template<typename T>
__device__ __host__
inline T
cudaGetVar(unsigned int ivec,  T a_s)
{
  return a_s;
}

///
template<CENTERING cent, typename data_t, unsigned int ncomp>
__device__ 
inline Var<data_t, ncomp>
cudaGetVar(unsigned int a_ivec,
           thrust::device_ptr<uglyStruct<cent, data_t, ncomp> > a_ptr)
{
  Var<data_t, ncomp> retval;

//  const uglyStruct<cent, data_t, ncomp>& ugly = a_dst[a_ivec];
  const uglyStruct<cent, data_t, ncomp>*  rawptr = thrust::raw_pointer_cast(a_ptr);
  const uglyStruct<cent, data_t, ncomp>&  ugly   = rawptr[a_ivec];

  for(int icomp = 0; icomp < ncomp; icomp++)
  {
    retval.m_ptrs[icomp] = ugly.m_startPtr + ugly.m_offset + (ugly.m_varsize*icomp);
  }
  return retval;
}

///going into this srcs are uglystruct* and other stuff
template<typename Func, typename... Srcs>
__global__
void
vec_indexer(unsigned int a_begin, unsigned int a_end, Func a_body, Srcs... a_srcs)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx >= a_begin && idx < a_end)
  {
    a_body(cudaGetVar(idx, a_srcs)...);
  }
}

///going into this srcs are thrust_device_vector<uglystruct> and other stuff
template<CENTERING cent, typename data_t,unsigned int ncomp,  typename Func, typename... Srcs>
__global__
void
vec_indexer_p(unsigned int a_begin, unsigned int a_end,Func a_body, 
              thrust::device_vector< uglyStruct<cent, data_t, ncomp> > a_dst, Srcs... a_srcs)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx >= a_begin && idx < a_end)
  {
    Point pt = a_dst[idx].m_index;
    a_body(pt, cudaGetVar(idx, a_dst), cudaGetVar(idx, a_srcs...));
  }
}

///going into this srcs are thrust_device_ptr<uglystruct> and other stuff
template<typename Func, typename... Srcs>
void
cudaVectorFunc(const Func& a_F, unsigned int a_Nvec, Srcs... a_srcs)
{
  cudaStream_t curstream = DisjointBoxLayout::getCurrentStream();
  const int N = a_Nvec;
  unsigned int stride = N;
  unsigned int blocks = 1;
  size_t smem = 0;
  vec_indexer<<<blocks, stride, smem, curstream>>>
    (0, N, mapper(a_F), a_srcs...);
                                                     
       
}


template <CENTERING cent, typename T>
inline T
cudaGetUglyStruct(const vector<EBIndex<cent> >& a_indices,
                  T& a_T)
{
  return a_T;
}
///
template <CENTERING cent, typename  data_t, unsigned int ncomp>
inline thrust::device_ptr< uglyStruct<cent, data_t, ncomp> >
cudaGetUglyStruct(const vector<EBIndex<cent> >& a_indices,
                  IrregData<cent, data_t, ncomp>& a_s )
{
  vector< uglyStruct<cent, data_t, ncomp> > 
    hostval = getUglyStruct(a_indices, a_s);
  //this copies from the host to the device
  thrust::device_vector< uglyStruct<cent, data_t, ncomp> > vec    = hostval;
  thrust::device_ptr<    uglyStruct<cent, data_t, ncomp> > retval = vec.data();
  return retval;
}
///going into this srcs are IrregDatas and other stuff
template<CENTERING cent, typename  data_t, unsigned int ncomp, typename Func, typename... Srcs>
inline void
cudaEBForAllIrreg(const Func& a_F, const Box& a_box,
                  IrregData<cent, data_t, ncomp>& a_dst,
                  Srcs&...  a_srcs)
{
  //indicies into irreg vector that correspond to input box
  const vector<EBIndex<cent> >& dstvofs = a_dst.getIndices(a_box);
  unsigned int vecsize = a_dst.vecsize();
  if(vecsize > 0)
  {
    cudaVectorFunc(a_F, vecsize, cudaGetUglyStruct(dstvofs, a_dst), (cudaGetUglyStruct(dstvofs, a_srcs))...);
   }
}

template<typename Func, typename... Srcs>
inline void
cudaEBforall(const Func & a_F,  Box a_box, Srcs&... a_srcs)
{
//call regular forall
//  forallInPlaceBase(a_F, a_box, (getBoxData(a_srcs))...);
  
//do the same thing for the irregular data
  cudaEBForAllIrreg(a_F, a_box, getIrregData(a_srcs)...);
}

#else
///cpu-only specific functions

///
template<typename T>
inline T
getVar(unsigned int ivec,  T a_s)
{
  return a_s;
}

///
template<CENTERING cent, typename data_t, unsigned int ncomp>
inline Var<data_t, ncomp>
getVar(unsigned int a_ivec,
       vector< uglyStruct<cent, data_t, ncomp> > a_dst)
{
  Var<data_t, ncomp> retval;
  const uglyStruct<cent, data_t, ncomp> ugly = a_dst[a_ivec];
  for(int icomp = 0; icomp < ncomp; icomp++)
  {
    retval.m_ptrs[icomp] = ugly.m_startPtr + ugly.m_offset + (ugly.m_varsize*icomp);
  }
  return retval;
}

///going into this srcs are vector<uglystruct> and other stuff
template<CENTERING cent, typename data_t,unsigned int ncomp,  typename Func, typename... Srcs>
void
hostVectorFunc(const Func& a_F, vector< uglyStruct<cent, data_t, ncomp> > a_dst, Srcs... a_srcs)
{
  for(unsigned int ivec = 0; ivec < a_dst.size(); ivec++)
  {
    a_F(getVar(ivec, a_dst), (getVar(ivec, a_srcs))...);
  }
       
}


///going into this srcs are vector<uglystruct> and other stuff
template<CENTERING cent, typename data_t,unsigned int ncomp,  typename Func, typename... Srcs>
void
hostVectorFunc_p(const Func& a_F, vector< uglyStruct<cent, data_t, ncomp> > a_dst, Srcs... a_srcs)
{
  for(unsigned int ivec = 0; ivec < a_dst.size(); ivec++)
  {
    Point pt = a_dst[ivec].m_index;
    a_F(pt, getVar(ivec, a_dst), (getVar(ivec, a_srcs))...);
  }
       
}

///going into this srcs are IrregDatas and other stuff
template<CENTERING cent, typename  data_t, unsigned int ncomp, typename Func, typename... Srcs>
inline void
hostEBForAllIrreg(const Func& a_F, const Box& a_box,
                  IrregData<cent, data_t, ncomp>& a_dst,
                  Srcs&...  a_srcs)
{
//indicies into irreg vector that correspond to input box
  vector<EBIndex<cent> > dstvofs = a_dst.getIndices(a_box);
  hostVectorFunc(a_F, getUglyStruct(dstvofs, a_dst), (getUglyStruct(dstvofs, a_srcs))...);
}


///going into this srcs are IrregDatas and other stuff
template<CENTERING cent, typename  data_t, unsigned int ncomp, typename Func, typename... Srcs>
inline void
hostEBForAllIrreg_p(const Func& a_F, const Box& a_box,
                    IrregData<cent, data_t, ncomp>& a_dst,
                    Srcs&...  a_srcs)
{
//indicies into irreg vector that correspond to input box
  vector<EBIndex<cent> > dstvofs = a_dst.getIndices(a_box);
  hostVectorFunc_p(a_F, getUglyStruct(dstvofs, a_dst), (getUglyStruct(dstvofs, a_srcs))...);
}


///going into this srcs are EBBoxDatas and other stuff
template<typename Func, typename... Srcs>
inline void
hostEBforall(const Func & a_F,  Box a_box, Srcs&... a_srcs)
{
//call regular forall
  forallInPlaceBase(a_F, a_box, (getBoxData(a_srcs))...);
  
//do the same thing for the irregular data
  hostEBForAllIrreg(a_F, a_box, getIrregData(a_srcs)...);
}

///going into this srcs are EBBoxDatas and other stuff
template<typename Func, typename... Srcs>
inline void
hostEBforall_p(const Func & a_F,  Box a_box, Srcs&... a_srcs)
{
//call regular forall
  forallInPlaceBase_p(a_F, a_box, (getBoxData(a_srcs))...);
  
//do the same thing for the irregular data
  hostEBForAllIrreg_p(a_F, a_box, getIrregData(a_srcs)...);
}

#endif

///version that does not send the point to the function
template<typename Func, typename... Srcs>
inline void EBforallInPlace(unsigned long long int a_num_flops_point,
                            const char*            a_timername,
                            const Func & a_F,  Box a_box, Srcs&... a_srcs)
{
  PR_TIME(a_timername);

  printf("in ebforall function pointer = %p\n", &a_F);
  unsigned long long int boxfloops = a_num_flops_point*a_box.size();

#ifdef PROTO_CUDA
  cudaEBforall(a_F, a_box, a_srcs...);
  cudaDeviceSynchronize();
#else
  hostEBforall(a_F, a_box, a_srcs...);
#endif
  PR_FLOPS(boxfloops);
}


/////version that sends the point to the function
//template<typename Func, typename... Srcs>
//inline void EBforallInPlace_p(unsigned long long int a_num_flops_point,
//                              const char*            a_timername,
//                              const Func & a_F,  Box a_box, Srcs&... a_srcs)
//{
//  PR_TIME(a_timername);
//
//
//  unsigned long long int boxfloops = a_num_flops_point*a_box.size();
//
//#ifdef PROTO_CUDA
//  cudaEBforall_p(a_F, a_box, a_srcs...);
//#else
//  hostEBforall_p(a_F, a_box, a_srcs...);
//#endif
//  PR_FLOPS(boxfloops);
//}


///  after this are specific to the test
typedef Var<double,DIM> V;


PROTO_KERNEL_START 
void UsetUF(V a_U, double  a_val)
{
  printf("in set u\n");
//  printf("setu: uptr[0] = %p, uptr[1] = %p\n",a_U.m_ptrs[0],a_U.m_ptrs[1]);
  for(int idir = 0; idir < DIM; idir++)
  {
    a_U(idir) = a_val;
//    if(a_U(idir) != a_val)
//    {
//      printf("p1: values do not match \n");
//      printf("setu: val = %f, uval = %f\n",a_val, a_U(idir));
//    }
  }
}
PROTO_KERNEL_END(UsetUF, UsetU)


//PROTO_KERNEL_START 
//void setUptF(Point a_p, V a_U, double  a_val)
//{
//  for(int idir = 0; idir < DIM; idir++)
//  {
//    a_U(idir) = a_val;
//  }
//}
//PROTO_KERNEL_END(setUptF, setUpt)


PROTO_KERNEL_START 
void VsetVF(V a_V, double  a_val)
{
  printf("in set v\n");
//  printf("setv: vptr[0] = %p, vptr[1] = %p\n",a_V.m_ptrs[0],a_V.m_ptrs[1]);
  for(int idir = 0; idir < DIM; idir++)
  {
    a_V(idir) = a_val;
//    if(a_V(idir) != a_val)
//        {
//      printf("p2: values do not match \n");
//      printf("setv: val = %f, vval = %f\n",a_val, a_V(idir));
//    }
  }
}
PROTO_KERNEL_END(VsetVF, VsetV)

//PROTO_KERNEL_START 
//void setVptF(Point a_p, V a_V, double  a_val)
//{
//  for(int idir = 0; idir < DIM; idir++)
//  {
//    a_V(idir) = a_val;
//  }
//}
//PROTO_KERNEL_END(setVptF, setVpt)


PROTO_KERNEL_START 
void WsetWtoUplusVF(V a_W,
                   V a_U,
                   V a_V,
                   double  a_val)
{
  printf("in set w\n");
//  printf("setw: uptr[0] = %p, uptr[1] = %p\n ",a_U.m_ptrs[0],a_U.m_ptrs[1]);
//  printf("setw: vptr[0] = %p, vptr[1] = %p\n" ,a_V.m_ptrs[0],a_V.m_ptrs[1]);
//  printf("setw: wptr[0] = %p, wptr[1] = %p\n" ,a_W.m_ptrs[0],a_W.m_ptrs[1]);
  for(int idir = 0; idir < DIM; idir++)
  {
    a_W(idir) = a_U(idir) + a_V(idir);
//    if(a_W(idir) != a_val)
//    {
//      printf("p3: values do not match \n");
//      printf("setwtouplusv: val = %f, wval = %f, uval = %f, vval = %f\n",a_val, a_W(idir), a_U(idir), a_V(idir));
//    }
  }
}
PROTO_KERNEL_END(WsetWtoUplusVF, WsetWtoUplusV)

//PROTO_KERNEL_START 
//void setWtoUplusVptF(Point a_pt,
//                     V a_W,
//                     V a_U,
//                     V a_V,
//                     double  a_val)
//{
//  for(int idir = 0; idir < DIM; idir++)
//  {
//    a_W(idir) = a_U(idir) + a_V(idir);
//    if(a_W(idir) != a_val)
//    {
//      printf("values do not match");
//    }
//  }
//}
//PROTO_KERNEL_END(setWtoUplusVptF, setWtoUplusVpt)

int main(int argc, char* argv[])
{

  int nx = 16;
  int maxGrid = nx; //this test is for a single grid
  Box domain(Point::Zeros(), Point::Ones(nx-1));
  double dx = 1.0/domain.size(0);
  std::array<bool, DIM> periodic;
  for(int idir = 0; idir < DIM; idir++) periodic[idir]=true;
  DisjointBoxLayout grids(domain, maxGrid, periodic);

  Point dataGhost = Point::Zeros();
  Point geomGhost = Point::Zeros();

  RealVect ABC = RealVect::Unit();
  RealVect X0 = RealVect::Unit();
//  X0 *= 100.;
  X0 *= 0.5;
  RealVect origin= RealVect::Zero();
  double R = 0.25;
  shared_ptr<BaseIF>              impfunc(new SimpleEllipsoidIF(ABC, X0, R, false));
  shared_ptr<GeometryService<2> > geoserv(new GeometryService<2>(impfunc, origin, dx, domain, grids, geomGhost, 0));
  shared_ptr<LevelData<EBGraph> > graphs = geoserv->getGraphs(domain);

  for(int ibox = 0; ibox < grids.size(); ibox++)
  {
    Box grid = grids[ibox];
    EBBoxData<CELL, double, DIM> U(grid,(*graphs)[ibox]);
    EBBoxData<CELL, double, DIM> V(grid,(*graphs)[ibox]);
    EBBoxData<CELL, double, DIM> W(grid,(*graphs)[ibox]);
    unsigned long long int numFlopsPt = 0;
    double uval = 1;
    double vval = 2;
    double wval = 3;
    printf("going into setU\n");
    EBforallInPlace(numFlopsPt, "setU", UsetU, grid, U, uval);
    printf("going into setV\n");
    EBforallInPlace(numFlopsPt, "setV", VsetV, grid, V, vval);
    printf("going into setWtoUPlusV\n");
    EBforallInPlace(numFlopsPt, "setWtoUPlusV", WsetWtoUplusV, grid, W, U, V, wval);

//    uval = 2;
//    vval = 5;
//    wval = 7;
//    EBforallInPlace_p(numFlopsPt, "setU", setUpt, grid, U, uval);
//    EBforallInPlace_p(numFlopsPt, "setV", setVpt, grid, V, vval);
//    EBforallInPlace_p(numFlopsPt, "setWtoUPlusV", setWtoUplusVpt, grid, W, U, V, wval);


  }

}








