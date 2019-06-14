#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define min(x,y) (((x)<(y))?(x):(y))
#define max(x,y) (((x)>(y))?(x):(y))
#define abs(x) ((x)<0?-(x):(x))
#define absmin(x,y) ((x)=min(abs((x)),abs((y))))
#define absmax(x,y) ((x)=max(abs((x)),abs((y))))
#define floord(x,y) ((x)/(y))
#define sgn(x) ((x)<0?-1:1)
#define offset2(i,j,M) ((j)+(i)*(M))
#define offset3(i,j,k,M,N) ((k)+((j)+(i)*(M))*(N))
#define offset4(i,j,k,l,M,N,P) ((l)+((k)+((j)+(i)*(M))*(N))*(P))
#define arrinit(ptr,val,size) for(unsigned __i__=0;__i__<(size);__i__++) (ptr)[__i__]=(val)
#define arrprnt(name,arr,size) {\
fprintf(stderr,"%s={",(name));\
for(unsigned __i__=0;__i__<(size);__i__++) fprintf(stderr,"%lg,",(arr)[__i__]);\
fprintf(stderr,"}\n");}
#define F_ave_f_d1(c,z,y,x) F_ave_f_d1[offset4((c),(z)+2,(y)+2,(x),(65+2+1),(65+2+1),(64+1))]
#define F_ave_f_d2(c,z,y,x) F_ave_f_d2[offset4((c),(z)+2,(y),(x)+2,(65+2+1),(64+1),(65+2+1))]
#define F_ave_f_d3(c,z,y,x) F_ave_f_d3[offset4((c),(z),(y)+2,(x)+2,(64+1),(65+2+1),(65+2+1))]
#define F_bar_f_d1(c,z,y,x) F_bar_f_d1[offset4((c),(z)+3,(y)+3,(x),(66+3+1),(66+3+1),(64+1))]
#define F_bar_f_d2(c,z,y,x) F_bar_f_d2[offset4((c),(z)+3,(y),(x)+3,(66+3+1),(64+1),(66+3+1))]
#define F_bar_f_d3(c,z,y,x) F_bar_f_d3[offset4((c),(z),(y)+3,(x)+3,(64+1),(66+3+1),(66+3+1))]
#define F_div_f_d1(c,z,y,x) F_div_f_d1[offset4((c),(z)+2,(y)+2,(x),(65+2+1),(65+2+1),(63+1))]
#define F_div_f_d2(c,z,y,x) F_div_f_d2[offset4((c),(z)+2,(y),(x)+2,(65+2+1),(63+1),(65+2+1))]
#define F_div_f_d3(c,z,y,x) F_div_f_d3[offset4((c),(z),(y)+2,(x)+2,(63+1),(65+2+1),(65+2+1))]
#define F_lap_f_d1(c,z,y,x) F_lap_f_d1[offset4((c),(z)+2,(y)+2,(x),(65+2+1),(65+2+1),(64+1))]
#define F_lap_f_d2(c,z,y,x) F_lap_f_d2[offset4((c),(z)+2,(y),(x)+2,(65+2+1),(64+1),(65+2+1))]
#define F_lap_f_d3(c,z,y,x) F_lap_f_d3[offset4((c),(z),(y)+2,(x)+2,(64+1),(65+2+1),(65+2+1))]
#define U(c,z,y,x) U[offset4((c),(z)+4,(y)+4,(x)+4,(67+4+1),(67+4+1),(67+4+1))]
#define W(c,z,y,x) W[offset4((c),(z)+3,(y)+3,(x)+3,(66+3+1),(66+3+1),(66+3+1))]
#define W_ave(c,z,y,x) W_ave[offset4((c),(z)+3,(y)+3,(x)+3,(66+3+1),(66+3+1),(66+3+1))]
#define W_aveH_d1(c,z,y,x) W_aveH_d1[offset4((c),(z)+3,(y)+3,(x)+1,(66+3+1),(66+3+1),(64+1+1))]
#define W_aveH_d2(c,z,y,x) W_aveH_d2[offset4((c),(z)+3,(y)+1,(x)+3,(66+3+1),(64+1+1),(66+3+1))]
#define W_aveH_d3(c,z,y,x) W_aveH_d3[offset4((c),(z)+1,(y)+3,(x)+3,(64+1+1),(66+3+1),(66+3+1))]
#define W_aveL_d1(c,z,y,x) W_aveL_d1[offset4((c),(z)+3,(y)+3,(x),(66+3+1),(66+3+1),(65+1))]
#define W_aveL_d2(c,z,y,x) W_aveL_d2[offset4((c),(z)+3,(y),(x)+3,(66+3+1),(65+1),(66+3+1))]
#define W_aveL_d3(c,z,y,x) W_aveL_d3[offset4((c),(z),(y)+3,(x)+3,(65+1),(66+3+1),(66+3+1))]
#define W_ave_f_d1(c,z,y,x) W_ave_f_d1[offset4((c),(z)+3,(y)+3,(x),(66+3+1),(66+3+1),(64+1))]
#define W_ave_f_d2(c,z,y,x) W_ave_f_d2[offset4((c),(z)+3,(y),(x)+3,(66+3+1),(64+1),(66+3+1))]
#define W_ave_f_d3(c,z,y,x) W_ave_f_d3[offset4((c),(z),(y)+3,(x)+3,(64+1),(66+3+1),(66+3+1))]
#define W_bar(c,z,y,x) W_bar[offset4((c),(z)+4,(y)+4,(x)+4,(67+4+1),(67+4+1),(67+4+1))]
#define W_f_d1(c,z,y,x) W_f_d1[offset4((c),(z)+2,(y)+2,(x),(65+2+1),(65+2+1),(64+1))]
#define W_f_d2(c,z,y,x) W_f_d2[offset4((c),(z)+2,(y),(x)+2,(65+2+1),(64+1),(65+2+1))]
#define W_f_d3(c,z,y,x) W_f_d3[offset4((c),(z),(y)+2,(x)+2,(64+1),(65+2+1),(65+2+1))]
#define rhs(c,z,y,x) rhs[offset4((c),(z),(y),(x),(63+1),(63+1),(63+1))]
#define u(c,z,y,x) u[offset4((c),(z)+3,(y)+3,(x)+3,(66+3+1),(66+3+1),(66+3+1))]
#define umax(z,y,x) umax[offset3((z),(y),(x),(63+1),(63+1))]

double euler_step(const double* U, double* rhs);
inline double euler_step(const double* U, double* rhs) {
    int t1,t2,t3,t4,t5;
    double* W_bar = (double*) calloc((((5)*(67+4+1))*(67+4+1))*(67+4+1),sizeof(double));
    double* u = (double*) calloc((((5)*(66+3+1))*(66+3+1))*(66+3+1),sizeof(double));
    double* W = (double*) calloc((((5)*(66+3+1))*(66+3+1))*(66+3+1),sizeof(double));
    double* umax = (double*) calloc((((63+1))*(63+1))*(63+1),sizeof(double));
    double retval = 0;
    double* W_ave = (double*) calloc((((5)*(66+3+1))*(66+3+1))*(66+3+1),sizeof(double));
    double* W_aveL_d1 = (double*) calloc((((5)*(66+3+1))*(66+3+1))*(65+1),sizeof(double));
    double* W_aveH_d1 = (double*) calloc((((5)*(66+3+1))*(66+3+1))*(64+1+1),sizeof(double));
    double* W_ave_f_d1 = (double*) calloc((((5)*(66+3+1))*(66+3+1))*(64+1),sizeof(double));
    double* F_bar_f_d1 = (double*) calloc((((5)*(66+3+1))*(66+3+1))*(64+1),sizeof(double));
    double* W_f_d1 = (double*) calloc((((5)*(65+2+1))*(65+2+1))*(64+1),sizeof(double));
    double* F_ave_f_d1 = (double*) calloc((((5)*(65+2+1))*(65+2+1))*(64+1),sizeof(double));
    double* F_lap_f_d1 = (double*) calloc((((5)*(65+2+1))*(65+2+1))*(64+1),sizeof(double));
    double* F_div_f_d1 = (double*) calloc((((5)*(65+2+1))*(65+2+1))*(63+1),sizeof(double));
    double* W_aveL_d2 = (double*) calloc((((5)*(66+3+1))*(65+1))*(66+3+1),sizeof(double));
    double* W_aveH_d2 = (double*) calloc((((5)*(66+3+1))*(64+1+1))*(66+3+1),sizeof(double));
    double* W_ave_f_d2 = (double*) calloc((((5)*(66+3+1))*(64+1))*(66+3+1),sizeof(double));
    double* F_bar_f_d2 = (double*) calloc((((5)*(66+3+1))*(64+1))*(66+3+1),sizeof(double));
    double* W_f_d2 = (double*) calloc((((5)*(65+2+1))*(64+1))*(65+2+1),sizeof(double));
    double* F_ave_f_d2 = (double*) calloc((((5)*(65+2+1))*(64+1))*(65+2+1),sizeof(double));
    double* F_lap_f_d2 = (double*) calloc((((5)*(65+2+1))*(64+1))*(65+2+1),sizeof(double));
    double* F_div_f_d2 = (double*) calloc((((5)*(65+2+1))*(63+1))*(65+2+1),sizeof(double));
    double* W_aveL_d3 = (double*) calloc((((5)*(65+1))*(66+3+1))*(66+3+1),sizeof(double));
    double* W_aveH_d3 = (double*) calloc((((5)*(64+1+1))*(66+3+1))*(66+3+1),sizeof(double));
    double* W_ave_f_d3 = (double*) calloc((((5)*(64+1))*(66+3+1))*(66+3+1),sizeof(double));
    double* F_bar_f_d3 = (double*) calloc((((5)*(64+1))*(66+3+1))*(66+3+1),sizeof(double));
    double* W_f_d3 = (double*) calloc((((5)*(64+1))*(65+2+1))*(65+2+1),sizeof(double));
    double* F_ave_f_d3 = (double*) calloc((((5)*(64+1))*(65+2+1))*(65+2+1),sizeof(double));
    double* F_lap_f_d3 = (double*) calloc((((5)*(64+1))*(65+2+1))*(65+2+1),sizeof(double));
    double* F_div_f_d3 = (double*) calloc((((5)*(63+1))*(65+2+1))*(65+2+1),sizeof(double));

// consToPrim1
#undef s0
#define s0(z,y,x) {\
W_bar(0,(z),(y),(x)) = U(0,(z),(y),(x));\
W_bar(1,(z),(y),(x)) = U(1,(z),(y),(x))/U(0,(z),(y),(x));\
W_bar(2,(z),(y),(x)) = U(2,(z),(y),(x))/U(0,(z),(y),(x));\
W_bar(3,(z),(y),(x)) = U(3,(z),(y),(x))/U(0,(z),(y),(x));\
W_bar(4,(z),(y),(x)) = (U(4,(z),(y),(x))-0.500000*U(0,(z),(y),(x))*(((U(1,(z),(y),(x))/U(0,(z),(y),(x)))*(U(1,(z),(y),(x))/U(0,(z),(y),(x))))+((U(2,(z),(y),(x))/U(0,(z),(y),(x)))*(U(2,(z),(y),(x))/U(0,(z),(y),(x))))+((U(3,(z),(y),(x))/U(0,(z),(y),(x)))*(U(3,(z),(y),(x))/U(0,(z),(y),(x))))))*(1.400000-1.000000);\
}

for(t1 = -4; t1 <= 67; t1++) {
  for(t2 = -4; t2 <= 67; t2++) {
    for(t3 = -4; t3 <= 67; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// deconvolve
#undef s0
#define s0(c,z,y,x) u((c),(z),(y),(x))=(1.250000*U((c),(z),(y),(x)))+((-0.041667)*U((c),(z),(y),(x)+1))+((-0.041667)*U((c),(z),(y),(x)-1))+((-0.041667)*U((c),(z),(y)+1,(x)))+((-0.041667)*U((c),(z),(y)-1,(x)))+((-0.041667)*U((c),(z)+1,(y),(x)))+((-0.041667)*U((c),(z)-1,(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// consToPrim2
#undef s0
#define s0(z,y,x) {\
W(0,(z),(y),(x)) = u(0,(z),(y),(x));\
W(1,(z),(y),(x)) = u(1,(z),(y),(x))/u(0,(z),(y),(x));\
W(2,(z),(y),(x)) = u(2,(z),(y),(x))/u(0,(z),(y),(x));\
W(3,(z),(y),(x)) = u(3,(z),(y),(x))/u(0,(z),(y),(x));\
W(4,(z),(y),(x)) = (u(4,(z),(y),(x))-0.500000*u(0,(z),(y),(x))*(((u(1,(z),(y),(x))/u(0,(z),(y),(x)))*(u(1,(z),(y),(x))/u(0,(z),(y),(x))))+((u(2,(z),(y),(x))/u(0,(z),(y),(x)))*(u(2,(z),(y),(x))/u(0,(z),(y),(x))))+((u(3,(z),(y),(x))/u(0,(z),(y),(x)))*(u(3,(z),(y),(x))/u(0,(z),(y),(x))))))*(1.400000-1.000000);\
}

for(t1 = -3; t1 <= 66; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// waveSpeedBound1
#undef s0
#define s0(z,y,x) {\
umax((z),(y),(x))=(3.000000*sqrt(1.400000*W(4,(z),(y),(x))/W(0,(z),(y),(x))))+W(1,(z),(y),(x))+W(2,(z),(y),(x))+W(3,(z),(y),(x));\
}

for(t1 = 0; t1 <= 63; t1++) {
  for(t2 = 0; t2 <= 63; t2++) {
    for(t3 = 0; t3 <= 63; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// absMax
#undef s0
#define s0(z,y,x) absmax(retval,umax((z),(y),(x)))

for(t1 = 0; t1 <= 63; t1++) {
  for(t2 = 0; t2 <= 63; t2++) {
    for(t3 = 0; t3 <= 63; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// laplacian
#undef s0
#define s0(c,z,y,x) W_ave((c),(z),(y),(x))=((-0.250000)*W_bar((c),(z),(y),(x)))+(0.041667*W_bar((c),(z),(y),(x)+1))+(0.041667*W_bar((c),(z),(y),(x)-1))+(0.041667*W_bar((c),(z),(y)+1,(x)))+(0.041667*W_bar((c),(z),(y)-1,(x)))+(0.041667*W_bar((c),(z)+1,(y),(x)))+(0.041667*W_bar((c),(z)-1,(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// increment
#undef s0
#define s0(c,z,y,x) W_ave((c),(z),(y),(x))+=W((c),(z),(y),(x))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// interpL_d1
#undef s0
#define s0(c,z,y,x) W_aveL_d1((c),(z),(y),(x))=(0.033333*W_ave((c),(z),(y),(x)-3))+((-0.050000)*W_ave((c),(z),(y),(x)+1))+((-0.216667)*W_ave((c),(z),(y),(x)-2))+(0.450000*W_ave((c),(z),(y),(x)))+(0.783333*W_ave((c),(z),(y),(x)-1))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = 0; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// interpH_d1
#undef s0
#define s0(c,z,y,x) W_aveH_d1((c),(z),(y),(x))=((-0.050000)*W_ave((c),(z),(y),(x)-2))+(0.033333*W_ave((c),(z),(y),(x)+2))+(0.450000*W_ave((c),(z),(y),(x)-1))+((-0.216667)*W_ave((c),(z),(y),(x)+1))+(0.783333*W_ave((c),(z),(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = -1; t4 <= 64; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// upwindState1
#undef s0
#define s0(z,y,x) {\
W_ave_f_d1(0,(z),(y),(x))=(0.000000 < ((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d1(0,(z),(y),(x))) : (W_aveH_d1(0,(z),(y),(x)));\
W_ave_f_d1(1,(z),(y),(x))=(0.000000 < ((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d1(1,(z),(y),(x))) : (W_aveH_d1(1,(z),(y),(x)));\
W_ave_f_d1(2,(z),(y),(x))=(0.000000 < ((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d1(2,(z),(y),(x))) : (W_aveH_d1(2,(z),(y),(x)));\
W_ave_f_d1(3,(z),(y),(x))=(0.000000 < ((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d1(3,(z),(y),(x))) : (W_aveH_d1(3,(z),(y),(x)));\
W_ave_f_d1(4,(z),(y),(x))=(0.000000 < ((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d1(4,(z),(y),(x))) : (W_aveH_d1(4,(z),(y),(x)));\
W_ave_f_d1(0,(z),(y),(x))+=(0.000000 < sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000) ? (((((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000+((((((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))*(W_aveL_d1(0+1,(z),(y),(x))-W_aveH_d1(0+1,(z),(y),(x))))*0.500000-W_ave_f_d1(4,(z),(y),(x))))/((sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000)))*sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000)))) : (0.000000);\
W_ave_f_d1(0+1,(z),(y),(x))=(0.000000 < sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000) ? (((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))) : (W_ave_f_d1(0+1,(z),(y),(x)));\
W_ave_f_d1(4,(z),(y),(x))=(0.000000 < sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000+((W_aveL_d1(4,(z),(y),(x))-W_aveH_d1(4,(z),(y),(x))))/((2.000000*(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d1(0+1,(z),(y),(x))+W_aveH_d1(0+1,(z),(y),(x))))*0.500000) ? (((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000+((((((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d1(4,(z),(y),(x))+W_aveH_d1(4,(z),(y),(x))))*0.500000))/(((W_aveL_d1(0,(z),(y),(x))+W_aveH_d1(0,(z),(y),(x))))*0.500000))))*(W_aveL_d1(0+1,(z),(y),(x))-W_aveH_d1(0+1,(z),(y),(x))))*0.500000) : (W_ave_f_d1(4,(z),(y),(x)));\
}

for(t1 = -3; t1 <= 66; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = 0; t3 <= 64; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// getFlux1
#undef s0
#define s0(z,y,x) {\
F_bar_f_d1(0,(z),(y),(x))=(W_ave_f_d1(0+1,(z),(y),(x)))*W_ave_f_d1(0,(z),(y),(x));\
F_bar_f_d1(1,(z),(y),(x))=W_ave_f_d1(1,(z),(y),(x))*F_bar_f_d1(0,(z),(y),(x));\
F_bar_f_d1(2,(z),(y),(x))=W_ave_f_d1(2,(z),(y),(x))*F_bar_f_d1(0,(z),(y),(x));\
F_bar_f_d1(3,(z),(y),(x))=W_ave_f_d1(3,(z),(y),(x))*F_bar_f_d1(0,(z),(y),(x));\
F_bar_f_d1(0+1,(z),(y),(x))+=W_ave_f_d1(4,(z),(y),(x));\
F_bar_f_d1(4,(z),(y),(x))=((1.400000/(1.400000-1))*W_ave_f_d1(0+1,(z),(y),(x)))*W_ave_f_d1(4,(z),(y),(x))+0.500000*F_bar_f_d1(0,(z),(y),(x))*((W_ave_f_d1(1,(z),(y),(x))*W_ave_f_d1(1,(z),(y),(x)))+(W_ave_f_d1(2,(z),(y),(x))*W_ave_f_d1(2,(z),(y),(x)))+(W_ave_f_d1(3,(z),(y),(x))*W_ave_f_d1(3,(z),(y),(x))));\
}

for(t1 = -3; t1 <= 66; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = 0; t3 <= 64; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// deconvolve_f_d1
#undef s0
#define s0(c,z,y,x) W_f_d1((c),(z),(y),(x))=(1.166667*W_ave_f_d1((c),(z),(y),(x)))+((-0.041667)*W_ave_f_d1((c),(z),(y)+1,(x)))+((-0.041667)*W_ave_f_d1((c),(z),(y)-1,(x)))+((-0.041667)*W_ave_f_d1((c),(z)+1,(y),(x)))+((-0.041667)*W_ave_f_d1((c),(z)-1,(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      for(t4 = 0; t4 <= 64; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// getFlux2
#undef s0
#define s0(z,y,x) {\
F_ave_f_d1(0,(z),(y),(x))=(W_f_d1(0+1,(z),(y),(x)))*W_f_d1(0,(z),(y),(x));\
F_ave_f_d1(1,(z),(y),(x))=W_f_d1(1,(z),(y),(x))*F_ave_f_d1(0,(z),(y),(x));\
F_ave_f_d1(2,(z),(y),(x))=W_f_d1(2,(z),(y),(x))*F_ave_f_d1(0,(z),(y),(x));\
F_ave_f_d1(3,(z),(y),(x))=W_f_d1(3,(z),(y),(x))*F_ave_f_d1(0,(z),(y),(x));\
F_ave_f_d1(0+1,(z),(y),(x))+=W_f_d1(4,(z),(y),(x));\
F_ave_f_d1(4,(z),(y),(x))=((1.400000/(1.400000-1))*W_f_d1(0+1,(z),(y),(x)))*W_f_d1(4,(z),(y),(x))+0.500000*F_ave_f_d1(0,(z),(y),(x))*((W_f_d1(1,(z),(y),(x))*W_f_d1(1,(z),(y),(x)))+(W_f_d1(2,(z),(y),(x))*W_f_d1(2,(z),(y),(x)))+(W_f_d1(3,(z),(y),(x))*W_f_d1(3,(z),(y),(x))));\
}

for(t1 = -2; t1 <= 65; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = 0; t3 <= 64; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// smul_d1
#undef s0
#define s0(c,z,y,x) F_bar_f_d1((c),(z),(y),(x))*=0.041667

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = 0; t4 <= 64; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// lap_f_d1
#undef s0
#define s0(c,z,y,x) F_lap_f_d1((c),(z),(y),(x))=((-0.166667)*F_bar_f_d1((c),(z),(y),(x)))+(0.041667*F_bar_f_d1((c),(z),(y)+1,(x)))+(0.041667*F_bar_f_d1((c),(z),(y)-1,(x)))+(0.041667*F_bar_f_d1((c),(z)+1,(y),(x)))+(0.041667*F_bar_f_d1((c),(z)-1,(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      for(t4 = 0; t4 <= 64; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// inc_f_d1
#undef s0
#define s0(c,z,y,x) F_ave_f_d1((c),(z),(y),(x))+=F_lap_f_d1((c),(z),(y),(x))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      for(t4 = 0; t4 <= 64; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// div_f_d1
#undef s0
#define s0(c,z,y,x) F_div_f_d1((c),(z),(y),(x))=((-1.000000)*F_ave_f_d1((c),(z),(y),(x)))+(1.000000*F_ave_f_d1((c),(z),(y),(x)+1))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      for(t4 = 0; t4 <= 63; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// inc_rhs_d1
#undef s0
#define s0(c,z,y,x) rhs((c),(z),(y),(x))+=F_div_f_d1((c),(z),(y),(x))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 63; t2++) {
    for(t3 = 0; t3 <= 63; t3++) {
      for(t4 = 0; t4 <= 63; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// interpL_d2
#undef s0
#define s0(c,z,y,x) W_aveL_d2((c),(z),(y),(x))=(0.033333*W_ave((c),(z),(y)-3,(x)))+((-0.050000)*W_ave((c),(z),(y)+1,(x)))+((-0.216667)*W_ave((c),(z),(y)-2,(x)))+(0.450000*W_ave((c),(z),(y),(x)))+(0.783333*W_ave((c),(z),(y)-1,(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = 0; t3 <= 65; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// interpH_d2
#undef s0
#define s0(c,z,y,x) W_aveH_d2((c),(z),(y),(x))=((-0.050000)*W_ave((c),(z),(y)-2,(x)))+(0.033333*W_ave((c),(z),(y)+2,(x)))+(0.450000*W_ave((c),(z),(y)-1,(x)))+((-0.216667)*W_ave((c),(z),(y)+1,(x)))+(0.783333*W_ave((c),(z),(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -1; t3 <= 64; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// upwindState2
#undef s0
#define s0(z,y,x) {\
W_ave_f_d2(0,(z),(y),(x))=(0.000000 < ((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d2(0,(z),(y),(x))) : (W_aveH_d2(0,(z),(y),(x)));\
W_ave_f_d2(1,(z),(y),(x))=(0.000000 < ((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d2(1,(z),(y),(x))) : (W_aveH_d2(1,(z),(y),(x)));\
W_ave_f_d2(2,(z),(y),(x))=(0.000000 < ((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d2(2,(z),(y),(x))) : (W_aveH_d2(2,(z),(y),(x)));\
W_ave_f_d2(3,(z),(y),(x))=(0.000000 < ((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d2(3,(z),(y),(x))) : (W_aveH_d2(3,(z),(y),(x)));\
W_ave_f_d2(4,(z),(y),(x))=(0.000000 < ((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d2(4,(z),(y),(x))) : (W_aveH_d2(4,(z),(y),(x)));\
W_ave_f_d2(0,(z),(y),(x))+=(0.000000 < sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000) ? (((((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000+((((((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))*(W_aveL_d2(1+1,(z),(y),(x))-W_aveH_d2(1+1,(z),(y),(x))))*0.500000-W_ave_f_d2(4,(z),(y),(x))))/((sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000)))*sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000)))) : (0.000000);\
W_ave_f_d2(1+1,(z),(y),(x))=(0.000000 < sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000) ? (((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))) : (W_ave_f_d2(1+1,(z),(y),(x)));\
W_ave_f_d2(4,(z),(y),(x))=(0.000000 < sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000+((W_aveL_d2(4,(z),(y),(x))-W_aveH_d2(4,(z),(y),(x))))/((2.000000*(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d2(1+1,(z),(y),(x))+W_aveH_d2(1+1,(z),(y),(x))))*0.500000) ? (((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000+((((((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d2(4,(z),(y),(x))+W_aveH_d2(4,(z),(y),(x))))*0.500000))/(((W_aveL_d2(0,(z),(y),(x))+W_aveH_d2(0,(z),(y),(x))))*0.500000))))*(W_aveL_d2(1+1,(z),(y),(x))-W_aveH_d2(1+1,(z),(y),(x))))*0.500000) : (W_ave_f_d2(4,(z),(y),(x)));\
}

for(t1 = -3; t1 <= 66; t1++) {
  for(t2 = 0; t2 <= 64; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// getFlux3
#undef s0
#define s0(z,y,x) {\
F_bar_f_d2(0,(z),(y),(x))=(W_ave_f_d2(1+1,(z),(y),(x)))*W_ave_f_d2(0,(z),(y),(x));\
F_bar_f_d2(1,(z),(y),(x))=W_ave_f_d2(1,(z),(y),(x))*F_bar_f_d2(0,(z),(y),(x));\
F_bar_f_d2(2,(z),(y),(x))=W_ave_f_d2(2,(z),(y),(x))*F_bar_f_d2(0,(z),(y),(x));\
F_bar_f_d2(3,(z),(y),(x))=W_ave_f_d2(3,(z),(y),(x))*F_bar_f_d2(0,(z),(y),(x));\
F_bar_f_d2(1+1,(z),(y),(x))+=W_ave_f_d2(4,(z),(y),(x));\
F_bar_f_d2(4,(z),(y),(x))=((1.400000/(1.400000-1))*W_ave_f_d2(1+1,(z),(y),(x)))*W_ave_f_d2(4,(z),(y),(x))+0.500000*F_bar_f_d2(0,(z),(y),(x))*((W_ave_f_d2(1,(z),(y),(x))*W_ave_f_d2(1,(z),(y),(x)))+(W_ave_f_d2(2,(z),(y),(x))*W_ave_f_d2(2,(z),(y),(x)))+(W_ave_f_d2(3,(z),(y),(x))*W_ave_f_d2(3,(z),(y),(x))));\
}

for(t1 = -3; t1 <= 66; t1++) {
  for(t2 = 0; t2 <= 64; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// deconvolve_f_d2
#undef s0
#define s0(c,z,y,x) W_f_d2((c),(z),(y),(x))=(1.166667*W_ave_f_d2((c),(z),(y),(x)))+((-0.041667)*W_ave_f_d2((c),(z),(y),(x)+1))+((-0.041667)*W_ave_f_d2((c),(z),(y),(x)-1))+((-0.041667)*W_ave_f_d2((c),(z)+1,(y),(x)))+((-0.041667)*W_ave_f_d2((c),(z)-1,(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = 0; t3 <= 64; t3++) {
      for(t4 = -2; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// getFlux4
#undef s0
#define s0(z,y,x) {\
F_ave_f_d2(0,(z),(y),(x))=(W_f_d2(1+1,(z),(y),(x)))*W_f_d2(0,(z),(y),(x));\
F_ave_f_d2(1,(z),(y),(x))=W_f_d2(1,(z),(y),(x))*F_ave_f_d2(0,(z),(y),(x));\
F_ave_f_d2(2,(z),(y),(x))=W_f_d2(2,(z),(y),(x))*F_ave_f_d2(0,(z),(y),(x));\
F_ave_f_d2(3,(z),(y),(x))=W_f_d2(3,(z),(y),(x))*F_ave_f_d2(0,(z),(y),(x));\
F_ave_f_d2(1+1,(z),(y),(x))+=W_f_d2(4,(z),(y),(x));\
F_ave_f_d2(4,(z),(y),(x))=((1.400000/(1.400000-1))*W_f_d2(1+1,(z),(y),(x)))*W_f_d2(4,(z),(y),(x))+0.500000*F_ave_f_d2(0,(z),(y),(x))*((W_f_d2(1,(z),(y),(x))*W_f_d2(1,(z),(y),(x)))+(W_f_d2(2,(z),(y),(x))*W_f_d2(2,(z),(y),(x)))+(W_f_d2(3,(z),(y),(x))*W_f_d2(3,(z),(y),(x))));\
}

for(t1 = -2; t1 <= 65; t1++) {
  for(t2 = 0; t2 <= 64; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// smul_d2
#undef s0
#define s0(c,z,y,x) F_bar_f_d2((c),(z),(y),(x))*=0.041667

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = 0; t3 <= 64; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// lap_f_d2
#undef s0
#define s0(c,z,y,x) F_lap_f_d2((c),(z),(y),(x))=((-0.166667)*F_bar_f_d2((c),(z),(y),(x)))+(0.041667*F_bar_f_d2((c),(z),(y),(x)+1))+(0.041667*F_bar_f_d2((c),(z),(y),(x)-1))+(0.041667*F_bar_f_d2((c),(z)+1,(y),(x)))+(0.041667*F_bar_f_d2((c),(z)-1,(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = 0; t3 <= 64; t3++) {
      for(t4 = -2; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// inc_f_d2
#undef s0
#define s0(c,z,y,x) F_ave_f_d2((c),(z),(y),(x))+=F_lap_f_d2((c),(z),(y),(x))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = 0; t3 <= 64; t3++) {
      for(t4 = -2; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// div_f_d2
#undef s0
#define s0(c,z,y,x) F_div_f_d2((c),(z),(y),(x))=((-1.000000)*F_ave_f_d2((c),(z),(y),(x)))+(1.000000*F_ave_f_d2((c),(z),(y)+1,(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = 0; t3 <= 63; t3++) {
      for(t4 = -2; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// inc_rhs_d2
#undef s0
#define s0(c,z,y,x) rhs((c),(z),(y),(x))+=F_div_f_d2((c),(z),(y),(x))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 63; t2++) {
    for(t3 = 0; t3 <= 63; t3++) {
      for(t4 = 0; t4 <= 63; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// interpL_d3
#undef s0
#define s0(c,z,y,x) W_aveL_d3((c),(z),(y),(x))=(0.033333*W_ave((c),(z)-3,(y),(x)))+((-0.050000)*W_ave((c),(z)+1,(y),(x)))+((-0.216667)*W_ave((c),(z)-2,(y),(x)))+(0.450000*W_ave((c),(z),(y),(x)))+(0.783333*W_ave((c),(z)-1,(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 65; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// interpH_d3
#undef s0
#define s0(c,z,y,x) W_aveH_d3((c),(z),(y),(x))=((-0.050000)*W_ave((c),(z)-2,(y),(x)))+(0.033333*W_ave((c),(z)+2,(y),(x)))+(0.450000*W_ave((c),(z)-1,(y),(x)))+((-0.216667)*W_ave((c),(z)+1,(y),(x)))+(0.783333*W_ave((c),(z),(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = -1; t2 <= 64; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// upwindState3
#undef s0
#define s0(z,y,x) {\
W_ave_f_d3(0,(z),(y),(x))=(0.000000 < ((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d3(0,(z),(y),(x))) : (W_aveH_d3(0,(z),(y),(x)));\
W_ave_f_d3(1,(z),(y),(x))=(0.000000 < ((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d3(1,(z),(y),(x))) : (W_aveH_d3(1,(z),(y),(x)));\
W_ave_f_d3(2,(z),(y),(x))=(0.000000 < ((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d3(2,(z),(y),(x))) : (W_aveH_d3(2,(z),(y),(x)));\
W_ave_f_d3(3,(z),(y),(x))=(0.000000 < ((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d3(3,(z),(y),(x))) : (W_aveH_d3(3,(z),(y),(x)));\
W_ave_f_d3(4,(z),(y),(x))=(0.000000 < ((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))) ? (W_aveL_d3(4,(z),(y),(x))) : (W_aveH_d3(4,(z),(y),(x)));\
W_ave_f_d3(0,(z),(y),(x))+=(0.000000 < sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000) ? (((((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000+((((((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))*(W_aveL_d3(2+1,(z),(y),(x))-W_aveH_d3(2+1,(z),(y),(x))))*0.500000-W_ave_f_d3(4,(z),(y),(x))))/((sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000)))*sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000)))) : (0.000000);\
W_ave_f_d3(2+1,(z),(y),(x))=(0.000000 < sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000) ? (((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))) : (W_ave_f_d3(2+1,(z),(y),(x)));\
W_ave_f_d3(4,(z),(y),(x))=(0.000000 < sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))+(-sgn(((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000+((W_aveL_d3(4,(z),(y),(x))-W_aveH_d3(4,(z),(y),(x))))/((2.000000*(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))))*((W_aveL_d3(2+1,(z),(y),(x))+W_aveH_d3(2+1,(z),(y),(x))))*0.500000) ? (((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000+((((((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))*(sqrt((1.400000*(((W_aveL_d3(4,(z),(y),(x))+W_aveH_d3(4,(z),(y),(x))))*0.500000))/(((W_aveL_d3(0,(z),(y),(x))+W_aveH_d3(0,(z),(y),(x))))*0.500000))))*(W_aveL_d3(2+1,(z),(y),(x))-W_aveH_d3(2+1,(z),(y),(x))))*0.500000) : (W_ave_f_d3(4,(z),(y),(x)));\
}

for(t1 = 0; t1 <= 64; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// getFlux5
#undef s0
#define s0(z,y,x) {\
F_bar_f_d3(0,(z),(y),(x))=(W_ave_f_d3(2+1,(z),(y),(x)))*W_ave_f_d3(0,(z),(y),(x));\
F_bar_f_d3(1,(z),(y),(x))=W_ave_f_d3(1,(z),(y),(x))*F_bar_f_d3(0,(z),(y),(x));\
F_bar_f_d3(2,(z),(y),(x))=W_ave_f_d3(2,(z),(y),(x))*F_bar_f_d3(0,(z),(y),(x));\
F_bar_f_d3(3,(z),(y),(x))=W_ave_f_d3(3,(z),(y),(x))*F_bar_f_d3(0,(z),(y),(x));\
F_bar_f_d3(2+1,(z),(y),(x))+=W_ave_f_d3(4,(z),(y),(x));\
F_bar_f_d3(4,(z),(y),(x))=((1.400000/(1.400000-1))*W_ave_f_d3(2+1,(z),(y),(x)))*W_ave_f_d3(4,(z),(y),(x))+0.500000*F_bar_f_d3(0,(z),(y),(x))*((W_ave_f_d3(1,(z),(y),(x))*W_ave_f_d3(1,(z),(y),(x)))+(W_ave_f_d3(2,(z),(y),(x))*W_ave_f_d3(2,(z),(y),(x)))+(W_ave_f_d3(3,(z),(y),(x))*W_ave_f_d3(3,(z),(y),(x))));\
}

for(t1 = 0; t1 <= 64; t1++) {
  for(t2 = -3; t2 <= 66; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// deconvolve_f_d3
#undef s0
#define s0(c,z,y,x) W_f_d3((c),(z),(y),(x))=(1.166667*W_ave_f_d3((c),(z),(y),(x)))+((-0.041667)*W_ave_f_d3((c),(z),(y),(x)+1))+((-0.041667)*W_ave_f_d3((c),(z),(y),(x)-1))+((-0.041667)*W_ave_f_d3((c),(z),(y)+1,(x)))+((-0.041667)*W_ave_f_d3((c),(z),(y)-1,(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 64; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      for(t4 = -2; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// getFlux6
#undef s0
#define s0(z,y,x) {\
F_ave_f_d3(0,(z),(y),(x))=(W_f_d3(2+1,(z),(y),(x)))*W_f_d3(0,(z),(y),(x));\
F_ave_f_d3(1,(z),(y),(x))=W_f_d3(1,(z),(y),(x))*F_ave_f_d3(0,(z),(y),(x));\
F_ave_f_d3(2,(z),(y),(x))=W_f_d3(2,(z),(y),(x))*F_ave_f_d3(0,(z),(y),(x));\
F_ave_f_d3(3,(z),(y),(x))=W_f_d3(3,(z),(y),(x))*F_ave_f_d3(0,(z),(y),(x));\
F_ave_f_d3(2+1,(z),(y),(x))+=W_f_d3(4,(z),(y),(x));\
F_ave_f_d3(4,(z),(y),(x))=((1.400000/(1.400000-1))*W_f_d3(2+1,(z),(y),(x)))*W_f_d3(4,(z),(y),(x))+0.500000*F_ave_f_d3(0,(z),(y),(x))*((W_f_d3(1,(z),(y),(x))*W_f_d3(1,(z),(y),(x)))+(W_f_d3(2,(z),(y),(x))*W_f_d3(2,(z),(y),(x)))+(W_f_d3(3,(z),(y),(x))*W_f_d3(3,(z),(y),(x))));\
}

for(t1 = 0; t1 <= 64; t1++) {
  for(t2 = -2; t2 <= 65; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      s0(t1,t2,t3);
    }
  }
}

// smul_d3
#undef s0
#define s0(c,z,y,x) F_bar_f_d3((c),(z),(y),(x))*=0.041667

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 64; t2++) {
    for(t3 = -3; t3 <= 66; t3++) {
      for(t4 = -3; t4 <= 66; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// lap_f_d3
#undef s0
#define s0(c,z,y,x) F_lap_f_d3((c),(z),(y),(x))=((-0.166667)*F_bar_f_d3((c),(z),(y),(x)))+(0.041667*F_bar_f_d3((c),(z),(y),(x)+1))+(0.041667*F_bar_f_d3((c),(z),(y),(x)-1))+(0.041667*F_bar_f_d3((c),(z),(y)+1,(x)))+(0.041667*F_bar_f_d3((c),(z),(y)-1,(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 64; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      for(t4 = -2; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// inc_f_d3
#undef s0
#define s0(c,z,y,x) F_ave_f_d3((c),(z),(y),(x))+=F_lap_f_d3((c),(z),(y),(x))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 64; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      for(t4 = -2; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// div_f_d3
#undef s0
#define s0(c,z,y,x) F_div_f_d3((c),(z),(y),(x))=((-1.000000)*F_ave_f_d3((c),(z),(y),(x)))+(1.000000*F_ave_f_d3((c),(z)+1,(y),(x)))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 63; t2++) {
    for(t3 = -2; t3 <= 65; t3++) {
      for(t4 = -2; t4 <= 65; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// inc_rhs_d3
#undef s0
#define s0(c,z,y,x) rhs((c),(z),(y),(x))+=F_div_f_d3((c),(z),(y),(x))

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 63; t2++) {
    for(t3 = 0; t3 <= 63; t3++) {
      for(t4 = 0; t4 <= 63; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

// muldx
#undef s0
#define s0(c,z,y,x) rhs((c),(z),(y),(x))*=-1.000000

for(t1 = 0; t1 <= 4; t1++) {
  for(t2 = 0; t2 <= 63; t2++) {
    for(t3 = 0; t3 <= 63; t3++) {
      for(t4 = 0; t4 <= 63; t4++) {
        s0(t1,t2,t3,t4);
      }
    }
  }
}

    free(W_bar);
    free(u);
    free(W);
    free(umax);
    free(W_ave);
    free(W_aveL_d1);
    free(W_aveH_d1);
    free(W_ave_f_d1);
    free(F_bar_f_d1);
    free(W_f_d1);
    free(F_ave_f_d1);
    free(F_lap_f_d1);
    free(F_div_f_d1);
    free(W_aveL_d2);
    free(W_aveH_d2);
    free(W_ave_f_d2);
    free(F_bar_f_d2);
    free(W_f_d2);
    free(F_ave_f_d2);
    free(F_lap_f_d2);
    free(F_div_f_d2);
    free(W_aveL_d3);
    free(W_aveH_d3);
    free(W_ave_f_d3);
    free(F_bar_f_d3);
    free(W_f_d3);
    free(F_ave_f_d3);
    free(F_lap_f_d3);
    free(F_div_f_d3);

    return (retval);
}    // euler_step

#undef min
#undef max
#undef abs
#undef absmin
#undef absmax
#undef floord
#undef sgn
#undef offset2
#undef offset3
#undef offset4
#undef arrinit
#undef arrprnt
#undef F_ave_f_d1
#undef F_ave_f_d2
#undef F_ave_f_d3
#undef F_bar_f_d1
#undef F_bar_f_d2
#undef F_bar_f_d3
#undef F_div_f_d1
#undef F_div_f_d2
#undef F_div_f_d3
#undef F_lap_f_d1
#undef F_lap_f_d2
#undef F_lap_f_d3
#undef U
#undef W
#undef W_ave
#undef W_aveH_d1
#undef W_aveH_d2
#undef W_aveH_d3
#undef W_aveL_d1
#undef W_aveL_d2
#undef W_aveL_d3
#undef W_ave_f_d1
#undef W_ave_f_d2
#undef W_ave_f_d3
#undef W_bar
#undef W_f_d1
#undef W_f_d2
#undef W_f_d3
#undef rhs
#undef u
#undef umax

