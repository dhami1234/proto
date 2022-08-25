#include "Proto.H"
#include "MHD_Mapping.H"
#include "CommonTemplates.H"
#include "Proto_Timer.H"
#include "Proto_WriteBoxData.H"
#include "MHDOp.H"
#include "MHD_Output_Writer.H"
#include "MHD_Input_Parsing.H"
#include "MHD_Constants.H"
//#include "MHDLevelDataRK4.H"
extern Parsefrominputs inputs;

typedef BoxData<double,1,HOST> Scalar;
typedef BoxData<double,NUMCOMPS,HOST> Vector;

namespace MHD_Mapping {

	PROTO_KERNEL_START
	void iotaFuncF(Point           & a_p,
	               V               & a_X,
	               const double a_dx,
	               const double a_dy,
	               const double a_dz)
	{
		for (int ii = 0; ii < DIM; ii++)
		{
			double dxd;
			if (ii == 0) dxd = a_dx;
			if (ii == 1) dxd = a_dy;
			if (ii == 2) dxd = a_dz;
			a_X(ii) = a_p[ii]*dxd + 0.5*dxd;
		}
	}
	PROTO_KERNEL_END(iotaFuncF,iotaFunc)

	PROTO_KERNEL_START
	void iotaFuncFaceF(Point           & a_p,
	                   V               & a_X,
	                   const double a_dx,
	                   const double a_dy,
	                   const double a_dz,
	                   int a_d)
	{
		for (int ii = 0; ii < DIM; ii++)
		{
			double dxd;
			if (ii == 0) dxd = a_dx;
			if (ii == 1) dxd = a_dy;
			if (ii == 2) dxd = a_dz;
			if (ii == a_d) {
				a_X(ii) = a_p[ii]*dxd;
			} else {
				a_X(ii) = a_p[ii]*dxd + 0.5*dxd;
			}
		}
	}
	PROTO_KERNEL_END(iotaFuncFaceF,iotaFuncFace)

	void eta_calc(BoxData<double,DIM>& a_eta,
	              const Box& a_bx,
	              const double a_dx,
	              const double a_dy,
	              const double a_dz)
	{
		forallInPlace_p(iotaFunc, a_bx, a_eta, a_dx, a_dy, a_dz);
	}

	void etaFace_calc(BoxData<double,DIM>& a_eta,
	                  const Box& a_bx,
	                  const double a_dx,
	                  const double a_dy,
	                  const double a_dz,
	                  int a_d)
	{
		forallInPlace_p(iotaFuncFace, a_bx, a_eta, a_dx, a_dy, a_dz, a_d);
	}

	PROTO_KERNEL_START
	void eta_to_xF(const Point& a_pt,
					Var<double,DIM>& a_x,
	               const Var<double,DIM>& a_eta)
	{

#if DIM == 2
		if (inputs.grid_type_global == 0) {
			a_x(0) = a_eta(0);
			a_x(1) = a_eta(1);
		}
		if (inputs.grid_type_global == 1) {
			double C1 = inputs.C1_fix;
			double C2 = C1;
			a_x(0) = a_eta(0) + C1*sin(2.0*PI*a_eta(0))*sin(2.0*PI*a_eta(1));
			a_x(1) = a_eta(1) + C2*sin(2.0*PI*a_eta(0))*sin(2.0*PI*a_eta(1));
		}
		if (inputs.grid_type_global == 2) {
			a_x(0) = (inputs.r_in*AU + a_eta(0)*(inputs.r_out*AU-inputs.r_in*AU))*cos(2.0*PI*a_eta(1));
			a_x(1) = (inputs.r_in*AU + a_eta(0)*(inputs.r_out*AU-inputs.r_in*AU))*sin(2.0*PI*a_eta(1));
		}
#endif
#if DIM == 3
		if (inputs.grid_type_global == 0) {
			a_x(0) = a_eta(0);
			a_x(1) = a_eta(1);
			a_x(2) = a_eta(2);
		}
		if (inputs.grid_type_global == 2) {
			if (inputs.Spherical_2nd_order == 0){
				double R_t = (inputs.r_out*AU - inputs.r_in*AU)/(exp(inputs.C_rad) - 1.0);
				double r = inputs.r_in*AU + R_t*(exp(inputs.C_rad*a_eta(0)) - 1.0);
				a_x(0) = r*sin(PI*a_eta(1))*cos(2.0*PI*a_eta(2));
				a_x(1) = r*sin(PI*a_eta(1))*sin(2.0*PI*a_eta(2));
				a_x(2) = r*cos(PI*a_eta(1));
			}
			if (inputs.Spherical_2nd_order == 1){
				double R_t = (inputs.r_out*AU - inputs.r_in*AU)/(exp(inputs.C_rad) - 1.0);
				double r = inputs.r_in*AU + R_t*(exp(inputs.C_rad*a_eta(0)) - 1.0);
				double theta = PI*a_eta(1);
				double phi = 2.0*PI*a_eta(2);

				if (theta < 0){
					theta = -theta;
					if (phi < PI){
						phi = phi + PI;
					} else {
						phi = phi - PI;
					}
				}
				if (theta > PI){
					theta = PI - (theta - PI);
					if (phi < PI){
						phi = phi + PI;
					} else {
						phi = phi - PI;
					}
				}

				a_x(0) = r*sin(theta)*cos(phi);
				a_x(1) = r*sin(theta)*sin(phi);
				a_x(2) = r*cos(theta);
			}

		}

#endif
	}
	PROTO_KERNEL_END(eta_to_xF, eta_to_x)

	PROTO_KERNEL_START
	void eta_to_x_aveF(Var<double,DIM>& a_x,
	               const Var<double,DIM>& a_eta,
				   const double a_dx,
	               const double a_dy,
	               const double a_dz)
	{
		double R_t = (inputs.r_out*AU - inputs.r_in*AU)/(exp(inputs.C_rad) - 1.0);
		double r = inputs.r_in*AU + R_t*(exp(inputs.C_rad*a_eta(0)) - 1.0);
 
		a_x(0) = (2*cos(2*a_eta(2)*PI)*sin(a_dz*PI)*sin((a_dy*PI)/2.)*sin(a_eta(1)*PI)*(inputs.C_rad*a_dx*(inputs.r_in*AU - R_t) + 2*exp(inputs.C_rad*a_eta(0))*R_t*sinh((inputs.C_rad*a_dx)/2.)))/(inputs.C_rad*a_dx*a_dz*a_dy*pow(PI,2));
		a_x(1) = (2*sin(a_dz*PI)*sin((a_dy*PI)/2.)*sin(2*a_eta(2)*PI)*sin(a_eta(1)*PI)*(inputs.C_rad*a_dx*(inputs.r_in*AU - R_t) + 2*exp(inputs.C_rad*a_eta(0))*R_t*sinh((inputs.C_rad*a_dx)/2.)))/(inputs.C_rad*a_dx*a_dz*a_dy*pow(PI,2));
		a_x(2) = (2*cos(a_eta(1)*PI)*sin((a_dy*PI)/2.)*(inputs.C_rad*a_dx*(inputs.r_in*AU - R_t) + 2*exp(inputs.C_rad*a_eta(0))*R_t*sinh((inputs.C_rad*a_dx)/2.)))/(inputs.C_rad*a_dx*a_dy*PI);
	}
	PROTO_KERNEL_END(eta_to_x_aveF, eta_to_x_ave)


	void eta_to_x_calc(BoxData<double,DIM>& a_x,
	                   const BoxData<double,DIM>& a_eta,
					   const Box& dbx0)
	{
		forallInPlace_p(eta_to_x,dbx0,a_x,a_eta);
	}

	void eta_to_x_ave_calc(BoxData<double,DIM>& a_x,
	                   const BoxData<double,DIM>& a_eta,
					   const double a_dx,
	                   const double a_dy,
	                   const double a_dz)
	{
		a_x = forall<double,DIM>(eta_to_x_ave,a_eta,a_dx,a_dy,a_dz);
	}

	void phys_coords_calc(BoxData<double,DIM>& a_x,
	                      const Box& dbx1,
	                      const double a_dx,
	                      const double a_dy,
	                      const double a_dz)
	{
		Box dbx0=a_x.box();
		BoxData<double,DIM> eta(dbx0);
		MHD_Mapping::eta_calc(eta,dbx0,a_dx, a_dy, a_dz);
		MHD_Mapping::eta_to_x_calc(a_x,eta, dbx0);		
	}

	void phys_coords_face_calc(BoxData<double,DIM>& a_x,
	                      const Box& dbx1,
	                      const double a_dx,
	                      const double a_dy,
	                      const double a_dz,
						  const int a_d)
	{
		Box dbx0=a_x.box();
		BoxData<double,DIM> eta(dbx0);
		MHD_Mapping::etaFace_calc(eta,dbx0,a_dx, a_dy, a_dz, a_d);
		MHD_Mapping::eta_to_x_calc(a_x,eta, dbx0);		
	}
	
	PROTO_KERNEL_START
	void dot_pro_calcXF(Var<double,1>& a_dot_pro,
	                    const Var<double,1>& a_d_perp_N_s,
	                    const Var<double,1>& a_d_perp_X)
	{
		a_dot_pro(0) = (a_d_perp_N_s(0)*a_d_perp_X(0));
	}
	PROTO_KERNEL_END(dot_pro_calcXF, dot_pro_calcX)


	PROTO_KERNEL_START
	void X_f_mapped1D_calcF(Var<double,1>& a_X_f_mapped1D,
	                        const Var<double,1>& a_X_ave_f,
	                        const Var<double,1>& a_N_s_d_ave_f,
	                        const Var<double,1>& a_dot_pro_sum)
	{
		a_X_f_mapped1D(0) = a_N_s_d_ave_f(0)*a_X_ave_f(0) + a_dot_pro_sum(0)/12.0;
	}
	PROTO_KERNEL_END(X_f_mapped1D_calcF, X_f_mapped1D_calc)




	PROTO_KERNEL_START
	void X_ave_f_calcF( const Point& a_pt,
	                    Var<double,1>& a_X_ave_f,
	                    int a_s,
	                    int a_d,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz)
	{

#if DIM == 2
		double x_hi = (a_pt[0] + 1.0)*a_dx;
		double x_lo = (a_pt[0] - 0.0)*a_dx;
		double y_hi = (a_pt[1] + 1.0)*a_dy;
		double y_lo = (a_pt[1] - 0.0)*a_dy;

		if (inputs.grid_type_global == 0) {
			if (a_s == 0 && a_d == 0) a_X_ave_f(0) = ((x_lo*y_hi)
				                                  -(x_lo*y_lo))/DIM/a_dy;
			if (a_s == 1 && a_d == 0) a_X_ave_f(0) = ((y_hi*y_hi/2.0)
				                                  -(y_lo*y_lo/2.0))/DIM/a_dy;
			if (a_s == 0 && a_d == 1) a_X_ave_f(0) = ((x_hi*x_hi/2.0)
				                                  -(x_lo*x_lo/2.0))/DIM/a_dx;
			if (a_s == 1 && a_d == 1) a_X_ave_f(0) = ((x_hi*y_lo)
				                                  -(x_lo*y_lo))/DIM/a_dx;
		}

		if (inputs.grid_type_global == 1) {
			double C1 = inputs.C1_fix;
			double C2 = C1;
			if (a_s == 0 && a_d == 0) a_X_ave_f(0) = ((x_lo*y_hi-(C1/2./PI)*sin(2.*PI*x_lo)*cos(2.*PI*y_hi))
				                                  -(x_lo*y_lo-(C1/2./PI)*sin(2.*PI*x_lo)*cos(2.*PI*y_lo)))/DIM/a_dy;
			if (a_s == 1 && a_d == 0) a_X_ave_f(0) = ((y_hi*y_hi/2.0-(C2/2./PI)*sin(2.*PI*x_lo)*cos(2.*PI*y_hi))
				                                  -(y_lo*y_lo/2.0-(C2/2./PI)*sin(2.*PI*x_lo)*cos(2.*PI*y_lo)))/DIM/a_dy;
			if (a_s == 0 && a_d == 1) a_X_ave_f(0) = ((x_hi*x_hi/2.0-(C1/2./PI)*cos(2.*PI*x_hi)*sin(2.*PI*y_lo))
				                                  -(x_lo*x_lo/2.0-(C1/2./PI)*cos(2.*PI*x_lo)*sin(2.*PI*y_lo)))/DIM/a_dx;
			if (a_s == 1 && a_d == 1) a_X_ave_f(0) = ((x_hi*y_lo-(C2/2./PI)*cos(2.*PI*x_hi)*sin(2.*PI*y_lo))
				                                  -(x_lo*y_lo-(C2/2./PI)*cos(2.*PI*x_lo)*sin(2.*PI*y_lo)))/DIM/a_dx;
		}

		if (inputs.grid_type_global == 2) {
			double r_diff = inputs.r_out*AU-inputs.r_in*AU;
			if (a_s == 0 && a_d == 0) a_X_ave_f(0) = (((inputs.r_in*AU+x_lo*r_diff)*sin(2.*PI*y_hi)/2./PI)
				                                  -((inputs.r_in*AU+x_lo*r_diff)*sin(2.*PI*y_lo)/2./PI))/DIM/a_dy;
			if (a_s == 1 && a_d == 0) a_X_ave_f(0) = -(((inputs.r_in*AU+x_lo*r_diff)*cos(2.*PI*y_hi)/2./PI)
				                                   -((inputs.r_in*AU+x_lo*r_diff)*cos(2.*PI*y_lo)/2./PI))/DIM/a_dy;
			if (a_s == 0 && a_d == 1) a_X_ave_f(0) = (((inputs.r_in*AU*x_hi+x_hi*x_hi*r_diff/2.0)*cos(2.*PI*y_lo))
				                                  -((inputs.r_in*AU*x_lo+x_lo*x_lo*r_diff/2.0)*cos(2.*PI*y_lo)))/DIM/a_dx;
			if (a_s == 1 && a_d == 1) a_X_ave_f(0) = (((inputs.r_in*AU*x_hi+x_hi*x_hi*r_diff/2.0)*sin(2.*PI*y_lo))
				                                  -((inputs.r_in*AU*x_lo+x_lo*x_lo*r_diff/2.0)*sin(2.*PI*y_lo)))/DIM/a_dx;
		}
#endif

#if DIM == 3


		if (inputs.grid_type_global == 0) {
			if (a_s == 0 && a_d == 0) a_X_ave_f(0) = (a_pt[0] + 0.0)*a_dx/DIM;
			if (a_s == 1 && a_d == 0) a_X_ave_f(0) = (a_pt[1] + 0.5)*a_dy/DIM;
			if (a_s == 2 && a_d == 0) a_X_ave_f(0) = (a_pt[2] + 0.5)*a_dz/DIM;
			if (a_s == 0 && a_d == 1) a_X_ave_f(0) = (a_pt[0] + 0.5)*a_dx/DIM;
			if (a_s == 1 && a_d == 1) a_X_ave_f(0) = (a_pt[1] + 0.0)*a_dy/DIM;
			if (a_s == 2 && a_d == 1) a_X_ave_f(0) = (a_pt[2] + 0.5)*a_dz/DIM;
			if (a_s == 0 && a_d == 2) a_X_ave_f(0) = (a_pt[0] + 0.5)*a_dx/DIM;
			if (a_s == 1 && a_d == 2) a_X_ave_f(0) = (a_pt[1] + 0.5)*a_dy/DIM;
			if (a_s == 2 && a_d == 2) a_X_ave_f(0) = (a_pt[2] + 0.0)*a_dz/DIM;
		}

#endif

	}
	PROTO_KERNEL_END(X_ave_f_calcF, X_ave_f_calc)



	PROTO_KERNEL_START
	void N_ave_f_calcF( const Point& a_pt,
	                    Var<double,1>& a_N_ave_f,
	                    int a_s,
	                    int a_d,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz)
	{
#if DIM == 2
		int x_loc = a_pt[0];
		int y_loc = a_pt[1];
		double x_hi = (x_loc + 1.0)*a_dx;
		double x_lo = (x_loc - 0.0)*a_dx;
		double y_hi = (y_loc + 1.0)*a_dy;
		double y_lo = (y_loc - 0.0)*a_dy;

		if (inputs.grid_type_global == 0) {
			if (a_s == 0 && a_d == 0) a_N_ave_f(0) = 1.0;
			if (a_s == 1 && a_d == 0) a_N_ave_f(0) = 0.0;
			if (a_s == 0 && a_d == 1) a_N_ave_f(0) = 0.0;
			if (a_s == 1 && a_d == 1) a_N_ave_f(0) = 1.0;
		}

		if (inputs.grid_type_global == 1) {
			double C1 = inputs.C1_fix;
			double C2 = C1;
			if (a_s == 0 && a_d == 0) a_N_ave_f(0) = ((y_hi+C2*sin(2.*PI*x_lo)*sin(2.*PI*y_hi))
				                                  -(y_lo+C2*sin(2.*PI*x_lo)*sin(2.*PI*y_lo)))/a_dy;
			if (a_s == 1 && a_d == 0) a_N_ave_f(0) = -((x_lo+C1*sin(2.*PI*x_lo)*sin(2.*PI*y_hi))
				                                   -(x_lo+C1*sin(2.*PI*x_lo)*sin(2.*PI*y_lo)))/a_dy;
			if (a_s == 0 && a_d == 1) a_N_ave_f(0) = -((y_lo+C2*sin(2.*PI*x_hi)*sin(2.*PI*y_lo))
				                                   -(y_lo+C2*sin(2.*PI*x_lo)*sin(2.*PI*y_lo)))/a_dx;
			if (a_s == 1 && a_d == 1) a_N_ave_f(0) = ((x_hi+C1*sin(2.*PI*x_hi)*sin(2.*PI*y_lo))
				                                  -(x_lo+C1*sin(2.*PI*x_lo)*sin(2.*PI*y_lo)))/a_dx;
		}

		if (inputs.grid_type_global == 2) {
			double r_diff = inputs.r_out*AU-inputs.r_in*AU;
			if (a_s == 0 && a_d == 0) a_N_ave_f(0) = (((inputs.r_in*AU + x_lo*r_diff)*sin(2.*PI*y_hi))
				                                  -((inputs.r_in*AU + x_lo*r_diff)*sin(2.*PI*y_lo)))/a_dy;
			if (a_s == 1 && a_d == 0) a_N_ave_f(0) = -(((inputs.r_in*AU + x_lo*r_diff)*cos(2.*PI*y_hi))
				                                   -((inputs.r_in*AU + x_lo*r_diff)*cos(2.*PI*y_lo)))/a_dy;
			if (a_s == 0 && a_d == 1) a_N_ave_f(0) = -(((inputs.r_in*AU + x_hi*r_diff)*sin(2.*PI*y_lo))
				                                   -((inputs.r_in*AU + x_lo*r_diff)*sin(2.*PI*y_lo)))/a_dx;
			if (a_s == 1 && a_d == 1) a_N_ave_f(0) = (((inputs.r_in*AU + x_hi*r_diff)*cos(2.*PI*y_lo))
				                                  -((inputs.r_in*AU + x_lo*r_diff)*cos(2.*PI*y_lo)))/a_dx;
		}
#endif

#if DIM == 3
		if (inputs.grid_type_global == 0) {
			if (a_s == 0 && a_d == 0) a_N_ave_f(0) = 1.0;
			if (a_s == 1 && a_d == 0) a_N_ave_f(0) = 0.0;
			if (a_s == 2 && a_d == 0) a_N_ave_f(0) = 0.0;
			if (a_s == 0 && a_d == 1) a_N_ave_f(0) = 0.0;
			if (a_s == 1 && a_d == 1) a_N_ave_f(0) = 1.0;
			if (a_s == 2 && a_d == 1) a_N_ave_f(0) = 0.0;
			if (a_s == 0 && a_d == 2) a_N_ave_f(0) = 0.0;
			if (a_s == 1 && a_d == 2) a_N_ave_f(0) = 0.0;
			if (a_s == 2 && a_d == 2) a_N_ave_f(0) = 1.0;
		}
#endif
	}
	PROTO_KERNEL_END(N_ave_f_calcF, N_ave_f_calc)

	PROTO_KERNEL_START
	void N_ave_f_calc2F( const Point& a_pt,
	                     Var<double,DIM*DIM>& a_N_ave_f,
	                     const double a_dx,
	                     const double a_dy,
	                     const double a_dz)
	{
#if DIM == 2
		int x_loc = a_pt[0];
		int y_loc = a_pt[1];
		double x_hi = (x_loc + 1.0)*a_dx;
		double x_lo = (x_loc - 0.0)*a_dx;
		double y_hi = (y_loc + 1.0)*a_dy;
		double y_lo = (y_loc - 0.0)*a_dy;

		if (inputs.grid_type_global == 0) {
			a_N_ave_f(0) = 1.0;
			a_N_ave_f(1) = 0.0;
			a_N_ave_f(2) = 0.0;
			a_N_ave_f(3) = 1.0;
		}

		if (inputs.grid_type_global == 1) {
			double C1 = inputs.C1_fix;
			double C2 = C1;
			a_N_ave_f(0) = ((y_hi+C2*sin(2.*PI*x_lo)*sin(2.*PI*y_hi))
			                -(y_lo+C2*sin(2.*PI*x_lo)*sin(2.*PI*y_lo)))/a_dy;
			a_N_ave_f(1) = -((x_lo+C1*sin(2.*PI*x_lo)*sin(2.*PI*y_hi))
			                 -(x_lo+C1*sin(2.*PI*x_lo)*sin(2.*PI*y_lo)))/a_dy;
			a_N_ave_f(2) = -((y_lo+C2*sin(2.*PI*x_hi)*sin(2.*PI*y_lo))
			                 -(y_lo+C2*sin(2.*PI*x_lo)*sin(2.*PI*y_lo)))/a_dx;
			a_N_ave_f(3) = ((x_hi+C1*sin(2.*PI*x_hi)*sin(2.*PI*y_lo))
			                -(x_lo+C1*sin(2.*PI*x_lo)*sin(2.*PI*y_lo)))/a_dx;
		}

		if (inputs.grid_type_global == 2) {
			double r_diff = inputs.r_out*AU-inputs.r_in*AU;
			a_N_ave_f(0) = (((inputs.r_in*AU + x_lo*r_diff)*sin(2.*PI*y_hi))
			                -((inputs.r_in*AU + x_lo*r_diff)*sin(2.*PI*y_lo)))/a_dy;
			a_N_ave_f(1) = -(((inputs.r_in*AU + x_lo*r_diff)*cos(2.*PI*y_hi))
			                 -((inputs.r_in*AU + x_lo*r_diff)*cos(2.*PI*y_lo)))/a_dy;
			a_N_ave_f(2) = -(((inputs.r_in*AU + x_hi*r_diff)*sin(2.*PI*y_lo))
			                 -((inputs.r_in*AU + x_lo*r_diff)*sin(2.*PI*y_lo)))/a_dx;
			a_N_ave_f(3) = (((inputs.r_in*AU + x_hi*r_diff)*cos(2.*PI*y_lo))
			                -((inputs.r_in*AU + x_lo*r_diff)*cos(2.*PI*y_lo)))/a_dx;
		}
#endif

#if DIM == 3
		if (inputs.grid_type_global == 0) {
			a_N_ave_f(0) = 1.0;
			a_N_ave_f(1) = 0.0;
			a_N_ave_f(2) = 0.0;
			a_N_ave_f(3) = 0.0;
			a_N_ave_f(4) = 1.0;
			a_N_ave_f(5) = 0.0;
			a_N_ave_f(6) = 0.0;
			a_N_ave_f(7) = 0.0;
			a_N_ave_f(8) = 1.0;
		}
#endif
	}
	PROTO_KERNEL_END(N_ave_f_calc2F, N_ave_f_calc2)







	PROTO_KERNEL_START
	void d_U_calcF(  State& a_d_U,
	                 const State& a_UJ_ahead,
	                 const State& a_UJ_behind,
	                 const Var<double,1>& a_J_ahead,
	                 const Var<double,1>& a_J_behind)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_d_U(i) = ((a_UJ_ahead(i)/a_J_ahead(0))-(a_UJ_behind(i)/a_J_behind(0)))/(2.0);
		}
	}
	PROTO_KERNEL_END(d_U_calcF, d_U_calc)


	PROTO_KERNEL_START
	void a_U_demapped_calcF(  State& a_U_demapped,
	                          const Var<double,1>& a_Jacobian_ave,
	                          const State& a_a_U,
	                          const State& a_dot_pro_sum2)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_U_demapped(i) = (a_a_U(i)-(a_dot_pro_sum2(i)/12.0))/a_Jacobian_ave(0);
		}

	}
	PROTO_KERNEL_END(a_U_demapped_calcF, a_U_demapped_calc)


	PROTO_KERNEL_START
	void dot_pro_calcFF(State& a_dot_pro,
	                    const Var<double,1>& a_d_perp_N_s,
	                    const State& a_d_perp_F)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_dot_pro(i) = (a_d_perp_N_s(0)*a_d_perp_F(i));
		}
	}
	PROTO_KERNEL_END(dot_pro_calcFF, dot_pro_calcF)


	void N_ave_f_calc_func(BoxData<double,1>& a_N_ave_f,
	                       const int a_s,
	                       const int a_d,
	                       const double a_dx,
	                       const double a_dy,
	                       const double a_dz)
	{
		forallInPlace_p(N_ave_f_calc, a_N_ave_f, a_s, a_d, a_dx, a_dy, a_dz);
	}

	void N_ave_f_calc_func(BoxData<double,DIM*DIM>& a_N_ave_f,
	                       const double a_dx,
	                       const double a_dy,
	                       const double a_dz)
	{
		forallInPlace_p(N_ave_f_calc2, a_N_ave_f, a_dx, a_dy, a_dz);
	}


	void Jacobian_Ave_calc(BoxData<double,1>& a_Jacobian_ave,
	                       const double a_dx,
	                       const double a_dy,
	                       const double a_dz,
	                       const Box& a_dbx0)
	{


		static Stencil<double> m_divergence[DIM];
		static Stencil<double> m_derivative[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			for (int dir = 0; dir < DIM; dir++)
			{
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
				m_derivative[dir] = Stencil<double>::Derivative(1,dir,2);
			}
			initialized =  true;
		}


		Scalar N_s_d_ave_f(a_dbx0), X_ave_f(a_dbx0);
		double dxd[3] = {a_dx, a_dy, a_dz};
		a_Jacobian_ave.setVal(0.0);

		for (int d = 0; d < DIM; d++) {
			Scalar X_f_mapped(a_dbx0);
			X_f_mapped.setVal(0.0);
			for (int s = 0; s < DIM; s++) {
				forallInPlace_p(N_ave_f_calc, N_s_d_ave_f, s, d, a_dx, a_dy, a_dz);
				forallInPlace_p(X_ave_f_calc, X_ave_f, s, d, a_dx, a_dy, a_dz);

				// if (s==0 && d==0){
				// std::string filename="N_s_d_ave_f";
				// WriteBoxData(filename.c_str(),N_s_d_ave_f,a_dx);
				// filename="X_ave_f";
				// WriteBoxData(filename.c_str(),X_ave_f,a_dx);
				// }

				Scalar dot_pro_sum(a_dbx0);
				dot_pro_sum.setVal(0.0);
				for (int s_temp = 0; s_temp < DIM; s_temp++) {
					if (s_temp != d) {
						Scalar d_perp_N_s = m_derivative[s_temp](N_s_d_ave_f);
						Scalar d_perp_X = m_derivative[s_temp](X_ave_f);
						Scalar dot_pro = forall<double,1>(dot_pro_calcX,d_perp_N_s,d_perp_X);
						dot_pro_sum += dot_pro;
					}
				}
				Scalar X_f_mapped1D = forall<double,1>(X_f_mapped1D_calc,X_ave_f,N_s_d_ave_f,dot_pro_sum);
				X_f_mapped += X_f_mapped1D;
			}
			Scalar Rhs_d = m_divergence[d](X_f_mapped);
			Rhs_d *= 1./dxd[d];
			a_Jacobian_ave += Rhs_d;
		}
		// std::string filename="a_Jacobian_ave";
		// WriteBoxData(filename.c_str(),a_Jacobian_ave,a_dx);

	}



	void JU_to_U_calc(BoxData<double,NUMCOMPS>& a_U_demapped,
	                  const BoxData<double,NUMCOMPS>& a_a_U,
	                  BoxData<double,1>& a_Jacobian_ave,
	                  const Box& a_dbx0)
	{


		static Stencil<double> m_ahead_shift[DIM];
		static Stencil<double> m_behind_shift[DIM];
		static Stencil<double> m_derivative[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			for (int dir = 0; dir < DIM; dir++)
			{
				m_ahead_shift[dir] = 1.0*Shift(Point::Basis(dir)*(1));
				m_behind_shift[dir] = 1.0*Shift(Point::Basis(dir)*(-1));
				m_derivative[dir] = Stencil<double>::Derivative(1,dir,2);
			}
			initialized =  true;
		}



		Vector dot_pro_sum2(a_dbx0);
		dot_pro_sum2.setVal(0.0);
		for (int d = 0; d < DIM; d++) {

			Vector UJ_ahead = m_ahead_shift[d](a_a_U);
			Vector UJ_behind = m_behind_shift[d](a_a_U);
			Scalar J_ahead = alias(a_Jacobian_ave,Point::Basis(d)*(-1));
			Scalar J_behind = alias(a_Jacobian_ave,Point::Basis(d)*(1));

			Vector d_U = forall<double,NUMCOMPS>(d_U_calc,UJ_ahead,UJ_behind,J_ahead,J_behind);
			Scalar d_J = m_derivative[d](a_Jacobian_ave);
			Vector dot_pro = forall<double,NUMCOMPS>(dot_pro_calcF,d_J,d_U);
			dot_pro_sum2 += dot_pro;
		}

		a_U_demapped = forall<double,NUMCOMPS>(a_U_demapped_calc,a_Jacobian_ave,a_a_U,dot_pro_sum2);
	}


	PROTO_KERNEL_START
	void a_U_demapped_2nd_order_calcF(  State& a_U_demapped,
	                                    const Var<double,1>& a_Jacobian_ave,
	                                    const State& a_a_U)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_U_demapped(i) = a_a_U(i)/a_Jacobian_ave(0);
		}

	}
	PROTO_KERNEL_END(a_U_demapped_2nd_order_calcF, a_U_demapped_2nd_order_calc)


	void JU_to_U_2ndOrdercalc(BoxData<double,NUMCOMPS>& a_U_demapped,
	                          const BoxData<double,NUMCOMPS>& a_a_U,
	                          const BoxData<double,1>& a_Jacobian_ave,
	                          const Box& a_dbx0)
	{
		a_U_demapped = forall<double,NUMCOMPS>(a_U_demapped_2nd_order_calc,a_Jacobian_ave,a_a_U);
	}


	void JU_to_W_calc(BoxData<double,NUMCOMPS>& a_W,
	                  const BoxData<double,NUMCOMPS>& a_JU,
	                  const double a_dx,
	                  const double a_dy,
	                  const double a_dz,
	                  const double a_gamma)
	{
		Box dbx0 = a_JU.box();
		static Stencil<double> m_laplacian;
		static Stencil<double> m_deconvolve;
		static Stencil<double> m_copy;
		static bool initialized = false;
		if(!initialized)
		{
			m_laplacian = Stencil<double>::Laplacian();
			m_deconvolve = (-1.0/24.0)*m_laplacian + (1.0)*Shift(Point::Zeros());
			m_copy = 1.0*Shift(Point::Zeros());
			initialized =  true;
		}


		using namespace std;
		a_W.setVal(0.0);
		double gamma = a_gamma;

		Scalar Jacobian_ave(dbx0);
		MHD_Mapping::Jacobian_Ave_calc(Jacobian_ave,a_dx, a_dy, a_dz,dbx0);

		Vector a_U(dbx0);

		MHD_Mapping::JU_to_U_calc(a_U, a_JU, Jacobian_ave, dbx0);

		Vector W_bar(dbx0);
		MHDOp::consToPrimcalc(W_bar,a_U,gamma);
		Vector U = m_deconvolve(a_U);
		Vector W(dbx0);
		MHDOp::consToPrimcalc(W,U,gamma);
		a_W = m_laplacian(W_bar,1.0/24.0);
		a_W += W;

	}




	void JU_to_W_bar_calc(BoxData<double,NUMCOMPS>& a_W_bar,
	                      const BoxData<double,NUMCOMPS>& a_JU,
	                      BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                      BoxData<double,1>& a_r2rdot_avg,
	                      BoxData<double,1>& a_detA_avg,
	                      const double a_dx,
	                      const double a_dy,
	                      const double a_dz,
	                      const double a_gamma)
	{
		Box dbx1 = a_JU.box();
		Box dbx0 = dbx1.grow(NGHOST);
		a_W_bar.setVal(0.0);
		double gamma = a_gamma;
		Scalar Jacobian_ave(dbx0);
		if (inputs.grid_type_global == 2){
			MHD_Mapping::Jacobian_ave_sph_calc_func(Jacobian_ave,a_dx, a_dy, a_dz);
		} else {
			MHD_Mapping::Jacobian_Ave_calc(Jacobian_ave,a_dx, a_dy, a_dz,dbx0);
		}
		Vector a_U(dbx0);

		if (inputs.grid_type_global == 2){
			MHD_Mapping::JU_to_U_ave_calc_func(a_U, a_JU, a_r2rdot_avg, a_detA_avg);
		} else {
			MHD_Mapping::JU_to_U_calc(a_U, a_JU, Jacobian_ave, dbx0);
		}
		MHDOp::consToPrimcalc(a_W_bar,a_U,gamma);
	}



	PROTO_KERNEL_START
	void out_data_joinF(Var<double,NUMCOMPS+DIM>& a_out_data,
	                    const Var<double,DIM>& a_phys_coords,
	                    const Var<double,NUMCOMPS>& a_W)
	{
		for (int i=0; i< DIM; i++) {
			a_out_data(i) = a_phys_coords(i);///14957550000000; // AU
		}
		
		a_out_data(DIM+0) = a_W(0)/mp; // /cm^3
		a_out_data(DIM+1) = a_W(1)/1e5; // km/s
		a_out_data(DIM+2) = a_W(2)/1e5; // km/s
		a_out_data(DIM+3) = a_W(3)/1e5; // km/s
		a_out_data(DIM+4) = a_W(4); // dyne/cm*2
		a_out_data(DIM+5) = a_W(5); // Gauss
		a_out_data(DIM+6) = a_W(6); // Gauss
		a_out_data(DIM+7) = a_W(7); // Gauss

	}
	PROTO_KERNEL_END(out_data_joinF, out_data_join)


	void out_data_calc(BoxData<double,NUMCOMPS+DIM>& a_out_data,
	                   const BoxData<double,DIM>& a_phys_coords,
	                   const BoxData<double,NUMCOMPS>& a_W)
	{
		a_out_data = forall<double,NUMCOMPS+DIM>(out_data_join, a_phys_coords, a_W);
	}


	PROTO_KERNEL_START
	void out_data_join2F(Var<double,DIM*DIM+DIM>& a_out_data,
	                     const Var<double,DIM>& a_phys_coords,
	                     const Var<double,DIM*DIM>& a_W)
	{
		for (int i=0; i< DIM; i++) {
			a_out_data(i) = a_phys_coords(i);
		}
		for (int i=DIM; i< DIM*DIM+DIM; i++) {
			a_out_data(i) = a_W(i-DIM);
		}

	}
	PROTO_KERNEL_END(out_data_join2F, out_data_join2)


	void out_data2_calc(BoxData<double,DIM*DIM+DIM>& a_out_data,
	                    const BoxData<double,DIM>& a_phys_coords,
	                    const BoxData<double,DIM*DIM>& a_W)
	{
		a_out_data = forall<double,DIM*DIM+DIM>(out_data_join2, a_phys_coords, a_W);
	}


	PROTO_KERNEL_START
	void Spherical_map_calcF( Point& a_pt,
	                          Var<double,1>& a_Jacobian_ave,
	                          Var<double,DIM*DIM>& a_A_1_avg,
	                          Var<double,DIM*DIM>& a_A_2_avg,
	                          Var<double,DIM*DIM>& a_A_3_avg,
							  Var<double,DIM*DIM>& a_A_inv_1_avg,
	                          Var<double,DIM*DIM>& a_A_inv_2_avg,
	                          Var<double,DIM*DIM>& a_A_inv_3_avg,
	                          Var<double,DIM*DIM>& a_detAA_avg,
	                          Var<double,DIM*DIM>& a_detAA_inv_avg,
	                          Var<double,1>& a_r2rdot_avg,
	                          Var<double,1>& a_detA_avg,
							  Var<double,DIM>& a_A_row_mag_avg,
	                          Var<double,1>& a_r2detA_1_avg,
	                          Var<double,DIM*DIM>& a_r2detAA_1_avg,
	                          Var<double,DIM>& a_r2detAn_1_avg,
							  Var<double,DIM>& a_A_row_mag_1_avg,
	                          Var<double,1>& a_rrdotdetA_2_avg,
	                          Var<double,DIM*DIM>& a_rrdotdetAA_2_avg,
	                          Var<double,DIM>& a_rrdotd3ncn_2_avg,
							  Var<double,DIM>& a_A_row_mag_2_avg,
	                          Var<double,1>& a_rrdotdetA_3_avg,
	                          Var<double,DIM*DIM>& a_rrdotdetAA_3_avg,
	                          Var<double,DIM>& a_rrdotncd2n_3_avg,
							  Var<double,DIM>& a_A_row_mag_3_avg,
	                          const double a_dx,
	                          const double a_dy,
	                          const double a_dz,
							  bool a_exchanged_yet,
							  bool a_r_dir_turn)
	{

		double R0 = inputs.r_in*AU;
		double R1 = inputs.r_out*AU;
		double c = inputs.C_rad;
		double Rt = (R1 - R0)/(exp(c) - 1.0);

		double dE1 = a_dx;
		double dE2 = a_dy;
		double dE3 = a_dz;

		double E1, E2, E3;
/////  CELL AVERAGED VALUES

		//For the special mapping that preserves radial flow, Phil has suggested to use E2 as theta and E3 as Phi.
		
		E1 = (a_pt[0] + 0.5)*dE1;
		E2 = (a_pt[1] + 0.5)*dE2;
		E3 = (a_pt[2] + 0.5)*dE3;
		if (a_exchanged_yet){
			if (E2 > 0.0 && E2 < 1.0) return;
		}
		
		if (E2 < 0.0){
			E2 = -E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}

		if (E2 > 1.0){
			E2 = 2.0 - E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}
		
		if (!a_r_dir_turn){
			a_detAA_avg(0)	=	-0.25*((-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI)))/(dE2*dE3);
			a_detAA_avg(1)	=	(PI*sin(dE2*PI)*sin(2*E2*PI)*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI)))/(2.*dE2*dE3);
			a_detAA_avg(2)	=	(PI*sin(dE3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*sin(2*E3*PI))/(dE2*dE3);
			a_detAA_avg(3)	=	-0.5*(sin(dE3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*sin(2*E3*PI))/(dE2*dE3);
			a_detAA_avg(4)	=	(PI*sin(dE2*PI)*sin(dE3*PI)*sin(2*E2*PI)*sin(2*E3*PI))/(dE2*dE3);
			a_detAA_avg(5)	=	(PI*(4*dE2*PI*cos(2*E3*PI)*sin(dE3*PI) - (sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI))))/(2.*dE2*dE3);
			a_detAA_avg(6)	=	(PI*sin(dE2*PI)*sin(2*E2*PI))/dE2;
			a_detAA_avg(7)	=	(pow(PI,2)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI)))/(2.*dE2);
			a_detAA_avg(8)	=	0;

			double det_detAA_avg = a_detAA_avg(0)*(a_detAA_avg(4)*a_detAA_avg(8) - a_detAA_avg(7)*a_detAA_avg(5)) - a_detAA_avg(1)*(a_detAA_avg(3)*a_detAA_avg(8) - a_detAA_avg(5)*a_detAA_avg(6)) + a_detAA_avg(2)*(a_detAA_avg(3)*a_detAA_avg(7) - a_detAA_avg(4)*a_detAA_avg(6));

			a_detAA_inv_avg(0) = (a_detAA_avg(4)*a_detAA_avg(8) - a_detAA_avg(5)*a_detAA_avg(7))/det_detAA_avg;
			a_detAA_inv_avg(1) = (a_detAA_avg(2)*a_detAA_avg(7) - a_detAA_avg(1)*a_detAA_avg(8))/det_detAA_avg;
			a_detAA_inv_avg(2) = (a_detAA_avg(1)*a_detAA_avg(5) - a_detAA_avg(2)*a_detAA_avg(4))/det_detAA_avg;
			a_detAA_inv_avg(3) = (a_detAA_avg(5)*a_detAA_avg(6) - a_detAA_avg(3)*a_detAA_avg(8))/det_detAA_avg;
			a_detAA_inv_avg(4) = (a_detAA_avg(0)*a_detAA_avg(8) - a_detAA_avg(2)*a_detAA_avg(6))/det_detAA_avg;
			a_detAA_inv_avg(5) = (a_detAA_avg(2)*a_detAA_avg(3) - a_detAA_avg(0)*a_detAA_avg(5))/det_detAA_avg;
			a_detAA_inv_avg(6) = (a_detAA_avg(3)*a_detAA_avg(7) - a_detAA_avg(4)*a_detAA_avg(6))/det_detAA_avg;
			a_detAA_inv_avg(7) = (a_detAA_avg(1)*a_detAA_avg(6) - a_detAA_avg(0)*a_detAA_avg(7))/det_detAA_avg;
			a_detAA_inv_avg(8) = (a_detAA_avg(0)*a_detAA_avg(4) - a_detAA_avg(1)*a_detAA_avg(3))/det_detAA_avg;

			a_r2rdot_avg(0) = 1.0;
			a_detA_avg(0)	=	(4*PI*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;
			a_A_row_mag_avg(0) = 1.0;
			a_A_row_mag_avg(1) = PI;
			a_A_row_mag_avg(2) = (4*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;
		}
		if (a_r_dir_turn){
			a_r2rdot_avg(0)	*=	(exp(c*((-3*dE1)/2. + E1))*(-1 + exp(c*dE1))*Rt*(3*exp(c*dE1)*pow(R0 - Rt,2) + 3*exp(c*((3*dE1)/2. + E1))*(R0 - Rt)*Rt + 3*exp((c*dE1)/2. + c*E1)*(R0 - Rt)*Rt + exp(2*c*E1)*pow(Rt,2) + exp(2*c*(dE1 + E1))*pow(Rt,2) + exp(c*(dE1 + 2*E1))*pow(Rt,2)))/(3.*dE1);
		}
		

		a_Jacobian_ave(0) = a_r2rdot_avg(0)*a_detA_avg(0);

// FACE E=1 AVERAGED VALUES 

		E1 = (a_pt[0])*dE1;
		E2 = (a_pt[1] + 0.5)*dE2;
		E3 = (a_pt[2] + 0.5)*dE3;

		if (E2 < 0.0){
			E2 = -E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}

		if (E2 > 1.0){
			E2 = 2.0 - E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}

		if (!a_r_dir_turn){
			a_r2detA_1_avg(0)	=	(4*PI*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;
			a_r2detAA_1_avg(0)	=	-0.5*(cos(2*E3*PI)*sin(dE3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI)))/(dE2*dE3);
			a_r2detAA_1_avg(1)	=	(PI*cos(2*E3*PI)*sin(dE2*PI)*sin(dE3*PI)*sin(2*E2*PI))/(dE2*dE3);
			a_r2detAA_1_avg(2)	=	(PI*sin(dE3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*sin(2*E3*PI))/(dE2*dE3);
			a_r2detAA_1_avg(3)	=	-0.5*(sin(dE3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*sin(2*E3*PI))/(dE2*dE3);
			a_r2detAA_1_avg(4)	=	(PI*sin(dE2*PI)*sin(dE3*PI)*sin(2*E2*PI)*sin(2*E3*PI))/(dE2*dE3);
			a_r2detAA_1_avg(5)	=	-((PI*cos(2*E3*PI)*sin(dE3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI)))/(dE2*dE3));
			a_r2detAA_1_avg(6)	=	(PI*sin(dE2*PI)*sin(2*E2*PI))/dE2;
			a_r2detAA_1_avg(7)	=	(pow(PI,2)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI)))/(2.*dE2);
			a_r2detAA_1_avg(8)	=	0.0;
			a_r2detAn_1_avg(0)	=	-0.5*(cos(2*E3*PI)*sin(dE3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI)))/(dE2*dE3);
			a_r2detAn_1_avg(1)	=	-0.5*(sin(dE3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*sin(2*E3*PI))/(dE2*dE3);
			a_r2detAn_1_avg(2)	=	(PI*sin(dE2*PI)*sin(2*E2*PI))/dE2;
			a_A_row_mag_1_avg(0) = 1.0;
			a_A_row_mag_1_avg(1) = PI;
			a_A_row_mag_1_avg(2) = (4*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;

			a_A_1_avg(0) = (2*cos(2*E3*PI)*sin((dE2*PI)/2.)*sin(dE3*PI)*sin(E2*PI))/(dE2*dE3*pow(PI,2));
			a_A_1_avg(1) = (2*cos(E2*PI)*cos(2*E3*PI)*sin((dE2*PI)/2.)*sin(dE3*PI))/(dE2*dE3*PI);
			a_A_1_avg(2) = (-4*sin((dE2*PI)/2.)*sin(dE3*PI)*sin(E2*PI)*sin(2*E3*PI))/(dE2*dE3*PI);
			a_A_1_avg(3) = (2*sin((dE2*PI)/2.)*sin(dE3*PI)*sin(E2*PI)*sin(2*E3*PI))/(dE2*dE3*pow(PI,2));
			a_A_1_avg(4) = (2*cos(E2*PI)*sin((dE2*PI)/2.)*sin(dE3*PI)*sin(2*E3*PI))/(dE2*dE3*PI);
			a_A_1_avg(5) = (4*cos(2*E3*PI)*sin((dE2*PI)/2.)*sin(dE3*PI)*sin(E2*PI))/(dE2*dE3*PI);
			a_A_1_avg(6) = (2*cos(E2*PI)*sin((dE2*PI)/2.))/(dE2*PI);
			a_A_1_avg(7) = (-2*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;
			a_A_1_avg(8) = 0.0;


			double det_A_1_avg = a_A_1_avg(0)*(a_A_1_avg(4)*a_A_1_avg(8) - a_A_1_avg(7)*a_A_1_avg(5)) - a_A_1_avg(1)*(a_A_1_avg(3)*a_A_1_avg(8) - a_A_1_avg(5)*a_A_1_avg(6)) + a_A_1_avg(2)*(a_A_1_avg(3)*a_A_1_avg(7) - a_A_1_avg(4)*a_A_1_avg(6));

			a_A_inv_1_avg(0) = (a_A_1_avg(4)*a_A_1_avg(8) - a_A_1_avg(5)*a_A_1_avg(7))/det_A_1_avg;
			a_A_inv_1_avg(1) = (a_A_1_avg(2)*a_A_1_avg(7) - a_A_1_avg(1)*a_A_1_avg(8))/det_A_1_avg;
			a_A_inv_1_avg(2) = (a_A_1_avg(1)*a_A_1_avg(5) - a_A_1_avg(2)*a_A_1_avg(4))/det_A_1_avg;
			a_A_inv_1_avg(3) = (a_A_1_avg(5)*a_A_1_avg(6) - a_A_1_avg(3)*a_A_1_avg(8))/det_A_1_avg;
			a_A_inv_1_avg(4) = (a_A_1_avg(0)*a_A_1_avg(8) - a_A_1_avg(2)*a_A_1_avg(6))/det_A_1_avg;
			a_A_inv_1_avg(5) = (a_A_1_avg(2)*a_A_1_avg(3) - a_A_1_avg(0)*a_A_1_avg(5))/det_A_1_avg;
			a_A_inv_1_avg(6) = (a_A_1_avg(3)*a_A_1_avg(7) - a_A_1_avg(4)*a_A_1_avg(6))/det_A_1_avg;
			a_A_inv_1_avg(7) = (a_A_1_avg(1)*a_A_1_avg(6) - a_A_1_avg(0)*a_A_1_avg(7))/det_A_1_avg;
			a_A_inv_1_avg(8) = (a_A_1_avg(0)*a_A_1_avg(4) - a_A_1_avg(1)*a_A_1_avg(3))/det_A_1_avg;
		}
		if (a_r_dir_turn){
			a_r2detA_1_avg(0)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(0)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(1)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(2)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(3)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(4)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(5)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(6)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(7)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAA_1_avg(8)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAn_1_avg(0)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAn_1_avg(1)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
			a_r2detAn_1_avg(2)	*=	pow(R0 + (-1 + exp(c*E1))*Rt,2);
		}

		


// FACE E=2 AVERAGED VALUES
		E1 = (a_pt[0] + 0.5)*dE1;
		E2 = (a_pt[1])*dE2;
		E3 = (a_pt[2] + 0.5)*dE3;


		if (E2 < 0.0){
			E2 = -E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}

		if (E2 > 1.0){
			E2 = 2.0 - E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}
		

		if (!a_r_dir_turn){
			a_rrdotdetA_2_avg(0)	=	(4*pow(PI,2)*sin(E2*PI));
			a_rrdotdetAA_2_avg(0)	=	(2*PI*pow(sin(E2*PI),2)*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI)))/(dE3);
			a_rrdotdetAA_2_avg(1)	=	(pow(PI,2)*sin(2*E2*PI)*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI)))/(dE3);
			a_rrdotdetAA_2_avg(2)	=	(-8*pow(PI,2)*sin(dE3*PI)*pow(sin(E2*PI),2)*sin(2*E3*PI))/(dE3);
			a_rrdotdetAA_2_avg(3)	=	(4*PI*sin(dE3*PI)*pow(sin(E2*PI),2)*sin(2*E3*PI))/(dE3);
			a_rrdotdetAA_2_avg(4)	=	(2*pow(PI,2)*sin(dE3*PI)*sin(2*E2*PI)*sin(2*E3*PI))/(dE3);
			a_rrdotdetAA_2_avg(5)	=	(4*pow(PI,2)*pow(sin(E2*PI),2)*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI)))/(dE3);
			a_rrdotdetAA_2_avg(6)	=	(2*pow(PI,2)*sin(2*E2*PI));
			a_rrdotdetAA_2_avg(7)	=	(-4*pow(PI,3)*pow(sin(E2*PI),2));
			a_rrdotdetAA_2_avg(8)	=	0.0;
			a_rrdotd3ncn_2_avg(0)	=	(sin(2*E2*PI)*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI)))/(dE3);
			a_rrdotd3ncn_2_avg(1)	=	(2*sin(dE3*PI)*sin(2*E2*PI)*sin(2*E3*PI))/(dE3);
			a_rrdotd3ncn_2_avg(2)	=	(-4*PI*pow(sin(E2*PI),2));
			a_A_row_mag_2_avg(0) = 1.0;
			a_A_row_mag_2_avg(1) = PI;
			a_A_row_mag_2_avg(2) = 2*PI*sin(E2*PI);

			a_A_2_avg(0) = (cos(2*E3*PI)*sin(dE3*PI)*sin(E2*PI))/(dE3*PI);
			a_A_2_avg(1) = (cos(E2*PI)*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI)))/(2.*dE3);
			a_A_2_avg(2) = (-2*sin(dE3*PI)*sin(E2*PI)*sin(2*E3*PI))/dE3;
			a_A_2_avg(3) = (sin(dE3*PI)*sin(E2*PI)*sin(2*E3*PI))/(dE3*PI);
			a_A_2_avg(4) = (cos(E2*PI)*sin(dE3*PI)*sin(2*E3*PI))/dE3;
			a_A_2_avg(5) = (sin(E2*PI)*(sin((dE3 - 2*E3)*PI) + sin((dE3 + 2*E3)*PI)))/dE3;
			a_A_2_avg(6) = cos(E2*PI);
			a_A_2_avg(7) = -(PI*sin(E2*PI));
			a_A_2_avg(8) = 0.0;

			double det_A_2_avg = a_A_2_avg(0)*(a_A_2_avg(4)*a_A_2_avg(8) - a_A_2_avg(7)*a_A_2_avg(5)) - a_A_2_avg(1)*(a_A_2_avg(3)*a_A_2_avg(8) - a_A_2_avg(5)*a_A_2_avg(6)) + a_A_2_avg(2)*(a_A_2_avg(3)*a_A_2_avg(7) - a_A_2_avg(4)*a_A_2_avg(6));

			a_A_inv_2_avg(0) = (a_A_2_avg(4)*a_A_2_avg(8) - a_A_2_avg(5)*a_A_2_avg(7))/det_A_2_avg;
			a_A_inv_2_avg(1) = (a_A_2_avg(2)*a_A_2_avg(7) - a_A_2_avg(1)*a_A_2_avg(8))/det_A_2_avg;
			a_A_inv_2_avg(2) = (a_A_2_avg(1)*a_A_2_avg(5) - a_A_2_avg(2)*a_A_2_avg(4))/det_A_2_avg;
			a_A_inv_2_avg(3) = (a_A_2_avg(5)*a_A_2_avg(6) - a_A_2_avg(3)*a_A_2_avg(8))/det_A_2_avg;
			a_A_inv_2_avg(4) = (a_A_2_avg(0)*a_A_2_avg(8) - a_A_2_avg(2)*a_A_2_avg(6))/det_A_2_avg;
			a_A_inv_2_avg(5) = (a_A_2_avg(2)*a_A_2_avg(3) - a_A_2_avg(0)*a_A_2_avg(5))/det_A_2_avg;
			a_A_inv_2_avg(6) = (a_A_2_avg(3)*a_A_2_avg(7) - a_A_2_avg(4)*a_A_2_avg(6))/det_A_2_avg;
			a_A_inv_2_avg(7) = (a_A_2_avg(1)*a_A_2_avg(6) - a_A_2_avg(0)*a_A_2_avg(7))/det_A_2_avg;
			a_A_inv_2_avg(8) = (a_A_2_avg(0)*a_A_2_avg(4) - a_A_2_avg(1)*a_A_2_avg(3))/det_A_2_avg;
		}
		if (a_r_dir_turn){
			a_rrdotdetA_2_avg(0)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(0)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(1)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(2)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(3)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(4)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(5)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(6)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(7)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_2_avg(8)	*=	0.0;
			a_rrdotd3ncn_2_avg(0)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotd3ncn_2_avg(1)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotd3ncn_2_avg(2)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
		}

		


// FACE E=3 AVERAGED VALUES

		E1 = (a_pt[0] + 0.5)*dE1;
		E2 = (a_pt[1] + 0.5)*dE2;
		E3 = (a_pt[2])*dE3;

		
		if (E2 < 0.0){
			E2 = -E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}

		if (E2 > 1.0){
			E2 = 2.0 - E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}
		
		
		if (!a_r_dir_turn){
			a_rrdotdetA_3_avg(0)	=	(8*PI*sin((dE2*PI)/2.)*sin(E2*PI))/(dE2);
			a_rrdotdetAA_3_avg(0)	=	-((cos(2*E3*PI)*PI*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI)))/(dE2));
			a_rrdotdetAA_3_avg(1)	=	(2*pow(PI,2)*cos(2*E3*PI)*sin(dE2*PI)*sin(2*E2*PI))/(dE2);
			a_rrdotdetAA_3_avg(2)	=	(2*pow(PI,2)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*sin(2*E3*PI))/(dE2);
			a_rrdotdetAA_3_avg(3)	=	-((PI*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI))*sin(2*E3*PI))/(dE2));
			a_rrdotdetAA_3_avg(4)	=	(2*pow(PI,2)*sin(dE2*PI)*sin(2*E2*PI)*sin(2*E3*PI))/(dE2);
			a_rrdotdetAA_3_avg(5)	=	(-2*pow(PI,2)*cos(2*E3*PI)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI)))/(dE2);
			a_rrdotdetAA_3_avg(6)	=	(2*PI*sin(dE2*PI)*sin(2*E2*PI))/(dE2);
			a_rrdotdetAA_3_avg(7)	=	(pow(PI,2)*(-2*dE2*PI + sin((dE2 - 2*E2)*PI) + sin((dE2 + 2*E2)*PI)))/(dE2);
			a_rrdotdetAA_3_avg(8)	=	0.0;
			a_rrdotncd2n_3_avg(0)	=	(-2*PI*sin(2*E3*PI));
			a_rrdotncd2n_3_avg(1)	=	(2*PI*cos(2*E3*PI));
			a_rrdotncd2n_3_avg(2)	=	0.0;
			a_A_row_mag_3_avg(0) = 1.0;
			a_A_row_mag_3_avg(1) = PI;
			a_A_row_mag_3_avg(2) = (4*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;

			a_A_3_avg(0) = (2*cos(2*E3*PI)*sin((dE2*PI)/2.)*sin(E2*PI))/(dE2*PI);
			a_A_3_avg(1) = (2*cos(E2*PI)*cos(2*E3*PI)*sin((dE2*PI)/2.))/dE2;
			a_A_3_avg(2) = (-4*sin((dE2*PI)/2.)*sin(E2*PI)*sin(2*E3*PI))/dE2;
			a_A_3_avg(3) = (2*sin((dE2*PI)/2.)*sin(E2*PI)*sin(2*E3*PI))/(dE2*PI);
			a_A_3_avg(4) = (2*cos(E2*PI)*sin((dE2*PI)/2.)*sin(2*E3*PI))/dE2;
			a_A_3_avg(5) = (4*cos(2*E3*PI)*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;
			a_A_3_avg(6) = (2*cos(E2*PI)*sin((dE2*PI)/2.))/(dE2*PI);
			a_A_3_avg(7) = (-2*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;
			a_A_3_avg(8) = 0.0;

			double det_A_3_avg = a_A_3_avg(0)*(a_A_3_avg(4)*a_A_3_avg(8) - a_A_3_avg(7)*a_A_3_avg(5)) - a_A_3_avg(1)*(a_A_3_avg(3)*a_A_3_avg(8) - a_A_3_avg(5)*a_A_3_avg(6)) + a_A_3_avg(2)*(a_A_3_avg(3)*a_A_3_avg(7) - a_A_3_avg(4)*a_A_3_avg(6));

			a_A_inv_3_avg(0) = (a_A_3_avg(4)*a_A_3_avg(8) - a_A_3_avg(5)*a_A_3_avg(7))/det_A_3_avg;
			a_A_inv_3_avg(1) = (a_A_3_avg(2)*a_A_3_avg(7) - a_A_3_avg(1)*a_A_3_avg(8))/det_A_3_avg;
			a_A_inv_3_avg(2) = (a_A_3_avg(1)*a_A_3_avg(5) - a_A_3_avg(2)*a_A_3_avg(4))/det_A_3_avg;
			a_A_inv_3_avg(3) = (a_A_3_avg(5)*a_A_3_avg(6) - a_A_3_avg(3)*a_A_3_avg(8))/det_A_3_avg;
			a_A_inv_3_avg(4) = (a_A_3_avg(0)*a_A_3_avg(8) - a_A_3_avg(2)*a_A_3_avg(6))/det_A_3_avg;
			a_A_inv_3_avg(5) = (a_A_3_avg(2)*a_A_3_avg(3) - a_A_3_avg(0)*a_A_3_avg(5))/det_A_3_avg;
			a_A_inv_3_avg(6) = (a_A_3_avg(3)*a_A_3_avg(7) - a_A_3_avg(4)*a_A_3_avg(6))/det_A_3_avg;
			a_A_inv_3_avg(7) = (a_A_3_avg(1)*a_A_3_avg(6) - a_A_3_avg(0)*a_A_3_avg(7))/det_A_3_avg;
			a_A_inv_3_avg(8) = (a_A_3_avg(0)*a_A_3_avg(4) - a_A_3_avg(1)*a_A_3_avg(3))/det_A_3_avg;
		}
		if (a_r_dir_turn){
			a_rrdotdetA_3_avg(0)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(0)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(1)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(2)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(3)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(4)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(5)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(6)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(7)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotdetAA_3_avg(8)	*=	0.0;
			a_rrdotncd2n_3_avg(0)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotncd2n_3_avg(1)	*=	(exp(c*E1)*Rt*(R0 - Rt + exp(c*E1)*Rt*cosh((c*dE1)/2.))*sinh((c*dE1)/2.))/dE1;
			a_rrdotncd2n_3_avg(2)	*=	0.0;
		}

		

	}
	PROTO_KERNEL_END(Spherical_map_calcF, Spherical_map_calc)



	void Spherical_map_calc_func(BoxData<double,1>& a_Jacobian_ave,
	                             BoxData<double,DIM*DIM>& a_A_1_avg,
	                             BoxData<double,DIM*DIM>& a_A_2_avg,
	                             BoxData<double,DIM*DIM>& a_A_3_avg,
								 BoxData<double,DIM*DIM>& a_A_inv_1_avg,
	                             BoxData<double,DIM*DIM>& a_A_inv_2_avg,
	                             BoxData<double,DIM*DIM>& a_A_inv_3_avg,
	                             BoxData<double,DIM*DIM>& a_detAA_avg,
	                             BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                             BoxData<double,1>& a_r2rdot_avg,
	                             BoxData<double,1>& a_detA_avg,
	                             BoxData<double,DIM>& a_A_row_mag_avg,
	                             BoxData<double,1>& a_r2detA_1_avg,
	                             BoxData<double,DIM*DIM>& a_r2detAA_1_avg,
	                             BoxData<double,DIM>& a_r2detAn_1_avg,
								 BoxData<double,DIM>& a_A_row_mag_1_avg,
	                             BoxData<double,1>& a_rrdotdetA_2_avg,
	                             BoxData<double,DIM*DIM>& a_rrdotdetAA_2_avg,
	                             BoxData<double,DIM>& a_rrdotd3ncn_2_avg,
								 BoxData<double,DIM>& a_A_row_mag_2_avg,
	                             BoxData<double,1>& a_rrdotdetA_3_avg,
	                             BoxData<double,DIM*DIM>& a_rrdotdetAA_3_avg,
	                             BoxData<double,DIM>& a_rrdotncd2n_3_avg,
								 BoxData<double,DIM>& a_A_row_mag_3_avg,
	                             const double a_dx,
	                             const double a_dy,
	                             const double a_dz,
								 bool a_exchanged_yet,
								 bool a_r_dir_turn)
	{
		forallInPlace_p(Spherical_map_calc, a_Jacobian_ave, a_A_1_avg, a_A_2_avg, a_A_3_avg, a_A_inv_1_avg, a_A_inv_2_avg, a_A_inv_3_avg, a_detAA_avg, a_detAA_inv_avg, a_r2rdot_avg, a_detA_avg, a_A_row_mag_avg, a_r2detA_1_avg, a_r2detAA_1_avg, a_r2detAn_1_avg, a_A_row_mag_1_avg, a_rrdotdetA_2_avg, a_rrdotdetAA_2_avg, a_rrdotd3ncn_2_avg, a_A_row_mag_2_avg, a_rrdotdetA_3_avg, a_rrdotdetAA_3_avg, a_rrdotncd2n_3_avg, a_A_row_mag_3_avg, a_dx, a_dy, a_dz,a_exchanged_yet,a_r_dir_turn);
	}

	PROTO_KERNEL_START
	void Jacobian_ave_sph_calcF( const Point& a_pt,
	                          Var<double,1>& a_Jacobian_ave,
	                          const double a_dx,
	                          const double a_dy,
	                          const double a_dz)
	{

		double R0 = inputs.r_in*AU;
		double R1 = inputs.r_out*AU;
		double c = inputs.C_rad;
		double Rt = (R1 - R0)/(exp(c) - 1.0);

		double dE1 = a_dx;
		double dE2 = a_dy;
		double dE3 = a_dz;

		double E1, E2, E3;
/////  CELL AVERAGED VALUES

		//For the special mapping that preserves radial flow, Phil has suggested to use E2 as theta and E3 as Phi.

		E1 = (a_pt[0] + 0.5)*dE1;
		E2 = (a_pt[1] + 0.5)*dE2;
		E3 = (a_pt[2] + 0.5)*dE3;

		
		if (E2 < 0.0){
			E2 = -E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}

		if (E2 > 1.0){
			E2 = 2.0 - E2;
			if (E3 < 0.5) {
				E3 += 0.5;
			} else {
				E3 -= 0.5;
			}
		}
		
		double a_r2rdot_avg	=	(exp(c*((-3*dE1)/2. + E1))*(-1 + exp(c*dE1))*Rt*(3*exp(c*dE1)*pow(R0 - Rt,2) + 3*exp(c*((3*dE1)/2. + E1))*(R0 - Rt)*Rt + 3*exp((c*dE1)/2. + c*E1)*(R0 - Rt)*Rt + exp(2*c*E1)*pow(Rt,2) + exp(2*c*(dE1 + E1))*pow(Rt,2) + exp(c*(dE1 + 2*E1))*pow(Rt,2)))/(3.*dE1);
		double a_detA_avg	=	(4*PI*sin((dE2*PI)/2.)*sin(E2*PI))/dE2;

		a_Jacobian_ave(0) = a_r2rdot_avg*a_detA_avg;
	}
	PROTO_KERNEL_END(Jacobian_ave_sph_calcF, Jacobian_ave_sph_calc)


	void Jacobian_ave_sph_calc_func(BoxData<double,1>& a_Jacobian_ave,
	                             const double a_dx,
	                             const double a_dy,
	                             const double a_dz)
	{
		forallInPlace_p(Jacobian_ave_sph_calc, a_Jacobian_ave, a_dx, a_dy, a_dz);
	}


	PROTO_KERNEL_START
	void JU_to_U_Sph_ave_calcF(const Point& a_pt,
							   State& a_U_Sph_ave,
	                          const Var<double,NUMCOMPS>& a_JU_ave,
	                          const Var<double,DIM*DIM>& a_detAA_inv_avg,
	                          const Var<double,1>& a_r2rdot_avg,
	                          const Var<double,1>& a_detA_avg,
	                          const Var<double,DIM>& a_A_row_mag_avg,
							  bool a_normalized)
	{
		double r2rdotrho_ave = a_JU_ave(0)/a_detA_avg(0);
		a_U_Sph_ave(0) = r2rdotrho_ave/a_r2rdot_avg(0);

		double r2rdotE_ave = a_JU_ave(4)/a_detA_avg(0);
		a_U_Sph_ave(4) = r2rdotE_ave/a_r2rdot_avg(0);

		double r2rdotrhou_ave = a_detAA_inv_avg(0)*a_JU_ave(1) + a_detAA_inv_avg(1)*a_JU_ave(2) + a_detAA_inv_avg(2)*a_JU_ave(3);
		double r2rdotrhov_ave = a_detAA_inv_avg(3)*a_JU_ave(1) + a_detAA_inv_avg(4)*a_JU_ave(2) + a_detAA_inv_avg(5)*a_JU_ave(3);
		double r2rdotrhow_ave = a_detAA_inv_avg(6)*a_JU_ave(1) + a_detAA_inv_avg(7)*a_JU_ave(2) + a_detAA_inv_avg(8)*a_JU_ave(3);


		a_U_Sph_ave(1)  = r2rdotrhou_ave/a_r2rdot_avg(0);
		a_U_Sph_ave(2)  = r2rdotrhov_ave/a_r2rdot_avg(0);
		a_U_Sph_ave(3)  = r2rdotrhow_ave/a_r2rdot_avg(0);

		double r2rdotBx_ave = a_detAA_inv_avg(0)*a_JU_ave(5) + a_detAA_inv_avg(1)*a_JU_ave(6) + a_detAA_inv_avg(2)*a_JU_ave(7);
		double r2rdotBy_ave = a_detAA_inv_avg(3)*a_JU_ave(5) + a_detAA_inv_avg(4)*a_JU_ave(6) + a_detAA_inv_avg(5)*a_JU_ave(7);
		double r2rdotBz_ave = a_detAA_inv_avg(6)*a_JU_ave(5) + a_detAA_inv_avg(7)*a_JU_ave(6) + a_detAA_inv_avg(8)*a_JU_ave(7);
		a_U_Sph_ave(5)  = r2rdotBx_ave/a_r2rdot_avg(0);
		a_U_Sph_ave(6)  = r2rdotBy_ave/a_r2rdot_avg(0);
		a_U_Sph_ave(7)  = r2rdotBz_ave/a_r2rdot_avg(0);

		if (a_normalized){
			a_U_Sph_ave(1)*=a_A_row_mag_avg(0);
			a_U_Sph_ave(2)*=a_A_row_mag_avg(1);
			a_U_Sph_ave(3)*=a_A_row_mag_avg(2);
			a_U_Sph_ave(5)*=a_A_row_mag_avg(0);
			a_U_Sph_ave(6)*=a_A_row_mag_avg(1);
			a_U_Sph_ave(7)*=a_A_row_mag_avg(2);
		}
	}
	PROTO_KERNEL_END(JU_to_U_Sph_ave_calcF, JU_to_U_Sph_ave_calc)

	void JU_to_U_Sph_ave_calc_func(BoxData<double,NUMCOMPS>& a_U_Sph_ave,
	                  const BoxData<double,NUMCOMPS>& a_JU_ave,
	                  BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                  BoxData<double,1>& a_r2rdot_avg,
	                  BoxData<double,1>& a_detA_avg,
	                  BoxData<double,DIM>& a_A_row_mag_avg,
					  bool a_normalized)
	{

		forallInPlace_p(JU_to_U_Sph_ave_calc, a_U_Sph_ave, a_JU_ave,a_detAA_inv_avg,a_r2rdot_avg,a_detA_avg, a_A_row_mag_avg, a_normalized);
	}

	void JU_to_W_Sph_ave_calc_func(BoxData<double,NUMCOMPS>& a_W_Sph_ave,
	                  const BoxData<double,NUMCOMPS>& a_JU_ave,
	                  BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                  BoxData<double,1>& a_r2rdot_avg,
	                  BoxData<double,1>& a_detA_avg,
					  BoxData<double,DIM>& a_A_row_mag_avg,
	                  const double a_gamma,
					  bool a_normalized)
	{
		double gamma = a_gamma;
		Vector a_U_Sph_ave(a_W_Sph_ave.box());
		forallInPlace_p(JU_to_U_Sph_ave_calc, a_U_Sph_ave, a_JU_ave,a_detAA_inv_avg,a_r2rdot_avg,a_detA_avg, a_A_row_mag_avg, a_normalized);
		MHDOp::consToPrimcalc(a_W_Sph_ave,a_U_Sph_ave,gamma);
	}

	PROTO_KERNEL_START
	void U_Sph_ave_to_JU_calcF(State& a_JU_ave,
	                          const Var<double,NUMCOMPS>& a_U_Sph_ave,
	                          const Var<double,DIM*DIM>& a_detAA_avg,
	                          const Var<double,DIM*DIM>& a_detAA_inv_avg,
	                          const Var<double,1>& a_r2rdot_avg,
	                          const Var<double,1>& a_detA_avg,
							  const Var<double,DIM>& a_A_row_mag_avg,
					  		  bool a_normalized)
	{

		double r2rdotrho_ave = a_U_Sph_ave(0)*a_r2rdot_avg(0);
		a_JU_ave(0) = r2rdotrho_ave * a_detA_avg(0);

		double r2rdotE_ave = a_U_Sph_ave(4)*a_r2rdot_avg(0);
		a_JU_ave(4) = r2rdotE_ave * a_detA_avg(0);

		double r2rdotrhou_ave = a_U_Sph_ave(1)*a_r2rdot_avg(0);
		double r2rdotrhov_ave = a_U_Sph_ave(2)*a_r2rdot_avg(0);
		double r2rdotrhow_ave = a_U_Sph_ave(3)*a_r2rdot_avg(0);

		double r2rdotBx_ave = a_U_Sph_ave(5)*a_r2rdot_avg(0);
		double r2rdotBy_ave = a_U_Sph_ave(6)*a_r2rdot_avg(0);
		double r2rdotBz_ave = a_U_Sph_ave(7)*a_r2rdot_avg(0);

		if (a_normalized){
			r2rdotrhou_ave/=a_A_row_mag_avg(0);
			r2rdotrhov_ave/=a_A_row_mag_avg(1);
			r2rdotrhow_ave/=a_A_row_mag_avg(2);

			r2rdotBx_ave/=a_A_row_mag_avg(0);
			r2rdotBy_ave/=a_A_row_mag_avg(1);
			r2rdotBz_ave/=a_A_row_mag_avg(2);
		}

		a_JU_ave(1) = a_detAA_avg(0)*r2rdotrhou_ave + a_detAA_avg(1)*r2rdotrhov_ave + a_detAA_avg(2)*r2rdotrhow_ave;
		a_JU_ave(2) = a_detAA_avg(3)*r2rdotrhou_ave + a_detAA_avg(4)*r2rdotrhov_ave + a_detAA_avg(5)*r2rdotrhow_ave;
		a_JU_ave(3) = a_detAA_avg(6)*r2rdotrhou_ave + a_detAA_avg(7)*r2rdotrhov_ave + a_detAA_avg(8)*r2rdotrhow_ave;

		a_JU_ave(5) = a_detAA_avg(0)*r2rdotBx_ave + a_detAA_avg(1)*r2rdotBy_ave + a_detAA_avg(2)*r2rdotBz_ave;
		a_JU_ave(6) = a_detAA_avg(3)*r2rdotBx_ave + a_detAA_avg(4)*r2rdotBy_ave + a_detAA_avg(5)*r2rdotBz_ave;
		a_JU_ave(7) = a_detAA_avg(6)*r2rdotBx_ave + a_detAA_avg(7)*r2rdotBy_ave + a_detAA_avg(8)*r2rdotBz_ave;
	}
	PROTO_KERNEL_END(U_Sph_ave_to_JU_calcF, U_Sph_ave_to_JU_calc)

	void U_Sph_ave_to_JU_calc_func(BoxData<double,NUMCOMPS>& a_JU_ave,
	                  const BoxData<double,NUMCOMPS>& a_U_Sph_ave,
	                  BoxData<double,DIM*DIM>& a_detAA_avg,
	                  BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                  BoxData<double,1>& a_r2rdot_avg,
	                  BoxData<double,1>& a_detA_avg,
					  BoxData<double,DIM>& a_A_row_mag_avg,
					  bool a_normalized)
	{

		a_JU_ave = forall<double,NUMCOMPS>(U_Sph_ave_to_JU_calc,a_U_Sph_ave,a_detAA_avg, a_detAA_inv_avg,a_r2rdot_avg,a_detA_avg,a_A_row_mag_avg,a_normalized);
	}


	PROTO_KERNEL_START
	void JU_to_U_ave_calcF(State& a_U_Sph_ave,
	                          const Var<double,NUMCOMPS>& a_JU_ave,
	                          const Var<double,1>& a_r2rdot_avg,
	                          const Var<double,1>& a_detA_avg)
	{
		double r2rdotrho_ave = a_JU_ave(0)/a_detA_avg(0);
		a_U_Sph_ave(0) = r2rdotrho_ave/a_r2rdot_avg(0);

		double r2rdotE_ave = a_JU_ave(4)/a_detA_avg(0);
		a_U_Sph_ave(4) = r2rdotE_ave/a_r2rdot_avg(0);

		double r2rdotrhou_ave = a_JU_ave(1)/a_detA_avg(0);
		double r2rdotrhov_ave = a_JU_ave(2)/a_detA_avg(0);
		double r2rdotrhow_ave = a_JU_ave(3)/a_detA_avg(0);
		a_U_Sph_ave(1)  = r2rdotrhou_ave/a_r2rdot_avg(0);
		a_U_Sph_ave(2)  = r2rdotrhov_ave/a_r2rdot_avg(0);
		a_U_Sph_ave(3)  = r2rdotrhow_ave/a_r2rdot_avg(0);

		double r2rdotBx_ave = a_JU_ave(5)/a_detA_avg(0);
		double r2rdotBy_ave = a_JU_ave(6)/a_detA_avg(0);
		double r2rdotBz_ave = a_JU_ave(7)/a_detA_avg(0);
		a_U_Sph_ave(5)  = r2rdotBx_ave/a_r2rdot_avg(0);
		a_U_Sph_ave(6)  = r2rdotBy_ave/a_r2rdot_avg(0);
		a_U_Sph_ave(7)  = r2rdotBz_ave/a_r2rdot_avg(0);
	}
	PROTO_KERNEL_END(JU_to_U_ave_calcF, JU_to_U_ave_calc)

	void JU_to_U_ave_calc_func(BoxData<double,NUMCOMPS>& a_U_ave,
	                  const BoxData<double,NUMCOMPS>& a_JU_ave,
	                  BoxData<double,1>& a_r2rdot_avg,
	                  BoxData<double,1>& a_detA_avg)
	{

		a_U_ave = forall<double,NUMCOMPS>(JU_to_U_ave_calc,a_JU_ave,a_r2rdot_avg,a_detA_avg);
	}

	PROTO_KERNEL_START
	void W_Sph_to_W_Cart_calcF( const Point& a_pt,
	                          Var<double,NUMCOMPS>& W_cart,
	                          Var<double,NUMCOMPS>& W,
							  const Var<double,DIM*DIM>& a_A_1_avg,
							  const Var<double,DIM*DIM>& a_A_2_avg,
							  const Var<double,DIM*DIM>& a_A_3_avg,
							  int a_d)
	{

		if (a_d == 0)
		{
			W_cart(0) = W(0);
			W_cart(4) = W(4);
			W_cart(1) = a_A_1_avg(0)*W(1) + a_A_1_avg(1)*W(2) + a_A_1_avg(2)*W(3);
			W_cart(2) = a_A_1_avg(3)*W(1) + a_A_1_avg(4)*W(2) + a_A_1_avg(5)*W(3);
			W_cart(3) = a_A_1_avg(6)*W(1) + a_A_1_avg(7)*W(2) + a_A_1_avg(8)*W(3);
			W_cart(5) = a_A_1_avg(0)*W(5) + a_A_1_avg(1)*W(6) + a_A_1_avg(2)*W(7);
			W_cart(6) = a_A_1_avg(3)*W(5) + a_A_1_avg(4)*W(6) + a_A_1_avg(5)*W(7);
			W_cart(7) = a_A_1_avg(6)*W(5) + a_A_1_avg(7)*W(6) + a_A_1_avg(8)*W(7);
		}

		if (a_d == 1)
		{
			W_cart(0) = W(0);
			W_cart(4) = W(4);
			W_cart(1) = a_A_2_avg(0)*W(1) + a_A_2_avg(1)*W(2) + a_A_2_avg(2)*W(3);
			W_cart(2) = a_A_2_avg(3)*W(1) + a_A_2_avg(4)*W(2) + a_A_2_avg(5)*W(3);
			W_cart(3) = a_A_2_avg(6)*W(1) + a_A_2_avg(7)*W(2) + a_A_2_avg(8)*W(3);
			W_cart(5) = a_A_2_avg(0)*W(5) + a_A_2_avg(1)*W(6) + a_A_2_avg(2)*W(7);
			W_cart(6) = a_A_2_avg(3)*W(5) + a_A_2_avg(4)*W(6) + a_A_2_avg(5)*W(7);
			W_cart(7) = a_A_2_avg(6)*W(5) + a_A_2_avg(7)*W(6) + a_A_2_avg(8)*W(7);
		}

		if (a_d == 2)
		{
			W_cart(0) = W(0);
			W_cart(4) = W(4);
			W_cart(1) = a_A_3_avg(0)*W(1) + a_A_3_avg(1)*W(2) + a_A_3_avg(2)*W(3);
			W_cart(2) = a_A_3_avg(3)*W(1) + a_A_3_avg(4)*W(2) + a_A_3_avg(5)*W(3);
			W_cart(3) = a_A_3_avg(6)*W(1) + a_A_3_avg(7)*W(2) + a_A_3_avg(8)*W(3);
			W_cart(5) = a_A_3_avg(0)*W(5) + a_A_3_avg(1)*W(6) + a_A_3_avg(2)*W(7);
			W_cart(6) = a_A_3_avg(3)*W(5) + a_A_3_avg(4)*W(6) + a_A_3_avg(5)*W(7);
			W_cart(7) = a_A_3_avg(6)*W(5) + a_A_3_avg(7)*W(6) + a_A_3_avg(8)*W(7);
		}
		
	}
	PROTO_KERNEL_END(W_Sph_to_W_Cart_calcF, W_Sph_to_W_Cart_calc)

	void W_Sph_to_W_Cart(BoxData<double,NUMCOMPS>& W_cart,
	                    const BoxData<double,NUMCOMPS>& W,
						BoxData<double,DIM*DIM>& a_A_1_avg,
						BoxData<double,DIM*DIM>& a_A_2_avg,
						BoxData<double,DIM*DIM>& a_A_3_avg,
	                    int a_d)
	{
		forallInPlace_p(W_Sph_to_W_Cart_calc, W_cart, W, a_A_1_avg, a_A_2_avg, a_A_3_avg, a_d);
	}	


	PROTO_KERNEL_START
	void W_Sph_to_W_normalized_sph_calcF( const Point& a_pt,
										Var<double,NUMCOMPS>& W_normalized_sph,
										Var<double,NUMCOMPS>& W_Sph,
										const Var<double,DIM>& a_A_row_mag_1_avg,
										const Var<double,DIM>& a_A_row_mag_2_avg,
										const Var<double,DIM>& a_A_row_mag_3_avg,
										int a_d)
	{
		double a,b,c;

		if (a_d == 0)
		{
			a = a_A_row_mag_1_avg(0);
			b = a_A_row_mag_1_avg(1);
			c = a_A_row_mag_1_avg(2);
		}
		if (a_d == 1)
		{
			a = a_A_row_mag_2_avg(0);
			b = a_A_row_mag_2_avg(1);
			c = a_A_row_mag_2_avg(2);
		}
		if (a_d == 2)
		{
			a = a_A_row_mag_3_avg(0);
			b = a_A_row_mag_3_avg(1);
			c = a_A_row_mag_3_avg(2);
		}

		W_normalized_sph(0) = W_Sph(0);
		W_normalized_sph(1) = W_Sph(1)*a;
		W_normalized_sph(2) = W_Sph(2)*b;
		W_normalized_sph(3) = W_Sph(3)*c;
		W_normalized_sph(4) = W_Sph(4);
		W_normalized_sph(5) = W_Sph(5)*a;
		W_normalized_sph(6) = W_Sph(6)*b;
		W_normalized_sph(7) = W_Sph(7)*c;
		
	}
	PROTO_KERNEL_END(W_Sph_to_W_normalized_sph_calcF, W_Sph_to_W_normalized_sph_calc)


	void W_Sph_to_W_normalized_sph(BoxData<double,NUMCOMPS>& W_normalized_sph,
	                    const BoxData<double,NUMCOMPS>& W_Sph,
						BoxData<double,DIM>& a_A_row_mag_1_avg,
						BoxData<double,DIM>& a_A_row_mag_2_avg,
						BoxData<double,DIM>& a_A_row_mag_3_avg,
	                    int a_d)
	{
		forallInPlace_p(W_Sph_to_W_normalized_sph_calc, W_normalized_sph, W_Sph, a_A_row_mag_1_avg, a_A_row_mag_2_avg, a_A_row_mag_3_avg, a_d);
	}



	void Regular_map_filling_func(MHDLevelDataState& a_state){
		for (auto dit : a_state.m_Jacobian_ave){		
			MHD_Mapping::Jacobian_Ave_calc((a_state.m_Jacobian_ave)[dit],a_state.m_dx,a_state.m_dy,a_state.m_dz,a_state.m_U[dit].box().grow(1));
			MHD_Mapping::N_ave_f_calc_func((a_state.m_N_ave_f)[dit],a_state.m_dx,a_state.m_dy,a_state.m_dz);
		}
		(a_state.m_Jacobian_ave).exchange();
	}

	void Spherical_map_filling_func(MHDLevelDataState& a_state)
	{
		bool exchanged_yet = false;
		bool r_dir_turn = false;
		#if DIM == 3
		for (auto dit : a_state.m_detAA_avg){	
			MHD_Mapping::Spherical_map_calc_func((a_state.m_Jacobian_ave)[dit], (a_state.m_A_1_avg)[dit], (a_state.m_A_2_avg)[dit], (a_state.m_A_3_avg)[dit], (a_state.m_A_inv_1_avg)[dit], (a_state.m_A_inv_2_avg)[dit], (a_state.m_A_inv_3_avg)[dit], (a_state.m_detAA_avg)[dit], (a_state.m_detAA_inv_avg)[dit], (a_state.m_r2rdot_avg)[dit], (a_state.m_detA_avg)[dit], (a_state.m_A_row_mag_avg)[dit], (a_state.m_r2detA_1_avg)[dit], (a_state.m_r2detAA_1_avg)[dit], (a_state.m_r2detAn_1_avg)[dit], (a_state.m_A_row_mag_1_avg)[dit], (a_state.m_rrdotdetA_2_avg)[dit], (a_state.m_rrdotdetAA_2_avg)[dit], (a_state.m_rrdotd3ncn_2_avg)[dit], (a_state.m_A_row_mag_2_avg)[dit], (a_state.m_rrdotdetA_3_avg)[dit], (a_state.m_rrdotdetAA_3_avg)[dit], (a_state.m_rrdotncd2n_3_avg)[dit], (a_state.m_A_row_mag_3_avg)[dit],a_state.m_dx,a_state.m_dy,a_state.m_dz, exchanged_yet, r_dir_turn);
		}	

		if (inputs.grid_type_global == 2){
			(a_state.m_Jacobian_ave).exchange();
			(a_state.m_A_1_avg).exchange();
			(a_state.m_A_2_avg).exchange();
			(a_state.m_A_3_avg).exchange();
			(a_state.m_A_inv_1_avg).exchange();
			(a_state.m_A_inv_2_avg).exchange();
			(a_state.m_A_inv_3_avg).exchange();
			(a_state.m_detAA_avg).exchange();
			(a_state.m_detAA_inv_avg).exchange();
			(a_state.m_r2rdot_avg).exchange();
			(a_state.m_detA_avg).exchange();
			(a_state.m_r2detA_1_avg).exchange();
			(a_state.m_r2detAA_1_avg).exchange();
			(a_state.m_r2detAn_1_avg).exchange();
			(a_state.m_rrdotdetA_2_avg).exchange();
			(a_state.m_rrdotdetAA_2_avg).exchange();
			(a_state.m_rrdotd3ncn_2_avg).exchange();
			(a_state.m_rrdotdetA_3_avg).exchange();
			(a_state.m_rrdotdetAA_3_avg).exchange();
			(a_state.m_rrdotncd2n_3_avg).exchange();
			exchanged_yet = true;
		}
		for (auto dit : a_state.m_detAA_avg){		
			MHD_Mapping::Spherical_map_calc_func((a_state.m_Jacobian_ave)[dit], (a_state.m_A_1_avg)[dit], (a_state.m_A_2_avg)[dit], (a_state.m_A_3_avg)[dit], (a_state.m_A_inv_1_avg)[dit], (a_state.m_A_inv_2_avg)[dit], (a_state.m_A_inv_3_avg)[dit], (a_state.m_detAA_avg)[dit], (a_state.m_detAA_inv_avg)[dit], (a_state.m_r2rdot_avg)[dit], (a_state.m_detA_avg)[dit], (a_state.m_A_row_mag_avg)[dit], (a_state.m_r2detA_1_avg)[dit], (a_state.m_r2detAA_1_avg)[dit], (a_state.m_r2detAn_1_avg)[dit], (a_state.m_A_row_mag_1_avg)[dit], (a_state.m_rrdotdetA_2_avg)[dit], (a_state.m_rrdotdetAA_2_avg)[dit], (a_state.m_rrdotd3ncn_2_avg)[dit], (a_state.m_A_row_mag_2_avg)[dit], (a_state.m_rrdotdetA_3_avg)[dit], (a_state.m_rrdotdetAA_3_avg)[dit], (a_state.m_rrdotncd2n_3_avg)[dit], (a_state.m_A_row_mag_3_avg)[dit],a_state.m_dx,a_state.m_dy,a_state.m_dz, exchanged_yet, r_dir_turn);
		}
		exchanged_yet = false;
		r_dir_turn = true;

		for (auto dit : a_state.m_detAA_avg){		
			MHD_Mapping::Spherical_map_calc_func((a_state.m_Jacobian_ave)[dit], (a_state.m_A_1_avg)[dit], (a_state.m_A_2_avg)[dit], (a_state.m_A_3_avg)[dit], (a_state.m_A_inv_1_avg)[dit], (a_state.m_A_inv_2_avg)[dit], (a_state.m_A_inv_3_avg)[dit], (a_state.m_detAA_avg)[dit], (a_state.m_detAA_inv_avg)[dit], (a_state.m_r2rdot_avg)[dit], (a_state.m_detA_avg)[dit], (a_state.m_A_row_mag_avg)[dit], (a_state.m_r2detA_1_avg)[dit], (a_state.m_r2detAA_1_avg)[dit], (a_state.m_r2detAn_1_avg)[dit], (a_state.m_A_row_mag_1_avg)[dit], (a_state.m_rrdotdetA_2_avg)[dit], (a_state.m_rrdotdetAA_2_avg)[dit], (a_state.m_rrdotd3ncn_2_avg)[dit], (a_state.m_A_row_mag_2_avg)[dit], (a_state.m_rrdotdetA_3_avg)[dit], (a_state.m_rrdotdetAA_3_avg)[dit], (a_state.m_rrdotncd2n_3_avg)[dit], (a_state.m_A_row_mag_3_avg)[dit],a_state.m_dx,a_state.m_dy,a_state.m_dz, exchanged_yet, r_dir_turn);
		}
		#endif
	}	

	PROTO_KERNEL_START
	void Cartesian_to_Spherical_calcF( const Point& a_pt,
	                          Var<double,NUMCOMPS>& W_sph,
	                          Var<double,NUMCOMPS>& W_cart,
							  const Var<double,DIM>& a_x_sph)
	{
		
		double r = a_x_sph(0);
		double theta = a_x_sph(1);
		double phi = a_x_sph(2);

		double A0 = sin(theta)*cos(phi);
		double A1 = sin(theta)*sin(phi);
		double A2 = cos(theta);
		double A3 = cos(theta)*cos(phi);
		double A4 = cos(theta)*sin(phi);
		double A5 = -sin(theta);
		double A6 = -sin(phi);
		double A7 = cos(phi);
		double A8 = 0.0;

		
		W_sph(0) = W_cart(0);
		W_sph(4) = W_cart(4);
		W_sph(1) = A0*W_cart(1) + A1*W_cart(2) + A2*W_cart(3);
		W_sph(2) = A3*W_cart(1) + A4*W_cart(2) + A5*W_cart(3);
		W_sph(3) = A6*W_cart(1) + A7*W_cart(2) + A8*W_cart(3);
		W_sph(5) = A0*W_cart(5) + A1*W_cart(6) + A2*W_cart(7);
		W_sph(6) = A3*W_cart(5) + A4*W_cart(6) + A5*W_cart(7);
		W_sph(7) = A6*W_cart(5) + A7*W_cart(6) + A8*W_cart(7);
	}
	PROTO_KERNEL_END(Cartesian_to_Spherical_calcF, Cartesian_to_Spherical_calc)

	void Cartesian_to_Spherical(BoxData<double,NUMCOMPS>& a_W_Sph,
	                    		const BoxData<double,NUMCOMPS>& a_W_Cart,
	                    		const BoxData<double,DIM>& a_x_Sph)
	{
		forallInPlace_p(Cartesian_to_Spherical_calc, a_W_Sph, a_W_Cart, a_x_Sph);
	}	


	PROTO_KERNEL_START
	void Spherical_to_Cartesian_calcF( const Point& a_pt,
	                          Var<double,NUMCOMPS>& W_cart,
	                          Var<double,NUMCOMPS>& W_sph,
							  const Var<double,DIM>& a_x_sph)
	{
		
		double r = a_x_sph(0);
		double theta = a_x_sph(1);
		double phi = a_x_sph(2);

		double A0 = sin(theta)*cos(phi);
		double A1 = cos(theta)*cos(phi);
		double A2 = -sin(phi);
		double A3 = sin(theta)*sin(phi);
		double A4 = cos(theta)*sin(phi);
		double A5 = cos(phi);
		double A6 = cos(theta);
		double A7 = -sin(theta);
		double A8 = 0.0;

		
		W_cart(0) = W_sph(0);
		W_cart(4) = W_sph(4);
		W_cart(1) = A0*W_sph(1) + A1*W_sph(2) + A2*W_sph(3);
		W_cart(2) = A3*W_sph(1) + A4*W_sph(2) + A5*W_sph(3);
		W_cart(3) = A6*W_sph(1) + A7*W_sph(2) + A8*W_sph(3);
		W_cart(5) = A0*W_sph(5) + A1*W_sph(6) + A2*W_sph(7);
		W_cart(6) = A3*W_sph(5) + A4*W_sph(6) + A5*W_sph(7);
		W_cart(7) = A6*W_sph(5) + A7*W_sph(6) + A8*W_sph(7);
	}
	PROTO_KERNEL_END(Spherical_to_Cartesian_calcF, Spherical_to_Cartesian_calc)

	void Spherical_to_Cartesian(BoxData<double,NUMCOMPS>& a_W_Cart,
	                    		const BoxData<double,NUMCOMPS>& a_W_Sph,
	                    		const BoxData<double,DIM>& a_x_Sph)
	{
		forallInPlace_p(Spherical_to_Cartesian_calc, a_W_Cart, a_W_Sph, a_x_Sph);
	}



	PROTO_KERNEL_START
	void get_sph_coords_fc_calcF(const Point& a_pt,
							Var<double,DIM>& a_x_sph,
	                        const double a_dx,
	                   		const double a_dy,
	                   		const double a_dz,
	                   		int a_d)
	{
		double eta[3];
		for (int i = 0; i < DIM; i++)
		{
			double dxd;
			if (i == 0) dxd = a_dx;
			if (i == 1) dxd = a_dy;
			if (i == 2) dxd = a_dz;
			if (i == a_d) {
				eta[i] = a_pt[i]*dxd;
			} else {
				eta[i] = a_pt[i]*dxd + 0.5*dxd;
			}
		}

		double R_t = (inputs.r_out*AU - inputs.r_in*AU)/(exp(inputs.C_rad) - 1.0);

		double r = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta[0]) - 1.0);
		double theta = PI*eta[1];
		double phi = 2.0*PI*eta[2];

		if (theta < 0){
			theta = -theta;
			if (phi < PI){
				phi = phi + PI;
			} else {
				phi = phi - PI;
			}
		}
		if (theta > PI){
			theta = PI - (theta - PI);
			if (phi < PI){
				phi = phi + PI;
			} else {
				phi = phi - PI;
			}
		}
		a_x_sph(0) = r;  //r
		a_x_sph(1) = theta; //theta
		a_x_sph(2) = phi; //phi
	}
	PROTO_KERNEL_END(get_sph_coords_fc_calcF, get_sph_coords_fc_calc)


	void get_sph_coords_fc(BoxData<double,DIM>& a_x_sph,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz,
	                    int a_d)
	{
		forallInPlace_p(get_sph_coords_fc_calc, a_bx, a_x_sph, a_dx, a_dy, a_dz, a_d);
	}

	

	PROTO_KERNEL_START
	void get_sph_coords_cc_calcF(const Point& a_pt,
							Var<double,DIM>& a_x_sph,
	                        const double a_dx,
	                   		const double a_dy,
	                   		const double a_dz)
	{
		double R_t = (inputs.r_out*AU - inputs.r_in*AU)/(exp(inputs.C_rad) - 1.0);
		double eta_here[3];
		double eta_ahead[3];
		for (int i = 0; i < DIM; i++)
		{
			double dxd;
			if (i == 0) dxd = a_dx;
			if (i == 1) dxd = a_dy;
			if (i == 2) dxd = a_dz;
			eta_here[i] = a_pt[i]*dxd;
			eta_ahead[i] = a_pt[i]*dxd + dxd;
		}

		double r_here = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta_here[0]) - 1.0);
		double theta_here = PI*eta_here[1];
		double phi_here = 2.0*PI*eta_here[2];

		double r_ahead = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta_ahead[0]) - 1.0);
		double theta_ahead = PI*eta_ahead[1];
		double phi_ahead = 2.0*PI*eta_ahead[2];

		double r = 0.5*(r_here + r_ahead);
		double theta = 0.5*(theta_here + theta_ahead);
		double phi = 0.5*(phi_here + phi_ahead);

		if (theta < 0){
			theta = -theta;
			if (phi < PI){
				phi = phi + PI;
			} else {
				phi = phi - PI;
			}
		}
		if (theta > PI){
			theta = PI - (theta - PI);
			if (phi < PI){
				phi = phi + PI;
			} else {
				phi = phi - PI;
			}
		}
		a_x_sph(0) = r;  //r
		a_x_sph(1) = theta; //theta
		a_x_sph(2) = phi; //phi
	}
	PROTO_KERNEL_END(get_sph_coords_cc_calcF, get_sph_coords_cc_calc)


	void get_sph_coords_cc(BoxData<double,DIM>& a_x_sph,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz)
	{
		forallInPlace_p(get_sph_coords_cc_calc, a_bx, a_x_sph, a_dx, a_dy, a_dz);
	}
	


	PROTO_KERNEL_START
	void get_cell_volume_calcF(const Point& a_pt,
							Var<double,1>& a_V,
	                        const double a_dx,
	                   		const double a_dy,
	                   		const double a_dz)
	{
		double R_t = (inputs.r_out*AU - inputs.r_in*AU)/(exp(inputs.C_rad) - 1.0);
		double eta_here[3];
		double eta_ahead[3];
		for (int i = 0; i < DIM; i++)
		{
			double dxd;
			if (i == 0) dxd = a_dx;
			if (i == 1) dxd = a_dy;
			if (i == 2) dxd = a_dz;
			eta_here[i] = a_pt[i]*dxd;
			eta_ahead[i] = a_pt[i]*dxd + dxd;
		}

		double r_here = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta_here[0]) - 1.0);
		double theta_here = PI*eta_here[1];
		double phi_here = 2.0*PI*eta_here[2];

		double r_ahead = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta_ahead[0]) - 1.0);
		double theta_ahead = PI*eta_ahead[1];
		double phi_ahead = 2.0*PI*eta_ahead[2];

		double volume = (1.0/3.0)*(pow(r_ahead,3)-pow(r_here,3))*(cos(theta_here)-cos(theta_ahead))*(phi_ahead-phi_here);
		
		a_V(0) = abs(volume);  
		
	}
	PROTO_KERNEL_END(get_cell_volume_calcF, get_cell_volume_calc)


	void get_cell_volume(BoxData<double,1>& a_V,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz)
	{
		forallInPlace_p(get_cell_volume_calc, a_bx, a_V, a_dx, a_dy, a_dz);
	}


	PROTO_KERNEL_START
	void get_face_area_calcF(const Point& a_pt,
							Var<double,DIM>& a_A,
	                        const double a_dx,
	                   		const double a_dy,
	                   		const double a_dz)
	{
		double R_t = (inputs.r_out*AU - inputs.r_in*AU)/(exp(inputs.C_rad) - 1.0);
		double eta_here[3];
		double eta_ahead[3];
		for (int i = 0; i < DIM; i++)
		{
			double dxd;
			if (i == 0) dxd = a_dx;
			if (i == 1) dxd = a_dy;
			if (i == 2) dxd = a_dz;
			eta_here[i] = a_pt[i]*dxd;
			eta_ahead[i] = a_pt[i]*dxd + dxd;
		}

		double r_here = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta_here[0]) - 1.0);
		double theta_here = PI*eta_here[1];
		double phi_here = 2.0*PI*eta_here[2];

		double r_ahead = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta_ahead[0]) - 1.0);
		double theta_ahead = PI*eta_ahead[1];
		double phi_ahead = 2.0*PI*eta_ahead[2];

		double A_r = pow(r_here,2)*(cos(theta_here)-cos(theta_ahead))*(phi_ahead-phi_here);
		double A_theta = 0.5*sin(theta_here)*(pow(r_ahead,2) - pow(r_here,2))*(phi_ahead-phi_here);
		double A_phi = 0.5*(pow(r_ahead,2) - pow(r_here,2))*(theta_ahead-theta_here);
		
		a_A(0) = abs(A_r);  
		a_A(1) = abs(A_theta);  
		a_A(2) = abs(A_phi);  
		
	}
	PROTO_KERNEL_END(get_face_area_calcF, get_face_area_calc)


	void get_face_area(BoxData<double,DIM>& a_A,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz)
	{
		forallInPlace_p(get_face_area_calc, a_bx, a_A, a_dx, a_dy, a_dz);
	}

	PROTO_KERNEL_START
	void get_delta_sph_coords_calcF(const Point& a_pt,
							Var<double,DIM>& a_dx_sph,
	                        const double a_dx,
	                   		const double a_dy,
	                   		const double a_dz)
	{
		double R_t = (inputs.r_out*AU - inputs.r_in*AU)/(exp(inputs.C_rad) - 1.0);
		double eta_here[3];
		double eta_ahead[3];
		for (int i = 0; i < DIM; i++)
		{
			double dxd;
			if (i == 0) dxd = a_dx;
			if (i == 1) dxd = a_dy;
			if (i == 2) dxd = a_dz;
			eta_here[i] = a_pt[i]*dxd;
			eta_ahead[i] = a_pt[i]*dxd + dxd;
		}

		double r_here = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta_here[0]) - 1.0);
		double theta_here = PI*eta_here[1];
		double phi_here = 2.0*PI*eta_here[2];

		double r_ahead = inputs.r_in*AU + R_t*(exp(inputs.C_rad*eta_ahead[0]) - 1.0);
		double theta_ahead = PI*eta_ahead[1];
		double phi_ahead = 2.0*PI*eta_ahead[2];

		double dr = r_ahead - r_here;
		double dtheta = theta_ahead - theta_here;
		double dphi = phi_ahead - phi_here;
		
		a_dx_sph(0) = dr;  //dr
		a_dx_sph(1) = dtheta; //dtheta
		a_dx_sph(2) = dphi; //dphi
	}
	PROTO_KERNEL_END(get_delta_sph_coords_calcF, get_delta_sph_coords_calc)


	void get_delta_sph_coords(BoxData<double,DIM>& a_dx_sph,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz)
	{
		forallInPlace_p(get_delta_sph_coords_calc, a_bx, a_dx_sph, a_dx, a_dy, a_dz);
	}
	PROTO_KERNEL_START
	void Correct_V_theta_phi_at_poles_calcF( const Point& a_pt,
	                          Var<double,NUMCOMPS>& a_U_Sph_ave,
	                          const double a_dx,
	                          const double a_dy,
	                          const double a_dz)
	{

		double E1, E2, E3;
		E1 = (a_pt[0] + 0.5)*a_dx;
		E2 = (a_pt[1] + 0.5)*a_dy;
		E3 = (a_pt[2] + 0.5)*a_dz;

		if (E2 < 0.0 || E2 > 1.0){
			a_U_Sph_ave(2) *= -1.0;
			a_U_Sph_ave(3) *= -1.0;

			a_U_Sph_ave(6) *= -1.0;
			a_U_Sph_ave(7) *= -1.0;
		}

	}
	PROTO_KERNEL_END(Correct_V_theta_phi_at_poles_calcF, Correct_V_theta_phi_at_poles_calc)

	void Correct_V_theta_phi_at_poles(BoxData<double,NUMCOMPS>& a_U_Sph_ave,
	                             const double a_dx,
	                             const double a_dy,
	                             const double a_dz)
	{
		forallInPlace_p(Correct_V_theta_phi_at_poles_calc, a_U_Sph_ave, a_dx, a_dy, a_dz);
	}
}
