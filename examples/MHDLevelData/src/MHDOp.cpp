#include "Proto.H"
#include "MHDOp.H"
#include "CommonTemplates.H"
#include "Proto_Timer.H"
#include "Proto_WriteBoxData.H"
// For Chrono Timer (Talwinder)
#include <chrono>
#include <iostream>
#include <iomanip>

//////////////////////////////
#include "MHD_Limiters.H"
#include "MHD_Mapping.H"
#include "MHD_Riemann_Solvers.H"
#include "MHD_Output_Writer.H"
#include "MHD_Input_Parsing.H"
#include "MHD_Constants.H"
#include "MHD_CFL.H"
#include "MHDLevelDataRK4.H"

extern Parsefrominputs inputs;

typedef BoxData<double,1,HOST> Scalar;
typedef BoxData<double,NUMCOMPS,HOST> Vector;

namespace MHDOp {

	PROTO_KERNEL_START
	void
	consToPrimF(State&         a_W,
	            const State&   a_U,
	            double a_gamma)
	{
		double rho = a_U(0);
		double v2 = 0.0;
		double B2 = 0.0;
		double gamma = a_gamma;
		a_W(0) = rho;

		for (int i = 1; i <= DIM; i++)
		{
			double v, B;
			v = a_U(i) / rho;
			B = a_U(DIM+1+i);
			a_W(i) = v;
			a_W(DIM+1+i) = a_U(DIM+1+i);
			v2 += v*v;
			B2 += B*B;
		}

		a_W(NUMCOMPS-1-DIM) = (a_U(NUMCOMPS-1-DIM) - .5 * rho * v2  - B2/8.0/c_PI) * (gamma - 1.0);

	}
	PROTO_KERNEL_END(consToPrimF, consToPrim)

	void consToPrimcalc(BoxData<double,NUMCOMPS>& W_bar,
	                    const BoxData<double,NUMCOMPS>& a_U_demapped,
	                    const double gamma)
	{
		W_bar = forall<double,NUMCOMPS>(consToPrim,a_U_demapped, gamma);
	}

	PROTO_KERNEL_START
	void
	consToPrimSphF(State&         a_W_sph,
	            const State&   a_U_sph,
	            const State&   a_U_sph_actual,
	            double a_gamma)
	{
		double rho = a_U_sph(0);
		double v2 = 0.0;
		double B2 = 0.0;
		double gamma = a_gamma;
		a_W_sph(0) = rho;

		for (int i = 1; i <= DIM; i++)
		{
			double v, v_actual, B, B_actual;
			v = a_U_sph(i) / rho;
			v_actual = a_U_sph_actual(i) / rho;
			B = a_U_sph(DIM+1+i);
			B_actual = a_U_sph_actual(DIM+1+i);
			a_W_sph(i) = v;
			a_W_sph(DIM+1+i) = a_U_sph(DIM+1+i);
			v2 += v_actual*v_actual;
			B2 += B_actual*B_actual;
		}

		a_W_sph(NUMCOMPS-1-DIM) = (a_U_sph(NUMCOMPS-1-DIM) - .5 * rho * v2  - B2/8.0/c_PI) * (gamma - 1.0);
		// a_W_sph(NUMCOMPS-1-DIM) = a_U_sph(NUMCOMPS-1-DIM);

	}
	PROTO_KERNEL_END(consToPrimSphF, consToPrimSph)

	void consToPrimSphcalc(BoxData<double,NUMCOMPS>& W_bar,
	                    const BoxData<double,NUMCOMPS>& a_U_sph,
	                    const BoxData<double,NUMCOMPS>& a_U_sph_actual,
	                    const double gamma)
	{
		W_bar = forall<double,NUMCOMPS>(consToPrimSph,a_U_sph,a_U_sph_actual, gamma);
	}


	PROTO_KERNEL_START
	void waveSpeedBoundF(Var<double,1>& a_speed,
	                     const State& a_W,
	                     double a_gamma)
	{
		double gamma = a_gamma;
		double rho=0., u=0., v=0., w=0., p=0., Bx=0., By=0., Bz=0., ce, af, B_mag, Bdir, udir;

#if DIM == 2
		rho = a_W(0);
		u   = a_W(1);
		v   = a_W(2);
		p   = a_W(3);
		Bx  = a_W(4);
		By  = a_W(5);
#endif
#if DIM == 3
		rho = a_W(0);
		u   = a_W(1);
		v   = a_W(2);
		w   = a_W(3);
		p   = a_W(4);
		Bx  = a_W(5);
		By  = a_W(6);
		Bz  = a_W(7);
#endif
		a_speed(0) = 0.0;
		for (int dir = 0; dir< DIM; dir++) {
			if (dir == 0) {
				Bdir = Bx;
				udir = u;
			};
			if (dir == 1) {
				Bdir = By;
				udir = v;
			};
			if (dir == 2) {
				Bdir = Bz;
				udir = w;
			};

			ce = sqrt(gamma*p/rho);
			B_mag = sqrt(Bx*Bx+By*By+Bz*Bz);
			af = 0.5*(sqrt((ce*ce)+( B_mag*B_mag/(4.0*c_PI*rho) )+( abs(Bdir)*ce/sqrt(c_PI*rho) ))+
			          sqrt((ce*ce)+( B_mag*B_mag/(4.0*c_PI*rho) )-( abs(Bdir)*ce/sqrt(c_PI*rho) ))) + abs(udir);
			if (af > a_speed(0)) {a_speed(0) = af;}
		}
	}
	PROTO_KERNEL_END(waveSpeedBoundF, waveSpeedBound)



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



	PROTO_KERNEL_START
	void F_f_mapped1D_calcF(State& a_F_f_mapped1D,
	                        const State& a_F_ave_f,
	                        const Var<double,1>& a_N_s_d_ave_f,
	                        const State& a_dot_pro_sum,
	                        const double a_dx_d)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			// a_F_f_mapped1D(i) = -(a_N_s_d_ave_f(0)*a_F_ave_f(i) + a_dot_pro_sum(i)/12.0)/a_dx_d;
			a_F_f_mapped1D(i) = (a_N_s_d_ave_f(0)*a_F_ave_f(i) + a_dot_pro_sum(i)/12.0);
		}
	}
	PROTO_KERNEL_END(F_f_mapped1D_calcF, F_f_mapped1D_calc)

	PROTO_KERNEL_START
	void viscosity1_calcF(Var<double,1>& a_viscosity,
	                      const Var<double,1>& a_v,
	                      const Var<double,1>& a_v_behind)
	{
		a_viscosity(0) = a_v(0)-a_v_behind(0);
	}
	PROTO_KERNEL_END(viscosity1_calcF, viscosity1_calc)

	PROTO_KERNEL_START
	void v_d2_div_calcF(Var<double,1>& a_v_d2_div,
	                    const Var<double,1>& v_d2_ahead,
	                    const Var<double,1>& v_d2_behind,
	                    const Var<double,1>& v_d2_behind_dp,
	                    const Var<double,1>& v_d2_behind_dm)
	{
		a_v_d2_div(0) = (v_d2_ahead(0)-v_d2_behind(0)+v_d2_behind_dp(0)-v_d2_behind_dm(0))/4.0;
	}
	PROTO_KERNEL_END(v_d2_div_calcF, v_d2_div_calc)


	PROTO_KERNEL_START
	void Fast_MS_speed_calcF(Var<double,1>& a_Fast_MS_speed,
	                         const State& a_W_bar,
	                         int a_d,
	                         double a_gamma)
	{
		double gamma = a_gamma;
		double rho=0., p=0., Bx=0., By=0., Bz=0., ce, B_mag, Bdir;
#if DIM == 2
		rho = a_W_bar(0);
		p   = a_W_bar(3);
		Bx  = a_W_bar(4);
		By  = a_W_bar(5);
#endif
#if DIM == 3
		rho = a_W_bar(0);
		p   = a_W_bar(4);
		Bx  = a_W_bar(5);
		By  = a_W_bar(6);
		Bz  = a_W_bar(7);
#endif
		if (a_d == 0) {
			Bdir = Bx;
		};
		if (a_d == 1) {
			Bdir = By;
		};
		if (a_d == 2) {
			Bdir = Bz;
		};
		if (p < 0.0) p = 0.0;
		ce = sqrt(gamma*p/rho);
		B_mag = sqrt(Bx*Bx+By*By+Bz*Bz);
		//a_Fast_MS_speed(0) = 0.5*(sqrt((ce*ce)+( B_mag*B_mag/(4.0*c_PI*rho) )+( abs(Bdir)*ce/sqrt(c_PI*rho) ))+
		//	  sqrt((ce*ce)+( B_mag*B_mag/(4.0*c_PI*rho) )-( abs(Bdir)*ce/sqrt(c_PI*rho) )));
		a_Fast_MS_speed(0) = sqrt(ce*ce + B_mag*B_mag/4.0/c_PI/rho);
	}
	PROTO_KERNEL_END(Fast_MS_speed_calcF, Fast_MS_speed_calc)

	PROTO_KERNEL_START
	void Fast_MS_speed_min_calcF(Var<double,1>& a_Fast_MS_speed_min,
	                             const Var<double,1>& a_Fast_MS_speed,
	                             const Var<double,1>& a_Fast_MS_speed_behind)
	{
		a_Fast_MS_speed_min(0) = std::min({a_Fast_MS_speed(0),a_Fast_MS_speed_behind(0)});
	}
	PROTO_KERNEL_END(Fast_MS_speed_min_calcF, Fast_MS_speed_min_calc)


	PROTO_KERNEL_START
	void Visc_coef_calcF(Var<double,1>& a_Visc_coef,
	                     const Var<double,1>& a_h_lambda,
	                     const Var<double,1>& a_Fast_MS_speed_min)
	{
		if (a_h_lambda(0) < 0) {
			double temp = a_h_lambda(0)*a_h_lambda(0)/0.3/a_Fast_MS_speed_min(0)/a_Fast_MS_speed_min(0);
			a_Visc_coef(0) = a_h_lambda(0)*std::min({temp,1.0});
		} else {
			a_Visc_coef(0) = 0.0;
		}
	}
	PROTO_KERNEL_END(Visc_coef_calcF, Visc_coef_calc)

	PROTO_KERNEL_START
	void mu_f_calcF(State& a_mu_f,
	                const Var<double,1>& a_Visc_coef,
	                const State& a_U,
	                const State& a_U_behind)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_mu_f(i) = 0.3*a_Visc_coef(0)*(a_U(i)-a_U_behind(i));
		}
	}
	PROTO_KERNEL_END(mu_f_calcF, mu_f_calc)

	PROTO_KERNEL_START
	void lambdacalcF(State& a_lambda,
	                 const State& a_W_edge,
	                 int a_d,
	                 double a_gamma)
	{
		double gamma = a_gamma;
		double rho=0., u=0., v=0., w=0., p=0., Bx=0., By=0., Bz=0., ce, af, B_mag, u_mag, Bdir, udir;
#if DIM == 2
		rho = a_W_edge(0);
		u   = a_W_edge(1);
		v   = a_W_edge(2);
		p   = a_W_edge(3);
		Bx  = a_W_edge(4);
		By  = a_W_edge(5);
#endif
#if DIM == 3
		rho = a_W_edge(0);
		u   = a_W_edge(1);
		v   = a_W_edge(2);
		w   = a_W_edge(3);
		p   = a_W_edge(4);
		Bx  = a_W_edge(5);
		By  = a_W_edge(6);
		Bz  = a_W_edge(7);
#endif
		if (p < 0.0) p = 0.0;
		ce = sqrt(gamma*p/rho);
		B_mag = sqrt(Bx*Bx+By*By+Bz*Bz);
		u_mag = sqrt(u*u+v*v+w*w);
		//af = 0.5*(sqrt((ce*ce)+( B_mag*B_mag/(4.0*c_PI*rho) )+( B_mag*ce/sqrt(c_PI*rho) ))+
		//	  sqrt((ce*ce)+( B_mag*B_mag/(4.0*c_PI*rho) )-( B_mag*ce/sqrt(c_PI*rho) )));
		af = sqrt(ce*ce + B_mag*B_mag/4.0/c_PI/rho);
		double lambda = af + u_mag;
		//lambda = af + abs(a_W_edge(1+a_d));
		for (int i=0; i< NUMCOMPS; i++) {
			a_lambda(i) = lambda;
		}
	}
	PROTO_KERNEL_END(lambdacalcF, lambdacalc)

	PROTO_KERNEL_START
	void N_d_sqcalcF(Var<double,1>& a_N_d_sq,
	                 const Var<double,1>& a_N_s_d_ave_f)
	{
		a_N_d_sq(0) += a_N_s_d_ave_f(0)*a_N_s_d_ave_f(0);
	}
	PROTO_KERNEL_END(N_d_sqcalcF, N_d_sqcalc)

	PROTO_KERNEL_START
	void sqrtCalcF(Var<double,NUMCOMPS>& a_N_d,
	               const Var<double,1>& a_N_d_sq,
				   const double a_dx_d)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			// a_N_d(i) = sqrt(a_N_d_sq(0))/(-a_dx_d);
			a_N_d(i) = sqrt(a_N_d_sq(0));
		}
	}
	PROTO_KERNEL_END(sqrtCalcF, sqrtCalc)



	void step(LevelBoxData<double,NUMCOMPS>& a_Rhs,
			  LevelBoxData<double,NUMCOMPS>& a_JU_ave,
			  MHDLevelDataState& a_State,
			  double& a_min_dt)
	{

		
		static Stencil<double> m_laplacian;
		static Stencil<double> m_deconvolve;
		static Stencil<double> m_copy;
		static Stencil<double> m_laplacian_f[DIM];
		static Stencil<double> m_deconvolve_f[DIM];
		static Stencil<double> m_convolve_f[DIM];
		static Stencil<double> m_interp_H[DIM];
		static Stencil<double> m_interp_L[DIM];
		static Stencil<double> m_interp_edge[DIM];
		static Stencil<double> m_divergence[DIM];
		static Stencil<double> m_derivative[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			m_laplacian = Stencil<double>::Laplacian();
			m_deconvolve = (-1.0/24.0)*m_laplacian + (1.0)*Shift(Point::Zeros());
			m_copy = 1.0*Shift(Point::Zeros());
			for (int dir = 0; dir < DIM; dir++)
			{
				m_laplacian_f[dir] = Stencil<double>::LaplacianFace(dir);
				m_deconvolve_f[dir] = (-1.0/24.0)*m_laplacian_f[dir] + 1.0*Shift(Point::Zeros());
				m_convolve_f[dir] = (1.0/24.0)*m_laplacian_f[dir] + 1.0*Shift(Point::Zeros());
				m_interp_H[dir] = Stencil<double>::CellToFaceH(dir);
				m_interp_L[dir] = Stencil<double>::CellToFaceL(dir);
				m_interp_edge[dir] = Stencil<double>::CellToFace(dir);
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
				m_derivative[dir] = Stencil<double>::Derivative(1,dir,2);
			}
			initialized =  true;
		}



		using namespace std;
		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		double gamma = a_State.m_gamma;
		double dxd[3] = {a_dx, a_dy, a_dz};
		double dt_new;
		for (auto dit : a_State.m_U){
			Box dbx0 = a_JU_ave[dit].box();
			//Box dbx1 = dbx0.grow(1-NGHOST);
			Box dbx1 = dbx0;
			Box dbx2 = dbx0.grow(0-NGHOST);

			a_Rhs[dit].setVal(0.0);
			// if (!a_State.m_Viscosity_calculated) a_State.m_Viscosity[dit].setVal(0.0);
			
			Vector a_U_ave(dbx0);
			MHD_Mapping::JU_to_U_calc(a_U_ave, a_JU_ave[dit], a_State.m_Jacobian_ave[dit], dbx0);
			Vector W_bar = forall<double,NUMCOMPS>(consToPrim,a_U_ave, gamma);
			Vector U = m_deconvolve(a_U_ave);
			Vector W  = forall<double,NUMCOMPS>(consToPrim,U, gamma);
			Vector W_ave = m_laplacian(W_bar,1.0/24.0);
			W_ave += W;
			if (!a_State.m_min_dt_calculated){ 
				MHD_CFL::Min_dt_calc_func(dt_new, W_ave, a_dx, a_dy, a_dz, gamma);	
				if (dt_new < a_min_dt) a_min_dt = dt_new;
			}

			for (int d = 0; d < DIM; d++)
			{
				Vector W_ave_low_temp(dbx0), W_ave_high_temp(dbx0), Lambda_f(dbx0);
				Vector W_ave_low(dbx0), W_ave_high(dbx0);
				Scalar N_d_sq(dbx1);
				// W_ave_low_temp = m_interp_edge[d](W_ave);
				// W_ave_high_temp = m_copy(W_ave_low_temp);
				W_ave_low_temp = m_interp_L[d](W_ave);
				W_ave_high_temp = m_interp_H[d](W_ave);

				// if (inputs.linear_visc_apply == 1 && !a_State.m_Viscosity_calculated){ 
				// 	Lambda_f = forall<double,NUMCOMPS>(lambdacalc, W_ave_low_temp, d, gamma);
				// 	N_d_sq.setVal(0.0);
				// }


				MHD_Limiters::MHD_Limiters_4O(W_ave_low,W_ave_high,W_ave_low_temp,W_ave_high_temp,W_ave,W_bar,d,a_dx, a_dy, a_dz);

				Vector W_low = m_deconvolve_f[d](W_ave_low);
				Vector W_high = m_deconvolve_f[d](W_ave_high);


				Vector F_f(dbx1), F_ave_f(dbx1);
				Vector F_f_mapped(dbx1);
				// Vector F_f_mapped_noghost(dbx2);
				F_f_mapped.setVal(0.0);
				// F_f_mapped_noghost.setVal(0.0);
				double dx_d = dxd[d];


				for (int s = 0; s < DIM; s++) {
					if (inputs.Riemann_solver_type == 1) {
						MHD_Riemann_Solvers::Rusanov_Solver(F_f,W_low,W_high,s,gamma);
					}
					if (inputs.Riemann_solver_type == 2) {
						MHD_Riemann_Solvers::Roe8Wave_Solver(F_f,W_low,W_high,s,gamma);
					}
					Scalar N_s_d_ave_f = slice(a_State.m_N_ave_f[dit],d*DIM+s);

					F_ave_f = m_convolve_f[d](F_f);

					Vector dot_pro_sum(dbx1);
					dot_pro_sum.setVal(0.0);
					for (int s_temp = 0; s_temp < DIM; s_temp++) {
						if (s_temp != d) {
							Scalar d_perp_N_s = m_derivative[s_temp](N_s_d_ave_f);
							Vector d_perp_F = m_derivative[s_temp](F_ave_f);
							Vector dot_pro = forall<double,NUMCOMPS>(dot_pro_calcF,d_perp_N_s,d_perp_F);
							dot_pro_sum += dot_pro;
						}
					}

					Vector F_f_mapped1D = forall<double,NUMCOMPS>(F_f_mapped1D_calc,F_ave_f,N_s_d_ave_f,dot_pro_sum,dx_d);


					F_f_mapped += F_f_mapped1D;

					// if (inputs.linear_visc_apply == 1 && !a_State.m_Viscosity_calculated){ 
					// 	forallInPlace(N_d_sqcalc,dbx1,N_d_sq,N_s_d_ave_f);
					// }

				}
				Vector Rhs_d = m_divergence[d](F_f_mapped);
				Rhs_d *= -1./dx_d;
				a_Rhs[dit] += Rhs_d;

				// if (inputs.linear_visc_apply == 1 && !a_State.m_Viscosity_calculated) {
				// 	Vector N_d = forall<double,NUMCOMPS>(sqrtCalc, N_d_sq, dx_d);
				// 	Stencil<double> SPlus = 1.0*Shift(Point::Basis(d)) ;
				// 	Stencil<double> SMinus = 1.0*Shift(-Point::Basis(d));
				// 	Stencil<double> IdOp = 1.0*Shift(Point::Zeros());
				// 	Stencil<double> D1 = SPlus - IdOp;
				// 	Stencil<double> D2 = SPlus - 2.0*IdOp + SMinus;
				// 	Stencil<double> D5 = (-1.0) * D1 * D2  * D2;
				// 	Vector F_f = D5(a_U_ave, 1.0/64);
				// 	Vector F_f_behind = SMinus(F_f);
				// 	F_f_behind *= Lambda_f;
				// 	F_f_behind *= N_d;

				// 	Vector Rhs_d = m_divergence[d](F_f_behind);
				// 	Rhs_d *= -1./dxd[d];
				// 	a_State.m_Viscosity[dit] += Rhs_d;
				// }



				// if (inputs.non_linear_visc_apply == 1 && !a_State.m_Viscosity_calculated) {
				// 	Vector F_f(dbx1), F_ave_f(dbx1);
				// 	Scalar Lambda_f(dbx1);
				// 	Vector F_f_mapped(dbx1);
				// 	F_f_mapped.setVal(0.0);
				// 	for (int s = 0; s < DIM; s++) {
				// 		Scalar v_s =  slice(W_bar,1+s);
				// 		Scalar v_s_behind = alias(v_s,Point::Basis(d)*(1));
				// 		Scalar h_lambda = forall<double>(viscosity1_calc,v_s,v_s_behind);
				// 		for (int s2 = 0; s2 < DIM; s2++) {
				// 			if (s2!=s) {
				// 				for (int d2 = 0; d2 < DIM; d2++) {
				// 					if (d2!=d) {
				// 						Scalar v_s2 = slice(W_bar,1+s2);
				// 						Scalar v_s2_ahead = alias(v_s2,Point::Basis(d2)*(-1));
				// 						Scalar v_s2_behind = alias(v_s2,Point::Basis(d2)*(1));
				// 						Scalar v_s2_behind_dp = alias(v_s2_ahead,Point::Basis(d)*(1));
				// 						Scalar v_s2_behind_dm = alias(v_s2_behind,Point::Basis(d)*(1));
				// 						Scalar v_s2_div = forall<double>(v_d2_div_calc,v_s2_ahead,v_s2_behind,v_s2_behind_dp,v_s2_behind_dm);
				// 						h_lambda += v_s2_div;
				// 					}
				// 				}
				// 			}
				// 		}
				// 		Scalar Fast_MS_speed = forall<double>(Fast_MS_speed_calc, W_bar, s, gamma);
				// 		Scalar Fast_MS_speed_behind = alias(Fast_MS_speed,Point::Basis(d)*(1));
				// 		Scalar Fast_MS_speed_min = forall<double>(Fast_MS_speed_min_calc,Fast_MS_speed,Fast_MS_speed_behind);
				// 		Scalar Visc_coef = forall<double>(Visc_coef_calc,h_lambda,Fast_MS_speed_min);
				// 		Vector a_U_behind = alias(a_U_ave,Point::Basis(d)*(1));
				// 		Vector mu_f = forall<double,NUMCOMPS>(mu_f_calc, Visc_coef, a_U_ave, a_U_behind);
				// 		Scalar N_s_d_ave_f = slice(a_State.m_N_ave_f[dit],d*DIM+s);

				// 		F_ave_f = m_convolve_f[d](mu_f);

				// 		Vector dot_pro_sum(dbx1);
				// 		dot_pro_sum.setVal(0.0);
				// 		for (int s_temp = 0; s_temp < DIM; s_temp++) {
				// 			if (s_temp != d) {
				// 				Scalar d_perp_N_s = m_derivative[s_temp](N_s_d_ave_f);
				// 				Vector d_perp_F = m_derivative[s_temp](F_ave_f);
				// 				Vector dot_pro = forall<double,NUMCOMPS>(dot_pro_calcF,d_perp_N_s,d_perp_F);
				// 				dot_pro_sum += dot_pro;
				// 			}
				// 		}
				// 		double dx_d = dxd[d];
				// 		Vector F_f_mapped1D = forall<double,NUMCOMPS>(F_f_mapped1D_calc,F_ave_f,N_s_d_ave_f,dot_pro_sum,dx_d);

				// 		F_f_mapped += F_f_mapped1D;
				// 	}
				// 	Vector Rhs_d = m_divergence[d](F_f_mapped);
				// 	Rhs_d *= -1./dxd[d];
				// 	a_State.m_Viscosity[dit] += Rhs_d;

				// }


			}
		}
	}

	void step_spherical(LevelBoxData<double,NUMCOMPS>& a_Rhs,
			  LevelBoxData<double,NUMCOMPS>& a_JU_ave,
			  MHDLevelDataState& a_State,
			  double& a_min_dt)
	{	
		static Stencil<double> m_laplacian;
		static Stencil<double> m_deconvolve;
		static Stencil<double> m_copy;
		static Stencil<double> m_laplacian_f[DIM];
		static Stencil<double> m_deconvolve_f[DIM];
		static Stencil<double> m_convolve_f[DIM];
		static Stencil<double> m_interp_H[DIM];
		static Stencil<double> m_interp_L[DIM];
		static Stencil<double> m_interp_edge[DIM];
		static Stencil<double> m_interp_f_2nd[DIM];
		static Stencil<double> m_divergence[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			m_laplacian = Stencil<double>::Laplacian();
			m_deconvolve = (-1.0/24.0)*m_laplacian + (1.0)*Shift(Point::Zeros());
			m_copy = 1.0*Shift(Point::Zeros());
			for (int dir = 0; dir < DIM; dir++)
			{
				m_laplacian_f[dir] = Stencil<double>::LaplacianFace(dir);
				m_deconvolve_f[dir] = (-1.0/24.0)*m_laplacian_f[dir] + 1.0*Shift(Point::Zeros());
				m_convolve_f[dir] = (1.0/24.0)*m_laplacian_f[dir] + 1.0*Shift(Point::Zeros());
				m_interp_f_2nd[dir] = 0.5*Shift(Point::Zeros()) + 0.5*Shift(-Point::Basis(dir)); 
				m_interp_H[dir] = Stencil<double>::CellToFaceH(dir);
				m_interp_L[dir] = Stencil<double>::CellToFaceL(dir);
				m_interp_edge[dir] = Stencil<double>::CellToFace(dir);
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
			}
			initialized =  true;
		}

		using namespace std;
		
		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		double gamma = a_State.m_gamma;
		double dxd[3] = {a_dx, a_dy, a_dz}; // Because now its r, theta, phi

		double dt_new;
		for (auto dit : a_State.m_U){

			Box dbx0 = a_JU_ave[dit].box();
			Box dbx1 = dbx0.grow(NGHOST-NGHOST);
			//Box dbx1 = dbx0;
			Box dbx2 = dbx0.grow(1-NGHOST);
			a_Rhs[dit].setVal(0.0);
			Vector a_U_Sph_ave(dbx0), a_U_Sph_actual_ave(dbx0);
			MHD_Mapping::JU_to_U_Sph_ave_calc_func(a_U_Sph_ave, a_JU_ave[dit], a_State.m_detAA_inv_avg[dit], a_State.m_r2rdot_avg[dit], a_State.m_detA_avg[dit], a_State.m_A_row_mag_avg[dit], false);
			MHD_Mapping::JU_to_U_Sph_ave_calc_func(a_U_Sph_actual_ave, a_JU_ave[dit], a_State.m_detAA_inv_avg[dit], a_State.m_r2rdot_avg[dit], a_State.m_detA_avg[dit], a_State.m_A_row_mag_avg[dit], true);
			// MHD_Mapping::JU_to_U_ave_calc_func(a_U_ave, a_JU_ave[dit], a_State.m_r2rdot_avg[dit], a_State.m_detA_avg[dit]);
			MHD_Mapping::Correct_V_theta_phi_at_poles(a_U_Sph_ave, a_dx, a_dy, a_dz);

			// Vector W_bar = forall<double,NUMCOMPS>(consToPrim, a_U_Sph_ave, gamma);
			Vector W_bar = forall<double,NUMCOMPS>(consToPrimSph, a_U_Sph_ave, a_U_Sph_actual_ave, gamma);

			// MHD_Output_Writer::WriteBoxData_array_nocoord(W_bar, a_dx, a_dy, a_dz, "W_bar");
			// Vector U = m_deconvolve(a_U_Sph_ave);
			// Vector U_Sph_actual = m_deconvolve(a_U_Sph_actual_ave);
			// // Vector W  = forall<double,NUMCOMPS>(consToPrimSph,U, gamma);
			// Vector W  = forall<double,NUMCOMPS>(consToPrimSph,U, U_Sph_actual, gamma);
			// Vector W_ave = m_laplacian(W_bar,1.0/24.0);
			// W_ave += W;

			Vector W_ave = m_copy(W_bar);

			if (!a_State.m_min_dt_calculated){ 
				MHD_CFL::Min_dt_calc_func(dt_new, W_ave, a_dx, a_dy, a_dz, gamma);	
				if (dt_new < a_min_dt) a_min_dt = dt_new;
			}
			
			for (int d = 0; d < DIM; d++)
			{
				Vector W_ave_low_temp(dbx0), W_ave_high_temp(dbx0);
				Vector W_ave_low(dbx0), W_ave_high(dbx0);
				Vector W_ave_low_actual(dbx0), W_ave_high_actual(dbx0);

				W_ave_low_temp = m_interp_L[d](W_ave);
				W_ave_high_temp = m_interp_H[d](W_ave);
				// W_ave_low_temp = m_interp_edge[d](W_ave);
				// W_ave_high_temp = m_copy(W_ave_low_temp);

				// W_ave_low_temp = m_interp_f_2nd[d](W_ave);
				// W_ave_high_temp = m_interp_f_2nd[d](W_ave);

				MHD_Limiters::MHD_Limiters_4O(W_ave_low,W_ave_high,W_ave_low_temp,W_ave_high_temp,W_ave,W_bar,d,a_dx, a_dy, a_dz);			
				// // Vector W_low = m_deconvolve_f[d](W_ave_low);
				// Vector W_low = m_copy(W_ave_low);
				// // Vector W_high = m_deconvolve_f[d](W_ave_high);
				// Vector W_high = m_copy(W_ave_high);

				MHD_Mapping::W_Sph_to_W_normalized_sph(W_ave_low_actual, W_ave_low, a_State.m_A_row_mag_1_avg[dit], a_State.m_A_row_mag_2_avg[dit], a_State.m_A_row_mag_3_avg[dit], d);
				MHD_Mapping::W_Sph_to_W_normalized_sph(W_ave_high_actual, W_ave_low, a_State.m_A_row_mag_1_avg[dit], a_State.m_A_row_mag_2_avg[dit], a_State.m_A_row_mag_3_avg[dit], d);


				Vector F_ave_f(dbx0), F_f(dbx0);
				F_ave_f.setVal(0.0);
				F_f.setVal(0.0);
				// Vector F_f_mapped_noghost(dbx2);
				// F_f_mapped_noghost.setVal(0.0);
				double dx_d = dxd[d];
				MHD_Riemann_Solvers::Spherical_Riemann_Solver(F_ave_f, W_ave_low, W_ave_high, W_ave_low_actual, W_ave_high_actual, a_State.m_r2detA_1_avg[dit], a_State.m_r2detAA_1_avg[dit], a_State.m_r2detAn_1_avg[dit], a_State.m_rrdotdetA_2_avg[dit], a_State.m_rrdotdetAA_2_avg[dit], a_State.m_rrdotd3ncn_2_avg[dit], a_State.m_rrdotdetA_3_avg[dit], a_State.m_rrdotdetAA_3_avg[dit], a_State.m_rrdotncd2n_3_avg[dit], d, gamma, a_dx, a_dy, a_dz);	
				Vector Rhs_d = m_divergence[d](F_ave_f);
				Rhs_d *= -1./dx_d;
				a_Rhs[dit] += Rhs_d;
				// F_f_mapped_noghost += F_ave_f;
				// if (d==0) MHD_Output_Writer::WriteBoxData_array_nocoord(F_f_mapped_noghost, a_dx, a_dy, a_dz, "F0_m1");
				// if (d==1) MHD_Output_Writer::WriteBoxData_array_nocoord(F_f_mapped_noghost, a_dx, a_dy, a_dz, "F1_m1");
				// if (d==2) MHD_Output_Writer::WriteBoxData_array_nocoord(F_f_mapped_noghost, a_dx, a_dy, a_dz, "F2_m1");

			}
		}
	}


	PROTO_KERNEL_START
	void
	Fix_negative_P_calcF(const Point& a_pt,
						State&         a_U,
	            		double a_gamma)
	{
		double rho = a_U(0);
		double v2 = 0.0;
		double B2 = 0.0;
		double gamma = a_gamma;

		for (int i = 1; i <= DIM; i++)
		{
			double v, B;
			v = a_U(i) / rho;
			B = a_U(DIM+1+i);
			v2 += v*v;
			B2 += B*B;
		}

		double p = std::max((a_U(NUMCOMPS-1-DIM) - .5 * rho * v2  - B2/8.0/c_PI) * (gamma - 1.0),1.0e-14);
		a_U(NUMCOMPS-1-DIM) = p/(gamma-1.0) + .5 * rho * v2  + B2/8.0/c_PI;


	}
	PROTO_KERNEL_END(Fix_negative_P_calcF, Fix_negative_P_calc)

	void Fix_negative_P(BoxData<double,NUMCOMPS>& a_U,
	                    const double gamma)
	{
		forallInPlace_p(Fix_negative_P_calc, a_U, gamma);
	}


	PROTO_KERNEL_START
	void Scale_with_A_Ff_calcF(const Point& a_pt,
						Var<double,NUMCOMPS>& a_F_scaled,
						Var<double,NUMCOMPS>& a_F,
	                   	Var<double,DIM>& a_face_area,
	                   	int a_d)
	{
		double area = a_face_area(a_d);

		for (int i = 0; i < NUMCOMPS; i++){
			a_F_scaled(i) = -a_F(i)*area;
		}
	}
	PROTO_KERNEL_END(Scale_with_A_Ff_calcF, Scale_with_A_Ff_calc)

	PROTO_KERNEL_START
	void Scale_with_A_Bf_calcF(const Point& a_pt,
						Var<double,1>& a_B_scaled,
						const State& a_W_sph1,
	               		const State& a_W_sph2,
	                   	Var<double,DIM>& a_face_area,
	                   	int a_d)
	{
		double area = a_face_area(a_d);
		double B_face = 0.5*(a_W_sph1(2+DIM+a_d)+a_W_sph2(2+DIM+a_d));
		a_B_scaled(0) = -B_face*area;
		
	}
	PROTO_KERNEL_END(Scale_with_A_Bf_calcF, Scale_with_A_Bf_calc)

	PROTO_KERNEL_START
	void Scale_with_V_calcF(const Point& a_pt,
						Var<double,NUMCOMPS>& a_F_scaled,
						Var<double,NUMCOMPS>& a_F,
	                   	Var<double,1>& a_cell_volume)
	{
		double volume = a_cell_volume(0);

		for (int i = 0; i < NUMCOMPS; i++){
			a_F_scaled(i) = a_F(i)/volume;
		}
	}
	PROTO_KERNEL_END(Scale_with_V_calcF, Scale_with_V_calc)


	PROTO_KERNEL_START
	void Scale_with_V2_calcF(const Point& a_pt,
						Var<double,NUMCOMPS>& a_F_scaled,
						Var<double,1>& a_F,
	                   	Var<double,1>& a_cell_volume)
	{
		double volume = a_cell_volume(0);

		for (int i = 0; i < NUMCOMPS; i++){
			a_F_scaled(i) = a_F(0)/volume;
		}
	}
	PROTO_KERNEL_END(Scale_with_V2_calcF, Scale_with_V2_calc)


	PROTO_KERNEL_START
	void Scale_Ff_calc2F(const Point& a_pt,
						Var<double,NUMCOMPS>& a_F_scaled,
						Var<double,NUMCOMPS>& a_F,
	                    Var<double,DIM>& a_x_sph_cc,
	                   	Var<double,DIM>& a_x_sph_fc,
	                   	Var<double,DIM>& a_dx_sp,
	                   	int a_d)
	{
		double r_cc = a_x_sph_cc(0);
		double theta_cc = a_x_sph_cc(1);

		double r_fc = a_x_sph_fc(0);
		double theta_fc = a_x_sph_fc(1);

		double dr = a_dx_sp(0);
		double dtheta = a_dx_sp(1);
		double dphi = a_dx_sp(2);

		if (a_d == 0){
			for (int i = 0; i < NUMCOMPS; i++){
				// a_F_scaled(i) = -a_F(i)*(r_fc*r_fc)/(dr*r_cc*r_cc);
				a_F_scaled(i) = (r_fc*r_fc)/(dr*r_cc*r_cc);
			}
		}

		if (a_d == 1){
			for (int i = 0; i < NUMCOMPS; i++){
				// a_F_scaled(i) = -a_F(i)*sin(theta_fc)/(r_cc*sin(theta_cc)*sin(dtheta));
				a_F_scaled(i) = sin(theta_fc)/(r_cc*sin(theta_cc)*sin(dtheta));
			}
		}

		if (a_d == 2){
			for (int i = 0; i < NUMCOMPS; i++){
				// a_F_scaled(i) = -a_F(i)/(r_cc*sin(theta_cc)*dphi);
				a_F_scaled(i) = 1.0/(r_cc*sin(theta_cc)*dphi);
			}
		}

	}
	PROTO_KERNEL_END(Scale_Ff_calc2F, Scale_Ff_calc2)


	PROTO_KERNEL_START
	void PowellF(const Point& a_pt,
				 State&         a_P,
	             const State&   a_W,
				 Var<double,1>& a_F,
				 Var<double,1>& a_cell_volume)
	{
	double volume = a_cell_volume(0);
#if DIM==2
		a_P(0) = (a_F(0)/volume)*0.;
		a_P(1) = (a_F(0)/volume)*a_W(4)/4.0/c_PI;
		a_P(2) = (a_F(0)/volume)*a_W(5)/4.0/c_PI;
		a_P(3) = (a_F(0)/volume)*(a_W(1)*a_W(4)/4.0/c_PI + a_W(2)*a_W(5)/4.0/c_PI);
		a_P(4) = (a_F(0)/volume)*a_W(1);
		a_P(5) = (a_F(0)/volume)*a_W(2);
#endif

#if DIM==3
		a_P(0) = (a_F(0)/volume)*0.;
		a_P(1) = (a_F(0)/volume)*a_W(5)/4.0/c_PI;
		a_P(2) = (a_F(0)/volume)*a_W(6)/4.0/c_PI;
		a_P(3) = (a_F(0)/volume)*a_W(7)/4.0/c_PI;
		a_P(4) = (a_F(0)/volume)*(a_W(1)*a_W(5)/4.0/c_PI + a_W(2)*a_W(6)/4.0/c_PI + a_W(3)*a_W(7)/4.0/c_PI);
		a_P(5) = (a_F(0)/volume)*a_W(1);
		a_P(6) = (a_F(0)/volume)*a_W(2);
		a_P(7) = (a_F(0)/volume)*a_W(3);

#endif
	}
	PROTO_KERNEL_END(PowellF, Powell)




	PROTO_KERNEL_START
	void B_face_calcF(const Point& a_pt,
                    Var<double,1>& a_B_face,
	               const State& a_W_sph1,
	               const State& a_W_sph2,
	               int a_d)
	{
		a_B_face(0) = 0.5*(a_W_sph1(2+DIM+a_d)+a_W_sph2(2+DIM+a_d));
	}
	PROTO_KERNEL_END(B_face_calcF, B_face_calc)

	void step_spherical_2O(LevelBoxData<double,NUMCOMPS>& a_Rhs,
			  LevelBoxData<double,NUMCOMPS>& a_U,
			  MHDLevelDataState& a_State,
			  double& a_min_dt)
	{
		
		using namespace std;
		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		double gamma = a_State.m_gamma;
		double dxd[3] = {a_dx, a_dy, a_dz};
		
		static Stencil<double> m_divergence[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			for (int dir = 0; dir < DIM; dir++)
			{
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
			}
			initialized =  true;
		}
		double dt_new;
		for (auto dit : a_State.m_U){	
			Box dbx0 = a_U[dit].box();
			Vector F_f_sph(dbx0), F_f(dbx0), F_f_scaled(dbx0), Rhs_d(dbx0), RhsV(dbx0);
			Scalar B_f_sph(dbx0), B_f_scaled(dbx0), Rhs_d_divB(dbx0), RhsV_divB(dbx0);
			RhsV.setVal(0.0);
			RhsV_divB.setVal(0.0);

			Vector W_low_temp(dbx0), W_high_temp(dbx0), W_low(dbx0), W_high(dbx0);
			BoxData<double,NUMCOMPS> W_sph(dbx0);
			Vector W_cart  = forall<double,NUMCOMPS>(consToPrim, a_U[dit], gamma);
			MHD_Mapping::Cartesian_to_Spherical(W_sph, W_cart, a_State.m_x_sph_cc[dit]);
			MHD_Mapping::Correct_V_theta_phi_at_poles(W_sph, a_dx, a_dy, a_dz);	

			if (!a_State.m_min_dt_calculated){ 
				MHD_CFL::Min_dt_calc_func(dt_new, W_sph, a_dx, a_dy, a_dz, gamma);	
				if (dt_new < a_min_dt) a_min_dt = dt_new;
			}

			for (int d = 0; d < DIM; d++)
			{
				MHD_Limiters::MHD_Limiters_minmod(W_low,W_high,W_sph,a_State.m_x_sph_cc[dit],a_State.m_dx_sph[dit],d);
				MHD_Riemann_Solvers::Roe8Wave_Solver(F_f_sph,W_low,W_high,d,gamma);
				if (d==0) MHD_Mapping::Spherical_to_Cartesian(F_f, F_f_sph, a_State.m_x_sph_fc_1[dit]);
				if (d==1) MHD_Mapping::Spherical_to_Cartesian(F_f, F_f_sph, a_State.m_x_sph_fc_2[dit]);
				if (d==2) MHD_Mapping::Spherical_to_Cartesian(F_f, F_f_sph, a_State.m_x_sph_fc_3[dit]);
				forallInPlace_p(Scale_with_A_Ff_calc, F_f_scaled, F_f, a_State.m_face_area[dit], d);
				Rhs_d = m_divergence[d](F_f_scaled);
				RhsV += Rhs_d;

				if (!a_State.m_divB_calculated){
					forallInPlace_p(Scale_with_A_Bf_calc, B_f_scaled, W_low,W_high, a_State.m_face_area[dit], d);
					Rhs_d_divB = m_divergence[d](B_f_scaled);
					RhsV_divB += Rhs_d_divB;
				}
			}
			forallInPlace_p(Scale_with_V_calc, a_Rhs[dit], RhsV, a_State.m_cell_volume[dit]);
			if (!a_State.m_divB_calculated) forallInPlace_p(Powell,a_State.m_divB[dit],W_cart,RhsV_divB,a_State.m_cell_volume[dit]);		
		}
	}
}
