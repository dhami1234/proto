#include "Proto.H"
#include "MHDOp.H"
#include "MHD_Mapping.H"
#include "MHD_Artificial_Viscosity.H"
#include "MHD_Input_Parsing.H"
#include "MHD_Output_Writer.H"
#include "MHD_Constants.H"
extern Parsefrominputs inputs;

typedef BoxData<double,1,HOST> Scalar;
typedef BoxData<double,NUMCOMPS,HOST> Vector;

namespace MHD_Artificial_Viscosity {

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
		//a_Fast_MS_speed(0) = 0.5*(sqrt((ce*ce)+( B_mag*B_mag/(4.0*PI*rho) )+( abs(Bdir)*ce/sqrt(PI*rho) ))+
		//	  sqrt((ce*ce)+( B_mag*B_mag/(4.0*PI*rho) )-( abs(Bdir)*ce/sqrt(PI*rho) )));
		a_Fast_MS_speed(0) = sqrt(ce*ce + B_mag*B_mag/4.0/PI/rho);
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
			a_F_f_mapped1D(i) = (a_N_s_d_ave_f(0)*a_F_ave_f(i) + a_dot_pro_sum(i)/12.0)/(-a_dx_d);
		}
	}
	PROTO_KERNEL_END(F_f_mapped1D_calcF, F_f_mapped1D_calc)


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
		//af = 0.5*(sqrt((ce*ce)+( B_mag*B_mag/(4.0*PI*rho) )+( B_mag*ce/sqrt(PI*rho) ))+
		//	  sqrt((ce*ce)+( B_mag*B_mag/(4.0*PI*rho) )-( B_mag*ce/sqrt(PI*rho) )));
		af = sqrt(ce*ce + B_mag*B_mag/4.0/PI/rho);
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
			a_N_d(i) = sqrt(a_N_d_sq(0))/(-a_dx_d);
		}
	}
	PROTO_KERNEL_END(sqrtCalcF, sqrtCalc)


// Used to implement artificial viscosity
	void step(BoxData<double,NUMCOMPS>& a_Rhs,
	          const BoxData<double,NUMCOMPS>& a_JU,
	          const Box& a_rangeBox,
	          const double a_dx,
	          const double a_dy,
	          const double a_dz,
	          const double a_gamma,
	          BoxData<double,1>& a_Jacobian_ave,
			  BoxData<double,DIM*DIM>& a_N_ave_f,
	          bool a_computeMaxWaveSpeed,
	          bool a_callBCs)
	{


		Box dbx0 = a_JU.box();
		Box dbx1 = dbx0.grow(5-NGHOST); // Are 2 ghost cells enough here? No, atleast 5 are required or strange V_theta appears in polar pulse problem.
		//Box dbx1 = a_JU.box();
		static Stencil<double> m_divergence[DIM];
		static Stencil<double> m_derivative[DIM];
		static Stencil<double> m_convolve_f[DIM];
		static Stencil<double> m_laplacian_f[DIM];
		static Stencil<double> m_laplacian;
		static Stencil<double> m_deconvolve;
		static Stencil<double> m_interp_edge[DIM];
		static Stencil<double> m_deconvolve_f[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			m_laplacian = Stencil<double>::Laplacian();
			m_deconvolve = (-1.0/24.0)*m_laplacian + (1.0)*Shift(Point::Zeros());
			for (int dir = 0; dir < DIM; dir++)
			{
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
				m_derivative[dir] = Stencil<double>::Derivative(1,dir,2);
				m_convolve_f[dir] = (1.0/24.0)*m_laplacian_f[dir] + 1.0*Shift(Point::Zeros());
				m_laplacian_f[dir] = Stencil<double>::LaplacianFace(dir);
				m_laplacian_f[dir] = Stencil<double>::LaplacianFace(dir);
				m_interp_edge[dir] = Stencil<double>::CellToFace(dir);
				m_deconvolve_f[dir] = (-1.0/24.0)*m_laplacian_f[dir] + 1.0*Shift(Point::Zeros());
			}
			initialized =  true;
		}


		using namespace std;
		//PR_TIME("MHDOp::operator");
		a_Rhs.setVal(0.0);
		double gamma = a_gamma;
		double dxd[3] = {a_dx, a_dy, a_dz};
		double retval;

		Vector a_U(dbx1);
		MHD_Mapping::JU_to_U_calc(a_U, a_JU, a_Jacobian_ave, dbx1);

		Vector W_bar(dbx1);
		MHDOp::consToPrimcalc(W_bar,a_U,gamma);


		if (inputs.linear_visc_apply == 1) {
			//Confirm with Phil that W_bar is fine in place of W_ave. Will help in reducing stencil.
			//Vector U = m_deconvolve(a_U);
			//Vector W(dbx1);
			//MHDOp::consToPrimcalc(W,U,gamma);
			//Vector W_ave = m_laplacian(W_bar,1.0/24.0);
			//W_ave += W;
			for (int d = 0; d < DIM; d++)
			{

				//Vector W_ave_edge = m_interp_edge[d](W_ave);
				Vector W_ave_edge = m_interp_edge[d](W_bar);
				Vector Lambda_f = forall<double,NUMCOMPS>(lambdacalc, W_ave_edge, d, gamma);
				//Scalar N_s_d_ave_f(dbx1);
				Scalar N_d_sq(dbx1);
				N_d_sq.setVal(0.0);
				for (int s = 0; s < DIM; s++) {
					//MHD_Mapping::N_ave_f_calc_func(N_s_d_ave_f,s,d,a_dx,a_dy,a_dz);
					Scalar N_s_d_ave_f = slice(a_N_ave_f,d*DIM+s);
					forallInPlace(N_d_sqcalc,dbx1,N_d_sq,N_s_d_ave_f);
				}
				double dx_d = dxd[d];
				Vector N_d = forall<double,NUMCOMPS>(sqrtCalc, N_d_sq, dx_d);
				Stencil<double> SPlus = 1.0*Shift(Point::Basis(d)) ;
				Stencil<double> SMinus = 1.0*Shift(-Point::Basis(d));
				Stencil<double> IdOp = 1.0*Shift(Point::Zeros());
				Stencil<double> D1 = SPlus - IdOp;
				Stencil<double> D2 = SPlus - 2.0*IdOp + SMinus;
				Stencil<double> D5 = (-1.0) * D1 * D2  * D2;
				Vector F_f = D5(a_U, 1.0/64);
				Vector F_f_behind = SMinus(F_f);
				F_f_behind *= Lambda_f;
				F_f_behind *= N_d;

				Vector Rhs_d = m_divergence[d](F_f_behind);
				//Rhs_d *= -1./dxd[d];
				a_Rhs += Rhs_d;
			}
		}



		if (inputs.non_linear_visc_apply == 1) {
			for (int d = 0; d < DIM; d++)
			{
				Vector F_f(dbx1), F_ave_f(dbx1);
				Scalar Lambda_f(dbx1);
				//Scalar N_s_d_ave_f(dbx1);
				Vector F_f_mapped(dbx1);
				F_f_mapped.setVal(0.0);
				for (int s = 0; s < DIM; s++) {
					Scalar v_s =  slice(W_bar,1+s);
					Scalar v_s_behind = alias(v_s,Point::Basis(d)*(1));
					Scalar h_lambda = forall<double>(viscosity1_calc,v_s,v_s_behind);
					for (int s2 = 0; s2 < DIM; s2++) {
						if (s2!=s) {
							for (int d2 = 0; d2 < DIM; d2++) {
								if (d2!=d) {
									Scalar v_s2 = slice(W_bar,1+s2);
									Scalar v_s2_ahead = alias(v_s2,Point::Basis(d2)*(-1));
									Scalar v_s2_behind = alias(v_s2,Point::Basis(d2)*(1));
									Scalar v_s2_behind_dp = alias(v_s2_ahead,Point::Basis(d)*(1));
									Scalar v_s2_behind_dm = alias(v_s2_behind,Point::Basis(d)*(1));
									Scalar v_s2_div = forall<double>(v_d2_div_calc,v_s2_ahead,v_s2_behind,v_s2_behind_dp,v_s2_behind_dm);
									h_lambda += v_s2_div;
								}
							}
						}
					}
					Scalar Fast_MS_speed = forall<double>(Fast_MS_speed_calc, W_bar, s, gamma);
					Scalar Fast_MS_speed_behind = alias(Fast_MS_speed,Point::Basis(d)*(1));
					Scalar Fast_MS_speed_min = forall<double>(Fast_MS_speed_min_calc,Fast_MS_speed,Fast_MS_speed_behind);
					Scalar Visc_coef = forall<double>(Visc_coef_calc,h_lambda,Fast_MS_speed_min);
					Vector a_U_behind = alias(a_U,Point::Basis(d)*(1));
					Vector mu_f = forall<double,NUMCOMPS>(mu_f_calc, Visc_coef, a_U, a_U_behind);
					//MHD_Mapping::N_ave_f_calc_func(N_s_d_ave_f,s,d,a_dx, a_dy, a_dz);
					Scalar N_s_d_ave_f = slice(a_N_ave_f,d*DIM+s);

					F_ave_f = m_convolve_f[d](mu_f);

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
					double dx_d = dxd[d];
					Vector F_f_mapped1D = forall<double,NUMCOMPS>(F_f_mapped1D_calc,F_ave_f,N_s_d_ave_f,dot_pro_sum,dx_d);

					F_f_mapped += F_f_mapped1D;
				}
				Vector Rhs_d = m_divergence[d](F_f_mapped);
				//Rhs_d *= -1./dxd[d];
				a_Rhs += Rhs_d;
			}
		}
	}













	PROTO_KERNEL_START
	void lambda_sph_calcF(Var<double,1>& a_lambda,
	                 const State& a_W_edge,
	                 const State& a_W_edge_actual,
					 const Var<double,1>& a_r2detA_1_avg,                 
					 const Var<double,1>& a_rrdotdetA_2_avg,
					 const Var<double,1>& a_rrdotdetA_3_avg,
	                 int a_d,
	                 double a_gamma)
	{
		double gamma = a_gamma;
		double rho=0., u=0., v=0., w=0., p=0., Bx=0., By=0., Bz=0., ce, af, B_mag, u_mag, Bdir, udir;

		rho = a_W_edge(0);
		u   = a_W_edge(1);
		v   = a_W_edge(2);
		w   = a_W_edge(3);
		p   = a_W_edge(4);
		Bx  = a_W_edge(5);
		By  = a_W_edge(6);
		Bz  = a_W_edge(7);

		double rho_actual=0., u_actual=0., v_actual=0., w_actual=0., p_actual=0., Bx_actual=0., By_actual=0., Bz_actual=0.;

		rho_actual = a_W_edge_actual(0);
		u_actual   = a_W_edge_actual(1);
		v_actual   = a_W_edge_actual(2);
		w_actual   = a_W_edge_actual(3);
		p_actual   = a_W_edge_actual(4);
		Bx_actual  = a_W_edge_actual(5);
		By_actual  = a_W_edge_actual(6);
		Bz_actual  = a_W_edge_actual(7);

		if (p_actual < 0.0) p_actual = 0.0;
		ce = sqrt(gamma*p_actual/rho_actual);
		B_mag = sqrt(Bx_actual*Bx_actual+By_actual*By_actual+Bz_actual*Bz_actual);
		if (a_d == 0) u_mag = abs(u);
		if (a_d == 1) u_mag = abs(v);
		if (a_d == 2) u_mag = abs(w);
		af = sqrt(ce*ce + B_mag*B_mag/4.0/PI/rho_actual);
		double lambda = af + u_mag;
		// if (a_d == 0) lambda = lambda * a_r2detA_1_avg(0);
		// if (a_d == 1) lambda = lambda * a_rrdotdetA_2_avg(0);
		// if (a_d == 2) lambda = lambda * a_rrdotdetA_3_avg(0);
		a_lambda(0) = lambda;
		
	}
	PROTO_KERNEL_END(lambda_sph_calcF, lambda_sph_calc)


	PROTO_KERNEL_START
	void Spherical_lin_visc_StateF(const Point& a_pt,
										State& a_F_ave_f,
	                                    const State& a_F_ave_f_low_unmapped,
	                                    const Var<double,1>& a_r2detA_1_avg,
	                                    const Var<double,DIM*DIM>& a_r2detAA_1_avg,
	                                    const Var<double,1>& a_rrdotdetA_2_avg,
	                                    const Var<double,DIM*DIM>& a_rrdotdetAA_2_avg,
	                                    const Var<double,1>& a_rrdotdetA_3_avg,
	                                    const Var<double,DIM*DIM>& a_rrdotdetAA_3_avg,
										const Var<double,1>& a_af,
	                                    int a_d,
	                                    double a_gamma,
										const double a_dx,
	                    		  		const double a_dy,
	                    		  		const double a_dz)
	{
		
		if (a_d == 0) {

			// double lambda = (a_W_low_avg(1) + a_af(0));
			double lambda = a_af(0);
			// lambda = 0.0;
			a_F_ave_f(0) = a_r2detA_1_avg(0)*lambda*a_F_ave_f_low_unmapped(0);
			a_F_ave_f(4) = a_r2detA_1_avg(0)*lambda*a_F_ave_f_low_unmapped(4);
			a_F_ave_f(1) = a_r2detAA_1_avg(0)*lambda*a_F_ave_f_low_unmapped(1) + a_r2detAA_1_avg(1)*lambda*a_F_ave_f_low_unmapped(2) + a_r2detAA_1_avg(2)*lambda*a_F_ave_f_low_unmapped(3);
			a_F_ave_f(2) = a_r2detAA_1_avg(3)*lambda*a_F_ave_f_low_unmapped(1) + a_r2detAA_1_avg(4)*lambda*a_F_ave_f_low_unmapped(2) + a_r2detAA_1_avg(5)*lambda*a_F_ave_f_low_unmapped(3);
			a_F_ave_f(3) = a_r2detAA_1_avg(6)*lambda*a_F_ave_f_low_unmapped(1) + a_r2detAA_1_avg(7)*lambda*a_F_ave_f_low_unmapped(2) + a_r2detAA_1_avg(8)*lambda*a_F_ave_f_low_unmapped(3);
		}

		if (a_d == 1) {  
			
			// double lambda = (a_W_low_avg(2) + a_af(0));
			double lambda = a_af(0);
			// lambda = 0.0;
			a_F_ave_f(0) = a_rrdotdetA_2_avg(0)*lambda*a_F_ave_f_low_unmapped(0);
			a_F_ave_f(4) = a_rrdotdetA_2_avg(0)*lambda*a_F_ave_f_low_unmapped(4);
			a_F_ave_f(1) = a_rrdotdetAA_2_avg(0)*lambda*a_F_ave_f_low_unmapped(1) + a_rrdotdetAA_2_avg(1)*lambda*a_F_ave_f_low_unmapped(2) + a_rrdotdetAA_2_avg(2)*lambda*a_F_ave_f_low_unmapped(3);
			a_F_ave_f(2) = a_rrdotdetAA_2_avg(3)*lambda*a_F_ave_f_low_unmapped(1) + a_rrdotdetAA_2_avg(4)*lambda*a_F_ave_f_low_unmapped(2) + a_rrdotdetAA_2_avg(5)*lambda*a_F_ave_f_low_unmapped(3);
			a_F_ave_f(3) = a_rrdotdetAA_2_avg(6)*lambda*a_F_ave_f_low_unmapped(1) + a_rrdotdetAA_2_avg(7)*lambda*a_F_ave_f_low_unmapped(2) + a_rrdotdetAA_2_avg(8)*lambda*a_F_ave_f_low_unmapped(3);
		}

		if (a_d == 2) {  
			
			// double lambda = (a_W_low_avg(3) + a_af(0));	
			double lambda = a_af(0);	
			// lambda = 0.0;
			a_F_ave_f(0) = a_rrdotdetA_3_avg(0)*lambda*a_F_ave_f_low_unmapped(0);
			a_F_ave_f(4) = a_rrdotdetA_3_avg(0)*lambda*a_F_ave_f_low_unmapped(4);
			a_F_ave_f(1) = a_rrdotdetAA_3_avg(0)*lambda*a_F_ave_f_low_unmapped(1) + a_rrdotdetAA_3_avg(1)*lambda*a_F_ave_f_low_unmapped(2) + a_rrdotdetAA_3_avg(2)*lambda*a_F_ave_f_low_unmapped(3);
			a_F_ave_f(2) = a_rrdotdetAA_3_avg(3)*lambda*a_F_ave_f_low_unmapped(1) + a_rrdotdetAA_3_avg(4)*lambda*a_F_ave_f_low_unmapped(2) + a_rrdotdetAA_3_avg(5)*lambda*a_F_ave_f_low_unmapped(3);
			a_F_ave_f(3) = a_rrdotdetAA_3_avg(6)*lambda*a_F_ave_f_low_unmapped(1) + a_rrdotdetAA_3_avg(7)*lambda*a_F_ave_f_low_unmapped(2) + a_rrdotdetAA_3_avg(8)*lambda*a_F_ave_f_low_unmapped(3);
		}


	}
	PROTO_KERNEL_END(Spherical_lin_visc_StateF, Spherical_lin_visc_State)


	PROTO_KERNEL_START
	void D5_mult_UF(const Point& a_pt,
					State& a_F,
					State& a_F2,
					const Var<double,DIM*DIM>& a_D5,
					const State& a_U
					)
	{
		double a0, a1, a2, a3, a4, a5, a6, a7;
		a0 = 0.0;
		a1 = a_D5(0)*a_U(1) + a_D5(1)*a_U(2) + a_D5(2)*a_U(3);
		a2 = a_D5(3)*a_U(1) + a_D5(4)*a_U(2) + a_D5(5)*a_U(3);
		a3 = a_D5(6)*a_U(1) + a_D5(7)*a_U(2) + a_D5(8)*a_U(3);
		a4 = 0.0;
		a5 = 0.0;
		a6 = 0.0;
		a7 = 0.0;

		a_F(0) = a_F2(0) - a0;
		a_F(1) = a_F2(1) - a1;
		a_F(2) = a_F2(2) - a2;
		a_F(3) = a_F2(3) - a3;
		a_F(4) = a_F2(4) - a4;
		a_F(5) = a_F2(5) - a5;
		a_F(6) = a_F2(6) - a6;
		a_F(7) = a_F2(7) - a7;

	}
	PROTO_KERNEL_END(D5_mult_UF, D5_mult_U)



	// Used to implement artificial viscosity
	void step_spherical(BoxData<double,NUMCOMPS>& a_Rhs,
	                    const BoxData<double,NUMCOMPS>& a_JU,
	                    const Box& a_rangeBox,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz,
	                    const double a_gamma,
	                    BoxData<double,1>& a_Jacobian_ave,
	                    BoxData<double,DIM*DIM>& a_N_ave_f,
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
	                    bool a_computeMaxWaveSpeed,
	                    bool a_callBCs)
	{


		Box dbx0 = a_JU.box();
		Box dbx1 = dbx0.grow(8-NGHOST); // Are 2 ghost cells enough here? No, atleast 5 are required or strange V_theta appears in polar pulse problem.
		//Box dbx1 = a_JU.box();
		static Stencil<double> m_divergence[DIM];
		static Stencil<double> m_interp_edge[DIM];
		static Stencil<double> m_interp_f_2nd[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			for (int dir = 0; dir < DIM; dir++)
			{
				m_interp_edge[dir] = Stencil<double>::CellToFace(dir);
				m_interp_f_2nd[dir] = 0.5*Shift(Point::Zeros()) + 0.5*Shift(-Point::Basis(dir)); 
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
			}
			initialized =  true;
		}


		using namespace std;
		a_Rhs.setVal(0.0);
		double gamma = a_gamma;
		double dxd[3] = {a_dx, a_dy, a_dz};

		Vector a_U_Sph_ave(dbx0), a_U_Sph_actual_ave(dbx0), a_U_ave(dbx0);
		MHD_Mapping::JU_to_U_Sph_ave_calc_func(a_U_Sph_ave, a_JU, a_detAA_inv_avg, a_r2rdot_avg, a_detA_avg, a_A_row_mag_avg, false);
		MHD_Mapping::JU_to_U_Sph_ave_calc_func(a_U_Sph_actual_ave, a_JU, a_detAA_inv_avg, a_r2rdot_avg, a_detA_avg, a_A_row_mag_avg, true);
		MHD_Mapping::JU_to_U_ave_calc_func(a_U_ave, a_JU, a_r2rdot_avg, a_detA_avg);

		MHD_Mapping::Correct_V_theta_phi_at_poles(a_U_Sph_ave, a_dx, a_dy, a_dz);
		MHD_Mapping::Correct_V_theta_phi_at_poles(a_U_Sph_actual_ave, a_dx, a_dy, a_dz);

		Vector W_bar(dbx1),W_bar_actual(dbx1);
		MHDOp::consToPrimSphcalc(W_bar,a_U_Sph_ave, a_U_Sph_actual_ave,gamma);
		MHDOp::consToPrimSphcalc(W_bar_actual,a_U_Sph_actual_ave, a_U_Sph_actual_ave, gamma);

		if (inputs.linear_visc_apply == 1) {
			//Confirm with Phil that W_bar is fine in place of W_ave. Will help in reducing stencil.
			for (int d = 0; d < DIM; d++)
			{
				Vector W_ave_edge = m_interp_edge[d](W_bar);
				Vector U_ave_edge = m_interp_edge[d](a_U_ave);
				Vector W_ave_edge_actual = m_interp_edge[d](W_bar_actual);

				// Vector W_ave_edge = m_interp_f_2nd[d](W_bar);
				// Vector W_ave_edge_actual = m_interp_f_2nd[d](W_bar_actual);

				Scalar Lambda_f = forall<double,1>(lambda_sph_calc, W_ave_edge, W_ave_edge_actual, a_r2detA_1_avg, a_rrdotdetA_2_avg, a_rrdotdetA_3_avg, d, gamma);  
				double dx_d = dxd[d];
				Stencil<double> SPlus = 1.0*Shift(Point::Basis(d)) ;
				Stencil<double> SMinus = 1.0*Shift(-Point::Basis(d));
				Stencil<double> IdOp = 1.0*Shift(Point::Zeros());
				Stencil<double> D1 = SPlus - IdOp;
				Stencil<double> D2 = SPlus - 2.0*IdOp + SMinus;
				Stencil<double> D5 = (-1.0) * D1 * D2  * D2;
				Vector F_f = D5(a_U_Sph_ave, 1.0/64);
				Vector F_f_behind = SMinus(F_f);

				/* Commenting this makes the viscosity work. It might not preserve constant flow though. Need to check this effect.
				// Vector F_f_behind2 = SMinus(F_f);
				// BoxData<double,DIM*DIM> a_D5_A_inv_avg(dbx0);
				// if (d==0) a_D5_A_inv_avg = D5(a_A_inv_1_avg, 1.0/64);
				// if (d==1) a_D5_A_inv_avg = D5(a_A_inv_2_avg, 1.0/64);
				// if (d==2) a_D5_A_inv_avg = D5(a_A_inv_3_avg, 1.0/64);
				// BoxData<double,DIM*DIM> a_D5_A_inv_avg_behind = SMinus(a_D5_A_inv_avg);
				// forallInPlace_p(D5_mult_U, F_f_behind, F_f_behind2, a_D5_A_inv_avg_behind, U_ave_edge);
				*/

				Vector F_f_behind_mapped(dbx0);
				F_f_behind_mapped.setVal(0.0);
				forallInPlace_p(Spherical_lin_visc_State, F_f_behind_mapped, F_f_behind, a_r2detA_1_avg, a_r2detAA_1_avg, a_rrdotdetA_2_avg, a_rrdotdetAA_2_avg, a_rrdotdetA_3_avg, a_rrdotdetAA_3_avg, Lambda_f, d, a_gamma, a_dx, a_dy, a_dz);
				// F_f_behind_mapped.setVal(0.0);
				Vector Rhs_d = m_divergence[d](F_f_behind_mapped);
				Rhs_d *= -1./dx_d;
				a_Rhs += Rhs_d;
			}

			// MHD_Output_Writer::WriteBoxData_array_nocoord(a_Rhs, a_dx, a_dy, a_dz, "a_Rhs");

		}


		if (inputs.non_linear_visc_apply == 1) {
			for (int d = 0; d < DIM; d++)
			{
				double dx_d = dxd[d];
				Stencil<double> SPlus = 1.0*Shift(Point::Basis(d)) ;
				Stencil<double> SMinus = 1.0*Shift(-Point::Basis(d));
				Stencil<double> IdOp = 1.0*Shift(Point::Zeros());
				Stencil<double> D1 = IdOp - SMinus;
				Vector F_f(dbx1), F_ave_f(dbx1);
				Scalar Lambda_f(dbx1);
				Vector F_f_mapped(dbx1);
				F_f_mapped.setVal(0.0);
				Scalar v_d =  slice(W_bar_actual,1+d);
				// Scalar v_d_behind = alias(v_d,Point::Basis(d)*(1));
				// Scalar h_lambda = forall<double>(viscosity1_calc,v_d,v_d_behind);
				Scalar h_lambda = D1(v_d);

				for (int d2 = 0; d2 < DIM; d2++) {
					if (d2!=d) {
						Scalar v_d2 = slice(W_bar_actual,1+d2);
						Scalar v_d2_ahead = alias(v_d2,Point::Basis(d2)*(-1));
						Scalar v_d2_behind = alias(v_d2,Point::Basis(d2)*(1));
						Scalar v_d2_behind_dp = alias(v_d2_ahead,Point::Basis(d)*(1));
						Scalar v_d2_behind_dm = alias(v_d2_behind,Point::Basis(d)*(1));
						Scalar v_d2_div = forall<double>(v_d2_div_calc,v_d2_ahead,v_d2_behind,v_d2_behind_dp,v_d2_behind_dm);
						h_lambda += v_d2_div;
					}
				}
				Scalar Fast_MS_speed = forall<double>(Fast_MS_speed_calc, W_bar_actual, d, gamma);
				Scalar Fast_MS_speed_behind = alias(Fast_MS_speed,Point::Basis(d)*(1));
				Scalar Fast_MS_speed_min = forall<double>(Fast_MS_speed_min_calc,Fast_MS_speed,Fast_MS_speed_behind);
				Scalar Visc_coef = forall<double>(Visc_coef_calc,h_lambda,Fast_MS_speed_min);
				
				F_f = D1(a_U_Sph_ave);
				forallInPlace_p(Spherical_lin_visc_State, F_f_mapped, F_f, a_r2detA_1_avg, a_r2detAA_1_avg, a_rrdotdetA_2_avg, a_rrdotdetAA_2_avg, a_rrdotdetA_3_avg, a_rrdotdetAA_3_avg, Visc_coef, d, a_gamma, a_dx, a_dy, a_dz);
				Vector Rhs_d = m_divergence[d](F_f_mapped);
				Rhs_d *= -1./dx_d;
				a_Rhs += Rhs_d;
			}
		}


	}





}
