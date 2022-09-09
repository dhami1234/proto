#include "Proto.H"
#include "MHDOp.H"
#include "MHD_Mapping.H"
#include "MHD_divB_Cleaning.H"
#include "MHD_Limiters.H"
#include "MHD_Constants.H"
typedef BoxData<double,1,HOST> Scalar;
typedef BoxData<double,NUMCOMPS,HOST> Vector;

namespace MHD_divB_Cleaning {


	PROTO_KERNEL_START
	void PowellF(State&         a_P,
	             const State&   a_W)
	             //const Var<double,1>&  dot_pro_sum3)
	{

#if DIM==2
		a_P(0) = 0.;
		a_P(1) = a_W(4)/4.0/c_PI;
		a_P(2) = a_W(5)/4.0/c_PI;
		a_P(3) = a_W(1)*a_W(4)/4.0/c_PI + a_W(2)*a_W(5)/4.0/c_PI;
		//a_P(3) = a_W(1)*a_W(4)/4.0/c_PI + a_W(2)*a_W(5)/4.0/c_PI + dot_pro_sum3(0)/4.0/c_PI;
		a_P(4) = a_W(1);
		a_P(5) = a_W(2);
#endif

#if DIM==3
		a_P(0) = 0.;
		a_P(1) = a_W(5)/4.0/c_PI;
		a_P(2) = a_W(6)/4.0/c_PI;
		a_P(3) = a_W(7)/4.0/c_PI;
		a_P(4) = a_W(1)*a_W(5)/4.0/c_PI + a_W(2)*a_W(6)/4.0/c_PI + a_W(3)*a_W(7)/4.0/c_PI;
		//a_P(4) = a_W(1)*a_W(5)/4.0/c_PI + a_W(2)*a_W(6)/4.0/c_PI + a_W(3)*a_W(7)/4.0/c_PI + dot_pro_sum3(0)/4.0/c_PI;
		a_P(5) = a_W(1);
		a_P(6) = a_W(2);
		a_P(7) = a_W(3);

#endif
	}
	PROTO_KERNEL_END(PowellF, Powell)



	PROTO_KERNEL_START
	void BavgcalcF(State& a_Bavg,
	               const State& a_W_ave,
	               int a_d)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_Bavg(i) = a_W_ave(2+DIM+a_d);
		}
	}
	PROTO_KERNEL_END(BavgcalcF, Bavgcalc)

	PROTO_KERNEL_START
	void dot_pro_calcFF(State& a_dot_pro,
	                    const Var<double,1>& a_d_perp_N_s,
	                    const State& a_d_perp_B)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_dot_pro(i) = (a_d_perp_N_s(0)*a_d_perp_B(i));
		}
	}
	PROTO_KERNEL_END(dot_pro_calcFF, dot_pro_calcF)


	PROTO_KERNEL_START
	void B_f_mapped1D_calcF(State& a_B_f_mapped1D,
	                        const State& a_B_ave_f,
	                        const Var<double,1>& a_N_s_d_ave_f,
	                        const State& a_dot_pro_sum,
							const double a_dx_d)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_B_f_mapped1D(i) = (a_N_s_d_ave_f(0)*a_B_ave_f(i) + a_dot_pro_sum(i)/12.0)/(-a_dx_d);
		}
	}
	PROTO_KERNEL_END(B_f_mapped1D_calcF, B_f_mapped1D_calc)


	// Used to find the Powell term
	void step(BoxData<double,NUMCOMPS>& a_Rhs,
	          const BoxData<double,NUMCOMPS>& a_U,
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

		Box dbx0 = a_U.box();
		Box dbx1 = dbx0.grow(3-NGHOST);
		//Box dbx1 = a_U.box();
		static Stencil<double> m_laplacian;
		static Stencil<double> m_deconvolve;
		static Stencil<double> m_divergence[DIM];
		static Stencil<double> m_interp_edge[DIM];
		static Stencil<double> m_derivative[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			m_laplacian = Stencil<double>::Laplacian();
			m_deconvolve = (-1.0/24.0)*m_laplacian + (1.0)*Shift(Point::Zeros());
			for (int dir = 0; dir < DIM; dir++)
			{
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
				m_interp_edge[dir] = Stencil<double>::CellToFace(dir);
				m_derivative[dir] = Stencil<double>::Derivative(1,dir,2);
			}
			initialized =  true;
		}

		using namespace std;
		a_Rhs.setVal(0.0);
		double gamma = a_gamma;
		double dxd[3] = {a_dx, a_dy, a_dz};
		double retval;
		Vector a_U_demapped(dbx1);
		MHD_Mapping::JU_to_U_calc(a_U_demapped, a_U, a_Jacobian_ave, dbx1);

		Vector W_bar(dbx1);
		MHDOp::consToPrimcalc(W_bar,a_U_demapped,gamma);

		Vector U = m_deconvolve(a_U_demapped);

		Vector W(dbx1);
		MHDOp::consToPrimcalc(W,U,gamma);
		Vector W_ave = m_laplacian(W_bar);
		W_ave *= (1.0/24.0);
		W_ave += W;
		
		//Scalar dot_pro_sum3(dbx0); // Used to get correct average of product from product of averages. Is it needed?
		//dot_pro_sum3.setVal(0.0);
		Box dbx2 = dbx0.grow(1-NGHOST);
		for (int d = 0; d < DIM; d++)
		{
			Vector W_ave_edge = m_interp_edge[d](W_ave);
			//Scalar N_s_d_ave_f(dbx2);
			Vector B_f_mapped(dbx2);
			B_f_mapped.setVal(0.0);
			//BoxData<double> V1 = slice(W_ave,1+d);
			//BoxData<double> B1 = slice(W_ave,2+DIM+d);
			//Scalar dot_pro_sum2(dbx2);
			//dot_pro_sum2.setVal(0.0);
			double dx_d = dxd[d];
			for (int s = 0; s < DIM; s++) {
				//Scalar d_V1 = m_derivative[s](V1);
				//Scalar d_B1 = m_derivative[s](B1);
				//Scalar dot_pro2(dbx2);
				//dot_pro2.setVal(1.0);
				//dot_pro2 *= d_V1;
				//dot_pro2 *= d_B1;
				//dot_pro2 *= 1.0/12.0;
				//dot_pro_sum2 += dot_pro2;


				//MHD_Mapping::N_ave_f_calc_func(N_s_d_ave_f,s,d,a_dx,a_dy,a_dz);
				Scalar N_s_d_ave_f = slice(a_N_ave_f,d*DIM+s);
				Vector dot_pro_sum(dbx2);
				dot_pro_sum.setVal(0.0);
				Vector B_ave_f = forall<double,NUMCOMPS>( Bavgcalc, W_ave_edge, s);
				for (int s_temp = 0; s_temp < DIM; s_temp++) {
					if (s_temp != d) {
						Scalar d_perp_N_s = m_derivative[s_temp](N_s_d_ave_f);
						Vector d_perp_B = m_derivative[s_temp](B_ave_f);
						Vector dot_pro = forall<double,NUMCOMPS>(dot_pro_calcF,d_perp_N_s,d_perp_B);
						dot_pro_sum += dot_pro;
					}
				}
				Vector B_f_mapped1D = forall<double,NUMCOMPS>(B_f_mapped1D_calc,B_ave_f,N_s_d_ave_f,dot_pro_sum,dx_d);
				B_f_mapped += B_f_mapped1D;

			}

			//dot_pro_sum3 += dot_pro_sum2;

			Vector Rhs_d = m_divergence[d](B_f_mapped);
			//Rhs_d *= -1./dxd[d];
			a_Rhs += Rhs_d;
		}
		//Vector Powell_term = forall<double,NUMCOMPS>(Powell,W_ave,dot_pro_sum3);
		Vector Powell_term = forall<double,NUMCOMPS>(Powell,W_ave);

		a_Rhs *= Powell_term;
	}



	PROTO_KERNEL_START
	void B_face_calcF(const Point& a_pt,
                    State& a_B_face,
	               const State& a_W_sph1,
	               const State& a_W_sph2,
	               int a_d)
	{
		for (int i=0; i< NUMCOMPS; i++) {
			a_B_face(i) = 0.5*(a_W_sph1(2+DIM+a_d)+a_W_sph1(2+DIM+a_d));
		}
	}
	PROTO_KERNEL_END(B_face_calcF, B_face_calc)


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


    // Used to find the Powell term
	void step_spherical_2O(BoxData<double,NUMCOMPS>& a_Rhs,
	          const BoxData<double,NUMCOMPS>& a_U,
	          const Box& a_rangeBox,
	          const double a_dx,
	          const double a_dy,
	          const double a_dz,
	          const double a_gamma,
	          bool a_computeMaxWaveSpeed,
	          bool a_callBCs)
	{

		Box dbx0 = a_U.box();

		static Stencil<double> m_interp_f_2nd[DIM];
		static Stencil<double> m_divergence[DIM];
		static bool initialized = false;
		if(!initialized)
		{
			for (int dir = 0; dir < DIM; dir++)
			{
				m_interp_f_2nd[dir] = 0.5*Shift(Point::Zeros()) + 0.5*Shift(-Point::Basis(dir)); 
				m_divergence[dir] = Stencil<double>::FluxDivergence(dir);
			}
			initialized =  true;
		}

		using namespace std;
		a_Rhs.setVal(0.0);
		
		double gamma = a_gamma;
		double dxd[3] = {a_dx, a_dy, a_dz};
		BoxData<double,DIM> x_sph_cc(dbx0), x_sph_fc(dbx0), dx_sph(dbx0), face_area(dbx0);
		BoxData<double,1> cell_volume(dbx0);
		Vector B_f_sph(dbx0), B_f(dbx0), B_f_scaled(dbx0), Rhs_d(dbx0), RhsV(dbx0), W_cart(dbx0);
		RhsV.setVal(0.0);

		MHD_Mapping::get_cell_volume(cell_volume,dbx0,a_dx,a_dy,a_dz);
		MHD_Mapping::get_face_area(face_area,dbx0,a_dx,a_dy,a_dz);
		MHD_Mapping::get_delta_sph_coords(dx_sph,dbx0,a_dx,a_dy,a_dz);

		BoxData<double,NUMCOMPS> W_sph(dbx0);
		MHD_Mapping::get_sph_coords_cc(x_sph_cc,dbx0,a_dx, a_dy, a_dz);
        MHDOp::consToPrimcalc(W_cart,a_U,gamma);
		MHD_Mapping::Cartesian_to_Spherical(W_sph, W_cart, x_sph_cc);

		MHD_Mapping::Correct_V_theta_phi_at_poles(W_sph, a_dx, a_dy, a_dz);

		for (int d = 0; d < DIM; d++)
		{
			Vector W_low_temp(dbx0), W_high_temp(dbx0);
			Vector W_low(dbx0), W_high(dbx0);
			MHD_Limiters::MHD_Limiters_minmod(W_low,W_high,W_sph,x_sph_cc,dx_sph,d);
            forallInPlace_p(B_face_calc, B_f_sph, W_low,W_high, d);
			forallInPlace_p(Scale_with_A_Ff_calc, B_f_scaled, B_f_sph, face_area, d);
			Rhs_d = m_divergence[d](B_f_scaled);
			RhsV += Rhs_d;
		}
		forallInPlace_p(Scale_with_V_calc, a_Rhs, RhsV, cell_volume);

		Vector Powell_term = forall<double,NUMCOMPS>(Powell,W_cart);

		a_Rhs *= Powell_term;
	}
}
