#include "Proto.H"
#include "MHD_Set_Boundary_Values.H"
#include "CommonTemplates.H"
#include "Proto_Timer.H"
#include "Proto_WriteBoxData.H"
#include "Proto_LevelBoxData.H"
#include "Proto_ProblemDomain.H"
#include "MHDOp.H"
#include "MHD_Mapping.H"
#include "MHD_Initialize.H"
#include "MHD_Input_Parsing.H"
#include "MHD_Output_Writer.H"
#include "MHD_Constants.H"
typedef BoxData<double,1,HOST> Scalar;
typedef BoxData<double,NUMCOMPS,HOST> Vector;

extern Parsefrominputs inputs;

namespace MHD_Set_Boundary_Values {

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


	void Set_Boundary_Values(LevelBoxData<double,NUMCOMPS>& a_JU,
	                         MHDLevelDataState& a_State)
	{
		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		double a_gamma = a_State.m_gamma;
		for (auto dit : a_State.m_U){	

			Box dbx0 = a_JU[dit].box();
			//Filling ghost cells for low side of dir == 0 here. This will be the inner boundary in r direction once we map to polar, spherical, or cubed sphere grids.
			if (dbx0.low()[0] < a_State.m_probDom.box().low()[0] && inputs.LowBoundType == 1) {
				Point ghost_low = dbx0.low();
	#if DIM == 2
				Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1]);
	#endif
	#if DIM == 3
				Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1], dbx0.high()[2]);
	#endif
				Box BoundBox(ghost_low,ghost_high);
				Vector a_U(BoundBox), a_JU_ghost(BoundBox);
				BoxData<double,DIM> eta(BoundBox);
				forallInPlace_p(iotaFunc, BoundBox, eta, a_dx, a_dy, a_dz);
				BoxData<double,DIM> x(BoundBox);
				MHD_Mapping::eta_to_x_calc(x,eta, BoundBox);
				MHD_Initialize::InitializeStatecalc(a_U,x,eta,a_gamma,BoundBox);
				
				if (inputs.grid_type_global == 2 && inputs.initialize_in_spherical_coords == 1){
					MHD_Mapping::U_Sph_ave_to_JU_calc_func(a_JU_ghost, a_U, a_State.m_detAA_avg[dit],a_State.m_detAA_inv_avg[dit], a_State.m_r2rdot_avg[dit], a_State.m_detA_avg[dit],a_State.m_A_row_mag_avg[dit], true);
				} else {
					a_JU_ghost = forall<double,NUMCOMPS>(dot_pro_calcF, a_State.m_Jacobian_ave[dit], a_U);
				}
				// a_JU_ghost = forall<double,NUMCOMPS>(dot_pro_calcF, a_State.m_Jacobian_ave[dit], a_U);
				a_JU_ghost.copyTo(a_JU[dit],BoundBox);
			}

			//Filling ghost cells for high side of dir == 0 here, to make an open boundary. Ghost cells are filled by outermost cells of problem domain. This will be the outer boundary in r direction once we map to polar, spherical, or cubed sphere grids.
			if (dbx0.high()[0] > a_State.m_probDom.box().high()[0] && inputs.HighBoundType == 2) {
	#if DIM == 2
				Point source_low = Point(a_State.m_probDom.box().high()[0],a_JU[dit].box().low()[1]);
				Point source_high = Point(a_State.m_probDom.box().high()[0],a_JU[dit].box().high()[1]);
	#endif
	#if DIM == 3
				Point source_low = Point(a_State.m_probDom.box().high()[0],a_JU[dit].box().low()[1],a_JU[dit].box().low()[2]);
				Point source_high = Point(a_State.m_probDom.box().high()[0],a_JU[dit].box().high()[1],a_JU[dit].box().high()[2]);
	#endif
				Box sourceBox(source_low,source_high);
				Box sourceBox1 = sourceBox.grow(1);

				Vector a_U(sourceBox1);
				Vector a_U_exterp(sourceBox1);

				Stencil<double> m_exterp_f_2nd;
		
				if (inputs.grid_type_global == 2 && inputs.initialize_in_spherical_coords == 1){
					MHD_Mapping::JU_to_U_Sph_ave_calc_func(a_U, a_JU[dit], a_State.m_detAA_inv_avg[dit], a_State.m_r2rdot_avg[dit], a_State.m_detA_avg[dit],a_State.m_A_row_mag_avg[dit], false);
				} else {
					MHD_Mapping::JU_to_U_2ndOrdercalc(a_U, a_JU[dit], a_State.m_Jacobian_ave[dit], sourceBox1);
				}
				// MHD_Mapping::JU_to_U_2ndOrdercalc(a_U, a_JU[dit], a_State.m_Jacobian_ave[dit], sourceBox1);

				

	#if DIM == 2
				Point ghost_low = Point(a_State.m_probDom.box().high()[0]+1, dbx0.low()[1]);
	#endif
	#if DIM == 3
				Point ghost_low = Point(a_State.m_probDom.box().high()[0]+1, dbx0.low()[1], dbx0.low()[2]);
	#endif
				Point ghost_high = dbx0.high();
				Box BoundBox(ghost_low,ghost_high);
				// Box BoundBox1 = BoundBox.grow(1);
				Box BoundBox1 = BoundBox.grow(0);
				Vector a_U_ghost(BoundBox), a_JU_ghost(BoundBox);
				Scalar Jacobian_ave2(BoundBox1);
				if (inputs.grid_type_global == 2){
					MHD_Mapping::Jacobian_ave_sph_calc_func(Jacobian_ave2,a_dx, a_dy, a_dz);
				} else {
					MHD_Mapping::Jacobian_Ave_calc(Jacobian_ave2,a_dx, a_dy, a_dz,BoundBox1);
				}

				for (int i = 1; i <= NGHOST; i++ ) {
					//Using outermost 2 layers to extrapolate to rest.
					m_exterp_f_2nd = (i+1.0)*Shift(Point::Zeros()) - (i*1.0)*Shift(-Point::Basis(0)); 
					a_U_exterp = m_exterp_f_2nd(a_U);
					a_U_exterp.copyTo(a_U_ghost,sourceBox,Point::Basis(0)*(i));// Using shifting option of copyTo
				}

				if (inputs.grid_type_global == 2 && inputs.initialize_in_spherical_coords == 1){
					MHD_Mapping::U_Sph_ave_to_JU_calc_func(a_JU_ghost, a_U_ghost, a_State.m_detAA_avg[dit], a_State.m_detAA_inv_avg[dit], a_State.m_r2rdot_avg[dit], a_State.m_detA_avg[dit], a_State.m_A_row_mag_avg[dit], false);
				} else {
					a_JU_ghost = forall<double,NUMCOMPS>(dot_pro_calcF, Jacobian_ave2, a_U_ghost);
				}
				// a_JU_ghost = forall<double,NUMCOMPS>(dot_pro_calcF, Jacobian_ave2, a_U_ghost);
				a_JU_ghost.copyTo(a_JU[dit],BoundBox);
			}
		}
	}





	void Set_Jacobian_Values(MHDLevelDataState& a_State)
	{

		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		for (auto dit : a_State.m_U){
			Box dbx0 = a_State.m_Jacobian_ave[dit].box();
			//Filling Jacobian values in ghost cells for low side of dir == 0 here. This will be the inner boundary in r direction once we map to polar, spherical, or cubed sphere grids.
			if (dbx0.low()[0] < a_State.m_probDom.box().low()[0] && inputs.LowBoundType != 0) {
				Point ghost_low = dbx0.low();
	#if DIM == 2
				Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1]);
	#endif
	#if DIM == 3
				Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1], dbx0.high()[2]);
	#endif
				Box BoundBox(ghost_low,ghost_high);
				Box BoundBox1 = BoundBox.grow(1);
				Scalar Jacobian_ave2(BoundBox1);
				if (inputs.grid_type_global == 2){
					MHD_Mapping::Jacobian_ave_sph_calc_func(Jacobian_ave2,a_dx, a_dy, a_dz);
				} else {
					MHD_Mapping::Jacobian_Ave_calc(Jacobian_ave2,a_dx, a_dy, a_dz,BoundBox1);
				}
				Jacobian_ave2.copyTo(a_State.m_Jacobian_ave[dit],BoundBox);
			}

			//Filling Jacobian values in ghost cells for high side of dir == 0 here. This will be the outer boundary in r direction once we map to polar, spherical, or cubed sphere grids.
			if (dbx0.high()[0] > a_State.m_probDom.box().high()[0] && inputs.HighBoundType != 0) {
	#if DIM == 2
				Point ghost_low = Point(a_State.m_probDom.box().high()[0]+1, dbx0.low()[1]);
	#endif
	#if DIM == 3
				Point ghost_low = Point(a_State.m_probDom.box().high()[0]+1, dbx0.low()[1], dbx0.low()[2]);
	#endif
				Point ghost_high = dbx0.high();
				Box BoundBox(ghost_low,ghost_high);
				Box BoundBox1 = BoundBox.grow(1);
				Scalar Jacobian_ave2(BoundBox1);
				if (inputs.grid_type_global == 2){
					MHD_Mapping::Jacobian_ave_sph_calc_func(Jacobian_ave2,a_dx, a_dy, a_dz);
				} else {
					MHD_Mapping::Jacobian_Ave_calc(Jacobian_ave2,a_dx, a_dy, a_dz,BoundBox1);
				}
				Jacobian_ave2.copyTo(a_State.m_Jacobian_ave[dit],BoundBox);
			}
		}
	}


	PROTO_KERNEL_START
	void scale_with_r_calcF(const Point& a_pt,
							Var<double,NUMCOMPS>& a_U_scaled,
							Var<double,NUMCOMPS>& a_W,
							Var<double,DIM>& a_x_sph,
							const double a_gamma)
	{
		double r_BC = inputs.r_in*c_AU; //in cm
		double rad = a_x_sph(0);
		double rho = a_W(0)*1.67262192e-24*pow(inputs.r_in*c_AU/rad,2.0);
		double u = a_W(1);
		double v = a_W(2);
		double w = a_W(3);
		double p = a_W(4)*1.0e-12*pow(inputs.r_in*c_AU/rad,2.0*a_gamma); // From picodyne to dyne 
		double Bx = a_W(5)*1.0e-6*pow(inputs.r_in*c_AU/rad,2.0); // MicroGauss to Gauss
		double By = a_W(6)*1.0e-6*pow(inputs.r_in*c_AU/rad,1.0);
		double Bz = a_W(7)*1.0e-6*pow(inputs.r_in*c_AU/rad,1.0);


		double e = p/(a_gamma-1.0) + rho*(u*u+v*v+w*w)/2.0 + (Bx*Bx+By*By+Bz*Bz)/8.0/c_PI;

		a_U_scaled(0) = rho; //rho
		a_U_scaled(1) = rho*u; //Momentum-x
		a_U_scaled(2) = rho*v; //Momentum-y
		a_U_scaled(3) = rho*w; //Momentum-z
		a_U_scaled(4) = e; //Energy
		a_U_scaled(5) = Bx; //Bx
		a_U_scaled(6) = By; //By
		a_U_scaled(7) = Bz; //Bz


	}
	PROTO_KERNEL_END(scale_with_r_calcF, scale_with_r_calc)

	

	void Set_Boundary_Values_Spherical_2O(LevelBoxData<double,NUMCOMPS>& a_U,
	                         			  MHDLevelDataState& a_State)
	{

		double a_dx = a_State.m_dx;
		double a_dy = a_State.m_dy;
		double a_dz = a_State.m_dz;
		double a_gamma = a_State.m_gamma;
		for (auto dit : a_U){	
			Box dbx0 = a_U[dit].box();
			//Filling ghost cells for low side of dir == 0 here. This will be the inner boundary in r direction once we map to polar, spherical, or cubed sphere grids.
			
			if (dbx0.low()[0] < a_State.m_probDom.box().low()[0] && inputs.LowBoundType == 1) {
				if (inputs.sph_inner_BC_hdf5 == 1){
					#if DIM == 2
						Point source_low = Point(a_State.m_probDom.box().low()[0],a_U[dit].box().low()[1]);
						Point source_high = Point(a_State.m_probDom.box().low()[0],a_U[dit].box().high()[1]);
					#endif
					#if DIM == 3
						Point source_low = Point(a_State.m_probDom.box().low()[0],a_U[dit].box().low()[1],a_U[dit].box().low()[2]);
						Point source_high = Point(a_State.m_probDom.box().low()[0],a_U[dit].box().high()[1],a_U[dit].box().high()[2]);
					#endif
					Box sourceBox(source_low,source_high);
					Box sourceBox1 = sourceBox.grow(1);
					Point ghost_low = dbx0.low();
					#if DIM == 2
						Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1]);
					#endif
					#if DIM == 3
						Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1], dbx0.high()[2]);
					#endif
					Box BoundBox(ghost_low,ghost_high);
					Vector a_U_ghost(BoundBox), a_U_sph(BoundBox), a_U_sph_scaled_r(BoundBox);

					for (int i = 1; i <= NGHOST; i++ ) {
						a_State.m_BC[dit].copyTo(a_U_sph,sourceBox,Point::Basis(0)*(-i));// Using shifting option of copyTo
					}
					BoxData<double,DIM> x_sph(BoundBox);
					MHD_Mapping::get_sph_coords_cc(x_sph,BoundBox,a_dx, a_dy, a_dz);
					forallInPlace_p(scale_with_r_calc, BoundBox, a_U_sph_scaled_r, a_U_sph, x_sph, a_gamma);
					MHD_Mapping::Spherical_to_Cartesian(a_U_ghost, a_U_sph_scaled_r, x_sph);
					a_U_ghost.copyTo(a_U[dit],BoundBox);
					


				} else {
					Point ghost_low = dbx0.low();
					#if DIM == 2
						Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1]);
					#endif
					#if DIM == 3
						Point ghost_high = Point(a_State.m_probDom.box().low()[0]-1, dbx0.high()[1], dbx0.high()[2]);
					#endif
					Box BoundBox(ghost_low,ghost_high);
					Vector a_U_sph(BoundBox), a_U_ghost(BoundBox);
					BoxData<double,DIM> x_sph(BoundBox);
					MHD_Mapping::get_sph_coords_cc(x_sph,BoundBox,a_dx, a_dy, a_dz);
					MHD_Initialize::InitializeState_Spherical_2O(a_U_sph, x_sph, a_gamma);
					MHD_Mapping::Spherical_to_Cartesian(a_U_ghost, a_U_sph, x_sph);
					a_U_ghost.copyTo(a_U[dit],BoundBox);
				}
			}
			

			//Filling ghost cells for high side of dir == 0 here, to make an open boundary. Ghost cells are filled by outermost cells of problem domain. This will be the outer boundary in r direction once we map to polar, spherical, or cubed sphere grids.
			if (dbx0.high()[0] > a_State.m_probDom.box().high()[0] && inputs.HighBoundType == 2) {
	#if DIM == 2
				Point source_low = Point(a_State.m_probDom.box().high()[0],a_U[dit].box().low()[1]);
				Point source_high = Point(a_State.m_probDom.box().high()[0],a_U[dit].box().high()[1]);
	#endif
	#if DIM == 3
				Point source_low = Point(a_State.m_probDom.box().high()[0],a_U[dit].box().low()[1],a_U[dit].box().low()[2]);
				Point source_high = Point(a_State.m_probDom.box().high()[0],a_U[dit].box().high()[1],a_U[dit].box().high()[2]);
	#endif
				Box sourceBox(source_low,source_high);
				Box sourceBox1 = sourceBox.grow(1);

				Vector a_U_exterp(sourceBox1);

				Stencil<double> m_exterp_f_2nd;


	#if DIM == 2
				Point ghost_low = Point(a_State.m_probDom.box().high()[0]+1, dbx0.low()[1]);
	#endif
	#if DIM == 3
				Point ghost_low = Point(a_State.m_probDom.box().high()[0]+1, dbx0.low()[1], dbx0.low()[2]);
	#endif
				Point ghost_high = dbx0.high();
				Box BoundBox(ghost_low,ghost_high);
				// Box BoundBox1 = BoundBox.grow(1);
				Box BoundBox1 = BoundBox.grow(0);
				Vector a_U_ghost(BoundBox);

				for (int i = 1; i <= NGHOST; i++ ) {
					//Using outermost 2 layers to extrapolate to rest.
					m_exterp_f_2nd = (i+1.0)*Shift(Point::Zeros()) - (i*1.0)*Shift(-Point::Basis(0)); 
					a_U_exterp = m_exterp_f_2nd(a_U[dit]);
					a_U_exterp.copyTo(a_U_ghost,sourceBox,Point::Basis(0)*(i));// Using shifting option of copyTo
				}
				MHDOp::Fix_negative_P(a_U_ghost,a_gamma); // Extrapolation can make -ve P in ghost cells.
				a_U_ghost.copyTo(a_U[dit],BoundBox);
			}

			// MHD_Output_Writer::WriteBoxData_array_nocoord(a_U[dit], a_dx, a_dy, a_dz, "STATE");
		}

	}





	void interpolate_h5_BC(MHDLevelDataState& state,
						const BoxData<double,NUMCOMPS>& BC_data,
						const double time)
	{
		BoxData<double, NUMCOMPS> BC_data_rotated;
		double carr_rot_time = 25.38*24*60*60; // Seconds
		// We should use sidereal time for this. 25.38 days. That's the rotation time from a fixed location.
		// Carrington rotation time (27.2753 days) is from Earth's prespective.
		double angle_to_rotate = fmod(360*time/carr_rot_time,360);
		int cells_to_rotate = angle_to_rotate/(360/inputs.domainSizez);
		double needed_fraction = angle_to_rotate/(360/inputs.domainSizez) - cells_to_rotate;
		cells_to_rotate = cells_to_rotate % inputs.domainSizez;
		cells_to_rotate = inputs.domainSizez - cells_to_rotate;
		static Stencil<double> m_right_shift;
		m_right_shift = (1.0-needed_fraction)*Shift(Point::Zeros()) + (needed_fraction)*Shift(-Point::Basis(2));
		Box dbx0 = BC_data.box();
		for (auto dit : state.m_U)
		{	
			BC_data.copyTo(state.m_BC[ dit],dbx0,Point::Basis(2)*(-cells_to_rotate));
			BC_data_rotated = m_right_shift(state.m_BC[ dit]);
			BC_data_rotated.copyTo(state.m_BC[ dit]);
		}
	}

}
