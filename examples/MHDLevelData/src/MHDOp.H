#pragma once

#include "Proto.H"

//TODO: These variables need to go into a common file or be defined in a more accessible way
constexpr unsigned int NUMCOMPS=2*DIM+2;

#define NGHOST 2

using namespace Proto;

typedef Var<double,DIM> V;
typedef Var<double,NUMCOMPS> State;

namespace MHDOp
{


	// Compute the right-hand side for the MHD equations (i.e. div(flux)).
	// Note: a_Rhs is NOT multiplied by the time step, which may be required
	// by the RK4 implementation.

	void step_spherical_2O(BoxData<double,NUMCOMPS>& a_Rhs,
	          const BoxData<double,NUMCOMPS>& a_U,
	          const Box& a_rangeBox,
	          const double a_dx,
	          const double a_dy,
	          const double a_dz,
	          const double a_gamma,
	          bool a_computeMaxWaveSpeed,
	          bool a_callBCs);

	void step(BoxData<double,NUMCOMPS>& a_Rhs,
	          const BoxData<double,NUMCOMPS>& a_U,
	          const Box& a_dbx0,
	          const double a_dx,
	          const double a_dy,
	          const double a_dz,
	          const double a_gamma,
	          BoxData<double,1>& Jacobian_ave,
	          BoxData<double,DIM*DIM>& a_N_ave_f,
	          bool a_computeMaxWaveSpeed=true,
	          bool a_callBCs=true);



	void step_spherical(BoxData<double,NUMCOMPS>& a_Rhs,
	                    const BoxData<double,NUMCOMPS>& a_JU_ave,
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
	                    bool a_callBCs);



	void consToPrimcalc(BoxData<double,NUMCOMPS>& W_bar,
	                    const BoxData<double,NUMCOMPS>& a_U_demapped,
	                    const double gamma);

	void Fix_negative_P(BoxData<double,NUMCOMPS>& a_U,
	                    const double gamma);					

	void consToPrimSphcalc(BoxData<double,NUMCOMPS>& W_bar,
	                    const BoxData<double,NUMCOMPS>& a_U_sph,
	                    const BoxData<double,NUMCOMPS>& a_U_sph_actual,
	                    const double gamma);

	// If one is so motivated, one can expose internal pointwise functions.
	// It is not necessary for correctness, these are all internal functions
	// used by the "step" function above.
	PROTO_KERNEL_START
	void consToPrimF(State&         a_W,
	                 const State&   a_U,
	                 double a_gamma);

	PROTO_KERNEL_START
	void upwindStateF(State& a_out,
	                  const State& a_low,
	                  const State& a_high,
	                  int a_dir,
	                  double a_gamma);


}
