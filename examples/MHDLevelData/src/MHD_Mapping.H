#pragma once

#include "Proto.H"
#include "MHDOp.H"
#include "MHDLevelDataRK4.H"
//TODO: These variables need to go into a common file or be defined in a more accessible way
//constexpr unsigned int NUMCOMPS=2*DIM+2;
typedef Var<double,NUMCOMPS> State;


namespace MHD_Mapping {
	void Jacobian_Ave_calc(BoxData<double,1>& a_Jacobian_ave,
	                       const double a_dx,
	                       const double a_dy,
	                       const double a_dz,
	                       const Box& a_dbx0);

	void JU_to_U_calc(BoxData<double,NUMCOMPS>& a_U_demapped,
	                  const BoxData<double,NUMCOMPS>& a_a_U,
	                  BoxData<double,1>& a_Jacobian_ave,
	                  const Box& a_dbx0);

	void JU_to_U_2ndOrdercalc(BoxData<double,NUMCOMPS>& a_U_demapped,
	                          const BoxData<double,NUMCOMPS>& a_a_U,
	                          const BoxData<double,1>& a_Jacobian_ave,
	                          const Box& a_dbx0);

	void JU_to_W_calc(BoxData<double,NUMCOMPS>& a_W,
	                  const BoxData<double,NUMCOMPS>& a_JU,
	                  const double a_dx,
	                  const double a_dy,
	                  const double a_dz,
	                  const double a_gamma);

	void JU_to_W_bar_calc(BoxData<double,NUMCOMPS>& a_W_bar,
	                      const BoxData<double,NUMCOMPS>& a_JU,
	                      BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                      BoxData<double,1>& a_r2rdot_avg,
	                      BoxData<double,1>& a_detA_avg,
	                      const double a_dx,
	                      const double a_dy,
	                      const double a_dz,
	                      const double a_gamma);


	void N_ave_f_calc_func(BoxData<double,1>& a_N_ave_f,
	                       const int a_s,
	                       const int a_d,
	                       const double a_dx,
	                       const double a_dy,
	                       const double a_dz);

	void N_ave_f_calc_func(BoxData<double,DIM*DIM>& a_N_ave_f,
	                       const double a_dx,
	                       const double a_dy,
	                       const double a_dz);

	void eta_calc(BoxData<double,DIM>& a_eta,
	              const Box& a_bx,
	              const double a_dx,
	              const double a_dy,
	              const double a_dz);

	void etaFace_calc(BoxData<double,DIM>& a_eta,
	                  const Box& a_bx,
	                  const double a_dx,
	                  const double a_dy,
	                  const double a_dz,
	                  int a_d);

	void eta_to_x_calc(BoxData<double,DIM>& a_x,
	                   const BoxData<double,DIM>& a_eta,
					   const Box& dbx0);

	void eta_to_x_ave_calc(BoxData<double,DIM>& a_x,
	                   const BoxData<double,DIM>& a_eta,
					   const double a_dx,
	                   const double a_dy,
	                   const double a_dz);				   

	void phys_coords_calc(BoxData<double,DIM>& a_x,
	                      const Box& dbx1,
	                      const double a_dx,
	                      const double a_dy,
	                      const double a_dz);	

	void phys_coords_face_calc(BoxData<double,DIM>& a_x,
	                      const Box& dbx1,
	                      const double a_dx,
	                      const double a_dy,
	                      const double a_dz,				  		  
	                      const int a_d);					  			  

	void out_data_calc(BoxData<double,NUMCOMPS+DIM>& a_out_data,
	                   const BoxData<double,DIM>& a_phys_coords,
	                   const BoxData<double,NUMCOMPS>& a_W);
		   

	void out_data2_calc(BoxData<double,DIM*DIM+DIM>& a_out_data,
	                    const BoxData<double,DIM>& a_phys_coords,
	                    const BoxData<double,DIM*DIM>& a_W);

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
								 bool a_r_dir_turn);

	void Regular_map_filling_func(MHDLevelDataState& a_state);

	void Spherical_map_filling_func(MHDLevelDataState& a_state);

	void Jacobian_ave_sph_calc_func(BoxData<double,1>& a_Jacobian_ave,
	                                const double a_dx,
	                                const double a_dy,
	                                const double a_dz);

	void JU_to_U_Sph_ave_calc_func(BoxData<double,NUMCOMPS>& a_U_Sph_ave,
	                  const BoxData<double,NUMCOMPS>& a_JU_ave,
	                  BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                  BoxData<double,1>& a_r2rdot_avg,
	                  BoxData<double,1>& a_detA_avg,
	                  BoxData<double,DIM>& a_A_row_mag_avg,
					  bool a_normalized);
								   
	void JU_to_W_Sph_ave_calc_func(BoxData<double,NUMCOMPS>& a_W_Sph_ave,
	                  const BoxData<double,NUMCOMPS>& a_JU_ave,
	                  BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                  BoxData<double,1>& a_r2rdot_avg,
	                  BoxData<double,1>& a_detA_avg,
					  BoxData<double,DIM>& a_A_row_mag_avg,
	                  const double a_gamma,
					  bool a_normalized);

    void U_Sph_ave_to_JU_calc_func(BoxData<double,NUMCOMPS>& a_JU_ave,
	                  const BoxData<double,NUMCOMPS>& a_U_Sph_ave,
	                  BoxData<double,DIM*DIM>& a_detAA_avg,
	                  BoxData<double,DIM*DIM>& a_detAA_inv_avg,
	                  BoxData<double,1>& a_r2rdot_avg,
	                  BoxData<double,1>& a_detA_avg,
					  BoxData<double,DIM>& a_A_row_mag_avg,
					  bool a_normalized);

	void JU_to_U_ave_calc_func(BoxData<double,NUMCOMPS>& a_U_ave,
	                  const BoxData<double,NUMCOMPS>& a_JU_ave,
	                  BoxData<double,1>& a_r2rdot_avg,
	                  BoxData<double,1>& a_detA_avg);		

	void W_Sph_to_W_Cart(BoxData<double,NUMCOMPS>& W_cart,
	                    const BoxData<double,NUMCOMPS>& W,
						BoxData<double,DIM*DIM>& a_A_1_avg,
						BoxData<double,DIM*DIM>& a_A_2_avg,
						BoxData<double,DIM*DIM>& a_A_3_avg,
	                    int a_d);		

	void W_Sph_to_W_normalized_sph(BoxData<double,NUMCOMPS>& W_normalized_sph,
	                    const BoxData<double,NUMCOMPS>& W_Sph,
						BoxData<double,DIM>& a_A_row_mag_1_avg,
						BoxData<double,DIM>& a_A_row_mag_2_avg,
						BoxData<double,DIM>& a_A_row_mag_3_avg,
	                    int a_d);

	void Cartesian_to_Spherical(BoxData<double,NUMCOMPS>& a_W_Sph,
	                    		const BoxData<double,NUMCOMPS>& a_W_Cart,
	                    		const BoxData<double,DIM>& a_x_Sph);

	void Spherical_to_Cartesian(BoxData<double,NUMCOMPS>& a_W_Cart,
	                    		const BoxData<double,NUMCOMPS>& a_W_Sph,
	                    		const BoxData<double,DIM>& a_x_Sph);

	void get_sph_coords_fc(BoxData<double,DIM>& a_x_sph,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz,
	                    int a_d);														

	void get_sph_coords_cc(BoxData<double,DIM>& a_x_sph,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz);

	void get_cell_volume(BoxData<double,1>& a_V,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz);

	void get_face_area(BoxData<double,DIM>& a_A,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz);	

	void get_delta_sph_coords(BoxData<double,DIM>& a_dx_sph,
	                    const Box& a_bx,
	                    const double a_dx,
	                    const double a_dy,
	                    const double a_dz);	
	
	void Correct_V_theta_phi_at_poles(BoxData<double,NUMCOMPS>& a_U_Sph_ave,
	                             const double a_dx,
	                             const double a_dy,
	                             const double a_dz);				
								 			   
	void Spherical_2O_map_filling_func(MHDLevelDataState& a_state);
}
