#pragma once

#include "Proto.H"
#include "MHDOp.H"

namespace MHD_Probe {
    void Probe(ofstream &outputFile,
        const BoxData<double,NUMCOMPS>& a_outdata,
                const double a_time,
                const double a_dx,
                const double a_dy,
                const double a_dz,
                const double a_gamma);
}