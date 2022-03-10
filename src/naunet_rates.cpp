#include <math.h>
/* */
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>  // access to sparse SUNMatrix
/* */
/*  */
#include "naunet_ode.h"
/*  */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

// clang-format off
int EvalRates(realtype *k, realtype *y, NaunetData *u_data) {

    realtype rG = u_data->rG;
    realtype gdens = u_data->gdens;
    realtype sites = u_data->sites;
    realtype fr = u_data->fr;
    realtype opt_thd = u_data->opt_thd;
    realtype opt_crd = u_data->opt_crd;
    realtype opt_h2d = u_data->opt_h2d;
    realtype opt_uvd = u_data->opt_uvd;
    realtype eb_h2d = u_data->eb_h2d;
    realtype eb_crd = u_data->eb_crd;
    realtype eb_uvd = u_data->eb_uvd;
    realtype crdeseff = u_data->crdeseff;
    realtype h2deseff = u_data->h2deseff;
    realtype nH = u_data->nH;
    realtype zeta = u_data->zeta;
    realtype Tgas = u_data->Tgas;
    realtype Av = u_data->Av;
    realtype omega = u_data->omega;
    realtype G0 = u_data->G0;
    realtype uvcreff = u_data->uvcreff;
    
    double mant = GetMantleDens(y);
    double mantabund = mant / nH;
    double garea = (pi*rG*rG) * gdens;
    double garea_per_H = garea / nH;
    double densites = 4.0 * garea * sites;
    double h2col = 0.5*1.59e21*Av;
    double cocol = 1e-5 * h2col;
    double lamdabar = GetCharactWavelength(h2col, cocol);
    double H2shielding = GetShieldingFactor(IDX_H2I, h2col, h2col, Tgas, 1);
    double H2formation = 1.0e-17 * sqrt(Tgas);
    double H2dissociation = 5.1e-11 * G0 * GetGrainScattering(Av, 1000.0) * H2shielding;
    
    // clang-format on

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    if (Tgas>10.0 && Tgas<2500.0) { k[0] = 1.09e-11 * pow(Tgas/300.0, -2.19)
        * exp(-165.1/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>1340.0 && Tgas<41000.0) { k[2] = 6e-09 * pow(Tgas/300.0, 0.0) *
        exp(-40200.0/Tgas);  }
        
    if (Tgas>2803.0 && Tgas<41000.0) { k[3] = 1e-08 * pow(Tgas/300.0, 0.0) *
        exp(-84100.0/Tgas);  }
        
    if (Tgas>1763.0 && Tgas<41000.0) { k[4] = 5.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-52900.0/Tgas);  }
        
    if (Tgas>20.0 && Tgas<41000.0) { k[5] = 3.8e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>1743.0 && Tgas<41000.0) { k[6] = 6e-09 * pow(Tgas/300.0, 0.0) *
        exp(-52300.0/Tgas);  }
        
    if (Tgas>1696.0 && Tgas<41000.0) { k[7] = 6e-09 * pow(Tgas/300.0, 0.0) *
        exp(-50900.0/Tgas);  }
        
    if (Tgas>3400.0 && Tgas<41000.0) { k[8] = 3.22e-09 * pow(Tgas/300.0,
        0.35) * exp(-102000.0/Tgas);  }
        
    if (Tgas>1340.0 && Tgas<41000.0) { k[9] = 6e-09 * pow(Tgas/300.0, 0.0) *
        exp(-40200.0/Tgas);  }
        
    if (Tgas>1833.0 && Tgas<41000.0) { k[10] = 4.67e-07 * pow(Tgas/300.0,
        -1.0) * exp(-55000.0/Tgas);  }
        
    if (Tgas>1763.0 && Tgas<41000.0) { k[11] = 5.8e-09 * pow(Tgas/300.0,
        0.0) * exp(-52900.0/Tgas);  }
        
    if (Tgas>1743.0 && Tgas<41000.0) { k[12] = 6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-52300.0/Tgas);  }
        
    if (Tgas>1696.0 && Tgas<41000.0) { k[13] = 6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-50900.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[14] = 5.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[15] = 3.8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[16] = 7.8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[17] = 6e-10 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[18] = 4.8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[19] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[20] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[21] = 6.72e-10 * pow(Tgas/300.0, 0.0)
        * exp(+0.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[22] = 7.05e-10 * pow(Tgas/300.0,
        -0.03) * exp(+16.7/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[23] = 7.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[24] = 4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[25] = 2.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[26] = 2.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[27] = 2e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[28] = 2e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[29] = 2.5e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[30] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[31] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[32] = 2.3e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[33] = 3.8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[34] = 3.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[35] = 5.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[36] = 8.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[37] = 8.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[38] = 8.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[39] = 4.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[40] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[41] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[42] = 8.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[43] = 2.2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[44] = 5e-10 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[45] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[46] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[47] = 8e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[48] = 3.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[49] = 7.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[50] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[51] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[52] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[53] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[54] = 5.2e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[55] = 4.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[56] = 3.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[57] = 4.59e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[58] = 7.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[59] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[60] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[61] = 4.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[62] = 4.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[63] = 8.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[64] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[65] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[66] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[67] = 8.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[68] = 4.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[69] = 9.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[70] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[71] = 4.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[72] = 4.4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[73] = 3.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[74] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[75] = 1.13e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[76] = 1.62e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[77] = 9.45e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[78] = 1.65e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[79] = 3.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[80] = 4.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[81] = 7.93e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[82] = 3.2e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[83] = 6.4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[84] = 3.2e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[85] = 3.1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[86] = 3.4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[87] = 3.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[88] = 6.3e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[89] = 3.5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[90] = 3.5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[91] = 3.1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[92] = 3.5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[93] = 6.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[94] = 5.2e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[95] = 1.79e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[96] = 3.7e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[97] = 5.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[98] = 2.58e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[99] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[100] = 1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[101] = 1.35e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[102] = 2.44e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[103] = 7.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[104] = 3.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[105] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[106] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[107] = 7.4e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[108] = 6.2e-11 * pow(Tgas/300.0,
        0.79) * exp(-6920.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[109] = 9.3e-11 * pow(Tgas/300.0,
        0.73) * exp(-232.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[110] = 3.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[111] = 5.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[112] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[113] = 4e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[114] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[115] = 3.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[116] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[117] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[118] = 4.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[119] = 2.96e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[120] = 4.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[121] = 6.9e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[122] = 3e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[123] = 5.28e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>205.0 && Tgas<565.0) { k[124] = 1.05e-08 * pow(Tgas/300.0,
        -0.13) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[125] = 9.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<520.0) { k[126] = 3.3e-09 * pow(Tgas/300.0, 1.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[127] = 3e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[128] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[129] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[130] = 2.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[131] = 3.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[132] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[133] = 2.9e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[134] = 4.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[135] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[136] = 6.86e-10 * pow(Tgas/300.0,
        0.26) * exp(-224.3/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[137] = 2.1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[138] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[139] = 3e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[140] = 1.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[141] = 5.78e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[142] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[143] = 9.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[144] = 3e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[145] = 3e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[146] = 3e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[147] = 1.5e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[148] = 1.5e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[149] = 1.5e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[150] = 1.7e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[151] = 3.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[152] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[153] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[154] = 4.82e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[155] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[156] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[157] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[158] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[159] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[160] = 6.44e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[161] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[162] = 3.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[163] = 2.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[164] = 2.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[165] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[166] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[167] = 5.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[168] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[169] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[170] = 8e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[171] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[172] = 7.2e-15 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[173] = 5.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[174] = 2.07e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[175] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[176] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[177] = 4.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[178] = 1.41e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[179] = 9.72e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[180] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[181] = 2.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[182] = 2.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[183] = 4.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[184] = 2.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[185] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[186] = 3e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[187] = 1.72e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[188] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[189] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[190] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[191] = 6.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[192] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[193] = 6.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[194] = 3.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[195] = 1.2e-15 * pow(Tgas/300.0,
        0.25) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[196] = 5.66e-10 * pow(Tgas/300.0,
        0.36) * exp(+8.6/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[197] = 8.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[198] = 3.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[199] = 5.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[200] = 3.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[201] = 3.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[202] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[203] = 7e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[204] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[205] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[206] = 6.6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[207] = 5e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[208] = 2.54e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[209] = 6.3e-15 * pow(Tgas/300.0, 0.75)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[210] = 5.1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[211] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[212] = 9.69e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[213] = 6.05e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[214] = 3.08e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[215] = 6.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[216] = 2.64e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[217] = 3.3e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[218] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[219] = 3.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[220] = 3e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[221] = 2.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[222] = 2.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[223] = 2.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[224] = 2.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[225] = 2.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[226] = 7e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[227] = 8.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[228] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[229] = 2.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[230] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[231] = 2.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[232] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[233] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[234] = 9.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[235] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[236] = 2.8e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[237] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[238] = 8.25e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[239] = 1.88e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[240] = 2.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[241] = 1.06e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[242] = 3.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[243] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[244] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[245] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[246] = 1.97e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[247] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[248] = 4.51e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[249] = 3.11e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[250] = 1.02e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[251] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[252] = 3.77e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[253] = 1.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[254] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[255] = 4.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[256] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[257] = 2.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[258] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[259] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[260] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[261] = 1.05e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[262] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[263] = 7.12e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[264] = 4.51e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[265] = 6.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[266] = 7.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[267] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[268] = 6.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[269] = 7e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[270] = 4.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[271] = 4.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[272] = 9.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[273] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[274] = 4.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[275] = 8.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[276] = 8.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[277] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[278] = 4.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[279] = 3.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[280] = 7.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[281] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[282] = 2.14e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[283] = 2.02e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[284] = 4.25e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[285] = 2.21e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[286] = 3.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[287] = 1.68e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[288] = 5.25e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[289] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[290] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[291] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[292] = 1.44e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[293] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[294] = 6.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[295] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[296] = 6.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[297] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[298] = 7.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[299] = 3.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[300] = 7e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[301] = 4.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[302] = 4.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[303] = 3.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[304] = 5.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[305] = 7.2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[306] = 4.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[307] = 3.9e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[308] = 4.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[309] = 8.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>2000.0 && Tgas<10000.0) { k[310] = 4.9e-12 * pow(Tgas/300.0,
        0.5) * exp(-4580.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[311] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[312] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[313] = 1.36e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[314] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[315] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[316] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[317] = 1.9e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[318] = 6.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[319] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[320] = 2.04e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[321] = 1.11e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[322] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[323] = 5.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[324] = 4.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[325] = 2.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[326] = 6.5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[327] = 1.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[328] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[329] = 4.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[330] = 4.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[331] = 7.44e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[332] = 1.59e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[333] = 1.23e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[334] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[335] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[336] = 3.59e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[337] = 5.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[338] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[339] = 6.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[340] = 6.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[341] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[342] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[343] = 3.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[344] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<4999.0) { k[345] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[346] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[347] = 9.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[348] = 1.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[349] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[350] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[351] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[352] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[353] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[354] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[355] = 4.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[356] = 2.3e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[357] = 3.9e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[358] = 3.9e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[359] = 2.86e-19 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[360] = 1.2e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[361] = 1.3e-18 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[362] = 5.98e-18 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[363] = 6.5e-18 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[364] = 2.7e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[365] = 3.4e-17 * (zeta / zism);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[366] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 119.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[367] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 654.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[368] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2577.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[369] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[370] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1881.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[371] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1881.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[372] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2153.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[373] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[374] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[375] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[376] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[377] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 875.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[378] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[379] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 255.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[380] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 88.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[381] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[382] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[383] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 456.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[384] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[385] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[386] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[387] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2388.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[388] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1584.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[389] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 752.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[390] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1169.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[391] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 365.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[392] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 5290.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[393] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 854.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[394] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 1.17) * 105.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[395] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[396] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[397] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 61.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[398] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[399] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1329.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[400] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[401] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 485.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[402] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[403] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 848.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[404] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2577.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[405] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[406] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 0.2 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[407] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 863.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[408] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1557.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[409] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 210.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[410] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 584.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[411] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1000.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[412] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[413] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1370.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[414] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[415] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[416] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 500.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[417] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[418] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[419] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 0.2 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[420] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 66.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[421] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 25.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[422] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1.1 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[423] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 474.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[424] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 324.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[425] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 40.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[426] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 657.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[427] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 288.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[428] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 270.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[429] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[430] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[431] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[432] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 247.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[433] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 231.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[434] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[435] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 58.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[436] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 375.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[437] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 375.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[438] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 1.4 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[439] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[440] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 722.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[441] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2680.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[442] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 254.5 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[443] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[444] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 480.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[445] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 922.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[446] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[447] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[448] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 2115.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[449] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[450] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[451] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[452] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[453] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[454] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 750.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[455] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[456] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[457] = 1.3e-17 * (zeta / zism) *
        pow(Tgas/300.0, 0.0) * 250.0 / (1.0 - omega);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[458] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[459] = 1.16e-07 * pow(Tgas/300.0,
        -0.76) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[460] = 1.53e-07 * pow(Tgas/300.0,
        -0.76) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[461] = 9e-08 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[462] = 9e-08 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[463] = 9e-08 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[464] = 1.77e-06 * pow(Tgas/300.0,
        -0.73) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[465] = 1.77e-06 * pow(Tgas/300.0,
        -0.73) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[466] = 1.77e-06 * pow(Tgas/300.0,
        -0.73) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[467] = 1.3e-07 * pow(Tgas/300.0,
        -0.73) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[468] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[469] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[470] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[471] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[472] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[473] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[474] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[475] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[476] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[477] = 1.5e-07 * pow(Tgas/300.0, -0.42)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[478] = 7.68e-08 * pow(Tgas/300.0,
        -0.6) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[479] = 4.03e-07 * pow(Tgas/300.0,
        -0.6) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[480] = 1.6e-07 * pow(Tgas/300.0, -0.6)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[481] = 7.75e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[482] = 1.95e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[483] = 2e-07 * pow(Tgas/300.0, -0.4) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[484] = 5.28e-07 * pow(Tgas/300.0,
        -0.69) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[485] = 2.85e-07 * pow(Tgas/300.0,
        -0.69) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[486] = 1.87e-07 * pow(Tgas/300.0,
        -0.59) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[487] = 8.01e-08 * pow(Tgas/300.0,
        -0.59) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[488] = 4.54e-07 * pow(Tgas/300.0,
        -0.59) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[489] = 2.67e-08 * pow(Tgas/300.0,
        -0.59) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[490] = 8.9e-08 * pow(Tgas/300.0, -0.59)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[491] = 1.75e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[492] = 1.75e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[493] = 4.76e-08 * pow(Tgas/300.0,
        -0.52) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[494] = 1.4e-08 * pow(Tgas/300.0, -0.52)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[495] = 1.96e-07 * pow(Tgas/300.0,
        -0.52) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[496] = 1.4e-08 * pow(Tgas/300.0, -0.52)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[497] = 8.4e-09 * pow(Tgas/300.0, -0.52)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[498] = 1.8e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[499] = 2e-07 * pow(Tgas/300.0, -0.48)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[500] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[501] = 1.6e-08 * pow(Tgas/300.0, -0.43)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[502] = 2.5e-08 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[503] = 7.5e-08 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[504] = 2.5e-07 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[505] = 1.6e-07 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[506] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[507] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[508] = 1.08e-07 * pow(Tgas/300.0,
        -0.85) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[509] = 1e-08 * pow(Tgas/300.0, -0.85) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[510] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[511] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[512] = 3.9e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[513] = 3.05e-07 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[514] = 8.6e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[515] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[516] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[517] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[518] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[519] = 2.34e-08 * pow(Tgas/300.0,
        -0.52) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[520] = 4.36e-08 * pow(Tgas/300.0,
        -0.52) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[521] = 4.2e-08 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[522] = 1.4e-08 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[523] = 2.1e-07 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[524] = 2.17e-07 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[525] = 2.17e-07 * pow(Tgas/300.0,
        -0.78) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[526] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[527] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[528] = 7.09e-08 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[529] = 5.6e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[530] = 5.37e-08 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[531] = 3.05e-07 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[532] = 4.76e-08 * pow(Tgas/300.0,
        -0.86) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[533] = 4.2e-08 * pow(Tgas/300.0,
        -0.86) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[534] = 1.62e-07 * pow(Tgas/300.0,
        -0.86) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[535] = 2.8e-08 * pow(Tgas/300.0,
        -0.86) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[536] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[537] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[538] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[539] = 9.3e-08 * pow(Tgas/300.0, -0.65)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[540] = 9.5e-08 * pow(Tgas/300.0, -0.65)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[541] = 9.5e-08 * pow(Tgas/300.0, -0.65)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[542] = 2.4e-07 * pow(Tgas/300.0, -0.69)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[543] = 6e-08 * pow(Tgas/300.0, -0.64) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[544] = 8.1e-07 * pow(Tgas/300.0, -0.64)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[545] = 3.2e-07 * pow(Tgas/300.0, -0.64)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[546] = 7.87e-07 * pow(Tgas/300.0,
        -0.57) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[547] = 1.84e-07 * pow(Tgas/300.0,
        -0.57) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[548] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[549] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[550] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[551] = 1.1e-07 * pow(Tgas/300.0, -1.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[552] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[553] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[554] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[555] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[556] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[557] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[558] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[559] = 1e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[560] = 1e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[561] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[562] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[563] = 1e-08 * pow(Tgas/300.0, -0.6) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[564] = 1.7e-07 * pow(Tgas/300.0, -0.3)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<150.0) { k[565] = 2.77e-07 * pow(Tgas/300.0,
        -0.74) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<150.0) { k[566] = 2.09e-08 * pow(Tgas/300.0,
        -0.74) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[567] = 4.3e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>12.0 && Tgas<12400.0) { k[568] = 1.78e-07 * pow(Tgas/300.0,
        -0.8) * exp(-17.1/Tgas);  }
        
    if (Tgas>12.0 && Tgas<12400.0) { k[569] = 9.21e-08 * pow(Tgas/300.0,
        -0.79) * exp(-17.1/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[570] = 1.55e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[571] = 1.55e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2000.0) { k[572] = 4.72e-08 * pow(Tgas/300.0,
        -0.6) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2000.0) { k[573] = 3.77e-08 * pow(Tgas/300.0,
        -0.6) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2000.0) { k[574] = 8.49e-07 * pow(Tgas/300.0,
        -0.6) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[575] = 4.3e-07 * pow(Tgas/300.0, -0.37)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[576] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[577] = 1.95e-07 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[578] = 3e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[579] = 1.05e-08 * pow(Tgas/300.0,
        -0.62) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[580] = 4.9e-08 * pow(Tgas/300.0, -0.62)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[581] = 2.91e-07 * pow(Tgas/300.0,
        -0.62) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[582] = 3.75e-08 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[583] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[584] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[585] = 1.79e-07 * pow(Tgas/300.0,
        -0.52) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[586] = 2.81e-07 * pow(Tgas/300.0,
        -0.52) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[587] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[588] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[589] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[590] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[591] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[592] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[593] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[594] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[595] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[596] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[597] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[598] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[599] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[600] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[601] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[602] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[603] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[604] = 1.5e-07 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[605] = 2e-07 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[606] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[607] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[608] = 5.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[609] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[610] = 1.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[611] = 1.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[612] = 5.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[613] = 2.08e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[614] = 3.89e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[615] = 3.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[616] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[617] = 2.34e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[618] = 7.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[619] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[620] = 9e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[621] = 2.09e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[622] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[623] = 1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[624] = 2.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[625] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[626] = 4.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[627] = 3.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[628] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[629] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[630] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(+0.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[631] = 7.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[632] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[633] = 3.42e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[634] = 4.54e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[635] = 3.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[636] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[637] = 7.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[638] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[639] = 2.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[640] = 2.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[641] = 2.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[642] = 2.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[643] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[644] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[645] = 5.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[646] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[647] = 8.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[648] = 3.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[649] = 8e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[650] = 5.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[651] = 8.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[652] = 8.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[653] = 8.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[654] = 8.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[655] = 8.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[656] = 4.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[657] = 8.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[658] = 8.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[659] = 7.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[660] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[661] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[662] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[663] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[664] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[665] = 1.06e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[666] = 8.36e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[667] = 4.6e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[668] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[669] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[670] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[671] = 6.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[672] = 4.4e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[673] = 1.65e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[674] = 7.26e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[675] = 5.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[676] = 6.18e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[677] = 3.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[678] = 7.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[679] = 7.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[680] = 7.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[681] = 7.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[682] = 7.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[683] = 7.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[684] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[685] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[686] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[687] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[688] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[689] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[690] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[691] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[692] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[693] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[694] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[695] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[696] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[697] = 9.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[698] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[699] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[700] = 5.2e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[701] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[702] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[703] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[704] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[705] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[706] = 9.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[707] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[708] = 1.16e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[709] = 1.45e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[710] = 2.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[711] = 1.43e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[712] = 7.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[713] = 5.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[714] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[715] = 9.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[716] = 9.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[717] = 9.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[718] = 5.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[719] = 5.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[720] = 2.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[721] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[722] = 1.47e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[723] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[724] = 2.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[725] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[726] = 4.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[727] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[728] = 1.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[729] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[730] = 4.05e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[731] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[732] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[733] = 9.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[734] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[735] = 3.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[736] = 1.05e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[737] = 8.55e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[738] = 7.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[739] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[740] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[741] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[742] = 2.81e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[743] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[744] = 1.84e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[745] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[746] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[747] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[748] = 1.26e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[749] = 9.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[750] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[751] = 7.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[752] = 1.08e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[753] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[754] = 4.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[755] = 9.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[756] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[757] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[758] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[759] = 9.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[760] = 8.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[761] = 4.35e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[762] = 4.35e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[763] = 8.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[764] = 8.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[765] = 8.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[766] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[767] = 4.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[768] = 9.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[769] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[770] = 8.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[771] = 4.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[772] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[773] = 8.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[774] = 5.24e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[775] = 1.04e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[776] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[777] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[778] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[779] = 4.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[780] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[781] = 3.04e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[782] = 5e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[783] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[784] = 4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[785] = 1.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[786] = 7.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[787] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[788] = 9.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[789] = 1.8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[790] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[791] = 4.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[792] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[793] = 7.83e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[794] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[795] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[796] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[797] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[798] = 1.98e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[799] = 2.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[800] = 1.16e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[801] = 1.15e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[802] = 9.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[803] = 2.38e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[804] = 1.82e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[805] = 3.74e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[806] = 6.64e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[807] = 4.55e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[808] = 5e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[809] = 9.35e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[810] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[811] = 1.04e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[812] = 7.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[813] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[814] = 2.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[815] = 7e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[816] = 9.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[817] = 8.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[818] = 4.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[819] = 1.95e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[820] = 1.31e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[821] = 3.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[822] = 2e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[823] = 9.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[824] = 9e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[825] = 3.2e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[826] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[827] = 4.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[828] = 3.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[829] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[830] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[831] = 8.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[832] = 2.06e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[833] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[834] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[835] = 1.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[836] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[837] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[838] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[839] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[840] = 6.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[841] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[842] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[843] = 3.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[844] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[845] = 6.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[846] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[847] = 3.15e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[848] = 3.15e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[849] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[850] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[851] = 5.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[852] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[853] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[854] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[855] = 3.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[856] = 6.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[857] = 3.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[858] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[859] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[860] = 3.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[861] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[862] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[863] = 6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[864] = 5.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[865] = 5.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[866] = 3.15e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[867] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[868] = 8.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[869] = 8.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[870] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[871] = 1.65e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[872] = 1.56e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[873] = 1.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<520.0) { k[874] = 7.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[875] = 7.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[876] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[877] = 8.8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[878] = 8.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[879] = 3e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[880] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[881] = 7.9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[882] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[883] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[884] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[885] = 1.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[886] = 3e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[887] = 5.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[888] = 3.84e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[889] = 8.85e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[890] = 2.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[891] = 3.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[892] = 1.06e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[893] = 3.57e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[894] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[895] = 3.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[896] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[897] = 9.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[898] = 9.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[899] = 4e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[900] = 7.94e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[901] = 4e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[902] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[903] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[904] = 2.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[905] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[906] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[907] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[908] = 1.7e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[909] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[910] = 8.82e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[911] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[912] = 2.4e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[913] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[914] = 2.3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[915] = 1.14e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[916] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[917] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[918] = 2.35e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[919] = 2.16e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[920] = 2.08e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[921] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[922] = 3.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[923] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[924] = 7.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[925] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[926] = 1.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[927] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[928] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[929] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[930] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[931] = 1.9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[932] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[933] = 7.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>154.0 && Tgas<3000.0) { k[934] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-4640.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[935] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[936] = 1.1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[937] = 1.2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[938] = 1.6e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>15.0 && Tgas<41000.0) { k[939] = 4.89e-11 * pow(Tgas/300.0,
        -0.14) * exp(+36.1/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[940] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[941] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[942] = 7.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[943] = 4.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>80.0 && Tgas<510.0) { k[944] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[945] = 6.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>96.0 && Tgas<300.0) { k[946] = 6e-10 * pow(Tgas/300.0, 0.0) *
        exp(-2900.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[947] = 9e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>80.0 && Tgas<510.0) { k[948] = 1.3e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>212.0 && Tgas<300.0) { k[949] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-6380.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[950] = 3.7e-14 * pow(Tgas/300.0, 0.0) *
        exp(-35.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[951] = 1.5e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[952] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-85.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[953] = 2e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[954] = 2.25e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[955] = 1.28e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[956] = 2.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[957] = 3.09e-13 * pow(Tgas/300.0,
        1.08) * exp(+50.9/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[958] = 1.7e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>80.0 && Tgas<300.0) { k[959] = 6.4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[960] = 1.01e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>328.0 && Tgas<41000.0) { k[961] = 1.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-9860.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[962] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[963] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[964] = 3.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[965] = 2.16e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[966] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[967] = 7.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[968] = 5.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[969] = 2.1e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[970] = 2.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[971] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[972] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[973] = 9.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[974] = 3.35e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[975] = 3.35e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[976] = 4.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[977] = 4.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[978] = 5e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[979] = 6.62e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[980] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[981] = 7.74e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[982] = 5.4e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[983] = 2.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[984] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[985] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[986] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[987] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[988] = 4.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[989] = 2.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[990] = 4.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[991] = 2.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[992] = 8.5e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[993] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[994] = 7.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[995] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[996] = 1.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[997] = 8.84e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[998] = 2.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<520.0) { k[999] = 2e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1000] = 8.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1001] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1002] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1003] = 2.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1004] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1005] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1006] = 3.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1007] = 7.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1008] = 2.13e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1009] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1010] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1011] = 2.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1012] = 8.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1013] = 2.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1014] = 8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1015] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1016] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1017] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1018] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1019] = 2.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1020] = 2.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1021] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1022] = 1.8e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1023] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1024] = 4e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1025] = 1.7e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1026] = 9e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1027] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1028] = 1.7e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1029] = 2.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1030] = 1.05e-08 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1031] = 3.71e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1032] = 8.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1033] = 2.4e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1034] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1035] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1036] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<400.0) { k[1037] = 1.36e-09 * pow(Tgas/300.0,
        -0.14) * exp(+3.4/Tgas);  }
        
    if (Tgas>10.0 && Tgas<400.0) { k[1038] = 8.49e-10 * pow(Tgas/300.0,
        0.07) * exp(-5.2/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1039] = 2.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1040] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1041] = 6.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1042] = 2.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1043] = 5.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1044] = 3.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1045] = 8.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1046] = 1.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1047] = 3e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1048] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<520.0) { k[1049] = 3.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1050] = 8.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1051] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1052] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1053] = 1.9e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1054] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1055] = 1.8e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1056] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1057] = 4.39e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1058] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1059] = 7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1060] = 1.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1061] = 2.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<3000.0) { k[1062] = 9.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-100.0/Tgas);  }
        
    if (Tgas>5.0 && Tgas<400.0) { k[1063] = 3.42e-10 * pow(Tgas/300.0,
        -0.16) * exp(-1.4/Tgas);  }
        
    if (Tgas>5.0 && Tgas<400.0) { k[1064] = 7.98e-10 * pow(Tgas/300.0,
        -0.16) * exp(-1.4/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1065] = 1.9e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1066] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1067] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1068] = 2.6e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1069] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1070] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1071] = 3.7e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1072] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1073] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1074] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1075] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1076] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1077] = 8.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1078] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1079] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1080] = 9.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1081] = 2.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1082] = 1.8e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1083] = 4.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1084] = 2.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1085] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1086] = 3.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1087] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1088] = 3.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1089] = 3e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1090] = 4e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1091] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1092] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1093] = 1.8e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1094] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1095] = 9.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1096] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1097] = 6.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[1098] = 9.06e-10 * pow(Tgas/300.0,
        -0.37) * exp(-29.1/Tgas);  }
        
    if (Tgas>236.0 && Tgas<300.0) { k[1099] = 1e-09 * pow(Tgas/300.0, 0.0) *
        exp(-7080.0/Tgas);  }
        
    if (Tgas>352.0 && Tgas<41000.0) { k[1100] = 7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-10560.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1101] = 1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1102] = 1.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1103] = 2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1104] = 6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1105] = 1.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1106] = 9.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1107] = 4.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1108] = 1.9e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1109] = 1.9e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1110] = 2.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1111] = 1.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1112] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1113] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1114] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1115] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1116] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1117] = 5.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1118] = 5.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1119] = 1.04e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1120] = 2.3e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1121] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1122] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1123] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>205.0 && Tgas<565.0) { k[1124] = 3.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1125] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1126] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1127] = 6.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1128] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1129] = 9.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1130] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1131] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1132] = 1.05e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1133] = 1.05e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1134] = 1.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1135] = 1.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1136] = 2.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1137] = 1.4e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1138] = 4.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1139] = 2.7e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1140] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1141] = 3.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1142] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1143] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1144] = 7.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1145] = 2.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1146] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1147] = 8.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1148] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1149] = 1.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1150] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1151] = 3.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1152] = 7.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1153] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1154] = 1.4e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1155] = 8.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1156] = 7.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1157] = 3.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1158] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1159] = 7.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1160] = 7.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1161] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1162] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1163] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1164] = 8.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1165] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1166] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1167] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1168] = 3.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1169] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1170] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1171] = 3.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1172] = 9.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1173] = 1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1174] = 1.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1175] = 4.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1176] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1177] = 1.6e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1178] = 1.61e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1179] = 8.75e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1180] = 7.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1181] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1182] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1183] = 4.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1184] = 2.2e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1185] = 4.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1186] = 5.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1187] = 5.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1188] = 5.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1189] = 8e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1190] = 8e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1191] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1192] = 7.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1193] = 7.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1194] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1195] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1196] = 1.8e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1197] = 5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1198] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1199] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1200] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1201] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1202] = 2.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1203] = 9.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1204] = 8.5e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1205] = 4.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1206] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1207] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1208] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1209] = 8.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1210] = 1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1211] = 1.1e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1212] = 4e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1213] = 1.6e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1214] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1215] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1216] = 1.88e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1217] = 1.14e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1218] = 1.71e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1219] = 8.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1220] = 8.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1221] = 8.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1222] = 2.86e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1223] = 2.04e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1224] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1225] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1226] = 4.84e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1227] = 3.61e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1228] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1229] = 2.84e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1230] = 2.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1231] = 1.46e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1232] = 2.17e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1233] = 7.75e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1234] = 6.51e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1235] = 4.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1236] = 3e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1237] = 4.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1238] = 3e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1239] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1240] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1241] = 3.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1242] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1243] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1244] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1245] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1246] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1247] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1248] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1249] = 1.7e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1250] = 9.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1251] = 2.7e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1252] = 8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1253] = 8e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1254] = 1.76e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1255] = 1.76e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1256] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1257] = 2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1258] = 1.4e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1259] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1260] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1261] = 1.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1262] = 3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1263] = 3e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1264] = 7.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1265] = 7.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1266] = 7.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1267] = 7.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1268] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1269] = 2e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1270] = 9e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1271] = 2.97e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1272] = 8.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1273] = 8.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1274] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1275] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1276] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1277] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1278] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1279] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1280] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1281] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1282] = 1.41e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1283] = 9.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1284] = 1.8e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1285] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1286] = 8.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1287] = 3.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1288] = 3.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1289] = 9.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1290] = 4.96e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1291] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1292] = 1.24e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1293] = 4.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1294] = 5.6e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1295] = 3.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1296] = 2.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1297] = 1.45e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1298] = 7.25e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1299] = 2.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1300] = 5.51e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1301] = 5.7e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1302] = 2.28e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1303] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1304] = 3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1305] = 1.4e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1306] = 2.16e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1307] = 2.16e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1308] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1309] = 7.9e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1310] = 2.63e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1311] = 3.66e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1312] = 7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1313] = 3.08e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1314] = 2.52e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1315] = 1.13e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1316] = 2.25e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1317] = 3.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1318] = 1.04e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1319] = 1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1320] = 8e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1321] = 4.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1322] = 9.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1323] = 3.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1324] = 1.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1325] = 4e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1326] = 1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1327] = 9e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1328] = 7.5e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1329] = 1.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1330] = 2.5e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1331] = 2.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1332] = 6.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1333] = 1.12e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1334] = 2.8e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1335] = 7.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1336] = 7.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1337] = 1.3e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1338] = 9.1e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1339] = 1.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1340] = 8.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1341] = 5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1342] = 7.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1343] = 9e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1344] = 2.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1345] = 4.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1346] = 4.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1347] = 4.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1348] = 1.4e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1349] = 1.6e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1350] = 3.85e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1351] = 3.85e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1352] = 3.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1353] = 4.41e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1354] = 4.95e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1355] = 1.82e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1356] = 1.05e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1357] = 3.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1358] = 1.75e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1359] = 8.75e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1360] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1361] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1362] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1363] = 6.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1364] = 1.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1365] = 6e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1366] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1367] = 1.78e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1368] = 2.05e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1369] = 1.64e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1370] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1371] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1372] = 6.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1373] = 6.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1374] = 9.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1375] = 9.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1376] = 2.24e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1377] = 5.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1378] = 2.76e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1379] = 1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1380] = 1.45e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1381] = 2.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1382] = 1.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1383] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1384] = 1.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1385] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1386] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1387] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1388] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1389] = 1.61e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1390] = 1.19e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1391] = 2.1e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1392] = 4.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1393] = 4.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1394] = 4.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1395] = 4.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1396] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1397] = 9.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1398] = 4.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1399] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1400] = 4.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1401] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1402] = 9.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1403] = 9e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1404] = 4.45e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1405] = 4.45e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1406] = 8.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1407] = 8.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1408] = 8.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1409] = 1e-11 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1410] = 8.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1411] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1412] = 1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1413] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1414] = 1.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1415] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1416] = 4.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1417] = 2.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1418] = 5.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1419] = 9.61e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1420] = 1.97e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1421] = 1.8e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1422] = 2.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1423] = 4.08e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1424] = 1.28e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1425] = 9.45e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1426] = 1.36e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1427] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1428] = 2.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1429] = 1.9e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1430] = 8.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1431] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1432] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1433] = 2.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1434] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1435] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1436] = 1.1e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1437] = 9.75e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1438] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1439] = 9.7e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1440] = 2.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1441] = 2e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1442] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1443] = 2.5e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1444] = 3.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1445] = 3.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1446] = 7.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1447] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1448] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1449] = 6.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1450] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1451] = 6.5e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1452] = 6.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1453] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1454] = 6.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1455] = 7.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1456] = 7.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1457] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1458] = 3.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1459] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1460] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1461] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1462] = 7.7e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1463] = 4.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1464] = 1.12e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1465] = 4.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1466] = 9.5e-11 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1467] = 1.33e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1468] = 1.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1469] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1470] = 9.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1471] = 1.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1472] = 4.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1473] = 2.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>205.0 && Tgas<565.0) { k[1474] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>20.0 && Tgas<5565.0) { k[1475] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1476] = 4.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>23.0 && Tgas<41000.0) { k[1477] = 2.42e-12 * pow(Tgas/300.0,
        -0.21) * exp(+44.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1478] = 8.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1479] = 2e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1480] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1481] = 3.6e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1482] = 6.5e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1483] = 5e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1484] = 5.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1485] = 1.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1486] = 1.5e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1487] = 6.23e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1488] = 2.67e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1489] = 1.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1490] = 3.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1491] = 3.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1492] = 8.5e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1493] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1494] = 4.4e-12 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1495] = 2.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1496] = 6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1497] = 4e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1498] = 3.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1499] = 3.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1500] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1501] = 5e-12 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1502] = 5e-12 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1503] = 2.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1504] = 2.9e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1505] = 1.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1506] = 1.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1507] = 7.2e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1508] = 1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1509] = 6.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1510] = 6.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1511] = 7.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1512] = 6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1513] = 4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1514] = 6.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1515] = 2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1516] = 2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1517] = 4.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1518] = 4.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1519] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1520] = 1.44e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1521] = 1.05e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1522] = 1.12e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1523] = 1.3e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1524] = 8.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1525] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1526] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1527] = 2.8e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1528] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1529] = 3.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1530] = 1.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1531] = 6.11e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1532] = 7e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1533] = 4.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1534] = 4.3e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1535] = 1.9e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1536] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1537] = 9.4e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1538] = 7e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1539] = 3.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1540] = 6.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1541] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1542] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1543] = 1e-09 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1544] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1545] = 6.2e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1546] = 7e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1547] = 6.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1548] = 6.1e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1549] = 6.3e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1550] = 1.89e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1551] = 5.9e-10 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1552] = 9.1e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1553] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1554] = 1.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1555] = 1e-09 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1556] = 6.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1557] = 1.1e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1558] = 2.42e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1559] = 2.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1560] = 6.6e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1561] = 6.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1562] = 2.69e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1563] = 2.2e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1564] = 1.65e-09 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1565] = 9e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1566] = 1.6e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1567] = 2.4e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1568] = 1.1e-09 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1569] = 4.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>24.0 && Tgas<300.0) { k[1570] = 2.88e-10 * pow(Tgas/300.0,
        -1.14) * exp(-77.0/Tgas);  }
        
    if (Tgas>90.0 && Tgas<200.0) { k[1571] = 1.11e-10 * pow(Tgas/300.0,
        -0.82) * exp(-9.7/Tgas);  }
        
    if (Tgas>298.0 && Tgas<1300.0) { k[1572] = 1.5e-11 * pow(Tgas/300.0,
        0.0) * exp(-4300.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1573] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>1100.0 && Tgas<1500.0) { k[1574] = 8.97e-12 * pow(Tgas/300.0,
        0.0) * exp(-18973.0/Tgas);  }
        
    if (Tgas>15.0 && Tgas<300.0) { k[1575] = 1.3e-10 * pow(Tgas/300.0,
        -0.71) * exp(-29.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<362.0) { k[1576] = 8.87e-12 * pow(Tgas/300.0,
        -0.73) * exp(-22.7/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3500.0) { k[1577] = 3.15e-14 * pow(Tgas/300.0,
        1.45) * exp(+52.0/Tgas);  }
        
    if (Tgas>297.0 && Tgas<360.0) { k[1578] = 5.3e-12 * pow(Tgas/300.0, 0.0)
        * exp(-770.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1579] = 2.5e-10 * pow(Tgas/300.0, -0.2)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1580] = 2e-13 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1500.0) { k[1581] = 3.67e-11 * pow(Tgas/300.0,
        -0.35) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1582] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1583] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1584] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1585] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1586] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<4000.0) { k[1587] = 2.69e-12 * pow(Tgas/300.0,
        0.0) * exp(-23550.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1588] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1589] = 6.59e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>603.0 && Tgas<41000.0) { k[1590] = 4.98e-10 * pow(Tgas/300.0,
        0.0) * exp(-18116.0/Tgas);  }
        
    if (Tgas>1934.0 && Tgas<41000.0) { k[1591] = 2.94e-11 * pow(Tgas/300.0,
        0.5) * exp(-58025.0/Tgas);  }
        
    if (Tgas>681.0 && Tgas<41000.0) { k[1592] = 1.44e-11 * pow(Tgas/300.0,
        0.5) * exp(-20435.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1593] = 2e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1594] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1595] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>196.0 && Tgas<300.0) { k[1596] = 1.2e-11 * pow(Tgas/300.0,
        0.58) * exp(-5880.0/Tgas);  }
        
    if (Tgas>2000.0 && Tgas<5000.0) { k[1597] = 8.69e-11 * pow(Tgas/300.0,
        0.0) * exp(-22600.0/Tgas);  }
        
    k[1598] = 3e-11 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>10.0 && Tgas<300.0) { k[1599] = 3.26e-11 * pow(Tgas/300.0,
        -0.1) * exp(+9.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1600] = 3.26e-11 * pow(Tgas/300.0,
        -0.1) * exp(+9.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<4000.0) { k[1601] = 9.62e-13 * pow(Tgas/300.0,
        0.0) * exp(-10517.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1602] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>133.0 && Tgas<300.0) { k[1603] = 1.73e-11 * pow(Tgas/300.0,
        0.5) * exp(-4000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1604] = 6e-11 * pow(Tgas/300.0,
        -0.16) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1605] = 9e-11 * pow(Tgas/300.0, -0.16)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>133.0 && Tgas<300.0) { k[1606] = 1.73e-11 * pow(Tgas/300.0,
        0.5) * exp(-4000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1607] = 1.5e-10 * pow(Tgas/300.0,
        -0.16) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<8000.0) { k[1608] = 5.56e-11 * pow(Tgas/300.0,
        0.41) * exp(+26.9/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1609] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1610] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1611] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>493.0 && Tgas<41000.0) { k[1612] = 2.25e-11 * pow(Tgas/300.0,
        0.5) * exp(-14800.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1613] = 7e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1614] = 7e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1615] = 3.5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1616] = 3.5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1617] = 6.59e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1100.0 && Tgas<2720.0) { k[1618] = 2.63e-09 * pow(Tgas/300.0,
        0.0) * exp(-6013.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3000.0) { k[1619] = 1.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-400.0/Tgas);  }
        
    if (Tgas>1600.0 && Tgas<2300.0) { k[1620] = 3.32e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1621] = 4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-5000.0/Tgas);  }
        
    if (Tgas>296.0 && Tgas<2500.0) { k[1622] = 7.13e-12 * pow(Tgas/300.0,
        0.0) * exp(-5050.0/Tgas);  }
        
    if (Tgas>83.0 && Tgas<300.0) { k[1623] = 5.3e-12 * pow(Tgas/300.0, 0.0)
        * exp(-2500.0/Tgas);  }
        
    if (Tgas>109.0 && Tgas<300.0) { k[1624] = 3.3e-13 * pow(Tgas/300.0, 0.0)
        * exp(-3270.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1625] = 3e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1626] = 1.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1500.0 && Tgas<2000.0) { k[1627] = 8e-12 * pow(Tgas/300.0, 0.0)
        * exp(-18000.0/Tgas);  }
        
    k[1628] = 6.91e-11 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>116.0 && Tgas<300.0) { k[1629] = 2.7e-12 * pow(Tgas/300.0, 0.0)
        * exp(-3500.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[1630] = 3.65e-12 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1631] = 3.65e-12 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<1828.0) { k[1632] = 2.92e-11 * pow(Tgas/300.0,
        -3.3) * exp(-1443.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<1828.0) { k[1633] = 3.65e-11 * pow(Tgas/300.0,
        -3.3) * exp(-1443.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<1828.0) { k[1634] = 2.48e-10 * pow(Tgas/300.0,
        -3.3) * exp(-1443.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<1828.0) { k[1635] = 3.65e-11 * pow(Tgas/300.0,
        -3.3) * exp(-1443.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1636] = 4.1e-11 * pow(Tgas/300.0,
        0.0) * exp(-750.0/Tgas);  }
        
    if (Tgas>1900.0 && Tgas<2600.0) { k[1637] = 8e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1638] = 1.33e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>1200.0 && Tgas<1812.0) { k[1639] = 5.01e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>1900.0 && Tgas<2300.0) { k[1640] = 4.98e-10 * pow(Tgas/300.0,
        0.0) * exp(-6000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1641] = 3e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<300.0) { k[1642] = 1.44e-11 * pow(Tgas/300.0,
        0.5) * exp(-3000.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<300.0) { k[1643] = 1.44e-11 * pow(Tgas/300.0,
        0.5) * exp(-3000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1644] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1645] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<900.0) { k[1646] = 1.7e-10 * pow(Tgas/300.0,
        -1.5) * exp(-300.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2651.0) { k[1647] = 1.66e-08 * pow(Tgas/300.0,
        0.0) * exp(-16556.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3000.0) { k[1648] = 1.46e-11 * pow(Tgas/300.0,
        0.1) * exp(-5335.0/Tgas);  }
        
    if (Tgas>1950.0 && Tgas<2300.0) { k[1649] = 7.13e-12 * pow(Tgas/300.0,
        0.0) * exp(-5052.0/Tgas);  }
        
    if (Tgas>50.0 && Tgas<300.0) { k[1650] = 9.21e-12 * pow(Tgas/300.0, 0.7)
        * exp(-1500.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1651] = 1.34e-15 * pow(Tgas/300.0,
        5.05) * exp(-1636.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3000.0) { k[1652] = 2.3e-15 * pow(Tgas/300.0,
        3.47) * exp(-6681.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1653] = 3.3e-13 * pow(Tgas/300.0,
        0.0) * exp(-1105.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1654] = 2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<913.0) { k[1655] = 3.32e-12 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[1656] = 4.76e-17 * pow(Tgas/300.0,
        5.77) * exp(+151.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1657] = 9.55e-14 * pow(Tgas/300.0,
        0.0) * exp(-4890.0/Tgas);  }
        
    k[1658] = 5.41e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>1100.0 && Tgas<2080.0) { k[1659] = 4e-12 * pow(Tgas/300.0, 0.0)
        * exp(-7900.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3000.0) { k[1660] = 5.64e-13 * pow(Tgas/300.0,
        0.0) * exp(-4500.0/Tgas);  }
        
    if (Tgas>1700.0 && Tgas<2000.0) { k[1661] = 1.66e-12 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>1165.0 && Tgas<41000.0) { k[1662] = 5.3e-12 * pow(Tgas/300.0,
        0.0) * exp(-34975.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1663] = 6e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>354.0 && Tgas<925.0) { k[1664] = 3.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-202.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1665] = 1.3e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1666] = 3.27e-14 * pow(Tgas/300.0,
        2.2) * exp(-2240.0/Tgas);  }
        
    k[1667] = 1.7e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>300.0 && Tgas<1000.0) { k[1668] = 1.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-1400.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1669] = 1.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>160.0 && Tgas<2500.0) { k[1670] = 3.14e-12 * pow(Tgas/300.0,
        1.53) * exp(-504.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1671] = 6.7e-11 * pow(Tgas/300.0,
        0.0) * exp(-28640.0/Tgas);  }
        
    if (Tgas>178.0 && Tgas<3000.0) { k[1672] = 3.77e-13 * pow(Tgas/300.0,
        2.42) * exp(-1162.0/Tgas);  }
        
    if (Tgas>1140.0 && Tgas<1480.0) { k[1673] = 3.39e-10 * pow(Tgas/300.0,
        0.0) * exp(-10019.0/Tgas);  }
        
    if (Tgas>23.0 && Tgas<710.0) { k[1674] = 3.39e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>23.0 && Tgas<726.0) { k[1675] = 2.21e-10 * pow(Tgas/300.0,
        -0.62) * exp(-32.9/Tgas);  }
        
    if (Tgas>23.0 && Tgas<300.0) { k[1676] = 1.05e-10 * pow(Tgas/300.0,
        -1.04) * exp(-36.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<300.0) { k[1677] = 2.94e-13 * pow(Tgas/300.0,
        0.5) * exp(-3000.0/Tgas);  }
        
    if (Tgas>66.0 && Tgas<300.0) { k[1678] = 9.21e-12 * pow(Tgas/300.0, 0.7)
        * exp(-2000.0/Tgas);  }
        
    if (Tgas>16.0 && Tgas<300.0) { k[1679] = 2.87e-12 * pow(Tgas/300.0, 0.7)
        * exp(-500.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1680] = 1.73e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<4000.0) { k[1681] = 5.6e-13 * pow(Tgas/300.0,
        0.88) * exp(-10128.0/Tgas);  }
        
    if (Tgas>222.0 && Tgas<584.0) { k[1682] = 1.66e-10 * pow(Tgas/300.0,
        -0.09) * exp(-0.0/Tgas);  }
        
    if (Tgas>990.0 && Tgas<1100.0) { k[1683] = 3.03e-11 * pow(Tgas/300.0,
        0.65) * exp(-1207.0/Tgas);  }
        
    if (Tgas>13.0 && Tgas<708.0) { k[1684] = 1.2e-10 * pow(Tgas/300.0,
        -0.13) * exp(-0.0/Tgas);  }
        
    if (Tgas>13.0 && Tgas<708.0) { k[1685] = 1.16e-11 * pow(Tgas/300.0,
        -0.13) * exp(-0.0/Tgas);  }
        
    if (Tgas>13.0 && Tgas<708.0) { k[1686] = 3.49e-11 * pow(Tgas/300.0,
        -0.13) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1687] = 1.14e-11 * pow(Tgas/300.0,
        -0.48) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1688] = 1.14e-11 * pow(Tgas/300.0,
        -0.48) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1689] = 7.6e-12 * pow(Tgas/300.0,
        -0.48) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1690] = 7.6e-12 * pow(Tgas/300.0,
        -0.48) * exp(-0.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<300.0) { k[1691] = 1.44e-11 * pow(Tgas/300.0,
        0.5) * exp(-3000.0/Tgas);  }
        
    if (Tgas>251.0 && Tgas<300.0) { k[1692] = 2.94e-13 * pow(Tgas/300.0,
        0.5) * exp(-7550.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2000.0) { k[1693] = 6.02e-11 * pow(Tgas/300.0,
        0.1) * exp(+4.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<6000.0) { k[1694] = 2.52e-11 * pow(Tgas/300.0,
        0.0) * exp(-2381.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1695] = 4e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>166.0 && Tgas<300.0) { k[1696] = 1.44e-11 * pow(Tgas/300.0,
        0.5) * exp(-5000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1697] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>133.0 && Tgas<300.0) { k[1698] = 1.73e-11 * pow(Tgas/300.0,
        0.5) * exp(-4000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1699] = 9e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1700] = 1.1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1701] = 2.72e-10 * pow(Tgas/300.0,
        -0.52) * exp(-19.0/Tgas);  }
        
    if (Tgas>25.0 && Tgas<2500.0) { k[1702] = 1.25e-10 * pow(Tgas/300.0,
        0.7) * exp(-30.0/Tgas);  }
        
    if (Tgas>3800.0 && Tgas<7000.0) { k[1703] = 2.66e-09 * pow(Tgas/300.0,
        0.0) * exp(-21638.0/Tgas);  }
        
    if (Tgas>297.0 && Tgas<2500.0) { k[1704] = 2.6e-10 * pow(Tgas/300.0,
        -0.47) * exp(-826.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1705] = 2.5e-17 * pow(Tgas/300.0,
        1.71) * exp(-770.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1706] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1707] = 2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1708] = 3e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>297.0 && Tgas<2500.0) { k[1709] = 7.02e-11 * pow(Tgas/300.0,
        -0.27) * exp(-8.3/Tgas);  }
        
    if (Tgas>300.0 && Tgas<1500.0) { k[1710] = 1.6e-13 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<4430.0) { k[1711] = 1.62e-10 * pow(Tgas/300.0,
        0.0) * exp(-21205.0/Tgas);  }
        
    if (Tgas>13.0 && Tgas<1565.0) { k[1712] = 5.12e-12 * pow(Tgas/300.0,
        -0.49) * exp(+5.2/Tgas);  }
        
    if (Tgas>13.0 && Tgas<4526.0) { k[1713] = 2.02e-11 * pow(Tgas/300.0,
        -0.19) * exp(+31.9/Tgas);  }
        
    if (Tgas>1067.0 && Tgas<41000.0) { k[1714] = 5.71e-11 * pow(Tgas/300.0,
        0.5) * exp(-32010.0/Tgas);  }
        
    k[1715] = 2.2e-10 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>10.0 && Tgas<1900.0) { k[1716] = 3.32e-12 * pow(Tgas/300.0,
        0.0) * exp(-6170.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1717] = 1.48e-10 * pow(Tgas/300.0,
        0.0) * exp(-17000.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<6000.0) { k[1718] = 5.99e-12 * pow(Tgas/300.0,
        0.0) * exp(-24075.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<2500.0) { k[1719] = 5.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-12160.0/Tgas);  }
        
    if (Tgas>354.0 && Tgas<2939.0) { k[1720] = 5.27e-12 * pow(Tgas/300.0,
        1.4) * exp(-1760.0/Tgas);  }
        
    if (Tgas>220.0 && Tgas<2200.0) { k[1721] = 9.11e-13 * pow(Tgas/300.0,
        2.57) * exp(-130.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1722] = 6.64e-10 * pow(Tgas/300.0,
        0.0) * exp(-11700.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1723] = 5.18e-11 * pow(Tgas/300.0,
        0.17) * exp(-6400.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1724] = 6.86e-14 * pow(Tgas/300.0,
        2.74) * exp(-4740.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1725] = 5.46e-10 * pow(Tgas/300.0,
        0.0) * exp(-1943.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3500.0) { k[1726] = 4.04e-13 * pow(Tgas/300.0,
        2.87) * exp(-820.0/Tgas);  }
        
    if (Tgas>268.0 && Tgas<300.0) { k[1727] = 6.52e-12 * pow(Tgas/300.0,
        0.09) * exp(-8050.0/Tgas);  }
        
    if (Tgas>1600.0 && Tgas<2850.0) { k[1728] = 1.69e-09 * pow(Tgas/300.0,
        0.0) * exp(-18095.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1729] = 2.05e-15 * pow(Tgas/300.0,
        3.89) * exp(-1400.0/Tgas);  }
        
    if (Tgas>2601.0 && Tgas<2788.0) { k[1730] = 5.96e-11 * pow(Tgas/300.0,
        0.0) * exp(-7782.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1731] = 2.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-28500.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1732] = 3.16e-10 * pow(Tgas/300.0,
        0.0) * exp(-21890.0/Tgas);  }
        
    if (Tgas>297.0 && Tgas<3532.0) { k[1733] = 3.14e-13 * pow(Tgas/300.0,
        2.7) * exp(-3150.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<2581.0) { k[1734] = 2.05e-12 * pow(Tgas/300.0,
        1.52) * exp(-1736.0/Tgas);  }
        
    if (Tgas>910.0 && Tgas<3137.0) { k[1735] = 1.76e-13 * pow(Tgas/300.0,
        2.88) * exp(-6126.0/Tgas);  }
        
    if (Tgas>1015.0 && Tgas<41000.0) { k[1736] = 4.67e-10 * pow(Tgas/300.0,
        0.5) * exp(-30450.0/Tgas);  }
        
    if (Tgas>295.0 && Tgas<6920.0) { k[1737] = 3.8e-10 * pow(Tgas/300.0,
        0.0) * exp(-13634.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1738] = 3.32e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1739] = 2.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2000.0) { k[1740] = 5.68e-11 * pow(Tgas/300.0,
        0.0) * exp(-1897.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1741] = 1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-7600.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1742] = 5.94e-13 * pow(Tgas/300.0,
        3.0) * exp(-4045.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[1743] = 1.31e-10 * pow(Tgas/300.0,
        0.0) * exp(-80.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<15223.0) { k[1744] = 3.38e-10 * pow(Tgas/300.0,
        0.0) * exp(-13163.0/Tgas);  }
        
    if (Tgas>2590.0 && Tgas<41000.0) { k[1745] = 1.1e-10 * pow(Tgas/300.0,
        0.5) * exp(-77700.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1746] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<5000.0) { k[1747] = 4.85e-12 * pow(Tgas/300.0,
        1.9) * exp(-1379.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<3000.0) { k[1748] = 1.59e-11 * pow(Tgas/300.0,
        1.2) * exp(-9610.0/Tgas);  }
        
    if (Tgas>190.0 && Tgas<2560.0) { k[1749] = 3.71e-12 * pow(Tgas/300.0,
        1.94) * exp(-455.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1750] = 6.2e-10 * pow(Tgas/300.0,
        0.0) * exp(-12500.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1751] = 1.5e-10 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>1200.0 && Tgas<1812.0) { k[1752] = 6.61e-11 * pow(Tgas/300.0,
        0.0) * exp(-51598.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1753] = 1.5e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[1754] = 1.14e-13 * pow(Tgas/300.0,
        4.23) * exp(+114.6/Tgas);  }
        
    if (Tgas>550.0 && Tgas<3000.0) { k[1755] = 1.05e-09 * pow(Tgas/300.0,
        -0.3) * exp(-14730.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3000.0) { k[1756] = 4.5e-11 * pow(Tgas/300.0,
        0.72) * exp(-329.0/Tgas);  }
        
    if (Tgas>350.0 && Tgas<3000.0) { k[1757] = 2.4e-09 * pow(Tgas/300.0,
        -0.5) * exp(-9010.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1758] = 2.5e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2900.0) { k[1759] = 1.48e-10 * pow(Tgas/300.0,
        0.0) * exp(-3588.0/Tgas);  }
        
    if (Tgas>73.0 && Tgas<3000.0) { k[1760] = 4.56e-12 * pow(Tgas/300.0,
        1.02) * exp(-2161.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1761] = 7.8e-13 * pow(Tgas/300.0,
        2.4) * exp(-4990.0/Tgas);  }
        
    if (Tgas>80.0 && Tgas<300.0) { k[1762] = 1.73e-11 * pow(Tgas/300.0, 0.5)
        * exp(-2400.0/Tgas);  }
        
    if (Tgas>24.0 && Tgas<300.0) { k[1763] = 1.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-740.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3000.0) { k[1764] = 9.29e-10 * pow(Tgas/300.0,
        -0.1) * exp(-35220.0/Tgas);  }
        
    if (Tgas>1500.0 && Tgas<4524.0) { k[1765] = 3.6e-10 * pow(Tgas/300.0,
        0.0) * exp(-24910.0/Tgas);  }
        
    if (Tgas>523.0 && Tgas<41000.0) { k[1766] = 7.27e-11 * pow(Tgas/300.0,
        0.5) * exp(-15700.0/Tgas);  }
        
    if (Tgas>691.0 && Tgas<41000.0) { k[1767] = 7.27e-11 * pow(Tgas/300.0,
        0.5) * exp(-20735.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<4000.0) { k[1768] = 2.61e-10 * pow(Tgas/300.0,
        0.0) * exp(-8156.0/Tgas);  }
        
    if (Tgas>245.0 && Tgas<2500.0) { k[1769] = 5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-866.0/Tgas);  }
        
    if (Tgas>230.0 && Tgas<2500.0) { k[1770] = 2.06e-11 * pow(Tgas/300.0,
        0.84) * exp(-277.0/Tgas);  }
        
    if (Tgas>230.0 && Tgas<2500.0) { k[1771] = 1.66e-10 * pow(Tgas/300.0,
        0.0) * exp(-413.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1772] = 1.87e-11 * pow(Tgas/300.0,
        0.9) * exp(-2924.0/Tgas);  }
        
    if (Tgas>295.0 && Tgas<1490.0) { k[1773] = 1.26e-10 * pow(Tgas/300.0,
        0.0) * exp(-515.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1774] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>261.0 && Tgas<525.0) { k[1775] = 1.23e-11 * pow(Tgas/300.0,
        0.0) * exp(-1949.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1776] = 6.99e-14 * pow(Tgas/300.0,
        2.8) * exp(-1950.0/Tgas);  }
        
    if (Tgas>278.0 && Tgas<300.0) { k[1777] = 2.25e-10 * pow(Tgas/300.0,
        0.5) * exp(-8355.0/Tgas);  }
        
    if (Tgas>664.0 && Tgas<41000.0) { k[1778] = 1.73e-11 * pow(Tgas/300.0,
        0.5) * exp(-19930.0/Tgas);  }
        
    if (Tgas>370.0 && Tgas<41000.0) { k[1779] = 5.9e-10 * pow(Tgas/300.0,
        -0.31) * exp(-11100.0/Tgas);  }
        
    if (Tgas>303.0 && Tgas<376.0) { k[1780] = 3.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1781] = 3e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1782] = 1e-12 * pow(Tgas/300.0, 0.0)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>295.0 && Tgas<2500.0) { k[1783] = 1.2e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    k[1784] = 7.6e-13 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1785] = 4.64e-12 * pow(Tgas/300.0,
        0.7) * exp(+25.6/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1786] = 5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<1200.0) { k[1787] = 2.43e-12 * pow(Tgas/300.0,
        1.44) * exp(-1240.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1788] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>293.0 && Tgas<2500.0) { k[1789] = 1.3e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1790] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1791] = 1.2e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>291.0 && Tgas<701.0) { k[1792] = 3.69e-14 * pow(Tgas/300.0,
        0.0) * exp(-161.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1793] = 7.15e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1794] = 3.85e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1795] = 1.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1796] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1797] = 1e-13 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1798] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1799] = 1.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1800] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1801] = 3.95e-11 * pow(Tgas/300.0,
        0.17) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1802] = 3.95e-11 * pow(Tgas/300.0,
        0.17) * exp(-0.0/Tgas);  }
        
    if (Tgas>1000.0 && Tgas<4000.0) { k[1803] = 9.96e-13 * pow(Tgas/300.0,
        0.0) * exp(-20380.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2000.0) { k[1804] = 7.4e-11 * pow(Tgas/300.0,
        0.26) * exp(-8.4/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1805] = 1.3e-11 * pow(Tgas/300.0, 0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>253.0 && Tgas<352.0) { k[1806] = 3.32e-13 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<299.0) { k[1807] = 1e-10 * pow(Tgas/300.0, 0.18) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>291.0 && Tgas<523.0) { k[1808] = 3.2e-13 * pow(Tgas/300.0, 0.0)
        * exp(-1710.0/Tgas);  }
        
    if (Tgas>38.0 && Tgas<300.0) { k[1809] = 3.8e-11 * pow(Tgas/300.0, 0.5)
        * exp(-1160.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1810] = 1e-10 * pow(Tgas/300.0, 0.0)
        * exp(-200.0/Tgas);  }
        
    if (Tgas>33.0 && Tgas<300.0) { k[1811] = 5.71e-12 * pow(Tgas/300.0, 0.5)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1812] = 1.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1813] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1814] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>33.0 && Tgas<300.0) { k[1815] = 2.94e-12 * pow(Tgas/300.0, 0.5)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1816] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>302.0 && Tgas<41000.0) { k[1817] = 1.73e-11 * pow(Tgas/300.0,
        0.5) * exp(-9060.0/Tgas);  }
        
    k[1818] = 1e-11 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>10.0 && Tgas<1400.0) { k[1819] = 4.98e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    k[1820] = 2.41e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>10.0 && Tgas<300.0) { k[1821] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1822] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>100.0 && Tgas<2500.0) { k[1823] = 3.38e-11 * pow(Tgas/300.0,
        -0.17) * exp(+2.8/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1824] = 3e-11 * pow(Tgas/300.0, -0.6) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<14000.0) { k[1825] = 2.26e-12 * pow(Tgas/300.0,
        0.86) * exp(-3134.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1826] = 1.7e-13 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>5.0 && Tgas<2500.0) { k[1827] = 6.05e-11 * pow(Tgas/300.0,
        -0.23) * exp(-14.9/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1828] = 1.88e-11 * pow(Tgas/300.0,
        0.1) * exp(-10700.0/Tgas);  }
        
    if (Tgas>133.0 && Tgas<300.0) { k[1829] = 1.73e-11 * pow(Tgas/300.0,
        0.5) * exp(-4000.0/Tgas);  }
        
    if (Tgas>275.0 && Tgas<300.0) { k[1830] = 4.68e-11 * pow(Tgas/300.0,
        0.5) * exp(-8254.0/Tgas);  }
        
    if (Tgas>25.0 && Tgas<300.0) { k[1831] = 1.73e-11 * pow(Tgas/300.0, 0.5)
        * exp(-750.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1832] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1833] = 2.9e-13 * pow(Tgas/300.0,
        2.87) * exp(-5380.0/Tgas);  }
        
    if (Tgas>210.0 && Tgas<3000.0) { k[1834] = 4.27e-11 * pow(Tgas/300.0,
        -2.5) * exp(-331.0/Tgas);  }
        
    if (Tgas>1500.0 && Tgas<2150.0) { k[1835] = 1.49e-12 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<3000.0) { k[1836] = 1.35e-12 * pow(Tgas/300.0,
        1.25) * exp(+43.5/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1837] = 2.08e-13 * pow(Tgas/300.0,
        0.76) * exp(-262.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1838] = 2.75e-11 * pow(Tgas/300.0,
        -1.14) * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[1839] = 6.63e-16 * pow(Tgas/300.0,
        6.13) * exp(-5895.0/Tgas);  }
        
    if (Tgas>33.0 && Tgas<300.0) { k[1840] = 2.94e-12 * pow(Tgas/300.0, 0.5)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>600.0 && Tgas<3000.0) { k[1841] = 1.83e-12 * pow(Tgas/300.0,
        1.6) * exp(-14090.0/Tgas);  }
        
    if (Tgas>1300.0 && Tgas<1700.0) { k[1842] = 5.25e-10 * pow(Tgas/300.0,
        0.0) * exp(-13470.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1843] = 1.7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    k[1844] = 1.16e-09 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>300.0 && Tgas<3000.0) { k[1845] = 1.81e-13 * pow(Tgas/300.0,
        1.8) * exp(+70.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1846] = 2.44e-11 * pow(Tgas/300.0,
        -1.94) * exp(-56.9/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3300.0) { k[1847] = 7.4e-10 * pow(Tgas/300.0,
        0.0) * exp(-10540.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3300.0) { k[1848] = 1.33e-11 * pow(Tgas/300.0,
        -0.78) * exp(-40.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3500.0) { k[1849] = 6.88e-14 * pow(Tgas/300.0,
        2.07) * exp(-3281.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3300.0) { k[1850] = 2.54e-14 * pow(Tgas/300.0,
        1.18) * exp(-312.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<299.0) { k[1851] = 6.6e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<3000.0) { k[1852] = 1.16e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3000.0) { k[1853] = 3.11e-12 * pow(Tgas/300.0,
        1.2) * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3000.0) { k[1854] = 3.32e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<3000.0) { k[1855] = 2.93e-12 * pow(Tgas/300.0,
        0.1) * exp(-5800.0/Tgas);  }
        
    if (Tgas>133.0 && Tgas<300.0) { k[1856] = 1.73e-11 * pow(Tgas/300.0,
        0.5) * exp(-4000.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1857] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>1573.0 && Tgas<4700.0) { k[1858] = 2.51e-11 * pow(Tgas/300.0,
        0.0) * exp(-30653.0/Tgas);  }
        
    if (Tgas>780.0 && Tgas<41000.0) { k[1859] = 2.8e-12 * pow(Tgas/300.0,
        0.0) * exp(-23400.0/Tgas);  }
        
    if (Tgas>290.0 && Tgas<2660.0) { k[1860] = 4.55e-11 * pow(Tgas/300.0,
        -1.33) * exp(-242.0/Tgas);  }
        
    if (Tgas>582.0 && Tgas<41000.0) { k[1861] = 2.94e-11 * pow(Tgas/300.0,
        0.5) * exp(-17465.0/Tgas);  }
        
    if (Tgas>2420.0 && Tgas<3870.0) { k[1862] = 1.75e-10 * pow(Tgas/300.0,
        0.0) * exp(-20200.0/Tgas);  }
        
    k[1863] = 1.32e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>25.0 && Tgas<300.0) { k[1864] = 8.1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-773.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3460.0) { k[1865] = 1.76e-12 * pow(Tgas/300.0,
        0.81) * exp(+30.8/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2980.0) { k[1866] = 1.1e-14 * pow(Tgas/300.0,
        1.89) * exp(-1538.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<8000.0) { k[1867] = 2e-10 * pow(Tgas/300.0, -0.12)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1868] = 1.15e-12 * pow(Tgas/300.0,
        1.4) * exp(-1110.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1869] = 1.6e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>290.0 && Tgas<2300.0) { k[1870] = 2.42e-13 * pow(Tgas/300.0,
        2.13) * exp(-1338.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2000.0) { k[1871] = 5.11e-14 * pow(Tgas/300.0,
        1.88) * exp(-92.0/Tgas);  }
        
    if (Tgas>226.0 && Tgas<2500.0) { k[1872] = 4.2e-11 * pow(Tgas/300.0,
        0.0) * exp(-2520.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1873] = 1.52e-12 * pow(Tgas/300.0,
        1.55) * exp(-215.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1874] = 2.67e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1875] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    k[1876] = 6e-12 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>10.0 && Tgas<300.0) { k[1877] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1878] = 6e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1879] = 2.29e-12 * pow(Tgas/300.0,
        2.2) * exp(-3820.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<294.0) { k[1880] = 2.54e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<5000.0) { k[1881] = 5.37e-11 * pow(Tgas/300.0,
        0.0) * exp(-13800.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<6000.0) { k[1882] = 2.46e-11 * pow(Tgas/300.0,
        0.0) * exp(-26567.0/Tgas);  }
        
    if (Tgas>150.0 && Tgas<906.0) { k[1883] = 2.48e-10 * pow(Tgas/300.0,
        -0.65) * exp(-783.0/Tgas);  }
        
    if (Tgas>964.0 && Tgas<41000.0) { k[1884] = 4.68e-11 * pow(Tgas/300.0,
        0.5) * exp(-28940.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1885] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>250.0 && Tgas<2500.0) { k[1886] = 1.07e-11 * pow(Tgas/300.0,
        1.17) * exp(-1242.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1887] = 1.85e-11 * pow(Tgas/300.0,
        0.95) * exp(-8571.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2000.0) { k[1888] = 2.98e-12 * pow(Tgas/300.0,
        1.62) * exp(-1462.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<4000.0) { k[1889] = 6.21e-10 * pow(Tgas/300.0,
        0.0) * exp(-12439.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2600.0) { k[1890] = 7.3e-13 * pow(Tgas/300.0,
        1.14) * exp(-3742.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2600.0) { k[1891] = 1.36e-12 * pow(Tgas/300.0,
        1.38) * exp(-3693.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1892] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1893] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1894] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1895] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1896] = 1e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1897] = 3.8e-11 * pow(Tgas/300.0,
        -0.08) * exp(-0.0/Tgas);  }
        
    if (Tgas>116.0 && Tgas<300.0) { k[1898] = 2.94e-12 * pow(Tgas/300.0,
        0.5) * exp(-3500.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1899] = 1.74e-11 * pow(Tgas/300.0,
        0.67) * exp(-956.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2000.0) { k[1900] = 1.74e-10 * pow(Tgas/300.0,
        -0.2) * exp(-5.7/Tgas);  }
        
    if (Tgas>1400.0 && Tgas<4700.0) { k[1901] = 2.51e-10 * pow(Tgas/300.0,
        0.0) * exp(-38602.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<3000.0) { k[1902] = 6.3e-11 * pow(Tgas/300.0,
        -0.1) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<3000.0) { k[1903] = 7e-12 * pow(Tgas/300.0, -0.1)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1904] = 1.89e-11 * pow(Tgas/300.0,
        0.0) * exp(-4003.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1905] = 9.82e-12 * pow(Tgas/300.0,
        -0.21) * exp(-5.2/Tgas);  }
        
    if (Tgas>200.0 && Tgas<5000.0) { k[1906] = 1.18e-11 * pow(Tgas/300.0,
        0.0) * exp(-20413.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1907] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>133.0 && Tgas<300.0) { k[1908] = 1e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1909] = 5.76e-11 * pow(Tgas/300.0,
        -0.3) * exp(-7.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1910] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1911] = 4.02e-10 * pow(Tgas/300.0,
        -1.43) * exp(-3501.0/Tgas);  }
        
    if (Tgas>1200.0 && Tgas<1900.0) { k[1912] = 8.3e-11 * pow(Tgas/300.0,
        0.0) * exp(-5530.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1913] = 1.6e-11 * pow(Tgas/300.0,
        0.0) * exp(-2150.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<500.0) { k[1914] = 3.69e-11 * pow(Tgas/300.0,
        -0.27) * exp(-12.9/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2500.0) { k[1915] = 1.7e-11 * pow(Tgas/300.0,
        0.0) * exp(-0.0/Tgas);  }
        
    if (Tgas>440.0 && Tgas<3760.0) { k[1916] = 9.01e-12 * pow(Tgas/300.0,
        0.0) * exp(-9837.0/Tgas);  }
        
    if (Tgas>92.0 && Tgas<300.0) { k[1917] = 6.6e-13 * pow(Tgas/300.0, 0.0)
        * exp(-2760.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1918] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1919] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1920] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1921] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1922] = 8e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1923] = 1.2e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1924] = 1.4e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>280.0 && Tgas<1000.0) { k[1925] = 1.98e-11 * pow(Tgas/300.0,
        0.0) * exp(-1183.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1926] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<2500.0) { k[1927] = 1.05e-13 * pow(Tgas/300.0,
        2.68) * exp(-6060.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1928] = 1.84e-11 * pow(Tgas/300.0,
        0.0) * exp(-5027.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1929] = 4.75e-17 * pow(Tgas/300.0,
        3.16) * exp(+128.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1930] = 5e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1931] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1932] = 1e-11 * pow(Tgas/300.0, 0.0)
        * exp(-1000.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1933] = 7e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>80.0 && Tgas<3150.0) { k[1934] = 2.81e-13 * pow(Tgas/300.0,
        0.0) * exp(-176.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1935] = 3e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1936] = 1.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3000.0) { k[1937] = 7.76e-12 * pow(Tgas/300.0,
        0.82) * exp(+30.6/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1938] = 6.3e-12 * pow(Tgas/300.0,
        0.0) * exp(-80.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<2840.0) { k[1939] = 1.87e-13 * pow(Tgas/300.0,
        1.5) * exp(-3887.0/Tgas);  }
        
    if (Tgas>500.0 && Tgas<2500.0) { k[1940] = 1.07e-13 * pow(Tgas/300.0,
        0.0) * exp(-5892.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<2500.0) { k[1941] = 1.7e-10 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>298.0 && Tgas<4000.0) { k[1942] = 6.17e-12 * pow(Tgas/300.0,
        1.23) * exp(+44.3/Tgas);  }
        
    if (Tgas>300.0 && Tgas<550.0) { k[1943] = 3.11e-13 * pow(Tgas/300.0,
        0.0) * exp(-1450.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<3000.0) { k[1944] = 1.47e-13 * pow(Tgas/300.0,
        2.05) * exp(-7.0/Tgas);  }
        
    if (Tgas>503.0 && Tgas<41000.0) { k[1945] = 5.2e-12 * pow(Tgas/300.0,
        0.0) * exp(-15100.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1946] = 8.58e-11 * pow(Tgas/300.0,
        -0.56) * exp(-14.8/Tgas);  }
        
    if (Tgas>200.0 && Tgas<2500.0) { k[1947] = 1.65e-12 * pow(Tgas/300.0,
        1.14) * exp(-50.0/Tgas);  }
        
    k[1948] = 6.6e-11 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    k[1949] = 8.6e-11 * pow(Tgas/300.0, 0.0) * exp(-0.0/Tgas);
        
    if (Tgas>10.0 && Tgas<3560.0) { k[1950] = 1e-10 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1951] = 8e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1952] = 4e-11 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1953] = 4.5e-11 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>1250.0 && Tgas<1600.0) { k[1954] = 9.76e-12 * pow(Tgas/300.0,
        0.0) * exp(-4545.0/Tgas);  }
        
    if (Tgas>383.0 && Tgas<41000.0) { k[1955] = 1.73e-11 * pow(Tgas/300.0,
        0.5) * exp(-11500.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<3160.0) { k[1956] = 2.72e-11 * pow(Tgas/300.0,
        0.0) * exp(-282.0/Tgas);  }
        
    if (Tgas>2720.0 && Tgas<5190.0) { k[1957] = 1.3e-09 * pow(Tgas/300.0,
        0.0) * exp(-34513.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1958] = 9e-11 * pow(Tgas/300.0, -0.96)
        * exp(-28.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[1959] = 1.72e-10 * pow(Tgas/300.0,
        -0.53) * exp(-17.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1960] = G0 * 1e-11 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1961] = G0 * 4.1e-10 * exp(-3.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1962] = G0 * 2.4e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1963] = G0 * 1e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1964] = G0 * 3.3e-10 * exp(-3.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1965] = G0 * 3.3e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1966] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1967] = G0 * 3e-09 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1968] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1969] = G0 * 2.9e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1970] = G0 * 5.2e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1971] = G0 * 1e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1972] = G0 * 5e-10 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1973] = G0 * 5e-10 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1974] = G0 * 5e-10 * exp(-1.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1975] = G0 * 1.85e-09 * exp(-2.0*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1976] = G0 * 3.1e-10 * exp(-3.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1977] = G0 * 3.3e-10 * exp(-2.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1978] = G0 * 4.67e-11 * exp(-2.2*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1979] = G0 * 4.67e-11 * exp(-2.2*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1980] = G0 * 4.67e-11 * exp(-2.2*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1981] = G0 * 1e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1982] = G0 * 5.8e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1983] = G0 * 1.4e-09 * exp(-2.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1984] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1985] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1986] = G0 * 1.35e-10 * exp(-2.3*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1987] = G0 * 1e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1988] = G0 * 1.35e-10 * exp(-2.3*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1989] = G0 * 2.5e-09 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1990] = G0 * 7e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1991] = G0 * 1.3e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1992] = G0 * 7e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1993] = G0 * 2.27e-10 * exp(-2.7*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1994] = G0 * 5.33e-11 * exp(-2.7*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1995] = G0 * 9.8e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1996] = G0 * 2.2e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1997] = G0 * 6.8e-12 * exp(-3.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1998] = G0 * 2.2e-10 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[1999] = G0 * 9.2e-10 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2000] = G0 * 7.6e-10 * exp(-3.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2001] = G0 * 2.9e-10 * exp(-3.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2002] = G0 * 1e-10 * exp(-2.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2003] = G0 * 8.9e-10 * exp(-3.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2004] = (2.0e-10) * G0 *
        GetShieldingFactor(IDX_COI, h2col, cocol, Tgas, 1) *
        GetGrainScattering(Av, lamdabar) / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2005] = G0 * 2e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2006] = G0 * 2e-10 * exp(-3.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2007] = G0 * 9.8e-10 * exp(-2.4*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2008] = G0 * 9.5e-11 * exp(-4.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2009] = G0 * 5.7e-10 * exp(-2.4*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2010] = G0 * 5.48e-10 * exp(-2.0*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2011] = G0 * 1e-09 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2012] = G0 * 7e-10 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2013] = G0 * 4.7e-10 * exp(-2.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2014] = G0 * 1.4e-11 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2015] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2016] = G0 * 1e-12 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2017] = G0 * 3.1e-11 * exp(-3.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2018] = G0 * 8e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2019] = G0 * 8.3e-10 * exp(-1.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2020] = G0 * 7.3e-10 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2021] = G0 * 1.55e-09 * exp(-2.3*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2022] = G0 * 1.55e-09 * exp(-2.3*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2023] = G0 * 4.4e-10 * exp(-1.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2024] = G0 * 4.4e-10 * exp(-1.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2025] = G0 * 5e-15 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2026] = G0 * 5e-15 * exp(-1.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2027] = G0 * 5.6e-09 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2028] = G0 * 1.6e-09 * exp(-2.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2029] = G0 * 5.4e-12 * exp(-3.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2030] = G0 * 1.1e-09 * exp(-1.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2031] = G0 * 5.6e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2032] = G0 * 1.38e-09 * exp(-1.7*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2033] = G0 * 2e-10 * exp(-2.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2034] = G0 * 1.7e-09 * exp(-1.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2035] = G0 * 1.9e-10 * exp(-1.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2036] = G0 * 1.5e-09 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2037] = G0 * 1e-09 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2038] = G0 * 1.7e-10 * exp(-0.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2039] = G0 * 3e-10 * exp(-1.8*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2040] = G0 * 2.5e-10 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2041] = G0 * 6.2e-12 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2042] = G0 * 3.3e-10 * exp(-1.4*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2043] = G0 * 9.8e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2044] = G0 * 7.9e-11 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2045] = G0 * 2.3e-10 * exp(-3.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2046] = G0 * 1e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2047] = G0 * 4.7e-11 * exp(-2.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2048] = G0 * 5.4e-11 * exp(-1.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2049] = G0 * 1.73e-10 * exp(-2.6*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2050] = G0 * 7.5e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2051] = G0 * 9.23e-10 * exp(-2.1*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2052] = G0 * 2.8e-10 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2053] = G0 * 2.76e-10 * exp(-2.1*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2054] = G0 * 5e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2055] = G0 * 1e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2056] = G0 * 1.4e-09 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2057] = G0 * 2.6e-10 * exp(-2.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2058] = G0 * 4.7e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2059] = G0 * 2e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2060] = G0 * 3.5e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2061] = G0 * 7.6e-11 * exp(-3.9*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2062] = G0 * 7.9e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2063] = G0 * 3.35e-10 * exp(-2.1*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2064] = G0 * 3.35e-10 * exp(-2.1*Av)
        / 1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2065] = G0 * 1e-11 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2066] = G0 * 6.9e-10 * exp(-3.4*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2067] = G0 * 3.7e-09 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2068] = G0 * 1.1e-11 * exp(-3.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2069] = G0 * 3.9e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2070] = G0 * 1.6e-12 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2071] = G0 * 6.2e-12 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2072] = G0 * 3.3e-10 * exp(-1.4*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2073] = G0 * 6e-10 * exp(-3.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2074] = G0 * 1.9e-09 * exp(-2.4*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2075] = G0 * 4.2e-09 * exp(-2.4*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2076] = G0 * 6e-10 * exp(-2.5*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2077] = G0 * 3.1e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2078] = G0 * 2.6e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2079] = G0 * 2e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2080] = G0 * 2e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2081] = G0 * 1e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2082] = G0 * 2.7e-09 * exp(-1.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2083] = G0 * 1e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2084] = G0 * 5e-11 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2085] = G0 * 3e-11 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2086] = G0 * 1e-10 * exp(-2.1*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2087] = G0 * 3e-11 * exp(-1.7*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2088] = G0 * 4.8e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2089] = G0 * 1.6e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2090] = G0 * 1.6e-10 * exp(-2.2*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2091] = G0 * 2.8e-09 * exp(-1.6*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2092] = G0 * 1e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2093] = G0 * 1.6e-09 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2094] = G0 * 2.4e-10 * exp(-2.0*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2095] = G0 * 1e-10 * exp(-2.3*Av) /
        1.7;  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2096] = 4.01e-18 * pow(Tgas/300.0,
        0.17) * exp(-101.5/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[2097] = 1.08e-18 * pow(Tgas/300.0,
        0.07) * exp(-57.5/Tgas);  }
        
    if (Tgas>10.0 && Tgas<13900.0) { k[2098] = 3.14e-18 * pow(Tgas/300.0,
        -0.15) * exp(-68.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[2099] = 7.22e-19 * pow(Tgas/300.0,
        0.15) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2100] = 1e-16 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2101] = 4.36e-18 * pow(Tgas/300.0,
        0.35) * exp(-161.3/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[2102] = 5.72e-19 * pow(Tgas/300.0,
        0.37) * exp(-51.0/Tgas);  }
        
    if (Tgas>2000.0 && Tgas<10000.0) { k[2103] = 5e-10 * pow(Tgas/300.0,
        -3.7) * exp(-800.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<14700.0) { k[2104] = 4.69e-19 * pow(Tgas/300.0,
        1.52) * exp(+50.5/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[2105] = 2.01e-18 * pow(Tgas/300.0,
        0.07) * exp(-301.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[2106] = 1.53e-18 * pow(Tgas/300.0,
        0.22) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2107] = 2e-12 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2108] = 9e-09 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2109] = 1e-16 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>200.0 && Tgas<32000.0) { k[2110] = 1.15e-18 * pow(Tgas/300.0,
        1.49) * exp(-228.0/Tgas);  }
        
    if (Tgas>16.0 && Tgas<100.0) { k[2111] = 5.26e-20 * pow(Tgas/300.0,
        -0.51) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2112] = 2e-16 * pow(Tgas/300.0, -1.3) *
        exp(-23.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2113] = 1e-17 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2114] = 3.92e-16 * pow(Tgas/300.0,
        -2.29) * exp(-21.3/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2115] = 5.09e-18 * pow(Tgas/300.0,
        -0.71) * exp(-11.6/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2116] = 2.4e-16 * pow(Tgas/300.0, -0.8)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2117] = 1e-17 * pow(Tgas/300.0, -0.2) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2118] = 3e-18 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2119] = 3e-17 * pow(Tgas/300.0, -1.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2120] = 1e-18 * pow(Tgas/300.0, -0.5) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2121] = 2.4e-14 * pow(Tgas/300.0, -2.8)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2122] = 1.7e-17 * pow(Tgas/300.0, 0.0)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2123] = 1e-17 * pow(Tgas/300.0, 0.0) *
        exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2124] = 9.9e-19 * pow(Tgas/300.0,
        -0.38) * exp(-0.0/Tgas);  }
        
    if (Tgas>20.0 && Tgas<300.0) { k[2125] = 5.26e-18 * pow(Tgas/300.0,
        -5.22) * exp(-90.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<10000.0) { k[2126] = 1.17e-17 * pow(Tgas/300.0,
        -0.14) * exp(-0.0/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[2127] = 3.71e-18 * pow(Tgas/300.0,
        0.24) * exp(-26.1/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2128] = 4.9e-20 * pow(Tgas/300.0, 1.58)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2129] = 3.2e-16 * pow(Tgas/300.0, -1.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<14700.0) { k[2130] = 9.22e-19 * pow(Tgas/300.0,
        -0.08) * exp(+21.2/Tgas);  }
        
    if (Tgas>300.0 && Tgas<14700.0) { k[2131] = 3.23e-17 * pow(Tgas/300.0,
        0.31) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2132] = 2.36e-12 * pow(Tgas/300.0,
        -0.29) * exp(+17.6/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2133] = 1.1e-10 * pow(Tgas/300.0, -0.5)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2134] = 4.14e-12 * pow(Tgas/300.0,
        -0.61) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<20000.0) { k[2135] = 3.5e-12 * pow(Tgas/300.0,
        -0.75) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2136] = 1.1e-10 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2137] = 1.1e-10 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2138] = 1.1e-10 * pow(Tgas/300.0, -0.7)
        * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[2139] = 5.36e-12 * pow(Tgas/300.0,
        -0.5) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[2140] = 2.78e-12 * pow(Tgas/300.0,
        -0.68) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2141] = 3.5e-12 * pow(Tgas/300.0,
        -0.53) * exp(+3.2/Tgas);  }
        
    if (Tgas>10.0 && Tgas<41000.0) { k[2142] = 3.24e-12 * pow(Tgas/300.0,
        -0.66) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[2143] = 5.49e-12 * pow(Tgas/300.0,
        -0.59) * exp(-0.0/Tgas);  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[2144] = 4.26e-12 * pow(Tgas/300.0,
        -0.62) * exp(-0.0/Tgas);  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2145] = 4.57e4 * 1.0 * sqrt(Tgas / 51.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2146] = 4.57e4 * 1.0 * sqrt(Tgas / 27.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2147] = 4.57e4 * 1.0 * sqrt(Tgas / 42.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2148] = 4.57e4 * 1.0 * sqrt(Tgas / 41.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2149] = 4.57e4 * 1.0 * sqrt(Tgas / 40.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2150] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2151] = 4.57e4 * 1.0 * sqrt(Tgas / 47.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2152] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2153] = 4.57e4 * 1.0 * sqrt(Tgas / 60.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2154] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2155] = 4.57e4 * 1.0 * sqrt(Tgas / 42.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2156] = 4.57e4 * 1.0 * sqrt(Tgas / 42.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2157] = 4.57e4 * 1.0 * sqrt(Tgas / 43.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2158] = 4.57e4 * 1.0 * sqrt(Tgas / 42.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2159] = 4.57e4 * 1.0 * sqrt(Tgas / 38.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2160] = 4.57e4 * 1.0 * sqrt(Tgas / 38.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2161] = 4.57e4 * 1.0 * sqrt(Tgas / 50.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2162] = 4.57e4 * 1.0 * sqrt(Tgas / 39.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2163] = 4.57e4 * 1.0 * sqrt(Tgas / 52.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2164] = 4.57e4 * 1.0 * sqrt(Tgas / 52.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2165] = 4.57e4 * 1.0 * sqrt(Tgas / 62.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2166] = 4.57e4 * 1.0 * sqrt(Tgas / 62.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2167] = 4.57e4 * 1.0 * sqrt(Tgas / 61.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2168] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2169] = 4.57e4 * 0.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2170] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2171] = 4.57e4 * 1.0 * sqrt(Tgas / 44.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2172] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2173] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2174] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2175] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2176] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2177] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2178] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2179] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2180] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2181] = 4.57e4 * 1.0 * sqrt(Tgas / 40.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2182] = 4.57e4 * 1.0 * sqrt(Tgas / 52.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2183] = 4.57e4 * 1.0 * sqrt(Tgas / 40.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2184] = 4.57e4 * 1.0 * sqrt(Tgas / 52.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2185] = 4.57e4 * 1.0 * sqrt(Tgas / 64.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2186] = 4.57e4 * 1.0 * sqrt(Tgas / 64.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2187] = 4.57e4 * 1.0 * sqrt(Tgas / 44.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2188] = 4.57e4 * 1.0 * sqrt(Tgas / 45.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2189] = 4.57e4 * 1.0 * sqrt(Tgas / 60.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2190] = 4.57e4 * 1.0 * sqrt(Tgas / 60.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2191] = 4.57e4 * 1.0 * sqrt(Tgas / 61.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2192] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2193] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2194] = 4.57e4 * 1.0 * sqrt(Tgas / 35.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2195] = 4.57e4 * 1.0 * sqrt(Tgas / 36.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2196] = 4.57e4 * 1.0 * sqrt(Tgas / 35.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2197] = 4.57e4 * 1.0 * sqrt(Tgas / 36.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2198] = 4.57e4 * 1.0 * sqrt(Tgas / 37.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2199] = 4.57e4 * 1.0 * sqrt(Tgas / 38.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2200] = 4.57e4 * 1.0 * sqrt(Tgas / 49.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2201] = 4.57e4 * 1.0 * sqrt(Tgas / 64.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2202] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2203] = 4.57e4 * 1.0 * sqrt(Tgas / 66.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2204] = 4.57e4 * 1.0 * sqrt(Tgas / 65.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2205] = 4.57e4 * 1.0 * sqrt(Tgas / 65.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2206] = 4.57e4 * 1.0 * sqrt(Tgas / 12.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2207] = 4.57e4 * 0.9 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2208] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2209] = 4.57e4 * 1.0 * sqrt(Tgas / 24.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2210] = 4.57e4 * 1.0 * sqrt(Tgas / 13.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2211] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2212] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2213] = 4.57e4 * 1.0 * sqrt(Tgas / 14.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2214] = 4.57e4 * 1.0 * sqrt(Tgas / 18.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2215] = 4.57e4 * 1.0 * sqrt(Tgas / 44.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2216] = 4.57e4 * 1.0 * sqrt(Tgas / 15.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2217] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2218] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2219] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2220] = 4.57e4 * 1.0 * sqrt(Tgas / 26.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2221] = 4.57e4 * 1.0 * sqrt(Tgas / 12.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2222] = 4.57e4 * 1.0 * sqrt(Tgas / 15.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2223] = 4.57e4 * 1.0 * sqrt(Tgas / 27.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2224] = 4.57e4 * 1.0 * sqrt(Tgas / 25.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2225] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2226] = 4.57e4 * 1.0 * sqrt(Tgas / 14.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2227] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2228] = 4.57e4 * 1.0 * sqrt(Tgas / 24.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2229] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2230] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2231] = 4.57e4 * 1.0 * sqrt(Tgas / 13.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2232] = 4.57e4 * 1.0 * sqrt(Tgas / 15.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2233] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2234] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2235] = 4.57e4 * 1.0 * sqrt(Tgas / 26.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2236] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2237] = 4.57e4 * 1.0 * sqrt(Tgas / 14.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2238] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2239] = 4.57e4 * 1.0 * sqrt(Tgas / 18.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2240] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2241] = 4.57e4 * 1.0 * sqrt(Tgas / 27.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2242] = 4.57e4 * 1.0 * sqrt(Tgas / 25.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2243] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2244] = 4.57e4 * 1.0 * sqrt(Tgas / 30.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2245] = 4.57e4 * 1.0 * sqrt(Tgas / 15.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2246] = 4.57e4 * 1.0 * sqrt(Tgas / 19.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2247] = 4.57e4 * 1.0 * sqrt(Tgas / 45.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2248] = 4.57e4 * 1.0 * sqrt(Tgas / 17.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2249] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2250] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2251] = 4.57e4 * 1.0 * sqrt(Tgas / 14.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2252] = 4.57e4 * 1.0 * sqrt(Tgas / 16.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2253] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2254] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2255] = 4.57e4 * 1.0 * sqrt(Tgas / 44.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2256] = 4.57e4 * 1.0 * sqrt(Tgas / 48.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2257] = 4.57e4 * 1.0 * sqrt(Tgas / 34.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2258] = 4.57e4 * 1.0 * sqrt(Tgas / 45.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2259] = 4.57e4 * 1.0 * sqrt(Tgas / 60.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2260] = 4.57e4 * 1.0 * sqrt(Tgas / 64.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2261] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2262] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2263] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2264] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2265] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2266] = 4.57e4 * 1.0 * sqrt(Tgas / 44.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2267] = 4.57e4 * 1.0 * sqrt(Tgas / 48.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2268] = 4.57e4 * 1.0 * sqrt(Tgas / 45.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2269] = 4.57e4 * 1.0 * sqrt(Tgas / 60.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2270] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2271] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2272] = 4.57e4 * 1.0 * sqrt(Tgas / 31.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2273] = 4.57e4 * 1.0 * sqrt(Tgas / 32.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2274] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2275] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2276] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2277] = 4.57e4 * 1.0 * sqrt(Tgas / 26.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2278] = 4.57e4 * 1.0 * sqrt(Tgas / 35.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2279] = 4.57e4 * 1.0 * sqrt(Tgas / 46.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2280] = 4.57e4 * 1.0 * sqrt(Tgas / 66.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2281] = 4.57e4 * 1.0 * sqrt(Tgas / 64.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2282] = 4.57e4 * 1.0 * sqrt(Tgas / 47.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2283] = 4.57e4 * 1.0 * sqrt(Tgas / 49.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2284] = 4.57e4 * 1.0 * sqrt(Tgas / 64.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2285] = 4.57e4 * 1.0 * sqrt(Tgas / 61.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2286] = 4.57e4 * 1.0 * sqrt(Tgas / 24.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2287] = 4.57e4 * 1.0 * sqrt(Tgas / 24.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2288] = 4.57e4 * 1.0 * sqrt(Tgas / 18.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2289] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2290] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2291] = 4.57e4 * 1.0 * sqrt(Tgas / 47.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2292] = 4.57e4 * 1.0 * sqrt(Tgas / 36.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2293] = 4.57e4 * 1.0 * sqrt(Tgas / 34.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2294] = 4.57e4 * 1.0 * sqrt(Tgas / 33.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2295] = 4.57e4 * 1.0 * sqrt(Tgas / 65.0)
        * garea * fr * ( 1.0 + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2296] = 4.57e4 * 0.1 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2297] = 4.57e4 * 1.0 * sqrt(Tgas / 27.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2298] = 4.57e4 * 1.0 * sqrt(Tgas / 26.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2299] = 4.57e4 * 1.0 * sqrt(Tgas / 41.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2300] = 4.57e4 * 1.0 * sqrt(Tgas / 28.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2301] = 4.57e4 * 1.0 * sqrt(Tgas / 29.0)
        * garea * fr;  }
        
    if (Tgas>0.0 && Tgas<30.0) { k[2302] = 4.57e4 * 1.0 * garea * fr * ( 1.0
        + 16.71e-4/(rG * Tgas) );  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2303] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2304] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2305] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2306] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCH4I/(pi*pi*amu*16.0)) * 2.0 * densites *
        exp(-eb_GCH4I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2307] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2308] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2309] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2310] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GNH3I/(pi*pi*amu*17.0)) * 2.0 * densites *
        exp(-eb_GNH3I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2311] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2312] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2313] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2314] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2OI/(pi*pi*amu*18.0)) * 2.0 * densites *
        exp(-eb_GH2OI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2315] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2316] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2317] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2318] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC2I/(pi*pi*amu*24.0)) * 2.0 * densites *
        exp(-eb_GC2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2319] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2320] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2321] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2322] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GMgI/(pi*pi*amu*24.0)) * 2.0 * densites *
        exp(-eb_GMgI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2323] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2324] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2325] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2326] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC2HI/(pi*pi*amu*25.0)) * 2.0 * densites *
        exp(-eb_GC2HI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2327] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2328] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2329] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2330] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC2H2I/(pi*pi*amu*26.0)) * 2.0 * densites *
        exp(-eb_GC2H2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2331] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2332] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2333] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2334] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHCNI/(pi*pi*amu*27.0)) * 2.0 * densites *
        exp(-eb_GHCNI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2335] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2336] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2337] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2338] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHNCI/(pi*pi*amu*27.0)) * 2.0 * densites *
        exp(-eb_GHNCI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2339] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2340] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2341] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2342] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC2H3I/(pi*pi*amu*27.0)) * 2.0 * densites *
        exp(-eb_GC2H3I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2343] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2344] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2345] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2346] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCOI/(pi*pi*amu*28.0)) * 2.0 * densites *
        exp(-eb_GCOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2347] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2348] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2349] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2350] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GN2I/(pi*pi*amu*28.0)) * 2.0 * densites *
        exp(-eb_GN2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2351] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2352] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2353] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2354] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC2H4I/(pi*pi*amu*28.0)) * 2.0 * densites *
        exp(-eb_GC2H4I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2355] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2356] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2357] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2358] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2CNI/(pi*pi*amu*28.0)) * 2.0 * densites *
        exp(-eb_GH2CNI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2359] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2360] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2361] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2362] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC2H5I/(pi*pi*amu*29.0)) * 2.0 * densites *
        exp(-eb_GC2H5I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2363] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2364] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2365] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2366] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GNOI/(pi*pi*amu*30.0)) * 2.0 * densites *
        exp(-eb_GNOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2367] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2368] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2369] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2370] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2COI/(pi*pi*amu*30.0)) * 2.0 * densites *
        exp(-eb_GH2COI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2371] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2372] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2373] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2374] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHNOI/(pi*pi*amu*31.0)) * 2.0 * densites *
        exp(-eb_GHNOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2375] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2376] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2377] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2378] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GO2I/(pi*pi*amu*32.0)) * 2.0 * densites *
        exp(-eb_GO2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2379] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2380] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2381] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2382] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCH3OHI/(pi*pi*amu*32.0)) * 2.0 * densites *
        exp(-eb_GCH3OHI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2383] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2384] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2385] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2386] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiH4I/(pi*pi*amu*32.0)) * 2.0 * densites *
        exp(-eb_GSiH4I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2387] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2388] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2389] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2390] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GO2HI/(pi*pi*amu*33.0)) * 2.0 * densites *
        exp(-eb_GO2HI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2391] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2392] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2393] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2394] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2SI/(pi*pi*amu*34.0)) * 2.0 * densites *
        exp(-eb_GH2SI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2395] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2396] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2397] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2398] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHClI/(pi*pi*amu*36.0)) * 2.0 * densites *
        exp(-eb_GHClI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2399] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2400] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2401] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2402] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC3H2I/(pi*pi*amu*38.0)) * 2.0 * densites *
        exp(-eb_GC3H2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2403] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2404] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2405] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2406] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCH3CCHI/(pi*pi*amu*40.0)) * 2.0 * densites *
        exp(-eb_GCH3CCHI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2407] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2408] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2409] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2410] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiCI/(pi*pi*amu*40.0)) * 2.0 * densites *
        exp(-eb_GSiCI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2411] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2412] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2413] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2414] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCH3CNI/(pi*pi*amu*41.0)) * 2.0 * densites *
        exp(-eb_GCH3CNI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2415] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2416] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2417] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2418] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCH2COI/(pi*pi*amu*42.0)) * 2.0 * densites *
        exp(-eb_GCH2COI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2419] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2420] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2421] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2422] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCH3CNHI/(pi*pi*amu*42.0)) * 2.0 * densites *
        exp(-eb_GCH3CNHI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2423] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2424] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2425] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2426] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHNCOI/(pi*pi*amu*43.0)) * 2.0 * densites *
        exp(-eb_GHNCOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2427] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2428] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2429] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2430] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiOI/(pi*pi*amu*44.0)) * 2.0 * densites *
        exp(-eb_GSiOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2431] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2432] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2433] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2434] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCSI/(pi*pi*amu*44.0)) * 2.0 * densites *
        exp(-eb_GCSI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2435] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2436] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2437] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2438] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GCO2I/(pi*pi*amu*44.0)) * 2.0 * densites *
        exp(-eb_GCO2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2439] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2440] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2441] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2442] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC2H5OHI/(pi*pi*amu*46.0)) * 2.0 * densites *
        exp(-eb_GC2H5OHI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2443] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2444] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2445] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2446] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2CSI/(pi*pi*amu*46.0)) * 2.0 * densites *
        exp(-eb_GH2CSI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2447] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2448] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2449] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2450] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GNO2I/(pi*pi*amu*46.0)) * 2.0 * densites *
        exp(-eb_GNO2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2451] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2452] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2453] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2454] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GNSI/(pi*pi*amu*46.0)) * 2.0 * densites *
        exp(-eb_GNSI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2455] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2456] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2457] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2458] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2SiOI/(pi*pi*amu*46.0)) * 2.0 * densites *
        exp(-eb_GH2SiOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2459] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2460] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2461] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2462] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSOI/(pi*pi*amu*48.0)) * 2.0 * densites *
        exp(-eb_GSOI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2463] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2464] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2465] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2466] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC4HI/(pi*pi*amu*49.0)) * 2.0 * densites *
        exp(-eb_GC4HI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2467] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2468] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2469] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2470] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHC3NI/(pi*pi*amu*51.0)) * 2.0 * densites *
        exp(-eb_GHC3NI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2471] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2472] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2473] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2474] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GNCCNI/(pi*pi*amu*52.0)) * 2.0 * densites *
        exp(-eb_GNCCNI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2475] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2476] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2477] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2478] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiC2I/(pi*pi*amu*52.0)) * 2.0 * densites *
        exp(-eb_GSiC2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2479] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2480] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2481] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2482] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GHCOOCH3I/(pi*pi*amu*60.0)) * 2.0 * densites *
        exp(-eb_GHCOOCH3I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2483] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2484] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2485] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2486] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiSI/(pi*pi*amu*60.0)) * 2.0 * densites *
        exp(-eb_GSiSI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2487] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2488] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2489] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2490] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GOCSI/(pi*pi*amu*60.0)) * 2.0 * densites *
        exp(-eb_GOCSI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2491] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2492] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2493] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2494] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GC4NI/(pi*pi*amu*62.0)) * 2.0 * densites *
        exp(-eb_GC4NI/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2495] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2496] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2497] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2498] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSiC3I/(pi*pi*amu*64.0)) * 2.0 * densites *
        exp(-eb_GSiC3I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2499] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2500] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2501] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2502] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GSO2I/(pi*pi*amu*64.0)) * 2.0 * densites *
        exp(-eb_GSO2I/Tgas)) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2503] = mantabund > 1e-30 ? (opt_h2d *
        h2deseff * 1.0e-17 * sqrt(Tgas) * y[IDX_HI] * nH / mant) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2504] = mantabund > 1e-30 ? (opt_crd *
        4.0 * pi * crdeseff * ((zeta / zism)) * 1.64e-4 * garea / mant) : 0.0;
    }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2505] = mantabund > 1e-30 ? (opt_uvd *
        (G0*1.0e8*exp(-Av*3.02) + 1.0e4 * (zeta / zism)) * 0.001 * nmono * 4.0 *
        garea) : 0.0;  }
        
    if (Tgas>0.0 && Tgas<10000.0) { k[2506] = mantabund > 1e-30 ? (opt_thd *
        sqrt(2.0*sites*kerg*eb_GH2S2I/(pi*pi*amu*66.0)) * 2.0 * densites *
        exp(-eb_GH2S2I/Tgas)) : 0.0;  }
        
    
        // clang-format on

    return NAUNET_SUCCESS;
}

// clang-format off
int EvalHeatingRates(realtype *kh, realtype *y, NaunetData *u_data) {

    realtype rG = u_data->rG;
    realtype gdens = u_data->gdens;
    realtype sites = u_data->sites;
    realtype fr = u_data->fr;
    realtype opt_thd = u_data->opt_thd;
    realtype opt_crd = u_data->opt_crd;
    realtype opt_h2d = u_data->opt_h2d;
    realtype opt_uvd = u_data->opt_uvd;
    realtype eb_h2d = u_data->eb_h2d;
    realtype eb_crd = u_data->eb_crd;
    realtype eb_uvd = u_data->eb_uvd;
    realtype crdeseff = u_data->crdeseff;
    realtype h2deseff = u_data->h2deseff;
    realtype nH = u_data->nH;
    realtype zeta = u_data->zeta;
    realtype Tgas = u_data->Tgas;
    realtype Av = u_data->Av;
    realtype omega = u_data->omega;
    realtype G0 = u_data->G0;
    realtype uvcreff = u_data->uvcreff;
    
    double mant = GetMantleDens(y);
    double mantabund = mant / nH;
    double garea = (pi*rG*rG) * gdens;
    double garea_per_H = garea / nH;
    double densites = 4.0 * garea * sites;
    double h2col = 0.5*1.59e21*Av;
    double cocol = 1e-5 * h2col;
    double lamdabar = GetCharactWavelength(h2col, cocol);
    double H2shielding = GetShieldingFactor(IDX_H2I, h2col, h2col, Tgas, 1);
    double H2formation = 1.0e-17 * sqrt(Tgas);
    double H2dissociation = 5.1e-11 * G0 * GetGrainScattering(Av, 1000.0) * H2shielding;
    
    // clang-format on

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    
    // clang-format on

    return NAUNET_SUCCESS;
}

// clang-format off
int EvalCoolingRates(realtype *kc, realtype *y, NaunetData *u_data) {

    realtype rG = u_data->rG;
    realtype gdens = u_data->gdens;
    realtype sites = u_data->sites;
    realtype fr = u_data->fr;
    realtype opt_thd = u_data->opt_thd;
    realtype opt_crd = u_data->opt_crd;
    realtype opt_h2d = u_data->opt_h2d;
    realtype opt_uvd = u_data->opt_uvd;
    realtype eb_h2d = u_data->eb_h2d;
    realtype eb_crd = u_data->eb_crd;
    realtype eb_uvd = u_data->eb_uvd;
    realtype crdeseff = u_data->crdeseff;
    realtype h2deseff = u_data->h2deseff;
    realtype nH = u_data->nH;
    realtype zeta = u_data->zeta;
    realtype Tgas = u_data->Tgas;
    realtype Av = u_data->Av;
    realtype omega = u_data->omega;
    realtype G0 = u_data->G0;
    realtype uvcreff = u_data->uvcreff;
    
    double mant = GetMantleDens(y);
    double mantabund = mant / nH;
    double garea = (pi*rG*rG) * gdens;
    double garea_per_H = garea / nH;
    double densites = 4.0 * garea * sites;
    double h2col = 0.5*1.59e21*Av;
    double cocol = 1e-5 * h2col;
    double lamdabar = GetCharactWavelength(h2col, cocol);
    double H2shielding = GetShieldingFactor(IDX_H2I, h2col, h2col, Tgas, 1);
    double H2formation = 1.0e-17 * sqrt(Tgas);
    double H2dissociation = 5.1e-11 * G0 * GetGrainScattering(Av, 1000.0) * H2shielding;
    
    // clang-format on

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    
    // clang-format on

    return NAUNET_SUCCESS;
}
