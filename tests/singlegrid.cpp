#include <stdio.h>

#include "naunet.h"
#include "naunet_data.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_timer.h"

int main() {
    double spy     = 86400.0 * 365.0;

    double nH       = 16.66231569;
    double Tgas     = 10.3985;
    // double Tgas     = 480.50409837;
    // double Tgas     = 1000.0;
    double zeta     = 1.0e-16;
    double Av       = 1.0237814e+00;
    double omega    = 0.5;
    double G0       = 4.0;
    double rG       = 1e-5;
    double gdens    = 7.6394373e-13 * nH;
    double sites    = 1.5e15;
    double fr       = 1.0;
    double opt_thd  = 1.0;
    double opt_crd  = 1.0;
    double opt_uvd  = 1.0;
    double opt_h2d  = 1.0;
    double crdeseff = 1.0e8;
    double h2deseff = 1.0e-2;
    double uvcreff  = 1.0e-3;
    double ksp      = 0.0;

    NaunetData data;
    // data.nH       = nH;
    // data.Tgas     = Tgas;
    // data.zeta     = zeta;
    // data.Av       = Av;
    // data.omega    = omega;
    // data.G0       = G0;
    // data.rG       = rG;
    // data.gdens    = gdens;
    // data.sites    = sites;
    // data.fr       = fr;
    // data.opt_thd  = opt_thd;
    // data.opt_crd  = opt_crd;
    // data.opt_uvd  = opt_uvd;
    // data.opt_h2d  = opt_h2d;
    // data.crdeseff = crdeseff;
    // data.h2deseff = h2deseff;
    // data.uvcreff  = uvcreff;
    // data.ksp      = ksp;
    // data.eb_crd   = 1.21e3;
    // data.eb_h2d   = 1.21e3;
    // data.eb_uvd   = 1.00e4;

    data.nH = 8.3085212e+01;
    data.Tgas = 1.8387799e+01;
    data.zeta = 1.0000000e-16;
    data.Av = 1.0116999e+00;
    data.omega = 5.0000000e-01;
    data.G0 = 4.0000000e+00;
    data.uvcreff = 1.0000000e-03;
    data.rG = 1.0000000e-05;
    data.gdens = 6.3472426e-11;
    data.sites = 1.5000000e+15;
    data.fr = 1.0000000e+00;
    data.opt_thd = 1.0000000e+00;
    data.opt_crd = 1.0000000e+00;
    data.opt_h2d = 1.0000000e+00;
    data.opt_uvd = 1.0000000e+00;
    data.eb_h2d = 1.2100000e+03;
    data.eb_crd = 2.0000000e+03;
    data.eb_uvd = 1.0000000e+04;
    data.crdeseff = 1.0000000e+08;
    data.h2deseff = 1.0000000e-02;
    data.ksp = 0.0000000e+00;
    
    Naunet naunet;
    naunet.Init(1, 1e-30, 1e-5, 1000);

#ifdef USE_CUDA
    naunet.Reset(1);
#endif

    double y[NEQUATIONS] = {0.0};

    // for (int i = 0; i < NEQUATIONS; i++)
    // {
    //     y[i] = 1.e-40;
    // }
    // y[IDX_H2I]           = 0.5 * nH;
    // y[IDX_HI]            = 5.0e-5 * nH;
    // y[IDX_HeI]           = 9.75e-2 * nH;
    // y[IDX_NI]            = 7.5e-5 * nH;
    // y[IDX_OI]            = 1.8e-4 * nH;
    // y[IDX_COI]           = 1.4e-4 * nH;
    // y[IDX_SiI]           = 8.0e-9 * nH;
    // y[IDX_MgI]           = 7.0e-9 * nH;

    y[0] = 3.6349780e-40;
    y[1] = 7.2699560e-40;
    y[2] = 4.1542606e-40;
    y[3] = 2.6436204e-40;
    y[4] = 4.1542606e-40;
    y[5] = 3.8773099e-40;
    y[6] = 6.4621831e-40;
    y[7] = 2.5286804e-40;
    y[8] = 4.3081221e-40;
    y[9] = 4.3081221e-40;
    y[10] = 2.7050999e-40;
    y[11] = 3.7522354e-40;
    y[12] = 4.8466373e-40;
    y[13] = 4.1542606e-40;
    y[14] = 6.8423115e-40;
    y[15] = 3.8773099e-40;
    y[16] = 2.5286804e-40;
    y[17] = 3.6349780e-40;
    y[18] = 3.5248272e-40;
    y[19] = 2.9079824e-40;
    y[20] = 2.2369095e-40;
    y[21] = 1.8174890e-40;
    y[22] = 3.6349780e-40;
    y[23] = 2.6436204e-40;
    y[24] = 9.6932747e-40;
    y[25] = 9.6932747e-40;
    y[26] = 8.9476382e-40;
    y[27] = 8.9476382e-40;
    y[28] = 8.3085212e-40;
    y[29] = 8.3085212e-40;
    y[30] = 7.7546198e-40;
    y[31] = 7.7546198e-40;
    y[32] = 3.6349780e-40;
    y[33] = 7.2699560e-40;
    y[34] = 7.2699560e-40;
    y[35] = 4.4738191e-40;
    y[36] = 4.4738191e-40;
    y[37] = 1.1649400e-02;
    y[38] = 4.1542606e-40;
    y[39] = 2.6436204e-40;
    y[40] = 8.3085989e-39;
    y[41] = 4.1605002e-03;
    y[42] = 1.1631930e-38;
    y[43] = 4.1605002e+01;
    y[44] = 5.8159648e-39;
    y[45] = 4.1542606e-40;
    y[46] = 3.8773099e-40;
    y[47] = 3.8773099e-40;
    y[48] = 3.6349780e-40;
    y[49] = 6.4621831e-40;
    y[50] = 6.4621831e-40;
    y[51] = 2.5286804e-40;
    y[52] = 3.8773099e-39;
    y[53] = 3.7522354e-40;
    y[54] = 6.1220682e-40;
    y[55] = 4.3081221e-40;
    y[56] = 4.3081221e-40;
    y[57] = 4.1542606e-40;
    y[58] = 4.0110102e-40;
    y[59] = 4.0110102e-40;
    y[60] = 2.5848733e-40;
    y[61] = 8.1129753e+00;
    y[62] = 2.9079824e-39;
    y[63] = 2.3263859e-39;
    y[64] = 4.3081221e-40;
    y[65] = 2.7050999e-40;
    y[66] = 3.7522354e-40;
    y[67] = 3.7522354e-40;
    y[68] = 4.0110102e-40;
    y[69] = 5.8247002e-07;
    y[70] = 4.8466373e-40;
    y[71] = 6.2407503e-03;
    y[72] = 8.3085212e-40;
    y[73] = 4.1542606e-40;
    y[74] = 4.1542606e-40;
    y[75] = 4.0110102e-40;
    y[76] = 7.7546198e-40;
    y[77] = 7.7546198e-40;
    y[78] = 7.2699560e-40;
    y[79] = 7.2699560e-40;
    y[80] = 6.8423115e-40;
    y[81] = 6.8423115e-40;
    y[82] = 3.8773099e-40;
    y[83] = 3.8773099e-40;
    y[84] = 2.5286804e-40;
    y[85] = 1.4977801e-02;
    y[86] = 7.2699560e-40;
    y[87] = 3.6349780e-40;
    y[88] = 3.6349780e-40;
    y[89] = 3.5248272e-40;
    y[90] = 3.5248272e-40;
    y[91] = 2.7695071e-40;
    y[92] = 6.8423115e-40;
    y[93] = 6.8423115e-40;
    y[94] = 6.6568003e-07;
    y[95] = 4.1542606e-40;
    y[96] = 2.9079824e-40;
    y[97] = 2.9079824e-40;
    y[98] = 2.2369095e-40;
    y[99] = 2.2369095e-40;
    y[100] = 1.8174890e-40;
    y[101] = 1.8174890e-40;
    y[102] = 4.0110102e-40;
    y[103] = 4.0110102e-40;
    y[104] = 3.8773099e-40;
    y[105] = 3.8773099e-40;
    y[106] = 3.7522354e-40;
    y[107] = 3.7522354e-40;
    y[108] = 3.6349780e-40;
    y[109] = 3.6349780e-40;
    y[110] = 3.5248272e-40;
    y[111] = 2.6436204e-40;
    y[112] = 2.6436204e-40;
    y[113] = 2.5848733e-40;

    FILE *fbin           = fopen("evolution_singlegrid.bin", "w");
    FILE *ftxt           = fopen("evolution_singlegrid.txt", "w");
    FILE *ttxt           = fopen("time_singlegrid.txt", "w");
#ifdef NAUNET_DEBUG
    printf("Initialization is done. Start to evolve.\n");
    FILE *rtxt = fopen("reactionrates.txt", "w");
    double rates[NREACTIONS] = {0.0};
#endif

    realtype y_init[NEQUATIONS];

    double atol = 1e-30;
    double logtstart = 2.0, logtend = 7.0;
    double dtyr = 0.0, time = 0.0;
    for (double logtime = logtstart; logtime < logtend; logtime += 0.1) {
#ifdef NAUNET_DEBUG
        EvalRates(rates, y, &data);
        for (int j = 0; j < NREACTIONS; j++)
        {
            fprintf(rtxt, "%13.7e ", rates[j]);
        }
        fprintf(rtxt, "\n");
#endif

        for (int idx = IDX_GCH3OHI; idx <= IDX_SiOHII; idx++) {
            y_init[idx] = y[idx];
        }

        // dtyr = pow(10.0, logtime) - time;
        dtyr = 8.2226644e+04;

        fwrite(&time, sizeof(double), 1, fbin);
        fwrite(y, sizeof(double), NEQUATIONS, fbin);

        fprintf(ftxt, "%13.7e ", time);
        for (int j = 0; j < NEQUATIONS; j++) {
            fprintf(ftxt, "%13.7e ", y[j]);
        }
        fprintf(ftxt, "\n");

        Timer timer;
        timer.start();
        int flag = naunet.Solve(y, dtyr * spy, &data);
        timer.stop();

        time += dtyr;

        // float duration = (float)timer.elapsed() / 1e6;
        double duration = timer.elapsed();
        fprintf(ttxt, "%8.5e \n", duration);
        printf("Time = %13.7e yr, elapsed: %8.5e sec\n", time, duration);

    }

    // save the final results
    fwrite(&time, sizeof(double), 1, fbin);
    fwrite(y, sizeof(double), NEQUATIONS, fbin);

    fprintf(ftxt, "%13.7e ", time);
    for (int j = 0; j < NEQUATIONS; j++) {
        fprintf(ftxt, "%13.7e ", y[j]);
    }
    fprintf(ftxt, "\n");

    fclose(fbin);
    fclose(ftxt);
    fclose(ttxt);
#ifdef NAUNET_DEBUG
    fclose(rtxt);
#endif
//



    naunet.Finalize();

    return 0;
}
