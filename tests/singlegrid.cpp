// 
#include <stdio.h>

#include <stdexcept>
#include <vector>

#include "naunet.h"
#include "naunet_data.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_timer.h"

int main() {
    double spy = 86400.0 * 365.0;

    NaunetData data;
    //
    double nH       = 2e4;
    double zeta     = 1.3e-17;
    double Tgas     = 15.0;
    double Av       = 10.0;
    double omega    = 0.5;
    double G0       = 1.0;
    double rG       = 1e-5;
    double gdens    = 7.6394373e-13 * nH;
    double sites    = 1.5e15;
    double fr       = 1.0;
    double opt_thd  = 1.0;
    double opt_crd  = 1.0;
    double opt_uvd  = 1.0;
    double opt_h2d  = 1.0;
    double eb_crd   = 1.21e3;
    double eb_h2d   = 1.21e3;
    double eb_uvd   = 1.00e4;
    double crdeseff = 1.0e5;
    double h2deseff = 1.0e-2;
    double uvcreff  = 1.0e-3;

    data.nH       = nH;
    data.zeta     = zeta;
    data.Tgas     = Tgas;
    data.Av       = Av;
    data.omega    = omega;
    data.G0       = G0;
    data.rG       = rG;
    data.gdens    = gdens;
    data.sites    = sites;
    data.fr       = fr;
    data.opt_thd  = opt_thd;
    data.opt_crd  = opt_crd;
    data.opt_uvd  = opt_uvd;
    data.opt_h2d  = opt_h2d;
    data.eb_crd   = eb_crd;
    data.eb_h2d   = eb_h2d;
    data.eb_uvd   = eb_uvd;
    data.crdeseff = crdeseff;
    data.h2deseff = h2deseff;
    data.uvcreff  = uvcreff;


    Naunet naunet;
    if (naunet.Init() == NAUNET_FAIL) {
        printf("Initialize Fail\n");
        return 1;
    }

#ifdef USE_CUDA
    if (naunet.Reset(1) == NAUNET_FAIL) {
        throw std::runtime_error("Fail to reset the number of systems");
    }
#endif

    //
    double y[NEQUATIONS] = {0.0};
    // for (int i = 0; i < NEQUATIONS; i++)
    // {
    //     y[i] = 1.e-40;
    // }
    y[IDX_H2I]           = 0.5 * nH;
    y[IDX_HI]            = 5.0e-5 * nH;
    y[IDX_HeI]           = 9.75e-2 * nH;
    y[IDX_NI]            = 7.5e-5 * nH;
    y[IDX_OI]            = 1.8e-4 * nH;
    y[IDX_COI]           = 1.4e-4 * nH;
    y[IDX_SI]            = 8.0e-8 * nH;
    y[IDX_SiI]           = 8.0e-9 * nH;
    y[IDX_MgI]           = 7.0e-9 * nH;
    y[IDX_ClI]           = 4.0e-9 * nH;


    FILE *fbin = fopen("evolution_singlegrid.bin", "w");
    FILE *ftxt = fopen("evolution_singlegrid.txt", "w");
    FILE *ttxt = fopen("time_singlegrid.txt", "w");
#ifdef NAUNET_DEBUG
    printf("Initialization is done. Start to evolve.\n");
    FILE *rtxt               = fopen("reactionrates.txt", "w");
    double rates[NREACTIONS] = {0.0};
#endif

    //
    std::vector<double> timesteps;
    double logtstart = 3.0, logtend = 7.0, logtstep = 0.1;
    double time = 0.0;
    for (double logtime = logtstart; logtime < logtend + 0.1 * logtstep;
         logtime += logtstep) {
        double dtyr = pow(10.0, logtime) - time;
        timesteps.push_back(dtyr);
        time += dtyr;
    }
    //

    double dtyr = 0.0, curtime = 0.0;

    // write the initial abundances
    fwrite(&curtime, sizeof(double), 1, fbin);
    fwrite(y, sizeof(double), NEQUATIONS, fbin);

    fprintf(ftxt, "%13.7e ", curtime);
    for (int j = 0; j < NEQUATIONS; j++) {
        fprintf(ftxt, "%13.7e ", y[j]);
    }
    fprintf(ftxt, "\n");

    for (auto step = timesteps.begin(); step != timesteps.end(); step++) {
#ifdef NAUNET_DEBUG
        EvalRates(rates, y, &data);
        for (int j = 0; j < NREACTIONS; j++) {
            fprintf(rtxt, "%13.7e ", rates[j]);
        }
        fprintf(rtxt, "\n");
#endif

        //
        //

        dtyr = *step;

        Timer timer;
        timer.start();
        naunet.Solve(y, dtyr * spy, &data);
        timer.stop();

        curtime += dtyr;

        // write the abundances after each step
        fwrite(&curtime, sizeof(double), 1, fbin);
        fwrite(y, sizeof(double), NEQUATIONS, fbin);

        fprintf(ftxt, "%13.7e ", curtime);
        for (int j = 0; j < NEQUATIONS; j++) {
            fprintf(ftxt, "%13.7e ", y[j]);
        }
        fprintf(ftxt, "\n");

        // float duration = (float)timer.elapsed() / 1e6;
        double duration = timer.elapsed();
        fprintf(ttxt, "%8.5e \n", duration);
        printf("Time = %13.7e yr, elapsed: %8.5e sec\n", curtime, duration);
    }

    fclose(fbin);
    fclose(ftxt);
    fclose(ttxt);
#ifdef NAUNET_DEBUG
    fclose(rtxt);
#endif

    if (naunet.Finalize() == NAUNET_FAIL) {
        printf("Finalize Fail\n");
        return 1;
    }

    return 0;
}