#include <math.h>
#include <algorithm>

#include "naunet_constants.h"
#include "naunet_utilities.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

// clang-format off
double GetMantleDens(double *y) {
    return y[IDX_GC2I] + y[IDX_GC2HI] + y[IDX_GC2H2I] + y[IDX_GC2H3I] +
        y[IDX_GC2H4I] + y[IDX_GC2H5I] + y[IDX_GC2H5OHI] + y[IDX_GC3H2I] +
        y[IDX_GC4HI] + y[IDX_GC4NI] + y[IDX_GCH2COI] + y[IDX_GCH3CCHI] +
        y[IDX_GCH3CNI] + y[IDX_GCH3CNHI] + y[IDX_GCH3OHI] + y[IDX_GCH4I] +
        y[IDX_GCOI] + y[IDX_GCO2I] + y[IDX_GCSI] + y[IDX_GH2CNI] + y[IDX_GH2COI]
        + y[IDX_GH2CSI] + y[IDX_GH2OI] + y[IDX_GH2SI] + y[IDX_GH2S2I] +
        y[IDX_GH2SiOI] + y[IDX_GHC3NI] + y[IDX_GHClI] + y[IDX_GHCNI] +
        y[IDX_GHCOOCH3I] + y[IDX_GHNCI] + y[IDX_GHNCOI] + y[IDX_GHNOI] +
        y[IDX_GMgI] + y[IDX_GN2I] + y[IDX_GNCCNI] + y[IDX_GNH3I] + y[IDX_GNOI] +
        y[IDX_GNO2I] + y[IDX_GNSI] + y[IDX_GO2I] + y[IDX_GO2HI] + y[IDX_GOCSI] +
        y[IDX_GSiCI] + y[IDX_GSiC2I] + y[IDX_GSiC3I] + y[IDX_GSiH4I] +
        y[IDX_GSiOI] + y[IDX_GSiSI] + y[IDX_GSOI] + y[IDX_GSO2I];
}

double GetMu(double *y) {
    return (y[IDX_GC2I]*24.0 + y[IDX_GC2HI]*25.0 + y[IDX_GC2H2I]*26.0 +
        y[IDX_GC2H3I]*27.0 + y[IDX_GC2H4I]*28.0 + y[IDX_GC2H5I]*29.0 +
        y[IDX_GC2H5OHI]*46.0 + y[IDX_GC3H2I]*38.0 + y[IDX_GC4HI]*49.0 +
        y[IDX_GC4NI]*62.0 + y[IDX_GCH2COI]*42.0 + y[IDX_GCH3CCHI]*40.0 +
        y[IDX_GCH3CNI]*41.0 + y[IDX_GCH3CNHI]*42.0 + y[IDX_GCH3OHI]*32.0 +
        y[IDX_GCH4I]*16.0 + y[IDX_GCOI]*28.0 + y[IDX_GCO2I]*44.0 +
        y[IDX_GCSI]*44.0 + y[IDX_GH2CNI]*28.0 + y[IDX_GH2COI]*30.0 +
        y[IDX_GH2CSI]*46.0 + y[IDX_GH2OI]*18.0 + y[IDX_GH2SI]*34.0 +
        y[IDX_GH2S2I]*66.0 + y[IDX_GH2SiOI]*46.0 + y[IDX_GHC3NI]*51.0 +
        y[IDX_GHClI]*36.0 + y[IDX_GHCNI]*27.0 + y[IDX_GHCOOCH3I]*60.0 +
        y[IDX_GHNCI]*27.0 + y[IDX_GHNCOI]*43.0 + y[IDX_GHNOI]*31.0 +
        y[IDX_GMgI]*24.0 + y[IDX_GN2I]*28.0 + y[IDX_GNCCNI]*52.0 +
        y[IDX_GNH3I]*17.0 + y[IDX_GNOI]*30.0 + y[IDX_GNO2I]*46.0 +
        y[IDX_GNSI]*46.0 + y[IDX_GO2I]*32.0 + y[IDX_GO2HI]*33.0 +
        y[IDX_GOCSI]*60.0 + y[IDX_GSiCI]*40.0 + y[IDX_GSiC2I]*52.0 +
        y[IDX_GSiC3I]*64.0 + y[IDX_GSiH4I]*32.0 + y[IDX_GSiOI]*44.0 +
        y[IDX_GSiSI]*60.0 + y[IDX_GSOI]*48.0 + y[IDX_GSO2I]*64.0 +
        y[IDX_CI]*12.0 + y[IDX_CII]*12.0 + y[IDX_C2I]*24.0 + y[IDX_C2II]*24.0 +
        y[IDX_C2HI]*25.0 + y[IDX_C2HII]*25.0 + y[IDX_C2H2I]*26.0 +
        y[IDX_C2H2II]*26.0 + y[IDX_C2H3I]*27.0 + y[IDX_C2H4I]*28.0 +
        y[IDX_C2H5I]*29.0 + y[IDX_C2H5OHI]*46.0 + y[IDX_C2H5OH2II]*47.0 +
        y[IDX_C2NI]*38.0 + y[IDX_C2NII]*38.0 + y[IDX_C2N2II]*52.0 +
        y[IDX_C2NHII]*39.0 + y[IDX_C3II]*36.0 + y[IDX_C3H2I]*38.0 +
        y[IDX_C3H5II]*41.0 + y[IDX_C3NI]*50.0 + y[IDX_C4HI]*49.0 +
        y[IDX_C4NI]*62.0 + y[IDX_C4NII]*62.0 + y[IDX_CHI]*13.0 +
        y[IDX_CHII]*13.0 + y[IDX_CH2I]*14.0 + y[IDX_CH2II]*14.0 +
        y[IDX_CH2COI]*42.0 + y[IDX_CH3I]*15.0 + y[IDX_CH3II]*15.0 +
        y[IDX_CH3CCHI]*40.0 + y[IDX_CH3CNI]*41.0 + y[IDX_CH3CNHI]*42.0 +
        y[IDX_CH3CNHII]*42.0 + y[IDX_CH3OHI]*32.0 + y[IDX_CH3OH2II]*33.0 +
        y[IDX_CH4I]*16.0 + y[IDX_CH4II]*16.0 + y[IDX_CH5II]*17.0 +
        y[IDX_ClI]*35.0 + y[IDX_ClII]*35.0 + y[IDX_CNI]*26.0 + y[IDX_CNII]*26.0
        + y[IDX_COI]*28.0 + y[IDX_COII]*28.0 + y[IDX_CO2I]*44.0 +
        y[IDX_CSI]*44.0 + y[IDX_CSII]*44.0 + y[IDX_EM]*0.0 + y[IDX_HI]*1.0 +
        y[IDX_HII]*1.0 + y[IDX_H2I]*2.0 + y[IDX_H2II]*2.0 + y[IDX_H2ClII]*37.0 +
        y[IDX_H2CNI]*28.0 + y[IDX_H2COI]*30.0 + y[IDX_H2COII]*30.0 +
        y[IDX_H2CSI]*46.0 + y[IDX_H2CSII]*46.0 + y[IDX_H2NOII]*32.0 +
        y[IDX_H2OI]*18.0 + y[IDX_H2OII]*18.0 + y[IDX_H2SI]*34.0 +
        y[IDX_H2SII]*34.0 + y[IDX_H2S2I]*66.0 + y[IDX_H2S2II]*66.0 +
        y[IDX_H2SiOI]*46.0 + y[IDX_H3II]*3.0 + y[IDX_H3COII]*31.0 +
        y[IDX_H3CSII]*47.0 + y[IDX_H3OII]*19.0 + y[IDX_H3SII]*35.0 +
        y[IDX_H5C2O2II]*61.0 + y[IDX_HC3NI]*51.0 + y[IDX_HClI]*36.0 +
        y[IDX_HClII]*36.0 + y[IDX_HCNI]*27.0 + y[IDX_HCNII]*27.0 +
        y[IDX_HCNHII]*28.0 + y[IDX_HCOI]*29.0 + y[IDX_HCOII]*29.0 +
        y[IDX_HCO2II]*45.0 + y[IDX_HCOOCH3I]*60.0 + y[IDX_HCSI]*45.0 +
        y[IDX_HCSII]*45.0 + y[IDX_HeI]*4.0 + y[IDX_HeII]*4.0 + y[IDX_HeHII]*5.0
        + y[IDX_HNCI]*27.0 + y[IDX_HNCOI]*43.0 + y[IDX_HNOI]*31.0 +
        y[IDX_HNOII]*31.0 + y[IDX_HNSII]*47.0 + y[IDX_HOCII]*29.0 +
        y[IDX_HOCSII]*61.0 + y[IDX_HSI]*33.0 + y[IDX_HSII]*33.0 +
        y[IDX_HS2I]*65.0 + y[IDX_HS2II]*65.0 + y[IDX_HSiSII]*61.0 +
        y[IDX_HSOII]*49.0 + y[IDX_HSO2II]*65.0 + y[IDX_MgI]*24.0 +
        y[IDX_MgII]*24.0 + y[IDX_NI]*14.0 + y[IDX_NII]*14.0 + y[IDX_N2I]*28.0 +
        y[IDX_N2II]*28.0 + y[IDX_N2HII]*29.0 + y[IDX_NCCNI]*52.0 +
        y[IDX_NHI]*15.0 + y[IDX_NHII]*15.0 + y[IDX_NH2I]*16.0 +
        y[IDX_NH2II]*16.0 + y[IDX_NH3I]*17.0 + y[IDX_NH3II]*17.0 +
        y[IDX_NH4II]*18.0 + y[IDX_NOI]*30.0 + y[IDX_NOII]*30.0 +
        y[IDX_NO2I]*46.0 + y[IDX_NSI]*46.0 + y[IDX_NSII]*46.0 + y[IDX_OI]*16.0 +
        y[IDX_OII]*16.0 + y[IDX_O2I]*32.0 + y[IDX_O2II]*32.0 + y[IDX_O2HI]*33.0
        + y[IDX_O2HII]*33.0 + y[IDX_OCNI]*42.0 + y[IDX_OCSI]*60.0 +
        y[IDX_OCSII]*60.0 + y[IDX_OHI]*17.0 + y[IDX_OHII]*17.0 + y[IDX_SI]*32.0
        + y[IDX_SII]*32.0 + y[IDX_S2I]*64.0 + y[IDX_S2II]*64.0 + y[IDX_SiI]*28.0
        + y[IDX_SiII]*28.0 + y[IDX_SiCI]*40.0 + y[IDX_SiCII]*40.0 +
        y[IDX_SiC2I]*52.0 + y[IDX_SiC2II]*52.0 + y[IDX_SiC3I]*64.0 +
        y[IDX_SiC3II]*64.0 + y[IDX_SiHI]*29.0 + y[IDX_SiHII]*29.0 +
        y[IDX_SiH2I]*30.0 + y[IDX_SiH2II]*30.0 + y[IDX_SiH3I]*31.0 +
        y[IDX_SiH3II]*31.0 + y[IDX_SiH4I]*32.0 + y[IDX_SiH4II]*32.0 +
        y[IDX_SiH5II]*33.0 + y[IDX_SiOI]*44.0 + y[IDX_SiOII]*44.0 +
        y[IDX_SiOHII]*45.0 + y[IDX_SiSI]*60.0 + y[IDX_SiSII]*60.0 +
        y[IDX_SOI]*48.0 + y[IDX_SOII]*48.0 + y[IDX_SO2I]*64.0 +
        y[IDX_SO2II]*64.0) / (y[IDX_GC2I] + y[IDX_GC2HI] + y[IDX_GC2H2I] +
        y[IDX_GC2H3I] + y[IDX_GC2H4I] + y[IDX_GC2H5I] + y[IDX_GC2H5OHI] +
        y[IDX_GC3H2I] + y[IDX_GC4HI] + y[IDX_GC4NI] + y[IDX_GCH2COI] +
        y[IDX_GCH3CCHI] + y[IDX_GCH3CNI] + y[IDX_GCH3CNHI] + y[IDX_GCH3OHI] +
        y[IDX_GCH4I] + y[IDX_GCOI] + y[IDX_GCO2I] + y[IDX_GCSI] + y[IDX_GH2CNI]
        + y[IDX_GH2COI] + y[IDX_GH2CSI] + y[IDX_GH2OI] + y[IDX_GH2SI] +
        y[IDX_GH2S2I] + y[IDX_GH2SiOI] + y[IDX_GHC3NI] + y[IDX_GHClI] +
        y[IDX_GHCNI] + y[IDX_GHCOOCH3I] + y[IDX_GHNCI] + y[IDX_GHNCOI] +
        y[IDX_GHNOI] + y[IDX_GMgI] + y[IDX_GN2I] + y[IDX_GNCCNI] + y[IDX_GNH3I]
        + y[IDX_GNOI] + y[IDX_GNO2I] + y[IDX_GNSI] + y[IDX_GO2I] + y[IDX_GO2HI]
        + y[IDX_GOCSI] + y[IDX_GSiCI] + y[IDX_GSiC2I] + y[IDX_GSiC3I] +
        y[IDX_GSiH4I] + y[IDX_GSiOI] + y[IDX_GSiSI] + y[IDX_GSOI] + y[IDX_GSO2I]
        + y[IDX_CI] + y[IDX_CII] + y[IDX_C2I] + y[IDX_C2II] + y[IDX_C2HI] +
        y[IDX_C2HII] + y[IDX_C2H2I] + y[IDX_C2H2II] + y[IDX_C2H3I] +
        y[IDX_C2H4I] + y[IDX_C2H5I] + y[IDX_C2H5OHI] + y[IDX_C2H5OH2II] +
        y[IDX_C2NI] + y[IDX_C2NII] + y[IDX_C2N2II] + y[IDX_C2NHII] + y[IDX_C3II]
        + y[IDX_C3H2I] + y[IDX_C3H5II] + y[IDX_C3NI] + y[IDX_C4HI] + y[IDX_C4NI]
        + y[IDX_C4NII] + y[IDX_CHI] + y[IDX_CHII] + y[IDX_CH2I] + y[IDX_CH2II] +
        y[IDX_CH2COI] + y[IDX_CH3I] + y[IDX_CH3II] + y[IDX_CH3CCHI] +
        y[IDX_CH3CNI] + y[IDX_CH3CNHI] + y[IDX_CH3CNHII] + y[IDX_CH3OHI] +
        y[IDX_CH3OH2II] + y[IDX_CH4I] + y[IDX_CH4II] + y[IDX_CH5II] + y[IDX_ClI]
        + y[IDX_ClII] + y[IDX_CNI] + y[IDX_CNII] + y[IDX_COI] + y[IDX_COII] +
        y[IDX_CO2I] + y[IDX_CSI] + y[IDX_CSII] + y[IDX_EM] + y[IDX_HI] +
        y[IDX_HII] + y[IDX_H2I] + y[IDX_H2II] + y[IDX_H2ClII] + y[IDX_H2CNI] +
        y[IDX_H2COI] + y[IDX_H2COII] + y[IDX_H2CSI] + y[IDX_H2CSII] +
        y[IDX_H2NOII] + y[IDX_H2OI] + y[IDX_H2OII] + y[IDX_H2SI] + y[IDX_H2SII]
        + y[IDX_H2S2I] + y[IDX_H2S2II] + y[IDX_H2SiOI] + y[IDX_H3II] +
        y[IDX_H3COII] + y[IDX_H3CSII] + y[IDX_H3OII] + y[IDX_H3SII] +
        y[IDX_H5C2O2II] + y[IDX_HC3NI] + y[IDX_HClI] + y[IDX_HClII] +
        y[IDX_HCNI] + y[IDX_HCNII] + y[IDX_HCNHII] + y[IDX_HCOI] + y[IDX_HCOII]
        + y[IDX_HCO2II] + y[IDX_HCOOCH3I] + y[IDX_HCSI] + y[IDX_HCSII] +
        y[IDX_HeI] + y[IDX_HeII] + y[IDX_HeHII] + y[IDX_HNCI] + y[IDX_HNCOI] +
        y[IDX_HNOI] + y[IDX_HNOII] + y[IDX_HNSII] + y[IDX_HOCII] + y[IDX_HOCSII]
        + y[IDX_HSI] + y[IDX_HSII] + y[IDX_HS2I] + y[IDX_HS2II] + y[IDX_HSiSII]
        + y[IDX_HSOII] + y[IDX_HSO2II] + y[IDX_MgI] + y[IDX_MgII] + y[IDX_NI] +
        y[IDX_NII] + y[IDX_N2I] + y[IDX_N2II] + y[IDX_N2HII] + y[IDX_NCCNI] +
        y[IDX_NHI] + y[IDX_NHII] + y[IDX_NH2I] + y[IDX_NH2II] + y[IDX_NH3I] +
        y[IDX_NH3II] + y[IDX_NH4II] + y[IDX_NOI] + y[IDX_NOII] + y[IDX_NO2I] +
        y[IDX_NSI] + y[IDX_NSII] + y[IDX_OI] + y[IDX_OII] + y[IDX_O2I] +
        y[IDX_O2II] + y[IDX_O2HI] + y[IDX_O2HII] + y[IDX_OCNI] + y[IDX_OCSI] +
        y[IDX_OCSII] + y[IDX_OHI] + y[IDX_OHII] + y[IDX_SI] + y[IDX_SII] +
        y[IDX_S2I] + y[IDX_S2II] + y[IDX_SiI] + y[IDX_SiII] + y[IDX_SiCI] +
        y[IDX_SiCII] + y[IDX_SiC2I] + y[IDX_SiC2II] + y[IDX_SiC3I] +
        y[IDX_SiC3II] + y[IDX_SiHI] + y[IDX_SiHII] + y[IDX_SiH2I] +
        y[IDX_SiH2II] + y[IDX_SiH3I] + y[IDX_SiH3II] + y[IDX_SiH4I] +
        y[IDX_SiH4II] + y[IDX_SiH5II] + y[IDX_SiOI] + y[IDX_SiOII] +
        y[IDX_SiOHII] + y[IDX_SiSI] + y[IDX_SiSII] + y[IDX_SOI] + y[IDX_SOII] +
        y[IDX_SO2I] + y[IDX_SO2II]);
}

double GetGamma(double *y) {
    return 5.0 / 3.0;
}

double GetNumDens(double *y) {
    double numdens = 0.0;

    for (int i = 0; i < NSPECIES; i++) numdens += y[i];
    return numdens;
}
// clang-format on

// clang-format off
double GetShieldingFactor(int specidx, double h2coldens, double spcoldens,
                          double tgas, int method) {
    // clang-format on
    double factor;
#ifdef IDX_H2I
    if (specidx == IDX_H2I) {
        factor = GetH2shielding(h2coldens, method);
    }
#endif
#ifdef IDX_COI
    if (specidx == IDX_COI) {
        factor = GetCOshielding(tgas, h2coldens, spcoldens, method);
    }
#endif
#ifdef IDX_N2I
    if (specidx == IDX_N2I) {
        factor = GetN2shielding(tgas, h2coldens, spcoldens, method);
    }
#endif

    return factor;
}

// clang-format off
double GetH2shielding(double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetH2shieldingInt(coldens);
            break;
        case 1:
            shielding = GetH2shieldingFGK(coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
double GetCOshielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetCOshieldingInt(tgas, h2col, coldens);
            break;
        case 1:
            shielding = GetCOshieldingInt1(h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
double GetN2shielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetN2shieldingInt(tgas, h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetH2shieldingInt(double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */

    return shielding;
}

// Calculates the line self shielding function
// Ref: Federman et al. apj vol.227 p.466.
// Originally implemented in UCLCHEM
double GetH2shieldingFGK(double coldens) {
    // clang-format on

    const double dopplerwidth = 3.0e10;
    const double radiativewidth = 8.0e7;
    const double oscillatorstrength = 1.0e-2;

    double shielding = -1.0;

    double taud = 0.5 * coldens * 1.5e-2 * oscillatorstrength / dopplerwidth;

    // Calculate wing contribution of self shielding function sr
    if (taud < 0.0) taud = 0.0;

    double sr = 0.0;
    if (radiativewidth != 0.0) {
        double r  = radiativewidth / (1.7724539*dopplerwidth);
        double t  = 3.02 * pow(1000.0*r, -0.064);
        double u  = pow(taud*r, 0.5) / t;
        double sr = pow((u*u + 0.78539816), -0.5) * r / t;
    }

    // Calculate doppler contribution of self shielding function sj
    double sj = 0.0;
    if (taud == 0.0) {
        sj = 1.0;
    }
    else if (taud < 2.0) {
        sj = exp(-0.6666667*taud) ;
    }
    else if (taud < 10.0) {
        sj = 0.638 * pow(taud, -1.25);
    }
    else if (taud < 100.0) {
        sj = 0.505 * pow(taud, -1.15);
    }
    else {
        sj = 0.344 * pow(taud, -1.0667);
    }

    shielding = sj + sr;

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetCOshieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */

    return shielding;
}

double GetCOshieldingInt1(double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */
    double logh2 = std::min(std::max(log10(h2col), COShieldingTableX[0]), COShieldingTableX[5]);
    double logco = std::min(std::max(log10(coldens), COShieldingTableY[0]), COShieldingTableY[6]);

    double *x1  = vector(1, 6);
    double *x2  = vector(1, 7);
    double **y  = matrix(1, 6, 1, 7);
    double **y2 = matrix(1, 6, 1, 7);

    for (int i=1; i<=6; i++) x1[i] = COShieldingTableX[i-1];
    for (int i=1; i<=7; i++) x2[i] = COShieldingTableY[i-1];

    for (int i=1; i<=6; i++) {
        for (int j=1; j<=7; j++) {
            y[i][j] = COShieldingTable[i-1][j-1];
        }
    }

    splie2(x1, x2, y, 6, 7, y2);

    splin2(x1, x2, y, y2, 6, 7, logh2, logco, &shielding);

    // splin2(COShieldingTableX, COShieldingTableY, COShieldingTable, COShieldingTableD2, 
    //        6, 7, logh2, logco, &shielding);

    shielding = pow(10.0, shielding);

    free_vector(x1, 1, 6);
    free_vector(x2, 1, 7);
    free_matrix(y, 1, 6, 1, 7);
    free_matrix(y2, 1, 6, 1, 7);

    /* */

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
double GetN2shieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */

    return shielding;
}

// Calculate xlamda := tau(lambda) / tau(visual)
// tau(lambda) is the opt. depth for dust extinction at
// wavelength x (cf. b.d.savage and j.s.mathis, annual
// review of astronomy and astrophysics vol.17(1979),p.84)
double xlamda(double wavelength) {
    double x[29] = {
        910.0, 950.0, 1000.0, 1050.0, 1110.0,
        1180.0, 1250.0, 1390.0, 1490.0, 1600.0,
        1700.0, 1800.0, 1900.0, 2000.0, 2100.0, 
        2190.0, 2300.0, 2400.0, 2500.0, 2740.0,
        3440.0, 4000.0, 4400.0, 5500.0, 7000.0,
        9000.0, 12500.0, 22000.0, 34000.0
    };

    double y[29] = {
        5.76, 5.18, 4.65, 4.16, 3.73, 
        3.4, 3.11, 2.74, 2.63, 2.62, 
        2.54, 2.5, 2.58, 2.78, 3.01, 
        3.12, 2.86, 2.58, 2.35, 2.0, 
        1.58, 1.42, 1.32, 1.0, 0.75,
        0.48, 0.28, 0.12, 0.05
    };

    if (wavelength < x[0]) {
        return 5.76;
    }

    else if (wavelength >= x[28]) {
        return 0.05 - 5.16e-11 * (wavelength-x[29]);
    }

    for (int i=0; i<28; i++) {
        if (wavelength >= x[i] && wavelength < x[i+1]) {
            return y[i] + (y[i+1] - y[i]) * (wavelength - x[i]) / (x[i+1] - x[i]);
        }
    }

}

// Calculate the influence of dust extinction (g=0.8, omega=0.3) 
// Ref: Wagenblast & Hartquist, mnras237, 1019 (1989)
// Originally implemented in UCLCHEM
double GetGrainScattering(double av, double wavelength) {

    double c[6] = {1.0e0, 2.006e0, -1.438e0, 7.364e-1, -5.076e-1, -5.920e-2};
    double k[6] = {7.514e-1, 8.490e-1, 1.013e0, 1.282e0, 2.005e0, 5.832e0};

    double tv = av / 1.086;
    double tl = tv * xlamda(wavelength);

    double scat = 0.0;
    double expo;
    if (tl < 1.0) {
        expo = k[0] * tl;
        if (expo < 35.0) {
            scat = c[0] * exp(-expo);
        }
    }
    else {
        for (int i=1; i<6; i++) {
            expo = k[i] * tl;
            if (expo < 35.0) {
                scat = scat + c[i] * exp(-expo);
            }
        }
    }

    return scat;

}

// Calculate lambda bar (in a) according to equ. 4 of van dishoeck
// and black, apj 334, p771 (1988)
double GetCharactWavelength(double h2col, double cocol) {
    double logco = log10(abs(cocol)+1.0);
    double logh2 = log10(abs(h2col)+1.0);

    double lbar = (5675.0 - 200.6*logh2) - (571.6 - 24.09*logh2) * logco 
                + (18.22 - 0.7664*logh2) * pow(logco, 2.0);

    // lbar represents the mean of the wavelengths of the 33
    // dissociating bands weighted by their fractional contribution
    // to the total rate of each depth. lbar cannot be larger than
    // the wavelength of band 33 (1076.1a) and not be smaller than
    // the wavelength of band 1 (913.6a).
    return std::min(1076.0, std::max(913.0, lbar));

}