#include "naunet_macros.h"
#include "naunet_physics.h"
#include "naunet_renorm.h"

// clang-format off
int InitRenorm(realtype *ab, SUNMatrix A) {
    // clang-format on
    realtype Hnuclei = GetHNuclei(ab);

    // clang-format off
            
    IJth(A, IDX_ELEM_CL, IDX_ELEM_CL) = 0.0 + 35.0 * ab[IDX_GHClI] / 36.0 / Hnuclei
                                    + 35.0 * ab[IDX_HClII] / 36.0 / Hnuclei +
                                    35.0 * ab[IDX_ClI] / 35.0 / Hnuclei + 35.0 *
                                    ab[IDX_ClII] / 35.0 / Hnuclei + 35.0 *
                                    ab[IDX_H2ClII] / 37.0 / Hnuclei + 35.0 *
                                    ab[IDX_HClI] / 36.0 / Hnuclei;
    IJth(A, IDX_ELEM_CL, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_CL, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_CL, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_CL, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_CL, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_CL, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_CL, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_CL, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_GHClI] / 36.0 / Hnuclei +
                                    1.0 * ab[IDX_HClII] / 36.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2ClII] / 37.0 / Hnuclei + 1.0 *
                                    ab[IDX_HClI] / 36.0 / Hnuclei;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_CL) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_MG) = 0.0 + 24.0 * ab[IDX_GMgI] / 24.0 / Hnuclei +
                                    24.0 * ab[IDX_MgI] / 24.0 / Hnuclei + 24.0 *
                                    ab[IDX_MgII] / 24.0 / Hnuclei;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_H) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_CL) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei
                                    + 28.0 * ab[IDX_GSiCI] / 40.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiC3I] / 64.0 / Hnuclei +
                                    28.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_GSiSI] / 60.0 / Hnuclei + 28.0
                                    * ab[IDX_H2SiOI] / 46.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3II] / 64.0 / Hnuclei + 28.0 *
                                    ab[IDX_GSiH4I] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH5II] / 33.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC3I] / 64.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2II] / 52.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiC2I] / 52.0 / Hnuclei + 28.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiSII] / 60.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiSI] / 60.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOII] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiI] / 28.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiII] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_S) = 0.0 + 32.0 * ab[IDX_GSiSI] / 60.0 / Hnuclei
                                    + 32.0 * ab[IDX_HSiSII] / 61.0 / Hnuclei +
                                    32.0 * ab[IDX_SiSII] / 60.0 / Hnuclei + 32.0
                                    * ab[IDX_SiSI] / 60.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei
                                    + 16.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    16.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei +
                                    16.0 * ab[IDX_SiOII] / 44.0 / Hnuclei + 16.0
                                    * ab[IDX_SiOHII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GSiCI] / 40.0 / Hnuclei
                                    + 24.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei +
                                    36.0 * ab[IDX_GSiC3I] / 64.0 / Hnuclei +
                                    36.0 * ab[IDX_SiC3II] / 64.0 / Hnuclei +
                                    36.0 * ab[IDX_SiC3I] / 64.0 / Hnuclei + 24.0
                                    * ab[IDX_SiC2II] / 52.0 / Hnuclei + 24.0 *
                                    ab[IDX_SiC2I] / 52.0 / Hnuclei + 12.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 12.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_H) = 0.0 + 2.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei
                                    + 2.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei +
                                    4.0 * ab[IDX_GSiH4I] / 32.0 / Hnuclei + 5.0
                                    * ab[IDX_SiH5II] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 3.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_CL) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GSiSI] / 60.0 / Hnuclei
                                    + 28.0 * ab[IDX_HSiSII] / 61.0 / Hnuclei +
                                    28.0 * ab[IDX_SiSII] / 60.0 / Hnuclei + 28.0
                                    * ab[IDX_SiSI] / 60.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_S) = 0.0 + 32.0 * ab[IDX_GCSI] / 44.0 / Hnuclei +
                                    32.0 * ab[IDX_GNSI] / 46.0 / Hnuclei + 32.0
                                    * ab[IDX_GOCSI] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_GSiSI] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_GSOI] / 48.0 / Hnuclei + 32.0 *
                                    ab[IDX_GSO2I] / 64.0 / Hnuclei + 128.0 *
                                    ab[IDX_GH2S2I] / 66.0 / Hnuclei + 32.0 *
                                    ab[IDX_GH2CSI] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_GH2SI] / 34.0 / Hnuclei + 128.0 *
                                    ab[IDX_H2S2I] / 66.0 / Hnuclei + 32.0 *
                                    ab[IDX_HNSII] / 47.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 128.0 *
                                    ab[IDX_H2S2II] / 66.0 / Hnuclei + 32.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 128.0 *
                                    ab[IDX_HS2I] / 65.0 / Hnuclei + 32.0 *
                                    ab[IDX_NSII] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 32.0 *
                                    ab[IDX_SO2II] / 64.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 32.0 *
                                    ab[IDX_SiSII] / 60.0 / Hnuclei + 128.0 *
                                    ab[IDX_HS2II] / 65.0 / Hnuclei + 128.0 *
                                    ab[IDX_S2II] / 64.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_SiSI] / 60.0 / Hnuclei + 128.0 *
                                    ab[IDX_S2I] / 64.0 / Hnuclei + 32.0 *
                                    ab[IDX_NSI] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_SO2I] / 64.0 / Hnuclei + 32.0 *
                                    ab[IDX_CSII] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_H3SII] / 35.0 / Hnuclei + 32.0 *
                                    ab[IDX_SOI] / 48.0 / Hnuclei + 32.0 *
                                    ab[IDX_CSI] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_SOII] / 48.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2SII] / 34.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSI] / 33.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSII] / 33.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2SI] / 34.0 / Hnuclei + 32.0 *
                                    ab[IDX_SII] / 32.0 / Hnuclei + 32.0 *
                                    ab[IDX_SI] / 32.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GNSI] / 46.0 / Hnuclei +
                                    14.0 * ab[IDX_HNSII] / 47.0 / Hnuclei + 14.0
                                    * ab[IDX_NSII] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_NSI] / 46.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_S, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GOCSI] / 60.0 / Hnuclei
                                    + 16.0 * ab[IDX_GSOI] / 48.0 / Hnuclei +
                                    32.0 * ab[IDX_GSO2I] / 64.0 / Hnuclei + 16.0
                                    * ab[IDX_HSOII] / 49.0 / Hnuclei + 32.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 16.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 32.0 *
                                    ab[IDX_SO2II] / 64.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_SO2I] / 64.0 / Hnuclei + 16.0 *
                                    ab[IDX_SOI] / 48.0 / Hnuclei + 16.0 *
                                    ab[IDX_SOII] / 48.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GCSI] / 44.0 / Hnuclei +
                                    12.0 * ab[IDX_GOCSI] / 60.0 / Hnuclei + 12.0
                                    * ab[IDX_GH2CSI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 12.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 12.0 *
                                    ab[IDX_CSII] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_CSI] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei;
    IJth(A, IDX_ELEM_S, IDX_ELEM_H) = 0.0 + 4.0 * ab[IDX_GH2S2I] / 66.0 / Hnuclei
                                    + 2.0 * ab[IDX_GH2CSI] / 46.0 / Hnuclei +
                                    2.0 * ab[IDX_GH2SI] / 34.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2S2I] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSII] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2S2II] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 2.0 *
                                    ab[IDX_HS2I] / 65.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 2.0 *
                                    ab[IDX_HS2II] / 65.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3SII] / 35.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SII] / 34.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSI] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSII] / 33.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SI] / 34.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_CL) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_S) = 0.0 + 32.0 * ab[IDX_GNSI] / 46.0 / Hnuclei +
                                    32.0 * ab[IDX_HNSII] / 47.0 / Hnuclei + 32.0
                                    * ab[IDX_NSII] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_NSI] / 46.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 14.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNO2I] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3CNHI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GC4NI] / 62.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_GNCCNI] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNOI] / 30.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNOI] / 31.0 /
                                    Hnuclei + 56.0 * ab[IDX_GN2I] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNSI] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_GCH3CNHI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_C4NI] / 62.0 /
                                    Hnuclei + 14.0 * ab[IDX_GNH3I] / 17.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2NOII] / 32.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNSII] / 47.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 56.0 * ab[IDX_C2N2II] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_C4NII] / 62.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 14.0 * ab[IDX_C3NI] / 50.0 /
                                    Hnuclei + 14.0 * ab[IDX_NSII] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_NO2I] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2NI] / 38.0 /
                                    Hnuclei + 56.0 * ab[IDX_NCCNI] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 14.0 * ab[IDX_NSI] / 46.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNOI] / 31.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2NII] / 38.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNII] / 26.0 /
                                    Hnuclei + 56.0 * ab[IDX_N2HII] / 29.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNOII] / 31.0 /
                                    Hnuclei + 56.0 * ab[IDX_N2II] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_NHII] / 15.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH2I] / 16.0 /
                                    Hnuclei + 14.0 * ab[IDX_NII] / 14.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH2II] / 16.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH4II] / 18.0 /
                                    Hnuclei + 14.0 * ab[IDX_NOII] / 30.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH3II] / 17.0 /
                                    Hnuclei + 56.0 * ab[IDX_N2I] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_NHI] / 15.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNI] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_NOI] / 30.0 /
                                    Hnuclei + 14.0 * ab[IDX_NH3I] / 17.0 /
                                    Hnuclei + 14.0 * ab[IDX_NI] / 14.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei
                                    + 16.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    16.0 * ab[IDX_GNOI] / 30.0 / Hnuclei + 16.0
                                    * ab[IDX_GHNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 32.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_C) = 0.0 + 24.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 12.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CNHI] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_GC4NI] / 62.0 /
                                    Hnuclei + 36.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 48.0 * ab[IDX_GNCCNI] / 52.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCH3CNHI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 48.0 * ab[IDX_C4NI] / 62.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 48.0 * ab[IDX_C2N2II] / 52.0 /
                                    Hnuclei + 48.0 * ab[IDX_C4NII] / 62.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 36.0 * ab[IDX_C3NI] / 50.0 /
                                    Hnuclei + 36.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2NI] / 38.0 /
                                    Hnuclei + 48.0 * ab[IDX_NCCNI] / 52.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2NII] / 38.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNII] / 26.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 12.0 * ab[IDX_CNI] / 26.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_H) = 0.0 + 3.0 * ab[IDX_GCH3CNI] / 41.0 / Hnuclei
                                    + 2.0 * ab[IDX_GH2CNI] / 28.0 / Hnuclei +
                                    1.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3CNHI] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHC3NI] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH3CNHI] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 3.0 *
                                    ab[IDX_GNH3I] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSII] / 47.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2NHII] / 39.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC3NI] / 51.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3CNHII] / 42.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3CNI] / 41.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 2.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 2.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH4II] / 18.0 / Hnuclei + 3.0 *
                                    ab[IDX_NH3II] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 3.0 *
                                    ab[IDX_NH3I] / 17.0 / Hnuclei;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_CL) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_S) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_HE) = 0.0 + 4.0 * ab[IDX_HeHII] / 5.0 / Hnuclei +
                                    4.0 * ab[IDX_HeII] / 4.0 / Hnuclei + 4.0 *
                                    ab[IDX_HeI] / 4.0 / Hnuclei;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_HeHII] / 5.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_CL) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei
                                    + 28.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei +
                                    28.0 * ab[IDX_SiOII] / 44.0 / Hnuclei + 28.0
                                    * ab[IDX_SiOHII] / 45.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiOI] / 44.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_S) = 0.0 + 32.0 * ab[IDX_GOCSI] / 60.0 / Hnuclei
                                    + 32.0 * ab[IDX_GSOI] / 48.0 / Hnuclei +
                                    64.0 * ab[IDX_GSO2I] / 64.0 / Hnuclei + 32.0
                                    * ab[IDX_HSOII] / 49.0 / Hnuclei + 64.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 32.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 64.0 *
                                    ab[IDX_SO2II] / 64.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 64.0 *
                                    ab[IDX_SO2I] / 64.0 / Hnuclei + 32.0 *
                                    ab[IDX_SOI] / 48.0 / Hnuclei + 32.0 *
                                    ab[IDX_SOII] / 48.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_N) = 0.0 + 28.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei
                                    + 14.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    14.0 * ab[IDX_GNOI] / 30.0 / Hnuclei + 14.0
                                    * ab[IDX_GHNOI] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 28.0 *
                                    ab[IDX_NO2I] / 46.0 / Hnuclei + 14.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOII] / 30.0 / Hnuclei + 14.0 *
                                    ab[IDX_NOI] / 30.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GCH2COI] / 42.0 /
                                    Hnuclei + 64.0 * ab[IDX_GNO2I] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_GSiOI] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCOI] / 28.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 16.0 * ab[IDX_GNOI] / 30.0 /
                                    Hnuclei + 64.0 * ab[IDX_GO2I] / 32.0 /
                                    Hnuclei + 64.0 * ab[IDX_GO2HI] / 33.0 /
                                    Hnuclei + 16.0 * ab[IDX_GC2H5OHI] / 46.0 /
                                    Hnuclei + 64.0 * ab[IDX_GCO2I] / 44.0 /
                                    Hnuclei + 64.0 * ab[IDX_GHCOOCH3I] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_GH2SiOI] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHNOI] / 31.0 /
                                    Hnuclei + 16.0 * ab[IDX_GOCSI] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_GSOI] / 48.0 /
                                    Hnuclei + 64.0 * ab[IDX_GSO2I] / 64.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCH3OHI] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_GH2COI] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_GH2OI] / 18.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2NOII] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2SiOI] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCII] / 29.0 /
                                    Hnuclei + 16.0 * ab[IDX_HSOII] / 49.0 /
                                    Hnuclei + 64.0 * ab[IDX_H5C2O2II] / 61.0 /
                                    Hnuclei + 16.0 * ab[IDX_C2H5OH2II] / 47.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH2COI] / 42.0 /
                                    Hnuclei + 64.0 * ab[IDX_HSO2II] / 65.0 /
                                    Hnuclei + 64.0 * ab[IDX_HCOOCH3I] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 64.0 * ab[IDX_SO2II] / 64.0 /
                                    Hnuclei + 64.0 * ab[IDX_O2HI] / 33.0 /
                                    Hnuclei + 64.0 * ab[IDX_NO2I] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_C2H5OHI] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH3OH2II] / 33.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCSII] / 60.0 /
                                    Hnuclei + 64.0 * ab[IDX_SO2I] / 64.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNOI] / 31.0 /
                                    Hnuclei + 16.0 * ab[IDX_SiOII] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_SiOHII] / 45.0 /
                                    Hnuclei + 64.0 * ab[IDX_HCO2II] / 45.0 /
                                    Hnuclei + 16.0 * ab[IDX_SiOI] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH3OHI] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_SOI] / 48.0 /
                                    Hnuclei + 16.0 * ab[IDX_SOII] / 48.0 /
                                    Hnuclei + 64.0 * ab[IDX_O2HII] / 33.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCSI] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNOII] / 31.0 /
                                    Hnuclei + 16.0 * ab[IDX_H3COII] / 31.0 /
                                    Hnuclei + 16.0 * ab[IDX_COII] / 28.0 /
                                    Hnuclei + 64.0 * ab[IDX_CO2I] / 44.0 /
                                    Hnuclei + 64.0 * ab[IDX_O2II] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_OII] / 16.0 /
                                    Hnuclei + 16.0 * ab[IDX_NOII] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2COII] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2OII] / 18.0 /
                                    Hnuclei + 16.0 * ab[IDX_OHII] / 17.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2COI] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCOI] / 29.0 /
                                    Hnuclei + 16.0 * ab[IDX_NOI] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_H3OII] / 19.0 /
                                    Hnuclei + 64.0 * ab[IDX_O2I] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_OHI] / 17.0 /
                                    Hnuclei + 16.0 * ab[IDX_OI] / 16.0 / Hnuclei
                                    + 16.0 * ab[IDX_HCOII] / 29.0 / Hnuclei +
                                    16.0 * ab[IDX_H2OI] / 18.0 / Hnuclei + 16.0
                                    * ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_C) = 0.0 + 24.0 * ab[IDX_GCH2COI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCOI] / 28.0 /
                                    Hnuclei + 12.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 24.0 * ab[IDX_GC2H5OHI] / 46.0 /
                                    Hnuclei + 24.0 * ab[IDX_GCO2I] / 44.0 /
                                    Hnuclei + 48.0 * ab[IDX_GHCOOCH3I] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_GOCSI] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_GCH3OHI] / 32.0 /
                                    Hnuclei + 12.0 * ab[IDX_GH2COI] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCII] / 29.0 /
                                    Hnuclei + 48.0 * ab[IDX_H5C2O2II] / 61.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2H5OH2II] / 47.0 /
                                    Hnuclei + 24.0 * ab[IDX_CH2COI] / 42.0 /
                                    Hnuclei + 48.0 * ab[IDX_HCOOCH3I] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 24.0 * ab[IDX_C2H5OHI] / 46.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3OH2II] / 33.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCSII] / 60.0 /
                                    Hnuclei + 24.0 * ab[IDX_HCO2II] / 45.0 /
                                    Hnuclei + 12.0 * ab[IDX_CH3OHI] / 32.0 /
                                    Hnuclei + 12.0 * ab[IDX_OCSI] / 60.0 /
                                    Hnuclei + 12.0 * ab[IDX_H3COII] / 31.0 /
                                    Hnuclei + 12.0 * ab[IDX_COII] / 28.0 /
                                    Hnuclei + 24.0 * ab[IDX_CO2I] / 44.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2COII] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_H2COI] / 30.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOI] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_HCOII] / 29.0 /
                                    Hnuclei + 12.0 * ab[IDX_COI] / 28.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_H) = 0.0 + 2.0 * ab[IDX_GCH2COI] / 42.0 / Hnuclei
                                    + 1.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei +
                                    2.0 * ab[IDX_GO2HI] / 33.0 / Hnuclei + 6.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 8.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 10.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 7.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 2.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 8.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 2.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 6.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 5.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_CL) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GSiCI] / 40.0 / Hnuclei
                                    + 56.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei +
                                    84.0 * ab[IDX_GSiC3I] / 64.0 / Hnuclei +
                                    84.0 * ab[IDX_SiC3II] / 64.0 / Hnuclei +
                                    84.0 * ab[IDX_SiC3I] / 64.0 / Hnuclei + 56.0
                                    * ab[IDX_SiC2II] / 52.0 / Hnuclei + 56.0 *
                                    ab[IDX_SiC2I] / 52.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 28.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_S) = 0.0 + 32.0 * ab[IDX_GCSI] / 44.0 / Hnuclei +
                                    32.0 * ab[IDX_GOCSI] / 60.0 / Hnuclei + 32.0
                                    * ab[IDX_GH2CSI] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 32.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 32.0 *
                                    ab[IDX_CSII] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 32.0 *
                                    ab[IDX_CSI] / 44.0 / Hnuclei + 32.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_N) = 0.0 + 28.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 14.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH3CNHI] / 42.0 /
                                    Hnuclei + 56.0 * ab[IDX_GC4NI] / 62.0 /
                                    Hnuclei + 42.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 56.0 * ab[IDX_GNCCNI] / 52.0 /
                                    Hnuclei + 28.0 * ab[IDX_GCH3CNHI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 56.0 * ab[IDX_C4NI] / 62.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 56.0 * ab[IDX_C2N2II] / 52.0 /
                                    Hnuclei + 56.0 * ab[IDX_C4NII] / 62.0 /
                                    Hnuclei + 28.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 42.0 * ab[IDX_C3NI] / 50.0 /
                                    Hnuclei + 42.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 28.0 * ab[IDX_C2NI] / 38.0 /
                                    Hnuclei + 56.0 * ab[IDX_NCCNI] / 52.0 /
                                    Hnuclei + 14.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 28.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 28.0 * ab[IDX_C2NII] / 38.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNII] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_CNI] / 26.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_GCH2COI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCOI] / 28.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 32.0 * ab[IDX_GC2H5OHI] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_GCO2I] / 44.0 /
                                    Hnuclei + 64.0 * ab[IDX_GHCOOCH3I] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_GOCSI] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_GCH3OHI] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_GH2COI] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCII] / 29.0 /
                                    Hnuclei + 64.0 * ab[IDX_H5C2O2II] / 61.0 /
                                    Hnuclei + 32.0 * ab[IDX_C2H5OH2II] / 47.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH2COI] / 42.0 /
                                    Hnuclei + 64.0 * ab[IDX_HCOOCH3I] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 32.0 * ab[IDX_C2H5OHI] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH3OH2II] / 33.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCNI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCSII] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCO2II] / 45.0 /
                                    Hnuclei + 16.0 * ab[IDX_CH3OHI] / 32.0 /
                                    Hnuclei + 16.0 * ab[IDX_OCSI] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_H3COII] / 31.0 /
                                    Hnuclei + 16.0 * ab[IDX_COII] / 28.0 /
                                    Hnuclei + 32.0 * ab[IDX_CO2I] / 44.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2COII] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_H2COI] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCOI] / 29.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCOII] / 29.0 /
                                    Hnuclei + 16.0 * ab[IDX_COI] / 28.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_C) = 0.0 + 48.0 * ab[IDX_GC2H3I] / 27.0 / Hnuclei
                                    + 48.0 * ab[IDX_GC2H4I] / 28.0 / Hnuclei +
                                    48.0 * ab[IDX_GC2H5I] / 29.0 / Hnuclei +
                                    108.0 * ab[IDX_GC3H2I] / 38.0 / Hnuclei +
                                    192.0 * ab[IDX_GC4HI] / 49.0 / Hnuclei +
                                    48.0 * ab[IDX_GCH2COI] / 42.0 / Hnuclei +
                                    48.0 * ab[IDX_GCH3CNI] / 41.0 / Hnuclei +
                                    12.0 * ab[IDX_GCSI] / 44.0 / Hnuclei + 12.0
                                    * ab[IDX_GH2CNI] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_GHNCI] / 27.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH3CNHI] / 42.0 / Hnuclei + 48.0 *
                                    ab[IDX_GC2HI] / 25.0 / Hnuclei + 48.0 *
                                    ab[IDX_GC2H2I] / 26.0 / Hnuclei + 192.0 *
                                    ab[IDX_GC4NI] / 62.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCOI] / 28.0 / Hnuclei + 108.0 *
                                    ab[IDX_GHC3NI] / 51.0 / Hnuclei + 12.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 48.0 *
                                    ab[IDX_GNCCNI] / 52.0 / Hnuclei + 12.0 *
                                    ab[IDX_GSiCI] / 40.0 / Hnuclei + 48.0 *
                                    ab[IDX_GSiC2I] / 52.0 / Hnuclei + 108.0 *
                                    ab[IDX_GSiC3I] / 64.0 / Hnuclei + 48.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 108.0 *
                                    ab[IDX_GCH3CCHI] / 40.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCO2I] / 44.0 / Hnuclei + 48.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 48.0 *
                                    ab[IDX_GC2I] / 24.0 / Hnuclei + 12.0 *
                                    ab[IDX_GOCSI] / 60.0 / Hnuclei + 48.0 *
                                    ab[IDX_GCH3CNHI] / 42.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 12.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_GH2CSI] / 46.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4NI] / 62.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3H2I] / 38.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 108.0 *
                                    ab[IDX_SiC3II] / 64.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4HI] / 49.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2N2II] / 52.0 / Hnuclei + 192.0 *
                                    ab[IDX_C4NII] / 62.0 / Hnuclei + 48.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 108.0 *
                                    ab[IDX_SiC3I] / 64.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2NHII] / 39.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3H5II] / 41.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3NI] / 50.0 / Hnuclei + 48.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 48.0 *
                                    ab[IDX_SiC2II] / 52.0 / Hnuclei + 108.0 *
                                    ab[IDX_C3II] / 36.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 108.0 *
                                    ab[IDX_CH3CCHI] / 40.0 / Hnuclei + 12.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 48.0 *
                                    ab[IDX_SiC2I] / 52.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 108.0 *
                                    ab[IDX_HC3NI] / 51.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH3CNHII] / 42.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H5I] / 29.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_SiCII] / 40.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2NI] / 38.0 / Hnuclei + 48.0 *
                                    ab[IDX_NCCNI] / 52.0 / Hnuclei + 12.0 *
                                    ab[IDX_SiCI] / 40.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCNI] / 42.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH3CNI] / 41.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCSII] / 60.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H3I] / 27.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2NII] / 38.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_CSII] / 44.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H4I] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_CSI] / 44.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2II] / 24.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNII] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_OCSI] / 60.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH5II] / 17.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H2I] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_COII] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_CO2I] / 44.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H2II] / 26.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2I] / 24.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_CNI] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_CII] / 12.0 / Hnuclei + 12.0 *
                                    ab[IDX_CI] / 12.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_COI] / 28.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_H) = 0.0 + 6.0 * ab[IDX_GC2H3I] / 27.0 / Hnuclei
                                    + 8.0 * ab[IDX_GC2H4I] / 28.0 / Hnuclei +
                                    10.0 * ab[IDX_GC2H5I] / 29.0 / Hnuclei + 6.0
                                    * ab[IDX_GC3H2I] / 38.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC4HI] / 49.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH2COI] / 42.0 / Hnuclei + 6.0 *
                                    ab[IDX_GCH3CNI] / 41.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2CNI] / 28.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCI] / 27.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3CNHI] / 42.0 / Hnuclei + 2.0 *
                                    ab[IDX_GC2HI] / 25.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC2H2I] / 26.0 / Hnuclei + 3.0 *
                                    ab[IDX_GHC3NI] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_GCH3CCHI] / 40.0 / Hnuclei + 8.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 8.0 *
                                    ab[IDX_GCH3CNHI] / 42.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_GH2CSI] / 46.0 / Hnuclei + 6.0 *
                                    ab[IDX_C3H2I] / 38.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_C4HI] / 49.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 10.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 14.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2NHII] / 39.0 / Hnuclei + 15.0 *
                                    ab[IDX_C3H5II] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 8.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 12.0 *
                                    ab[IDX_CH3CCHI] / 40.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 3.0 *
                                    ab[IDX_HC3NI] / 51.0 / Hnuclei + 8.0 *
                                    ab[IDX_CH3CNHII] / 42.0 / Hnuclei + 10.0 *
                                    ab[IDX_C2H5I] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 5.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 6.0 *
                                    ab[IDX_CH3CNI] / 41.0 / Hnuclei + 6.0 *
                                    ab[IDX_C2H3I] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 8.0 *
                                    ab[IDX_C2H4I] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 5.0 *
                                    ab[IDX_CH5II] / 17.0 / Hnuclei + 2.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 3.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2H2I] / 26.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 2.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2H2II] / 26.0 / Hnuclei + 2.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 3.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 2.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_CL) = 0.0 + 35.0 * ab[IDX_GHClI] / 36.0 / Hnuclei
                                    + 35.0 * ab[IDX_HClII] / 36.0 / Hnuclei +
                                    70.0 * ab[IDX_H2ClII] / 37.0 / Hnuclei +
                                    35.0 * ab[IDX_HClI] / 36.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_SI) = 0.0 + 56.0 * ab[IDX_GH2SiOI] / 46.0 /
                                    Hnuclei + 56.0 * ab[IDX_H2SiOI] / 46.0 /
                                    Hnuclei + 112.0 * ab[IDX_GSiH4I] / 32.0 /
                                    Hnuclei + 140.0 * ab[IDX_SiH5II] / 33.0 /
                                    Hnuclei + 112.0 * ab[IDX_SiH4II] / 32.0 /
                                    Hnuclei + 28.0 * ab[IDX_HSiSII] / 61.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiH2I] / 30.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiH3II] / 31.0 /
                                    Hnuclei + 84.0 * ab[IDX_SiH3I] / 31.0 /
                                    Hnuclei + 56.0 * ab[IDX_SiH2II] / 30.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiHI] / 29.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiOHII] / 45.0 /
                                    Hnuclei + 28.0 * ab[IDX_SiHII] / 29.0 /
                                    Hnuclei + 112.0 * ab[IDX_SiH4I] / 32.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_S) = 0.0 + 128.0 * ab[IDX_GH2S2I] / 66.0 /
                                    Hnuclei + 64.0 * ab[IDX_GH2CSI] / 46.0 /
                                    Hnuclei + 64.0 * ab[IDX_GH2SI] / 34.0 /
                                    Hnuclei + 128.0 * ab[IDX_H2S2I] / 66.0 /
                                    Hnuclei + 32.0 * ab[IDX_HNSII] / 47.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSOII] / 49.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSO2II] / 65.0 /
                                    Hnuclei + 128.0 * ab[IDX_H2S2II] / 66.0 /
                                    Hnuclei + 32.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 64.0 * ab[IDX_HS2I] / 65.0 /
                                    Hnuclei + 96.0 * ab[IDX_H3CSII] / 47.0 /
                                    Hnuclei + 64.0 * ab[IDX_H2CSI] / 46.0 /
                                    Hnuclei + 64.0 * ab[IDX_H2CSII] / 46.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSiSII] / 61.0 /
                                    Hnuclei + 64.0 * ab[IDX_HS2II] / 65.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCSI] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCSII] / 45.0 /
                                    Hnuclei + 96.0 * ab[IDX_H3SII] / 35.0 /
                                    Hnuclei + 64.0 * ab[IDX_H2SII] / 34.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSI] / 33.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSII] / 33.0 /
                                    Hnuclei + 64.0 * ab[IDX_H2SI] / 34.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_N) = 0.0 + 42.0 * ab[IDX_GCH3CNI] / 41.0 /
                                    Hnuclei + 28.0 * ab[IDX_GH2CNI] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCI] / 27.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3CNHI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHC3NI] / 51.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHNOI] / 31.0 /
                                    Hnuclei + 56.0 * ab[IDX_GCH3CNHI] / 42.0 /
                                    Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 /
                                    Hnuclei + 42.0 * ab[IDX_GNH3I] / 17.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2NOII] / 32.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNSII] / 47.0 /
                                    Hnuclei + 28.0 * ab[IDX_H2CNI] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_C2NHII] / 39.0 /
                                    Hnuclei + 14.0 * ab[IDX_HC3NI] / 51.0 /
                                    Hnuclei + 56.0 * ab[IDX_CH3CNHII] / 42.0 /
                                    Hnuclei + 42.0 * ab[IDX_CH3CNI] / 41.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNOI] / 31.0 /
                                    Hnuclei + 28.0 * ab[IDX_N2HII] / 29.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNOII] / 31.0 /
                                    Hnuclei + 28.0 * ab[IDX_HCNHII] / 28.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNII] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_HNCI] / 27.0 /
                                    Hnuclei + 14.0 * ab[IDX_NHII] / 15.0 /
                                    Hnuclei + 28.0 * ab[IDX_NH2I] / 16.0 /
                                    Hnuclei + 28.0 * ab[IDX_NH2II] / 16.0 /
                                    Hnuclei + 56.0 * ab[IDX_NH4II] / 18.0 /
                                    Hnuclei + 42.0 * ab[IDX_NH3II] / 17.0 /
                                    Hnuclei + 14.0 * ab[IDX_NHI] / 15.0 /
                                    Hnuclei + 14.0 * ab[IDX_HCNI] / 27.0 /
                                    Hnuclei + 42.0 * ab[IDX_NH3I] / 17.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_HE) = 0.0 + 4.0 * ab[IDX_HeHII] / 5.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_O) = 0.0 + 32.0 * ab[IDX_GCH2COI] / 42.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHNCOI] / 43.0 /
                                    Hnuclei + 32.0 * ab[IDX_GO2HI] / 33.0 /
                                    Hnuclei + 96.0 * ab[IDX_GC2H5OHI] / 46.0 /
                                    Hnuclei + 128.0 * ab[IDX_GHCOOCH3I] / 60.0 /
                                    Hnuclei + 32.0 * ab[IDX_GH2SiOI] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_GHNOI] / 31.0 /
                                    Hnuclei + 64.0 * ab[IDX_GCH3OHI] / 32.0 /
                                    Hnuclei + 32.0 * ab[IDX_GH2COI] / 30.0 /
                                    Hnuclei + 32.0 * ab[IDX_GH2OI] / 18.0 /
                                    Hnuclei + 32.0 * ab[IDX_H2NOII] / 32.0 /
                                    Hnuclei + 32.0 * ab[IDX_H2SiOI] / 46.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNCOI] / 43.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCII] / 29.0 /
                                    Hnuclei + 16.0 * ab[IDX_HSOII] / 49.0 /
                                    Hnuclei + 160.0 * ab[IDX_H5C2O2II] / 61.0 /
                                    Hnuclei + 112.0 * ab[IDX_C2H5OH2II] / 47.0 /
                                    Hnuclei + 32.0 * ab[IDX_CH2COI] / 42.0 /
                                    Hnuclei + 32.0 * ab[IDX_HSO2II] / 65.0 /
                                    Hnuclei + 128.0 * ab[IDX_HCOOCH3I] / 60.0 /
                                    Hnuclei + 16.0 * ab[IDX_HOCSII] / 61.0 /
                                    Hnuclei + 32.0 * ab[IDX_O2HI] / 33.0 /
                                    Hnuclei + 96.0 * ab[IDX_C2H5OHI] / 46.0 /
                                    Hnuclei + 80.0 * ab[IDX_CH3OH2II] / 33.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNOI] / 31.0 /
                                    Hnuclei + 16.0 * ab[IDX_SiOHII] / 45.0 /
                                    Hnuclei + 32.0 * ab[IDX_HCO2II] / 45.0 /
                                    Hnuclei + 64.0 * ab[IDX_CH3OHI] / 32.0 /
                                    Hnuclei + 32.0 * ab[IDX_O2HII] / 33.0 /
                                    Hnuclei + 16.0 * ab[IDX_HNOII] / 31.0 /
                                    Hnuclei + 48.0 * ab[IDX_H3COII] / 31.0 /
                                    Hnuclei + 32.0 * ab[IDX_H2COII] / 30.0 /
                                    Hnuclei + 32.0 * ab[IDX_H2OII] / 18.0 /
                                    Hnuclei + 16.0 * ab[IDX_OHII] / 17.0 /
                                    Hnuclei + 32.0 * ab[IDX_H2COI] / 30.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCOI] / 29.0 /
                                    Hnuclei + 48.0 * ab[IDX_H3OII] / 19.0 /
                                    Hnuclei + 16.0 * ab[IDX_OHI] / 17.0 /
                                    Hnuclei + 16.0 * ab[IDX_HCOII] / 29.0 /
                                    Hnuclei + 32.0 * ab[IDX_H2OI] / 18.0 /
                                    Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_C) = 0.0 + 72.0 * ab[IDX_GC2H3I] / 27.0 / Hnuclei
                                    + 96.0 * ab[IDX_GC2H4I] / 28.0 / Hnuclei +
                                    120.0 * ab[IDX_GC2H5I] / 29.0 / Hnuclei +
                                    72.0 * ab[IDX_GC3H2I] / 38.0 / Hnuclei +
                                    48.0 * ab[IDX_GC4HI] / 49.0 / Hnuclei + 48.0
                                    * ab[IDX_GCH2COI] / 42.0 / Hnuclei + 72.0 *
                                    ab[IDX_GCH3CNI] / 41.0 / Hnuclei + 24.0 *
                                    ab[IDX_GH2CNI] / 28.0 / Hnuclei + 12.0 *
                                    ab[IDX_GHNCI] / 27.0 / Hnuclei + 96.0 *
                                    ab[IDX_CH3CNHI] / 42.0 / Hnuclei + 24.0 *
                                    ab[IDX_GC2HI] / 25.0 / Hnuclei + 48.0 *
                                    ab[IDX_GC2H2I] / 26.0 / Hnuclei + 36.0 *
                                    ab[IDX_GHC3NI] / 51.0 / Hnuclei + 12.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 144.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 144.0 *
                                    ab[IDX_GCH3CCHI] / 40.0 / Hnuclei + 96.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 96.0 *
                                    ab[IDX_GCH3CNHI] / 42.0 / Hnuclei + 48.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 24.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 12.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 24.0 *
                                    ab[IDX_GH2CSI] / 46.0 / Hnuclei + 72.0 *
                                    ab[IDX_C3H2I] / 38.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 48.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 48.0 *
                                    ab[IDX_C4HI] / 49.0 / Hnuclei + 24.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 120.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 168.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2NHII] / 39.0 / Hnuclei + 180.0 *
                                    ab[IDX_C3H5II] / 41.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 96.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 12.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 144.0 *
                                    ab[IDX_CH3CCHI] / 40.0 / Hnuclei + 36.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 24.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 24.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 36.0 *
                                    ab[IDX_HC3NI] / 51.0 / Hnuclei + 96.0 *
                                    ab[IDX_CH3CNHII] / 42.0 / Hnuclei + 120.0 *
                                    ab[IDX_C2H5I] / 29.0 / Hnuclei + 144.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 60.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 72.0 *
                                    ab[IDX_CH3CNI] / 41.0 / Hnuclei + 72.0 *
                                    ab[IDX_C2H3I] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 96.0 *
                                    ab[IDX_C2H4I] / 28.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 60.0 *
                                    ab[IDX_CH5II] / 17.0 / Hnuclei + 24.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 36.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H2I] / 26.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 24.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 24.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 24.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 48.0 *
                                    ab[IDX_C2H2II] / 26.0 / Hnuclei + 24.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 36.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 24.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 48.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 12.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_H) = 0.0 + 9.0 * ab[IDX_GC2H3I] / 27.0 / Hnuclei
                                    + 16.0 * ab[IDX_GC2H4I] / 28.0 / Hnuclei +
                                    25.0 * ab[IDX_GC2H5I] / 29.0 / Hnuclei + 4.0
                                    * ab[IDX_GC3H2I] / 38.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC4HI] / 49.0 / Hnuclei + 4.0 *
                                    ab[IDX_GCH2COI] / 42.0 / Hnuclei + 9.0 *
                                    ab[IDX_GCH3CNI] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2CNI] / 28.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCI] / 27.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CNHI] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_GC2HI] / 25.0 / Hnuclei + 4.0 *
                                    ab[IDX_GC2H2I] / 26.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHC3NI] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_GO2HI] / 33.0 / Hnuclei + 36.0 *
                                    ab[IDX_GC2H5OHI] / 46.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3CCHI] / 40.0 / Hnuclei + 16.0 *
                                    ab[IDX_GHCOOCH3I] / 60.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHNOI] / 31.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3CNHI] / 42.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH3OHI] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2COI] / 30.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2S2I] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHClI] / 36.0 / Hnuclei + 1.0 *
                                    ab[IDX_GHCNI] / 27.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2CSI] / 46.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2OI] / 18.0 / Hnuclei + 4.0 *
                                    ab[IDX_GH2SI] / 34.0 / Hnuclei + 4.0 *
                                    ab[IDX_C3H2I] / 38.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2S2I] / 66.0 / Hnuclei + 9.0 *
                                    ab[IDX_GNH3I] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2NOII] / 32.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2SiOI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HClII] / 36.0 / Hnuclei + 1.0 *
                                    ab[IDX_HeHII] / 5.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNSII] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSOII] / 49.0 / Hnuclei + 16.0 *
                                    ab[IDX_GCH4I] / 16.0 / Hnuclei + 16.0 *
                                    ab[IDX_GSiH4I] / 32.0 / Hnuclei + 1.0 *
                                    ab[IDX_C4HI] / 49.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CNI] / 28.0 / Hnuclei + 25.0 *
                                    ab[IDX_SiH5II] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2ClII] / 37.0 / Hnuclei + 25.0 *
                                    ab[IDX_H5C2O2II] / 61.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiH4II] / 32.0 / Hnuclei + 49.0 *
                                    ab[IDX_C2H5OH2II] / 47.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2NHII] / 39.0 / Hnuclei + 25.0 *
                                    ab[IDX_C3H5II] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2COI] / 42.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSO2II] / 65.0 / Hnuclei + 16.0 *
                                    ab[IDX_HCOOCH3I] / 60.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2S2II] / 66.0 / Hnuclei + 1.0 *
                                    ab[IDX_HClI] / 36.0 / Hnuclei + 1.0 *
                                    ab[IDX_HOCSII] / 61.0 / Hnuclei + 1.0 *
                                    ab[IDX_HS2I] / 65.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CCHI] / 40.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3CSII] / 47.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CSI] / 46.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2CSII] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HC3NI] / 51.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSiSII] / 61.0 / Hnuclei + 1.0 *
                                    ab[IDX_HS2II] / 65.0 / Hnuclei + 1.0 *
                                    ab[IDX_O2HI] / 33.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3CNHII] / 42.0 / Hnuclei + 25.0 *
                                    ab[IDX_C2H5I] / 29.0 / Hnuclei + 36.0 *
                                    ab[IDX_C2H5OHI] / 46.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSI] / 45.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH2I] / 30.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiH3II] / 31.0 / Hnuclei + 25.0 *
                                    ab[IDX_CH3OH2II] / 33.0 / Hnuclei + 9.0 *
                                    ab[IDX_SiH3I] / 31.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3CNI] / 41.0 / Hnuclei + 4.0 *
                                    ab[IDX_SiH2II] / 30.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOI] / 31.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiOHII] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_SiHII] / 29.0 / Hnuclei + 16.0 *
                                    ab[IDX_SiH4I] / 32.0 / Hnuclei + 9.0 *
                                    ab[IDX_C2H3I] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCO2II] / 45.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCSII] / 45.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH3OHI] / 32.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3SII] / 35.0 / Hnuclei + 16.0 *
                                    ab[IDX_C2H4I] / 28.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH4II] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_N2HII] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2SII] / 34.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNOII] / 31.0 / Hnuclei + 25.0 *
                                    ab[IDX_CH5II] / 17.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSI] / 33.0 / Hnuclei + 4.0 *
                                    ab[IDX_HCNHII] / 28.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2II] / 2.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3COII] / 31.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2H2I] / 26.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNII] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HNCI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_HSII] / 33.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHII] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2I] / 16.0 / Hnuclei + 4.0 *
                                    ab[IDX_NH2II] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2HII] / 25.0 / Hnuclei + 16.0 *
                                    ab[IDX_NH4II] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_C2HI] / 25.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2COII] / 30.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2II] / 14.0 / Hnuclei + 9.0 *
                                    ab[IDX_NH3II] / 17.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2SI] / 34.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OII] / 18.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHII] / 17.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3II] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_C2H2II] / 26.0 / Hnuclei + 4.0 *
                                    ab[IDX_CH2I] / 14.0 / Hnuclei + 9.0 *
                                    ab[IDX_CH3I] / 15.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHII] / 13.0 / Hnuclei + 1.0 *
                                    ab[IDX_NHI] / 15.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2COI] / 30.0 / Hnuclei + 16.0 *
                                    ab[IDX_CH4I] / 16.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOI] / 29.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 *
                                    ab[IDX_CHI] / 13.0 / Hnuclei + 9.0 *
                                    ab[IDX_NH3I] / 17.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3OII] / 19.0 / Hnuclei + 1.0 *
                                    ab[IDX_OHI] / 17.0 / Hnuclei + 9.0 *
                                    ab[IDX_H3II] / 3.0 / Hnuclei + 1.0 *
                                    ab[IDX_HII] / 1.0 / Hnuclei + 1.0 *
                                    ab[IDX_HCOII] / 29.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2OI] / 18.0 / Hnuclei + 4.0 *
                                    ab[IDX_H2I] / 2.0 / Hnuclei + 1.0 *
                                    ab[IDX_HI] / 1.0 / Hnuclei;
        // clang-format on

    return NAUNET_SUCCESS;
}

// clang-format off
int RenormAbundance(realtype *rptr, realtype *ab) {
    
    ab[IDX_GC2H3I] = ab[IDX_GC2H3I] * (24.0 * rptr[IDX_ELEM_C] / 27.0 + 3.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_GC2H4I] = ab[IDX_GC2H4I] * (24.0 * rptr[IDX_ELEM_C] / 28.0 + 4.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_GC2H5I] = ab[IDX_GC2H5I] * (24.0 * rptr[IDX_ELEM_C] / 29.0 + 5.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_GC3H2I] = ab[IDX_GC3H2I] * (36.0 * rptr[IDX_ELEM_C] / 38.0 + 2.0 * rptr[IDX_ELEM_H] / 38.0);
    ab[IDX_GC4HI] = ab[IDX_GC4HI] * (48.0 * rptr[IDX_ELEM_C] / 49.0 + 1.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_GCH2COI] = ab[IDX_GCH2COI] * (16.0 * rptr[IDX_ELEM_O] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_GCH3CNI] = ab[IDX_GCH3CNI] * (14.0 * rptr[IDX_ELEM_N] / 41.0 + 24.0 * rptr[IDX_ELEM_C] / 41.0 + 3.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_GCSI] = ab[IDX_GCSI] * (32.0 * rptr[IDX_ELEM_S] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_GH2CNI] = ab[IDX_GH2CNI] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_GHNCI] = ab[IDX_GHNCI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_GNO2I] = ab[IDX_GNO2I] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_GSiOI] = ab[IDX_GSiOI] * (28.0 * rptr[IDX_ELEM_SI] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_CH3CNHI] = ab[IDX_CH3CNHI] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 4.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_GC2HI] = ab[IDX_GC2HI] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_GC2H2I] = ab[IDX_GC2H2I] * (24.0 * rptr[IDX_ELEM_C] / 26.0 + 2.0 * rptr[IDX_ELEM_H] / 26.0);
    ab[IDX_GC4NI] = ab[IDX_GC4NI] * (14.0 * rptr[IDX_ELEM_N] / 62.0 + 48.0 * rptr[IDX_ELEM_C] / 62.0);
    ab[IDX_GCOI] = ab[IDX_GCOI] * (16.0 * rptr[IDX_ELEM_O] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0);
    ab[IDX_GHC3NI] = ab[IDX_GHC3NI] * (14.0 * rptr[IDX_ELEM_N] / 51.0 + 36.0 * rptr[IDX_ELEM_C] / 51.0 + 1.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_GHNCOI] = ab[IDX_GHNCOI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_GMgI] = ab[IDX_GMgI] * (24.0 * rptr[IDX_ELEM_MG] / 24.0);
    ab[IDX_GNCCNI] = ab[IDX_GNCCNI] * (28.0 * rptr[IDX_ELEM_N] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_GNOI] = ab[IDX_GNOI] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_GO2I] = ab[IDX_GO2I] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_GO2HI] = ab[IDX_GO2HI] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_GSiCI] = ab[IDX_GSiCI] * (28.0 * rptr[IDX_ELEM_SI] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_GSiC2I] = ab[IDX_GSiC2I] * (28.0 * rptr[IDX_ELEM_SI] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_GSiC3I] = ab[IDX_GSiC3I] * (28.0 * rptr[IDX_ELEM_SI] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_GC2H5OHI] = ab[IDX_GC2H5OHI] * (16.0 * rptr[IDX_ELEM_O] / 46.0 + 24.0 * rptr[IDX_ELEM_C] / 46.0 + 6.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GCH3CCHI] = ab[IDX_GCH3CCHI] * (36.0 * rptr[IDX_ELEM_C] / 40.0 + 4.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_GCO2I] = ab[IDX_GCO2I] * (32.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_GHCOOCH3I] = ab[IDX_GHCOOCH3I] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_GC2I] = ab[IDX_GC2I] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_GH2SiOI] = ab[IDX_GH2SiOI] * (28.0 * rptr[IDX_ELEM_SI] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GHNOI] = ab[IDX_GHNOI] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_GN2I] = ab[IDX_GN2I] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_GNSI] = ab[IDX_GNSI] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0);
    ab[IDX_GOCSI] = ab[IDX_GOCSI] * (32.0 * rptr[IDX_ELEM_S] / 60.0 + 16.0 * rptr[IDX_ELEM_O] / 60.0 + 12.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_GSiSI] = ab[IDX_GSiSI] * (28.0 * rptr[IDX_ELEM_SI] / 60.0 + 32.0 * rptr[IDX_ELEM_S] / 60.0);
    ab[IDX_GSOI] = ab[IDX_GSOI] * (32.0 * rptr[IDX_ELEM_S] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0);
    ab[IDX_GSO2I] = ab[IDX_GSO2I] * (32.0 * rptr[IDX_ELEM_S] / 64.0 + 32.0 * rptr[IDX_ELEM_O] / 64.0);
    ab[IDX_GCH3CNHI] = ab[IDX_GCH3CNHI] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 4.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_GCH3OHI] = ab[IDX_GCH3OHI] * (16.0 * rptr[IDX_ELEM_O] / 32.0 + 12.0 * rptr[IDX_ELEM_C] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_GH2COI] = ab[IDX_GH2COI] * (16.0 * rptr[IDX_ELEM_O] / 30.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_GH2S2I] = ab[IDX_GH2S2I] * (64.0 * rptr[IDX_ELEM_S] / 66.0 + 2.0 * rptr[IDX_ELEM_H] / 66.0);
    ab[IDX_GHClI] = ab[IDX_GHClI] * (35.0 * rptr[IDX_ELEM_CL] / 36.0 + 1.0 * rptr[IDX_ELEM_H] / 36.0);
    ab[IDX_GHCNI] = ab[IDX_GHCNI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_GH2CSI] = ab[IDX_GH2CSI] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_GH2OI] = ab[IDX_GH2OI] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_GH2SI] = ab[IDX_GH2SI] * (32.0 * rptr[IDX_ELEM_S] / 34.0 + 2.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_C4NI] = ab[IDX_C4NI] * (14.0 * rptr[IDX_ELEM_N] / 62.0 + 48.0 * rptr[IDX_ELEM_C] / 62.0);
    ab[IDX_C3H2I] = ab[IDX_C3H2I] * (36.0 * rptr[IDX_ELEM_C] / 38.0 + 2.0 * rptr[IDX_ELEM_H] / 38.0);
    ab[IDX_H2S2I] = ab[IDX_H2S2I] * (64.0 * rptr[IDX_ELEM_S] / 66.0 + 2.0 * rptr[IDX_ELEM_H] / 66.0);
    ab[IDX_GNH3I] = ab[IDX_GNH3I] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_H2NOII] = ab[IDX_H2NOII] * (14.0 * rptr[IDX_ELEM_N] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0 + 2.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_H2SiOI] = ab[IDX_H2SiOI] * (28.0 * rptr[IDX_ELEM_SI] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_HClII] = ab[IDX_HClII] * (35.0 * rptr[IDX_ELEM_CL] / 36.0 + 1.0 * rptr[IDX_ELEM_H] / 36.0);
    ab[IDX_HeHII] = ab[IDX_HeHII] * (4.0 * rptr[IDX_ELEM_HE] / 5.0 + 1.0 * rptr[IDX_ELEM_H] / 5.0);
    ab[IDX_HNCOI] = ab[IDX_HNCOI] * (14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0);
    ab[IDX_HNSII] = ab[IDX_HNSII] * (32.0 * rptr[IDX_ELEM_S] / 47.0 + 14.0 * rptr[IDX_ELEM_N] / 47.0 + 1.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_HOCII] = ab[IDX_HOCII] * (16.0 * rptr[IDX_ELEM_O] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_HSOII] = ab[IDX_HSOII] * (32.0 * rptr[IDX_ELEM_S] / 49.0 + 16.0 * rptr[IDX_ELEM_O] / 49.0 + 1.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_SiC3II] = ab[IDX_SiC3II] * (28.0 * rptr[IDX_ELEM_SI] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_GCH4I] = ab[IDX_GCH4I] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_GSiH4I] = ab[IDX_GSiH4I] * (28.0 * rptr[IDX_ELEM_SI] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_C4HI] = ab[IDX_C4HI] * (48.0 * rptr[IDX_ELEM_C] / 49.0 + 1.0 * rptr[IDX_ELEM_H] / 49.0);
    ab[IDX_ClI] = ab[IDX_ClI] * (35.0 * rptr[IDX_ELEM_CL] / 35.0);
    ab[IDX_ClII] = ab[IDX_ClII] * (35.0 * rptr[IDX_ELEM_CL] / 35.0);
    ab[IDX_H2CNI] = ab[IDX_H2CNI] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_C2N2II] = ab[IDX_C2N2II] * (28.0 * rptr[IDX_ELEM_N] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_C4NII] = ab[IDX_C4NII] * (14.0 * rptr[IDX_ELEM_N] / 62.0 + 48.0 * rptr[IDX_ELEM_C] / 62.0);
    ab[IDX_SiH5II] = ab[IDX_SiH5II] * (28.0 * rptr[IDX_ELEM_SI] / 33.0 + 5.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_H2ClII] = ab[IDX_H2ClII] * (35.0 * rptr[IDX_ELEM_CL] / 37.0 + 2.0 * rptr[IDX_ELEM_H] / 37.0);
    ab[IDX_H5C2O2II] = ab[IDX_H5C2O2II] * (32.0 * rptr[IDX_ELEM_O] / 61.0 + 24.0 * rptr[IDX_ELEM_C] / 61.0 + 5.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_SiC3I] = ab[IDX_SiC3I] * (28.0 * rptr[IDX_ELEM_SI] / 64.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0);
    ab[IDX_SiH4II] = ab[IDX_SiH4II] * (28.0 * rptr[IDX_ELEM_SI] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_C2H5OH2II] = ab[IDX_C2H5OH2II] * (16.0 * rptr[IDX_ELEM_O] / 47.0 + 24.0 * rptr[IDX_ELEM_C] / 47.0 + 7.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_C2NHII] = ab[IDX_C2NHII] * (14.0 * rptr[IDX_ELEM_N] / 39.0 + 24.0 * rptr[IDX_ELEM_C] / 39.0 + 1.0 * rptr[IDX_ELEM_H] / 39.0);
    ab[IDX_C3H5II] = ab[IDX_C3H5II] * (36.0 * rptr[IDX_ELEM_C] / 41.0 + 5.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_CH2COI] = ab[IDX_CH2COI] * (16.0 * rptr[IDX_ELEM_O] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 2.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_HSO2II] = ab[IDX_HSO2II] * (32.0 * rptr[IDX_ELEM_S] / 65.0 + 32.0 * rptr[IDX_ELEM_O] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_C3NI] = ab[IDX_C3NI] * (14.0 * rptr[IDX_ELEM_N] / 50.0 + 36.0 * rptr[IDX_ELEM_C] / 50.0);
    ab[IDX_HCOOCH3I] = ab[IDX_HCOOCH3I] * (32.0 * rptr[IDX_ELEM_O] / 60.0 + 24.0 * rptr[IDX_ELEM_C] / 60.0 + 4.0 * rptr[IDX_ELEM_H] / 60.0);
    ab[IDX_SiC2II] = ab[IDX_SiC2II] * (28.0 * rptr[IDX_ELEM_SI] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_C3II] = ab[IDX_C3II] * (36.0 * rptr[IDX_ELEM_C] / 36.0);
    ab[IDX_H2S2II] = ab[IDX_H2S2II] * (64.0 * rptr[IDX_ELEM_S] / 66.0 + 2.0 * rptr[IDX_ELEM_H] / 66.0);
    ab[IDX_HClI] = ab[IDX_HClI] * (35.0 * rptr[IDX_ELEM_CL] / 36.0 + 1.0 * rptr[IDX_ELEM_H] / 36.0);
    ab[IDX_HOCSII] = ab[IDX_HOCSII] * (32.0 * rptr[IDX_ELEM_S] / 61.0 + 16.0 * rptr[IDX_ELEM_O] / 61.0 + 12.0 * rptr[IDX_ELEM_C] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_HS2I] = ab[IDX_HS2I] * (64.0 * rptr[IDX_ELEM_S] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_NSII] = ab[IDX_NSII] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0);
    ab[IDX_CH3CCHI] = ab[IDX_CH3CCHI] * (36.0 * rptr[IDX_ELEM_C] / 40.0 + 4.0 * rptr[IDX_ELEM_H] / 40.0);
    ab[IDX_H3CSII] = ab[IDX_H3CSII] * (32.0 * rptr[IDX_ELEM_S] / 47.0 + 12.0 * rptr[IDX_ELEM_C] / 47.0 + 3.0 * rptr[IDX_ELEM_H] / 47.0);
    ab[IDX_SiC2I] = ab[IDX_SiC2I] * (28.0 * rptr[IDX_ELEM_SI] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_SO2II] = ab[IDX_SO2II] * (32.0 * rptr[IDX_ELEM_S] / 64.0 + 32.0 * rptr[IDX_ELEM_O] / 64.0);
    ab[IDX_H2CSI] = ab[IDX_H2CSI] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_H2CSII] = ab[IDX_H2CSII] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 12.0 * rptr[IDX_ELEM_C] / 46.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_HC3NI] = ab[IDX_HC3NI] * (14.0 * rptr[IDX_ELEM_N] / 51.0 + 36.0 * rptr[IDX_ELEM_C] / 51.0 + 1.0 * rptr[IDX_ELEM_H] / 51.0);
    ab[IDX_HSiSII] = ab[IDX_HSiSII] * (28.0 * rptr[IDX_ELEM_SI] / 61.0 + 32.0 * rptr[IDX_ELEM_S] / 61.0 + 1.0 * rptr[IDX_ELEM_H] / 61.0);
    ab[IDX_SiSII] = ab[IDX_SiSII] * (28.0 * rptr[IDX_ELEM_SI] / 60.0 + 32.0 * rptr[IDX_ELEM_S] / 60.0);
    ab[IDX_HS2II] = ab[IDX_HS2II] * (64.0 * rptr[IDX_ELEM_S] / 65.0 + 1.0 * rptr[IDX_ELEM_H] / 65.0);
    ab[IDX_O2HI] = ab[IDX_O2HI] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_CH3CNHII] = ab[IDX_CH3CNHII] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 24.0 * rptr[IDX_ELEM_C] / 42.0 + 4.0 * rptr[IDX_ELEM_H] / 42.0);
    ab[IDX_NO2I] = ab[IDX_NO2I] * (14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0);
    ab[IDX_S2II] = ab[IDX_S2II] * (64.0 * rptr[IDX_ELEM_S] / 64.0);
    ab[IDX_C2H5I] = ab[IDX_C2H5I] * (24.0 * rptr[IDX_ELEM_C] / 29.0 + 5.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_C2H5OHI] = ab[IDX_C2H5OHI] * (16.0 * rptr[IDX_ELEM_O] / 46.0 + 24.0 * rptr[IDX_ELEM_C] / 46.0 + 6.0 * rptr[IDX_ELEM_H] / 46.0);
    ab[IDX_SiCII] = ab[IDX_SiCII] * (28.0 * rptr[IDX_ELEM_SI] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_C2NI] = ab[IDX_C2NI] * (14.0 * rptr[IDX_ELEM_N] / 38.0 + 24.0 * rptr[IDX_ELEM_C] / 38.0);
    ab[IDX_NCCNI] = ab[IDX_NCCNI] * (28.0 * rptr[IDX_ELEM_N] / 52.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0);
    ab[IDX_SiCI] = ab[IDX_SiCI] * (28.0 * rptr[IDX_ELEM_SI] / 40.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0);
    ab[IDX_HCSI] = ab[IDX_HCSI] * (32.0 * rptr[IDX_ELEM_S] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_SiH2I] = ab[IDX_SiH2I] * (28.0 * rptr[IDX_ELEM_SI] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_SiH3II] = ab[IDX_SiH3II] * (28.0 * rptr[IDX_ELEM_SI] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_SiSI] = ab[IDX_SiSI] * (28.0 * rptr[IDX_ELEM_SI] / 60.0 + 32.0 * rptr[IDX_ELEM_S] / 60.0);
    ab[IDX_CH3OH2II] = ab[IDX_CH3OH2II] * (16.0 * rptr[IDX_ELEM_O] / 33.0 + 12.0 * rptr[IDX_ELEM_C] / 33.0 + 5.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_OCNI] = ab[IDX_OCNI] * (14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0);
    ab[IDX_S2I] = ab[IDX_S2I] * (64.0 * rptr[IDX_ELEM_S] / 64.0);
    ab[IDX_SiH3I] = ab[IDX_SiH3I] * (28.0 * rptr[IDX_ELEM_SI] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_CH3CNI] = ab[IDX_CH3CNI] * (14.0 * rptr[IDX_ELEM_N] / 41.0 + 24.0 * rptr[IDX_ELEM_C] / 41.0 + 3.0 * rptr[IDX_ELEM_H] / 41.0);
    ab[IDX_NSI] = ab[IDX_NSI] * (32.0 * rptr[IDX_ELEM_S] / 46.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0);
    ab[IDX_OCSII] = ab[IDX_OCSII] * (32.0 * rptr[IDX_ELEM_S] / 60.0 + 16.0 * rptr[IDX_ELEM_O] / 60.0 + 12.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_SiH2II] = ab[IDX_SiH2II] * (28.0 * rptr[IDX_ELEM_SI] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_SO2I] = ab[IDX_SO2I] * (32.0 * rptr[IDX_ELEM_S] / 64.0 + 32.0 * rptr[IDX_ELEM_O] / 64.0);
    ab[IDX_HNOI] = ab[IDX_HNOI] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_SiHI] = ab[IDX_SiHI] * (28.0 * rptr[IDX_ELEM_SI] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_SiOII] = ab[IDX_SiOII] * (28.0 * rptr[IDX_ELEM_SI] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_SiOHII] = ab[IDX_SiOHII] * (28.0 * rptr[IDX_ELEM_SI] / 45.0 + 16.0 * rptr[IDX_ELEM_O] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_SiHII] = ab[IDX_SiHII] * (28.0 * rptr[IDX_ELEM_SI] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_SiH4I] = ab[IDX_SiH4I] * (28.0 * rptr[IDX_ELEM_SI] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_C2H3I] = ab[IDX_C2H3I] * (24.0 * rptr[IDX_ELEM_C] / 27.0 + 3.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_C2NII] = ab[IDX_C2NII] * (14.0 * rptr[IDX_ELEM_N] / 38.0 + 24.0 * rptr[IDX_ELEM_C] / 38.0);
    ab[IDX_HCO2II] = ab[IDX_HCO2II] * (32.0 * rptr[IDX_ELEM_O] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_CSII] = ab[IDX_CSII] * (32.0 * rptr[IDX_ELEM_S] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_HCSII] = ab[IDX_HCSII] * (32.0 * rptr[IDX_ELEM_S] / 45.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0);
    ab[IDX_SiOI] = ab[IDX_SiOI] * (28.0 * rptr[IDX_ELEM_SI] / 44.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0);
    ab[IDX_CH3OHI] = ab[IDX_CH3OHI] * (16.0 * rptr[IDX_ELEM_O] / 32.0 + 12.0 * rptr[IDX_ELEM_C] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0);
    ab[IDX_H3SII] = ab[IDX_H3SII] * (32.0 * rptr[IDX_ELEM_S] / 35.0 + 3.0 * rptr[IDX_ELEM_H] / 35.0);
    ab[IDX_SOI] = ab[IDX_SOI] * (32.0 * rptr[IDX_ELEM_S] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0);
    ab[IDX_C2H4I] = ab[IDX_C2H4I] * (24.0 * rptr[IDX_ELEM_C] / 28.0 + 4.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_CH4II] = ab[IDX_CH4II] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_CSI] = ab[IDX_CSI] * (32.0 * rptr[IDX_ELEM_S] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_MgI] = ab[IDX_MgI] * (24.0 * rptr[IDX_ELEM_MG] / 24.0);
    ab[IDX_MgII] = ab[IDX_MgII] * (24.0 * rptr[IDX_ELEM_MG] / 24.0);
    ab[IDX_SOII] = ab[IDX_SOII] * (32.0 * rptr[IDX_ELEM_S] / 48.0 + 16.0 * rptr[IDX_ELEM_O] / 48.0);
    ab[IDX_C2II] = ab[IDX_C2II] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_O2HII] = ab[IDX_O2HII] * (32.0 * rptr[IDX_ELEM_O] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_CNII] = ab[IDX_CNII] * (14.0 * rptr[IDX_ELEM_N] / 26.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0);
    ab[IDX_N2HII] = ab[IDX_N2HII] * (28.0 * rptr[IDX_ELEM_N] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_OCSI] = ab[IDX_OCSI] * (32.0 * rptr[IDX_ELEM_S] / 60.0 + 16.0 * rptr[IDX_ELEM_O] / 60.0 + 12.0 * rptr[IDX_ELEM_C] / 60.0);
    ab[IDX_H2SII] = ab[IDX_H2SII] * (32.0 * rptr[IDX_ELEM_S] / 34.0 + 2.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_HNOII] = ab[IDX_HNOII] * (14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_CH5II] = ab[IDX_CH5II] * (12.0 * rptr[IDX_ELEM_C] / 17.0 + 5.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_N2II] = ab[IDX_N2II] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_HSI] = ab[IDX_HSI] * (32.0 * rptr[IDX_ELEM_S] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_HCNHII] = ab[IDX_HCNHII] * (14.0 * rptr[IDX_ELEM_N] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0);
    ab[IDX_H2II] = ab[IDX_H2II] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_H3COII] = ab[IDX_H3COII] * (16.0 * rptr[IDX_ELEM_O] / 31.0 + 12.0 * rptr[IDX_ELEM_C] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0);
    ab[IDX_SiI] = ab[IDX_SiI] * (28.0 * rptr[IDX_ELEM_SI] / 28.0);
    ab[IDX_C2H2I] = ab[IDX_C2H2I] * (24.0 * rptr[IDX_ELEM_C] / 26.0 + 2.0 * rptr[IDX_ELEM_H] / 26.0);
    ab[IDX_HCNII] = ab[IDX_HCNII] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_HNCI] = ab[IDX_HNCI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_SiII] = ab[IDX_SiII] * (28.0 * rptr[IDX_ELEM_SI] / 28.0);
    ab[IDX_COII] = ab[IDX_COII] * (16.0 * rptr[IDX_ELEM_O] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0);
    ab[IDX_HSII] = ab[IDX_HSII] * (32.0 * rptr[IDX_ELEM_S] / 33.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0);
    ab[IDX_CO2I] = ab[IDX_CO2I] * (32.0 * rptr[IDX_ELEM_O] / 44.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0);
    ab[IDX_NHII] = ab[IDX_NHII] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_NH2I] = ab[IDX_NH2I] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_O2II] = ab[IDX_O2II] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_NII] = ab[IDX_NII] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_NH2II] = ab[IDX_NH2II] * (14.0 * rptr[IDX_ELEM_N] / 16.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_C2HII] = ab[IDX_C2HII] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_NH4II] = ab[IDX_NH4II] * (14.0 * rptr[IDX_ELEM_N] / 18.0 + 4.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_C2HI] = ab[IDX_C2HI] * (24.0 * rptr[IDX_ELEM_C] / 25.0 + 1.0 * rptr[IDX_ELEM_H] / 25.0);
    ab[IDX_OII] = ab[IDX_OII] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_NOII] = ab[IDX_NOII] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_H2COII] = ab[IDX_H2COII] * (16.0 * rptr[IDX_ELEM_O] / 30.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_CH2II] = ab[IDX_CH2II] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_NH3II] = ab[IDX_NH3II] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_H2SI] = ab[IDX_H2SI] * (32.0 * rptr[IDX_ELEM_S] / 34.0 + 2.0 * rptr[IDX_ELEM_H] / 34.0);
    ab[IDX_H2OII] = ab[IDX_H2OII] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_OHII] = ab[IDX_OHII] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_CH3II] = ab[IDX_CH3II] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_N2I] = ab[IDX_N2I] * (28.0 * rptr[IDX_ELEM_N] / 28.0);
    ab[IDX_C2H2II] = ab[IDX_C2H2II] * (24.0 * rptr[IDX_ELEM_C] / 26.0 + 2.0 * rptr[IDX_ELEM_H] / 26.0);
    ab[IDX_C2I] = ab[IDX_C2I] * (24.0 * rptr[IDX_ELEM_C] / 24.0);
    ab[IDX_CH2I] = ab[IDX_CH2I] * (12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0);
    ab[IDX_CH3I] = ab[IDX_CH3I] * (12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_CHII] = ab[IDX_CHII] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_NHI] = ab[IDX_NHI] * (14.0 * rptr[IDX_ELEM_N] / 15.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0);
    ab[IDX_SII] = ab[IDX_SII] * (32.0 * rptr[IDX_ELEM_S] / 32.0);
    ab[IDX_CNI] = ab[IDX_CNI] * (14.0 * rptr[IDX_ELEM_N] / 26.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0);
    ab[IDX_H2COI] = ab[IDX_H2COI] * (16.0 * rptr[IDX_ELEM_O] / 30.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0);
    ab[IDX_CH4I] = ab[IDX_CH4I] * (12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0);
    ab[IDX_HCOI] = ab[IDX_HCOI] * (16.0 * rptr[IDX_ELEM_O] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_HCNI] = ab[IDX_HCNI] * (14.0 * rptr[IDX_ELEM_N] / 27.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0);
    ab[IDX_NOI] = ab[IDX_NOI] * (14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0);
    ab[IDX_CHI] = ab[IDX_CHI] * (12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0);
    ab[IDX_NH3I] = ab[IDX_NH3I] * (14.0 * rptr[IDX_ELEM_N] / 17.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_H3OII] = ab[IDX_H3OII] * (16.0 * rptr[IDX_ELEM_O] / 19.0 + 3.0 * rptr[IDX_ELEM_H] / 19.0);
    ab[IDX_O2I] = ab[IDX_O2I] * (32.0 * rptr[IDX_ELEM_O] / 32.0);
    ab[IDX_CII] = ab[IDX_CII] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_SI] = ab[IDX_SI] * (32.0 * rptr[IDX_ELEM_S] / 32.0);
    ab[IDX_OHI] = ab[IDX_OHI] * (16.0 * rptr[IDX_ELEM_O] / 17.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0);
    ab[IDX_NI] = ab[IDX_NI] * (14.0 * rptr[IDX_ELEM_N] / 14.0);
    ab[IDX_HeII] = ab[IDX_HeII] * (4.0 * rptr[IDX_ELEM_HE] / 4.0);
    ab[IDX_HeI] = ab[IDX_HeI] * (4.0 * rptr[IDX_ELEM_HE] / 4.0);
    ab[IDX_H3II] = ab[IDX_H3II] * (3.0 * rptr[IDX_ELEM_H] / 3.0);
    ab[IDX_HII] = ab[IDX_HII] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
    ab[IDX_OI] = ab[IDX_OI] * (16.0 * rptr[IDX_ELEM_O] / 16.0);
    ab[IDX_CI] = ab[IDX_CI] * (12.0 * rptr[IDX_ELEM_C] / 12.0);
    ab[IDX_HCOII] = ab[IDX_HCOII] * (16.0 * rptr[IDX_ELEM_O] / 29.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0);
    ab[IDX_H2OI] = ab[IDX_H2OI] * (16.0 * rptr[IDX_ELEM_O] / 18.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0);
    ab[IDX_COI] = ab[IDX_COI] * (16.0 * rptr[IDX_ELEM_O] / 28.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0);
    ab[IDX_H2I] = ab[IDX_H2I] * (2.0 * rptr[IDX_ELEM_H] / 2.0);
    ab[IDX_EM] = ab[IDX_EM] * (1.0);
    ab[IDX_HI] = ab[IDX_HI] * (1.0 * rptr[IDX_ELEM_H] / 1.0);
        // clang-format on

    return NAUNET_SUCCESS;
}