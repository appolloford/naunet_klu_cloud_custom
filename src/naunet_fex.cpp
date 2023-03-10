#include <math.h>
/* */
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_sparse.h>  // access to sparse SUNMatrix
/* */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_physics.h"

#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)
#define NVEC_CUDA_CONTENT(x) ((N_VectorContent_Cuda)(x->content))
#define NVEC_CUDA_STREAM(x) (NVEC_CUDA_CONTENT(x)->stream_exec_policy->stream())
#define NVEC_CUDA_BLOCKSIZE(x) \
    (NVEC_CUDA_CONTENT(x)->stream_exec_policy->blockSize())
#define NVEC_CUDA_GRIDSIZE(x, n) \
    (NVEC_CUDA_CONTENT(x)->stream_exec_policy->gridSize(n))

/* */

int Fex(realtype t, N_Vector u, N_Vector udot, void *user_data) {
    /* */
    realtype *y            = N_VGetArrayPointer(u);
    realtype *ydot         = N_VGetArrayPointer(udot);
    NaunetData *u_data     = (NaunetData *)user_data;
    // clang-format off
    realtype nH = u_data->nH;
    realtype Tgas = u_data->Tgas;
    realtype zeta = u_data->zeta;
    realtype Av = u_data->Av;
    realtype omega = u_data->omega;
    realtype G0 = u_data->G0;
    realtype gdens = u_data->gdens;
    realtype rG = u_data->rG;
    realtype sites = u_data->sites;
    realtype fr = u_data->fr;
    realtype opt_crd = u_data->opt_crd;
    realtype opt_uvd = u_data->opt_uvd;
    realtype opt_h2d = u_data->opt_h2d;
    realtype eb_crd = u_data->eb_crd;
    realtype eb_uvd = u_data->eb_uvd;
    realtype eb_h2d = u_data->eb_h2d;
    realtype crdeseff = u_data->crdeseff;
    realtype uvcreff = u_data->uvcreff;
    realtype h2deseff = u_data->h2deseff;
    realtype opt_thd = u_data->opt_thd;
    
    realtype h2col = 0.5*1.59e21*Av;
    realtype cocol = 1e-5 * h2col;
    realtype lambdabar = GetCharactWavelength(h2col, cocol);
    realtype H2shielding = GetShieldingFactor(IDX_H2I, h2col, h2col, Tgas, 1);
    realtype H2formation = 1.0e-17 * sqrt(Tgas) * nH;
    realtype H2dissociation = 5.1e-11 * G0 * GetGrainScattering(Av, 1000.0) * H2shielding;
    realtype mant = GetMantleDens(y);
    realtype mantabund = mant / nH;
    realtype gxsec = (pi*rG*rG) * gdens;
    realtype garea = 4.0 * gxsec;
    realtype unisites = sites * (4*pi*rG*rG);
    realtype densites = garea * sites;
    
#if (NHEATPROCS || NCOOLPROCS)
    if (mu < 0) mu = GetMu(y);
    if (gamma < 0) gamma = GetGamma(y);
#endif

    // clang-format on

    realtype k[NREACTIONS] = {0.0};
    EvalRates(k, y, u_data);

#if NHEATPROCS
    realtype kh[NHEATPROCS] = {0.0};
    EvalHeatingRates(kh, y, u_data);
#endif

#if NCOOLPROCS
    realtype kc[NCOOLPROCS] = {0.0};
    EvalCoolingRates(kc, y, u_data);
#endif

    // clang-format off
    ydot[IDX_GC2H3I] = 0.0 + k[2146]*y[IDX_C2H3I] - k[2339]*y[IDX_GC2H3I] -
        k[2340]*y[IDX_GC2H3I] - k[2341]*y[IDX_GC2H3I] - k[2342]*y[IDX_GC2H3I];
    ydot[IDX_GC2H4I] = 0.0 + k[2300]*y[IDX_C2H4I] - k[2351]*y[IDX_GC2H4I] -
        k[2352]*y[IDX_GC2H4I] - k[2353]*y[IDX_GC2H4I] - k[2354]*y[IDX_GC2H4I];
    ydot[IDX_GC2H5I] = 0.0 + k[2301]*y[IDX_C2H5I] - k[2359]*y[IDX_GC2H5I] -
        k[2360]*y[IDX_GC2H5I] - k[2361]*y[IDX_GC2H5I] - k[2362]*y[IDX_GC2H5I];
    ydot[IDX_GC3H2I] = 0.0 + k[2199]*y[IDX_C3H2I] - k[2399]*y[IDX_GC3H2I] -
        k[2400]*y[IDX_GC3H2I] - k[2401]*y[IDX_GC3H2I] - k[2402]*y[IDX_GC3H2I];
    ydot[IDX_GC4HI] = 0.0 + k[2200]*y[IDX_C4HI] - k[2463]*y[IDX_GC4HI] -
        k[2464]*y[IDX_GC4HI] - k[2465]*y[IDX_GC4HI] - k[2466]*y[IDX_GC4HI];
    ydot[IDX_GCH2COI] = 0.0 + k[2147]*y[IDX_CH2COI] - k[2415]*y[IDX_GCH2COI]
        - k[2416]*y[IDX_GCH2COI] - k[2417]*y[IDX_GCH2COI] -
        k[2418]*y[IDX_GCH2COI];
    ydot[IDX_GCH3CNI] = 0.0 + k[2299]*y[IDX_CH3CNI] - k[2411]*y[IDX_GCH3CNI]
        - k[2412]*y[IDX_GCH3CNI] - k[2413]*y[IDX_GCH3CNI] -
        k[2414]*y[IDX_GCH3CNI];
    ydot[IDX_GCSI] = 0.0 - k[2431]*y[IDX_GCSI] - k[2432]*y[IDX_GCSI] -
        k[2433]*y[IDX_GCSI] - k[2434]*y[IDX_GCSI];
    ydot[IDX_GH2CNI] = 0.0 + k[2275]*y[IDX_H2CNI] - k[2355]*y[IDX_GH2CNI] -
        k[2356]*y[IDX_GH2CNI] - k[2357]*y[IDX_GH2CNI] - k[2358]*y[IDX_GH2CNI];
    ydot[IDX_GHNCI] = 0.0 + k[2297]*y[IDX_HNCI] - k[2335]*y[IDX_GHNCI] -
        k[2336]*y[IDX_GHNCI] - k[2337]*y[IDX_GHNCI] - k[2338]*y[IDX_GHNCI];
    ydot[IDX_GNO2I] = 0.0 + k[2270]*y[IDX_NO2I] - k[2447]*y[IDX_GNO2I] -
        k[2448]*y[IDX_GNO2I] - k[2449]*y[IDX_GNO2I] - k[2450]*y[IDX_GNO2I];
    ydot[IDX_GSiOI] = 0.0 - k[2427]*y[IDX_GSiOI] - k[2428]*y[IDX_GSiOI] -
        k[2429]*y[IDX_GSiOI] - k[2430]*y[IDX_GSiOI];
    ydot[IDX_CH3CNHI] = 0.0 - k[2155]*y[IDX_CH3CNHI] +
        k[2419]*y[IDX_GCH3CNHI] + k[2420]*y[IDX_GCH3CNHI] +
        k[2421]*y[IDX_GCH3CNHI] + k[2422]*y[IDX_GCH3CNHI];
    ydot[IDX_GC2HI] = 0.0 + k[2224]*y[IDX_C2HI] + k[2242]*y[IDX_C2HII] -
        k[2323]*y[IDX_GC2HI] - k[2324]*y[IDX_GC2HI] - k[2325]*y[IDX_GC2HI] -
        k[2326]*y[IDX_GC2HI];
    ydot[IDX_GC2H2I] = 0.0 + k[2277]*y[IDX_C2H2II] + k[2298]*y[IDX_C2H2I] -
        k[2327]*y[IDX_GC2H2I] - k[2328]*y[IDX_GC2H2I] - k[2329]*y[IDX_GC2H2I] -
        k[2330]*y[IDX_GC2H2I];
    ydot[IDX_GC4NI] = 0.0 + k[2165]*y[IDX_C4NII] + k[2166]*y[IDX_C4NI] -
        k[2491]*y[IDX_GC4NI] - k[2492]*y[IDX_GC4NI] - k[2493]*y[IDX_GC4NI] -
        k[2494]*y[IDX_GC4NI];
    ydot[IDX_GCOI] = 0.0 + k[2207]*y[IDX_COI] + k[2234]*y[IDX_COII] -
        k[2343]*y[IDX_GCOI] - k[2344]*y[IDX_GCOI] - k[2345]*y[IDX_GCOI] -
        k[2346]*y[IDX_GCOI];
    ydot[IDX_GHC3NI] = 0.0 + k[2145]*y[IDX_HC3NI] + k[2161]*y[IDX_C3NI] -
        k[2467]*y[IDX_GHC3NI] - k[2468]*y[IDX_GHC3NI] - k[2469]*y[IDX_GHC3NI] -
        k[2470]*y[IDX_GHC3NI];
    ydot[IDX_GHNCOI] = 0.0 + k[2157]*y[IDX_HNCOI] + k[2158]*y[IDX_OCNI] -
        k[2423]*y[IDX_GHNCOI] - k[2424]*y[IDX_GHNCOI] - k[2425]*y[IDX_GHNCOI] -
        k[2426]*y[IDX_GHNCOI];
    ydot[IDX_GMgI] = 0.0 + k[2286]*y[IDX_MgII] + k[2287]*y[IDX_MgI] -
        k[2319]*y[IDX_GMgI] - k[2320]*y[IDX_GMgI] - k[2321]*y[IDX_GMgI] -
        k[2322]*y[IDX_GMgI];
    ydot[IDX_GNCCNI] = 0.0 + k[2163]*y[IDX_C2N2II] + k[2164]*y[IDX_NCCNI] -
        k[2471]*y[IDX_GNCCNI] - k[2472]*y[IDX_GNCCNI] - k[2473]*y[IDX_GNCCNI] -
        k[2474]*y[IDX_GNCCNI];
    ydot[IDX_GNOI] = 0.0 + k[2212]*y[IDX_NOI] + k[2236]*y[IDX_NOII] -
        k[2363]*y[IDX_GNOI] - k[2364]*y[IDX_GNOI] - k[2365]*y[IDX_GNOI] -
        k[2366]*y[IDX_GNOI];
    ydot[IDX_GO2I] = 0.0 + k[2229]*y[IDX_O2II] + k[2253]*y[IDX_O2I] -
        k[2375]*y[IDX_GO2I] - k[2376]*y[IDX_GO2I] - k[2377]*y[IDX_GO2I] -
        k[2378]*y[IDX_GO2I];
    ydot[IDX_GO2HI] = 0.0 + k[2274]*y[IDX_O2HII] + k[2294]*y[IDX_O2HI] -
        k[2387]*y[IDX_GO2HI] - k[2388]*y[IDX_GO2HI] - k[2389]*y[IDX_GO2HI] -
        k[2390]*y[IDX_GO2HI];
    ydot[IDX_GSiCI] = 0.0 + k[2181]*y[IDX_SiCI] + k[2183]*y[IDX_SiCII] -
        k[2407]*y[IDX_GSiCI] - k[2408]*y[IDX_GSiCI] - k[2409]*y[IDX_GSiCI] -
        k[2410]*y[IDX_GSiCI];
    ydot[IDX_GSiC2I] = 0.0 + k[2182]*y[IDX_SiC2I] + k[2184]*y[IDX_SiC2II] -
        k[2475]*y[IDX_GSiC2I] - k[2476]*y[IDX_GSiC2I] - k[2477]*y[IDX_GSiC2I] -
        k[2478]*y[IDX_GSiC2I];
    ydot[IDX_GSiC3I] = 0.0 + k[2185]*y[IDX_SiC3I] + k[2186]*y[IDX_SiC3II] -
        k[2495]*y[IDX_GSiC3I] - k[2496]*y[IDX_GSiC3I] - k[2497]*y[IDX_GSiC3I] -
        k[2498]*y[IDX_GSiC3I];
    ydot[IDX_GC2H5OHI] = 0.0 + k[2150]*y[IDX_C2H5OHI] +
        k[2151]*y[IDX_C2H5OH2II] - k[2439]*y[IDX_GC2H5OHI] -
        k[2440]*y[IDX_GC2H5OHI] - k[2441]*y[IDX_GC2H5OHI] -
        k[2442]*y[IDX_GC2H5OHI];
    ydot[IDX_GCH3CCHI] = 0.0 + k[2148]*y[IDX_C3H5II] +
        k[2149]*y[IDX_CH3CCHI] - k[2403]*y[IDX_GCH3CCHI] -
        k[2404]*y[IDX_GCH3CCHI] - k[2405]*y[IDX_GCH3CCHI] -
        k[2406]*y[IDX_GCH3CCHI];
    ydot[IDX_GCO2I] = 0.0 + k[2215]*y[IDX_CO2I] + k[2247]*y[IDX_HCO2II] -
        k[2435]*y[IDX_GCO2I] - k[2436]*y[IDX_GCO2I] - k[2437]*y[IDX_GCO2I] -
        k[2438]*y[IDX_GCO2I];
    ydot[IDX_GHCOOCH3I] = 0.0 + k[2153]*y[IDX_HCOOCH3I] +
        k[2167]*y[IDX_H5C2O2II] - k[2479]*y[IDX_GHCOOCH3I] -
        k[2480]*y[IDX_GHCOOCH3I] - k[2481]*y[IDX_GHCOOCH3I] -
        k[2482]*y[IDX_GHCOOCH3I];
    ydot[IDX_GC2I] = 0.0 + k[2209]*y[IDX_C2I] + k[2228]*y[IDX_C2II] +
        k[2292]*y[IDX_C3II] - k[2315]*y[IDX_GC2I] - k[2316]*y[IDX_GC2I] -
        k[2317]*y[IDX_GC2I] - k[2318]*y[IDX_GC2I];
    ydot[IDX_GH2SiOI] = 0.0 + k[2171]*y[IDX_SiOI] + k[2187]*y[IDX_SiOII] +
        k[2188]*y[IDX_SiOHII] + k[2192]*y[IDX_H2SiOI] - k[2455]*y[IDX_GH2SiOI] -
        k[2456]*y[IDX_GH2SiOI] - k[2457]*y[IDX_GH2SiOI] -
        k[2458]*y[IDX_GH2SiOI];
    ydot[IDX_GHNOI] = 0.0 + k[2271]*y[IDX_HNOI] + k[2272]*y[IDX_HNOII] +
        k[2273]*y[IDX_H2NOII] - k[2371]*y[IDX_GHNOI] - k[2372]*y[IDX_GHNOI] -
        k[2373]*y[IDX_GHNOI] - k[2374]*y[IDX_GHNOI];
    ydot[IDX_GN2I] = 0.0 + k[2219]*y[IDX_N2I] + k[2230]*y[IDX_N2II] +
        k[2290]*y[IDX_N2HII] - k[2347]*y[IDX_GN2I] - k[2348]*y[IDX_GN2I] -
        k[2349]*y[IDX_GN2I] - k[2350]*y[IDX_GN2I];
    ydot[IDX_GNSI] = 0.0 + k[2262]*y[IDX_NSI] + k[2276]*y[IDX_NSII] +
        k[2291]*y[IDX_HNSII] - k[2451]*y[IDX_GNSI] - k[2452]*y[IDX_GNSI] -
        k[2453]*y[IDX_GNSI] - k[2454]*y[IDX_GNSI];
    ydot[IDX_GOCSI] = 0.0 + k[2259]*y[IDX_OCSI] + k[2269]*y[IDX_OCSII] +
        k[2285]*y[IDX_HOCSII] - k[2487]*y[IDX_GOCSI] - k[2488]*y[IDX_GOCSI] -
        k[2489]*y[IDX_GOCSI] - k[2490]*y[IDX_GOCSI];
    ydot[IDX_GSiSI] = 0.0 + k[2189]*y[IDX_SiSI] + k[2190]*y[IDX_SiSII] +
        k[2191]*y[IDX_HSiSII] - k[2483]*y[IDX_GSiSI] - k[2484]*y[IDX_GSiSI] -
        k[2485]*y[IDX_GSiSI] - k[2486]*y[IDX_GSiSI];
    ydot[IDX_GSOI] = 0.0 + k[2256]*y[IDX_SOI] + k[2267]*y[IDX_SOII] +
        k[2283]*y[IDX_HSOII] - k[2459]*y[IDX_GSOI] - k[2460]*y[IDX_GSOI] -
        k[2461]*y[IDX_GSOI] - k[2462]*y[IDX_GSOI];
    ydot[IDX_GSO2I] = 0.0 + k[2260]*y[IDX_SO2I] + k[2284]*y[IDX_SO2II] +
        k[2295]*y[IDX_HSO2II] - k[2499]*y[IDX_GSO2I] - k[2500]*y[IDX_GSO2I] -
        k[2501]*y[IDX_GSO2I] - k[2502]*y[IDX_GSO2I];
    ydot[IDX_GCH3CNHI] = 0.0 + k[2155]*y[IDX_CH3CNHI] +
        k[2156]*y[IDX_CH3CNHII] + k[2159]*y[IDX_C2NI] + k[2160]*y[IDX_C2NII] +
        k[2162]*y[IDX_C2NHII] - k[2419]*y[IDX_GCH3CNHI] -
        k[2420]*y[IDX_GCH3CNHI] - k[2421]*y[IDX_GCH3CNHI] -
        k[2422]*y[IDX_GCH3CNHI];
    ydot[IDX_GCH3OHI] = 0.0 + k[2152]*y[IDX_CH3OH2II] +
        k[2154]*y[IDX_CH3OHI] + k[2202]*y[IDX_H3COII] + k[2296]*y[IDX_COI] -
        k[2379]*y[IDX_GCH3OHI] - k[2380]*y[IDX_GCH3OHI] - k[2381]*y[IDX_GCH3OHI]
        - k[2382]*y[IDX_GCH3OHI];
    ydot[IDX_GH2COI] = 0.0 + k[2168]*y[IDX_HOCII] + k[2169]*y[IDX_COI] +
        k[2208]*y[IDX_H2COI] + k[2218]*y[IDX_HCOI] + k[2240]*y[IDX_HCOII] +
        k[2244]*y[IDX_H2COII] - k[2367]*y[IDX_GH2COI] - k[2368]*y[IDX_GH2COI] -
        k[2369]*y[IDX_GH2COI] - k[2370]*y[IDX_GH2COI];
    ydot[IDX_GH2S2I] = 0.0 + k[2201]*y[IDX_S2I] + k[2203]*y[IDX_H2S2I] +
        k[2204]*y[IDX_HS2I] + k[2205]*y[IDX_HS2II] + k[2280]*y[IDX_H2S2II] +
        k[2281]*y[IDX_S2II] - k[2503]*y[IDX_GH2S2I] - k[2504]*y[IDX_GH2S2I] -
        k[2505]*y[IDX_GH2S2I] - k[2506]*y[IDX_GH2S2I];
    ydot[IDX_GHClI] = 0.0 + k[2194]*y[IDX_ClI] + k[2195]*y[IDX_HClI] +
        k[2196]*y[IDX_ClII] + k[2197]*y[IDX_HClII] + k[2198]*y[IDX_H2ClII] -
        k[2395]*y[IDX_GHClI] - k[2396]*y[IDX_GHClI] - k[2397]*y[IDX_GHClI] -
        k[2398]*y[IDX_GHClI];
    ydot[IDX_GHCNI] = 0.0 + k[2220]*y[IDX_CNI] + k[2223]*y[IDX_HCNI] +
        k[2235]*y[IDX_CNII] + k[2241]*y[IDX_HCNII] + k[2289]*y[IDX_HCNHII] -
        k[2331]*y[IDX_GHCNI] - k[2332]*y[IDX_GHCNI] - k[2333]*y[IDX_GHCNI] -
        k[2334]*y[IDX_GHCNI];
    ydot[IDX_GH2CSI] = 0.0 + k[2255]*y[IDX_CSI] + k[2258]*y[IDX_HCSI] +
        k[2263]*y[IDX_H2CSI] + k[2266]*y[IDX_CSII] + k[2268]*y[IDX_HCSII] +
        k[2279]*y[IDX_H2CSII] + k[2282]*y[IDX_H3CSII] - k[2443]*y[IDX_GH2CSI] -
        k[2444]*y[IDX_GH2CSI] - k[2445]*y[IDX_GH2CSI] - k[2446]*y[IDX_GH2CSI];
    ydot[IDX_GH2OI] = 0.0 + k[2211]*y[IDX_OHI] + k[2214]*y[IDX_H2OI] +
        k[2227]*y[IDX_OII] + k[2233]*y[IDX_OHII] + k[2239]*y[IDX_H2OII] +
        k[2246]*y[IDX_H3OII] + k[2252]*y[IDX_OI] - k[2311]*y[IDX_GH2OI] -
        k[2312]*y[IDX_GH2OI] - k[2313]*y[IDX_GH2OI] - k[2314]*y[IDX_GH2OI];
    ydot[IDX_GH2SI] = 0.0 + k[2254]*y[IDX_HSI] + k[2257]*y[IDX_H2SI] +
        k[2261]*y[IDX_SI] + k[2264]*y[IDX_SII] + k[2265]*y[IDX_HSII] +
        k[2278]*y[IDX_H3SII] + k[2293]*y[IDX_H2SII] - k[2391]*y[IDX_GH2SI] -
        k[2392]*y[IDX_GH2SI] - k[2393]*y[IDX_GH2SI] - k[2394]*y[IDX_GH2SI];
    ydot[IDX_C4NI] = 0.0 + k[1799]*y[IDX_NI]*y[IDX_C4HI] -
        k[1800]*y[IDX_NI]*y[IDX_C4NI] - k[1878]*y[IDX_OI]*y[IDX_C4NI] -
        k[2166]*y[IDX_C4NI] + k[2491]*y[IDX_GC4NI] + k[2492]*y[IDX_GC4NI] +
        k[2493]*y[IDX_GC4NI] + k[2494]*y[IDX_GC4NI];
    ydot[IDX_C3H2I] = 0.0 + k[1582]*y[IDX_CI]*y[IDX_C2H3I] -
        k[1585]*y[IDX_CI]*y[IDX_C3H2I] + k[1674]*y[IDX_CHI]*y[IDX_C2H2I] -
        k[1797]*y[IDX_NI]*y[IDX_C3H2I] - k[2199]*y[IDX_C3H2I] +
        k[2399]*y[IDX_GC3H2I] + k[2400]*y[IDX_GC3H2I] + k[2401]*y[IDX_GC3H2I] +
        k[2402]*y[IDX_GC3H2I];
    ydot[IDX_H2S2I] = 0.0 - k[122]*y[IDX_HII]*y[IDX_H2S2I] -
        k[402]*y[IDX_H2S2I] - k[1224]*y[IDX_HeII]*y[IDX_H2S2I] -
        k[1225]*y[IDX_HeII]*y[IDX_H2S2I] - k[2019]*y[IDX_H2S2I] -
        k[2203]*y[IDX_H2S2I] + k[2503]*y[IDX_GH2S2I] + k[2504]*y[IDX_GH2S2I] +
        k[2505]*y[IDX_GH2S2I] + k[2506]*y[IDX_GH2S2I];
    ydot[IDX_GNH3I] = 0.0 + k[2222]*y[IDX_NHI] + k[2225]*y[IDX_NH3I] +
        k[2226]*y[IDX_NII] + k[2232]*y[IDX_NHII] + k[2238]*y[IDX_NH2II] +
        k[2243]*y[IDX_NH3II] + k[2250]*y[IDX_NH2I] + k[2251]*y[IDX_NI] +
        k[2288]*y[IDX_NH4II] - k[2307]*y[IDX_GNH3I] - k[2308]*y[IDX_GNH3I] -
        k[2309]*y[IDX_GNH3I] - k[2310]*y[IDX_GNH3I];
    ydot[IDX_H2NOII] = 0.0 - k[510]*y[IDX_H2NOII]*y[IDX_EM] -
        k[511]*y[IDX_H2NOII]*y[IDX_EM] + k[1051]*y[IDX_H3II]*y[IDX_HNOI] +
        k[1390]*y[IDX_NH2II]*y[IDX_O2I] - k[2273]*y[IDX_H2NOII];
    ydot[IDX_H2SiOI] = 0.0 - k[405]*y[IDX_H2SiOI] -
        k[896]*y[IDX_HII]*y[IDX_H2SiOI] - k[1228]*y[IDX_HeII]*y[IDX_H2SiOI] +
        k[1924]*y[IDX_OI]*y[IDX_SiH3I] - k[2023]*y[IDX_H2SiOI] -
        k[2024]*y[IDX_H2SiOI] - k[2192]*y[IDX_H2SiOI] + k[2455]*y[IDX_GH2SiOI] +
        k[2456]*y[IDX_GH2SiOI] + k[2457]*y[IDX_GH2SiOI] +
        k[2458]*y[IDX_GH2SiOI];
    ydot[IDX_HClII] = 0.0 + k[126]*y[IDX_HII]*y[IDX_HClI] -
        k[548]*y[IDX_HClII]*y[IDX_EM] + k[944]*y[IDX_H2I]*y[IDX_ClII] -
        k[948]*y[IDX_H2I]*y[IDX_HClII] + k[1040]*y[IDX_H3II]*y[IDX_ClI] +
        k[2035]*y[IDX_HClI] - k[2197]*y[IDX_HClII];
    ydot[IDX_HeHII] = 0.0 - k[563]*y[IDX_HeHII]*y[IDX_EM] +
        k[926]*y[IDX_H2II]*y[IDX_HeI] - k[951]*y[IDX_H2I]*y[IDX_HeHII] -
        k[1106]*y[IDX_HI]*y[IDX_HeHII] + k[1236]*y[IDX_HeII]*y[IDX_HCOI] +
        k[2111]*y[IDX_HII]*y[IDX_HeI];
    ydot[IDX_HNCOI] = 0.0 - k[415]*y[IDX_HNCOI] -
        k[900]*y[IDX_HII]*y[IDX_HNCOI] + k[1631]*y[IDX_CH2I]*y[IDX_NOI] -
        k[1788]*y[IDX_HNCOI]*y[IDX_CI] - k[2037]*y[IDX_HNCOI] -
        k[2157]*y[IDX_HNCOI] + k[2423]*y[IDX_GHNCOI] + k[2424]*y[IDX_GHNCOI] +
        k[2425]*y[IDX_GHNCOI] + k[2426]*y[IDX_GHNCOI];
    ydot[IDX_HNSII] = 0.0 - k[550]*y[IDX_HNSII]*y[IDX_EM] +
        k[1061]*y[IDX_H3II]*y[IDX_NSI] + k[1148]*y[IDX_HCOII]*y[IDX_NSI] +
        k[1392]*y[IDX_NH2II]*y[IDX_SI] - k[2291]*y[IDX_HNSII];
    ydot[IDX_HOCII] = 0.0 - k[5]*y[IDX_H2I]*y[IDX_HOCII] -
        k[551]*y[IDX_HOCII]*y[IDX_EM] + k[621]*y[IDX_CII]*y[IDX_H2OI] +
        k[942]*y[IDX_H2I]*y[IDX_COII] + k[1038]*y[IDX_H3II]*y[IDX_COI] -
        k[2168]*y[IDX_HOCII];
    ydot[IDX_HSOII] = 0.0 - k[557]*y[IDX_HSOII]*y[IDX_EM] +
        k[988]*y[IDX_H2OII]*y[IDX_SI] + k[1070]*y[IDX_H3II]*y[IDX_SOI] +
        k[1152]*y[IDX_HCOII]*y[IDX_SOI] - k[2283]*y[IDX_HSOII];
    ydot[IDX_SiC3II] = 0.0 + k[28]*y[IDX_CII]*y[IDX_SiC3I] +
        k[145]*y[IDX_HII]*y[IDX_SiC3I] - k[590]*y[IDX_SiC3II]*y[IDX_EM] -
        k[591]*y[IDX_SiC3II]*y[IDX_EM] - k[2186]*y[IDX_SiC3II];
    ydot[IDX_GCH4I] = 0.0 + k[2206]*y[IDX_CI] + k[2210]*y[IDX_CHI] +
        k[2213]*y[IDX_CH2I] + k[2216]*y[IDX_CH3I] + k[2217]*y[IDX_CH4I] +
        k[2221]*y[IDX_CII] + k[2231]*y[IDX_CHII] + k[2237]*y[IDX_CH2II] +
        k[2245]*y[IDX_CH3II] + k[2248]*y[IDX_CH5II] + k[2249]*y[IDX_CH4II] -
        k[2303]*y[IDX_GCH4I] - k[2304]*y[IDX_GCH4I] - k[2305]*y[IDX_GCH4I] -
        k[2306]*y[IDX_GCH4I];
    ydot[IDX_GSiH4I] = 0.0 + k[2170]*y[IDX_SiI] + k[2172]*y[IDX_SiHI] +
        k[2173]*y[IDX_SiII] + k[2174]*y[IDX_SiHII] + k[2175]*y[IDX_SiH2I] +
        k[2176]*y[IDX_SiH2II] + k[2177]*y[IDX_SiH3I] + k[2178]*y[IDX_SiH3II] +
        k[2179]*y[IDX_SiH4I] + k[2180]*y[IDX_SiH4II] + k[2193]*y[IDX_SiH5II] -
        k[2383]*y[IDX_GSiH4I] - k[2384]*y[IDX_GSiH4I] - k[2385]*y[IDX_GSiH4I] -
        k[2386]*y[IDX_GSiH4I];
    ydot[IDX_C4HI] = 0.0 - k[378]*y[IDX_C4HI] -
        k[1191]*y[IDX_HeII]*y[IDX_C4HI] + k[1570]*y[IDX_C2I]*y[IDX_C2H2I] +
        k[1585]*y[IDX_CI]*y[IDX_C3H2I] - k[1799]*y[IDX_NI]*y[IDX_C4HI] -
        k[1975]*y[IDX_C4HI] - k[2200]*y[IDX_C4HI] + k[2463]*y[IDX_GC4HI] +
        k[2464]*y[IDX_GC4HI] + k[2465]*y[IDX_GC4HI] + k[2466]*y[IDX_GC4HI];
    ydot[IDX_ClI] = 0.0 + k[108]*y[IDX_ClII]*y[IDX_HI] -
        k[109]*y[IDX_ClI]*y[IDX_HII] + k[324]*y[IDX_O2I]*y[IDX_ClII] -
        k[358]*y[IDX_ClI] - k[397]*y[IDX_ClI] + k[413]*y[IDX_HClI] +
        k[508]*y[IDX_H2ClII]*y[IDX_EM] + k[548]*y[IDX_HClII]*y[IDX_EM] -
        k[1040]*y[IDX_H3II]*y[IDX_ClI] - k[1720]*y[IDX_ClI]*y[IDX_H2I] +
        k[1787]*y[IDX_HClI]*y[IDX_HI] - k[2008]*y[IDX_ClI] + k[2034]*y[IDX_HClI]
        + k[2134]*y[IDX_ClII]*y[IDX_EM] - k[2194]*y[IDX_ClI];
    ydot[IDX_ClII] = 0.0 - k[108]*y[IDX_ClII]*y[IDX_HI] +
        k[109]*y[IDX_ClI]*y[IDX_HII] - k[324]*y[IDX_O2I]*y[IDX_ClII] +
        k[358]*y[IDX_ClI] + k[397]*y[IDX_ClI] - k[944]*y[IDX_H2I]*y[IDX_ClII] +
        k[1241]*y[IDX_HeII]*y[IDX_HClI] + k[2008]*y[IDX_ClI] -
        k[2134]*y[IDX_ClII]*y[IDX_EM] - k[2196]*y[IDX_ClII];
    ydot[IDX_H2CNI] = 0.0 - k[398]*y[IDX_H2CNI] -
        k[1593]*y[IDX_CI]*y[IDX_H2CNI] - k[1746]*y[IDX_HI]*y[IDX_H2CNI] +
        k[1794]*y[IDX_NI]*y[IDX_C2H5I] + k[1804]*y[IDX_NI]*y[IDX_CH3I] -
        k[1810]*y[IDX_NI]*y[IDX_H2CNI] - k[1885]*y[IDX_OI]*y[IDX_H2CNI] -
        k[2010]*y[IDX_H2CNI] - k[2275]*y[IDX_H2CNI] + k[2355]*y[IDX_GH2CNI] +
        k[2356]*y[IDX_GH2CNI] + k[2357]*y[IDX_GH2CNI] + k[2358]*y[IDX_GH2CNI];
    ydot[IDX_C2N2II] = 0.0 - k[470]*y[IDX_C2N2II]*y[IDX_EM] -
        k[471]*y[IDX_C2N2II]*y[IDX_EM] - k[675]*y[IDX_C2H2I]*y[IDX_C2N2II] +
        k[866]*y[IDX_CNII]*y[IDX_HCNI] - k[1097]*y[IDX_HI]*y[IDX_C2N2II] -
        k[1118]*y[IDX_HCNI]*y[IDX_C2N2II] + k[1305]*y[IDX_NII]*y[IDX_NCCNI] +
        k[2046]*y[IDX_NCCNI] - k[2163]*y[IDX_C2N2II];
    ydot[IDX_C4NII] = 0.0 - k[475]*y[IDX_C4NII]*y[IDX_EM] -
        k[476]*y[IDX_C4NII]*y[IDX_EM] + k[625]*y[IDX_CII]*y[IDX_HC3NI] -
        k[994]*y[IDX_H2OI]*y[IDX_C4NII] + k[1119]*y[IDX_HCNI]*y[IDX_C3II] -
        k[2165]*y[IDX_C4NII];
    ydot[IDX_SiH5II] = 0.0 - k[600]*y[IDX_SiH5II]*y[IDX_EM] -
        k[601]*y[IDX_SiH5II]*y[IDX_EM] + k[963]*y[IDX_H2I]*y[IDX_SiH4II] -
        k[1016]*y[IDX_H2OI]*y[IDX_SiH5II] + k[1074]*y[IDX_H3II]*y[IDX_SiH4I] +
        k[1154]*y[IDX_HCOII]*y[IDX_SiH4I] + k[2120]*y[IDX_H2I]*y[IDX_SiH3II] -
        k[2193]*y[IDX_SiH5II];
    ydot[IDX_H2ClII] = 0.0 - k[508]*y[IDX_H2ClII]*y[IDX_EM] -
        k[509]*y[IDX_H2ClII]*y[IDX_EM] + k[832]*y[IDX_CH5II]*y[IDX_HClI] -
        k[874]*y[IDX_COI]*y[IDX_H2ClII] + k[948]*y[IDX_H2I]*y[IDX_HClII] -
        k[999]*y[IDX_H2OI]*y[IDX_H2ClII] + k[1049]*y[IDX_H3II]*y[IDX_HClI] -
        k[2198]*y[IDX_H2ClII];
    ydot[IDX_H5C2O2II] = 0.0 - k[536]*y[IDX_H5C2O2II]*y[IDX_EM] -
        k[537]*y[IDX_H5C2O2II]*y[IDX_EM] + k[969]*y[IDX_H2COI]*y[IDX_CH3OH2II] +
        k[1047]*y[IDX_H3II]*y[IDX_HCOOCH3I] +
        k[1089]*y[IDX_H3OII]*y[IDX_HCOOCH3I] +
        k[1145]*y[IDX_HCOII]*y[IDX_HCOOCH3I] - k[2167]*y[IDX_H5C2O2II];
    ydot[IDX_SiC3I] = 0.0 - k[28]*y[IDX_CII]*y[IDX_SiC3I] -
        k[145]*y[IDX_HII]*y[IDX_SiC3I] - k[450]*y[IDX_SiC3I] -
        k[1275]*y[IDX_HeII]*y[IDX_SiC3I] - k[1919]*y[IDX_OI]*y[IDX_SiC3I] -
        k[2079]*y[IDX_SiC3I] - k[2080]*y[IDX_SiC3I] - k[2185]*y[IDX_SiC3I] +
        k[2495]*y[IDX_GSiC3I] + k[2496]*y[IDX_GSiC3I] + k[2497]*y[IDX_GSiC3I] +
        k[2498]*y[IDX_GSiC3I];
    ydot[IDX_SiH4II] = 0.0 + k[149]*y[IDX_HII]*y[IDX_SiH4I] -
        k[598]*y[IDX_SiH4II]*y[IDX_EM] - k[599]*y[IDX_SiH4II]*y[IDX_EM] -
        k[880]*y[IDX_COI]*y[IDX_SiH4II] - k[963]*y[IDX_H2I]*y[IDX_SiH4II] -
        k[1015]*y[IDX_H2OI]*y[IDX_SiH4II] + k[1073]*y[IDX_H3II]*y[IDX_SiH3I] -
        k[2180]*y[IDX_SiH4II];
    ydot[IDX_C2H5OH2II] = 0.0 - k[464]*y[IDX_C2H5OH2II]*y[IDX_EM] -
        k[465]*y[IDX_C2H5OH2II]*y[IDX_EM] - k[466]*y[IDX_C2H5OH2II]*y[IDX_EM] -
        k[467]*y[IDX_C2H5OH2II]*y[IDX_EM] + k[1081]*y[IDX_H3OII]*y[IDX_C2H5OHI]
        + k[1136]*y[IDX_HCOII]*y[IDX_C2H5OHI] +
        k[1164]*y[IDX_HCSII]*y[IDX_C2H5OHI] -
        k[1420]*y[IDX_NH3I]*y[IDX_C2H5OH2II] + k[2121]*y[IDX_H3OII]*y[IDX_C2H4I]
        - k[2151]*y[IDX_C2H5OH2II];
    ydot[IDX_C2NHII] = 0.0 - k[472]*y[IDX_C2NHII]*y[IDX_EM] +
        k[724]*y[IDX_CHII]*y[IDX_HCNI] + k[992]*y[IDX_H2OI]*y[IDX_C2NII] +
        k[1026]*y[IDX_H3II]*y[IDX_C2NI] + k[1329]*y[IDX_NI]*y[IDX_C2H2II] +
        k[1394]*y[IDX_NH2I]*y[IDX_C2II] - k[2162]*y[IDX_C2NHII];
    ydot[IDX_C3H5II] = 0.0 - k[474]*y[IDX_C3H5II]*y[IDX_EM] +
        k[665]*y[IDX_C2H2II]*y[IDX_CH3CNI] + k[774]*y[IDX_CH3II]*y[IDX_C2H4I] +
        k[806]*y[IDX_CH4I]*y[IDX_C2H2II] + k[1082]*y[IDX_H3OII]*y[IDX_CH3CCHI] +
        k[1137]*y[IDX_HCOII]*y[IDX_CH3CCHI] - k[2148]*y[IDX_C3H5II];
    ydot[IDX_CH2COI] = 0.0 - k[383]*y[IDX_CH2COI] -
        k[1194]*y[IDX_HeII]*y[IDX_CH2COI] - k[1195]*y[IDX_HeII]*y[IDX_CH2COI] -
        k[1740]*y[IDX_HI]*y[IDX_CH2COI] + k[1869]*y[IDX_OI]*y[IDX_C2H3I] +
        k[1871]*y[IDX_OI]*y[IDX_C2H4I] + k[1928]*y[IDX_OHI]*y[IDX_C2H2I] -
        k[1983]*y[IDX_CH2COI] - k[2147]*y[IDX_CH2COI] + k[2415]*y[IDX_GCH2COI] +
        k[2416]*y[IDX_GCH2COI] + k[2417]*y[IDX_GCH2COI] +
        k[2418]*y[IDX_GCH2COI];
    ydot[IDX_HSO2II] = 0.0 - k[558]*y[IDX_HSO2II]*y[IDX_EM] -
        k[559]*y[IDX_HSO2II]*y[IDX_EM] - k[560]*y[IDX_HSO2II]*y[IDX_EM] +
        k[962]*y[IDX_H2I]*y[IDX_SO2II] + k[989]*y[IDX_H2OII]*y[IDX_SO2I] -
        k[1008]*y[IDX_H2OI]*y[IDX_HSO2II] + k[1069]*y[IDX_H3II]*y[IDX_SO2I] -
        k[1438]*y[IDX_NH3I]*y[IDX_HSO2II] - k[2295]*y[IDX_HSO2II];
    ydot[IDX_C3NI] = 0.0 - k[377]*y[IDX_C3NI] +
        k[476]*y[IDX_C4NII]*y[IDX_EM] - k[1190]*y[IDX_HeII]*y[IDX_C3NI] +
        k[1571]*y[IDX_C2I]*y[IDX_HCNI] - k[1798]*y[IDX_NI]*y[IDX_C3NI] +
        k[1800]*y[IDX_NI]*y[IDX_C4NI] - k[1877]*y[IDX_OI]*y[IDX_C3NI] +
        k[1878]*y[IDX_OI]*y[IDX_C4NI] - k[1974]*y[IDX_C3NI] -
        k[2161]*y[IDX_C3NI];
    ydot[IDX_HCOOCH3I] = 0.0 - k[411]*y[IDX_HCOOCH3I] +
        k[537]*y[IDX_H5C2O2II]*y[IDX_EM] - k[1047]*y[IDX_H3II]*y[IDX_HCOOCH3I] -
        k[1089]*y[IDX_H3OII]*y[IDX_HCOOCH3I] -
        k[1145]*y[IDX_HCOII]*y[IDX_HCOOCH3I] -
        k[1238]*y[IDX_HeII]*y[IDX_HCOOCH3I] - k[2032]*y[IDX_HCOOCH3I] -
        k[2153]*y[IDX_HCOOCH3I] + k[2479]*y[IDX_GHCOOCH3I] +
        k[2480]*y[IDX_GHCOOCH3I] + k[2481]*y[IDX_GHCOOCH3I] +
        k[2482]*y[IDX_GHCOOCH3I];
    ydot[IDX_SiC2II] = 0.0 + k[27]*y[IDX_CII]*y[IDX_SiC2I] +
        k[144]*y[IDX_HII]*y[IDX_SiC2I] - k[588]*y[IDX_SiC2II]*y[IDX_EM] -
        k[589]*y[IDX_SiC2II]*y[IDX_EM] + k[670]*y[IDX_C2H2II]*y[IDX_SiI] +
        k[684]*y[IDX_C2HI]*y[IDX_SiII] + k[1275]*y[IDX_HeII]*y[IDX_SiC3I] -
        k[2184]*y[IDX_SiC2II];
    ydot[IDX_C3II] = 0.0 - k[473]*y[IDX_C3II]*y[IDX_EM] +
        k[607]*y[IDX_CII]*y[IDX_C2HI] + k[624]*y[IDX_CII]*y[IDX_HC3NI] +
        k[647]*y[IDX_C2II]*y[IDX_C2I] + k[685]*y[IDX_CI]*y[IDX_C2HII] +
        k[705]*y[IDX_CHII]*y[IDX_C2I] + k[706]*y[IDX_CHII]*y[IDX_C2HI] +
        k[837]*y[IDX_CHI]*y[IDX_C2II] - k[1119]*y[IDX_HCNI]*y[IDX_C3II] +
        k[1197]*y[IDX_HeII]*y[IDX_CH3CCHI] - k[2292]*y[IDX_C3II];
    ydot[IDX_H2S2II] = 0.0 + k[122]*y[IDX_HII]*y[IDX_H2S2I] -
        k[517]*y[IDX_H2S2II]*y[IDX_EM] - k[518]*y[IDX_H2S2II]*y[IDX_EM] +
        k[793]*y[IDX_CH3OHI]*y[IDX_S2II] + k[1052]*y[IDX_H3II]*y[IDX_HS2I] +
        k[1091]*y[IDX_H3OII]*y[IDX_HS2I] + k[1146]*y[IDX_HCOII]*y[IDX_HS2I] +
        k[1553]*y[IDX_SI]*y[IDX_H3SII] - k[2280]*y[IDX_H2S2II];
    ydot[IDX_HClI] = 0.0 - k[126]*y[IDX_HII]*y[IDX_HClI] -
        k[413]*y[IDX_HClI] + k[509]*y[IDX_H2ClII]*y[IDX_EM] -
        k[832]*y[IDX_CH5II]*y[IDX_HClI] + k[874]*y[IDX_COI]*y[IDX_H2ClII] +
        k[999]*y[IDX_H2OI]*y[IDX_H2ClII] - k[1049]*y[IDX_H3II]*y[IDX_HClI] -
        k[1241]*y[IDX_HeII]*y[IDX_HClI] + k[1720]*y[IDX_ClI]*y[IDX_H2I] -
        k[1787]*y[IDX_HClI]*y[IDX_HI] - k[2034]*y[IDX_HClI] -
        k[2035]*y[IDX_HClI] - k[2195]*y[IDX_HClI] + k[2395]*y[IDX_GHClI] +
        k[2396]*y[IDX_GHClI] + k[2397]*y[IDX_GHClI] + k[2398]*y[IDX_GHClI];
    ydot[IDX_HOCSII] = 0.0 - k[552]*y[IDX_HOCSII]*y[IDX_EM] -
        k[553]*y[IDX_HOCSII]*y[IDX_EM] + k[737]*y[IDX_CHII]*y[IDX_OCSI] +
        k[788]*y[IDX_CH3II]*y[IDX_SOI] + k[802]*y[IDX_CH4II]*y[IDX_OCSI] -
        k[1006]*y[IDX_H2OI]*y[IDX_HOCSII] + k[1065]*y[IDX_H3II]*y[IDX_OCSI] +
        k[1149]*y[IDX_HCOII]*y[IDX_OCSI] - k[2285]*y[IDX_HOCSII];
    ydot[IDX_HS2I] = 0.0 - k[127]*y[IDX_HII]*y[IDX_HS2I] -
        k[417]*y[IDX_HS2I] + k[517]*y[IDX_H2S2II]*y[IDX_EM] -
        k[1052]*y[IDX_H3II]*y[IDX_HS2I] - k[1091]*y[IDX_H3OII]*y[IDX_HS2I] -
        k[1146]*y[IDX_HCOII]*y[IDX_HS2I] - k[1247]*y[IDX_HeII]*y[IDX_HS2I] -
        k[1248]*y[IDX_HeII]*y[IDX_HS2I] - k[2041]*y[IDX_HS2I] -
        k[2042]*y[IDX_HS2I] - k[2204]*y[IDX_HS2I];
    ydot[IDX_NSII] = 0.0 + k[23]*y[IDX_CII]*y[IDX_NSI] +
        k[134]*y[IDX_HII]*y[IDX_NSI] - k[576]*y[IDX_NSII]*y[IDX_EM] +
        k[1335]*y[IDX_NI]*y[IDX_H2SII] + k[1336]*y[IDX_NI]*y[IDX_HSII] +
        k[1341]*y[IDX_NI]*y[IDX_SOII] + k[1373]*y[IDX_NHII]*y[IDX_SI] +
        k[1461]*y[IDX_NHI]*y[IDX_SII] - k[1509]*y[IDX_OI]*y[IDX_NSII] -
        k[2276]*y[IDX_NSII];
    ydot[IDX_CH3CCHI] = 0.0 + k[474]*y[IDX_C3H5II]*y[IDX_EM] -
        k[611]*y[IDX_CII]*y[IDX_CH3CCHI] - k[1082]*y[IDX_H3OII]*y[IDX_CH3CCHI] -
        k[1137]*y[IDX_HCOII]*y[IDX_CH3CCHI] - k[1197]*y[IDX_HeII]*y[IDX_CH3CCHI]
        + k[1583]*y[IDX_CI]*y[IDX_C2H5I] + k[1675]*y[IDX_CHI]*y[IDX_C2H4I] -
        k[2149]*y[IDX_CH3CCHI] + k[2403]*y[IDX_GCH3CCHI] +
        k[2404]*y[IDX_GCH3CCHI] + k[2405]*y[IDX_GCH3CCHI] +
        k[2406]*y[IDX_GCH3CCHI];
    ydot[IDX_H3CSII] = 0.0 - k[526]*y[IDX_H3CSII]*y[IDX_EM] -
        k[527]*y[IDX_H3CSII]*y[IDX_EM] + k[744]*y[IDX_CH2II]*y[IDX_H2SI] +
        k[778]*y[IDX_CH3II]*y[IDX_H2SI] + k[785]*y[IDX_CH3II]*y[IDX_OCSI] +
        k[814]*y[IDX_CH4I]*y[IDX_HSII] + k[821]*y[IDX_CH4I]*y[IDX_SII] +
        k[1042]*y[IDX_H3II]*y[IDX_H2CSI] + k[1142]*y[IDX_HCOII]*y[IDX_H2CSI] +
        k[1561]*y[IDX_SOII]*y[IDX_C2H4I] - k[2282]*y[IDX_H3CSII];
    ydot[IDX_SiC2I] = 0.0 - k[27]*y[IDX_CII]*y[IDX_SiC2I] -
        k[144]*y[IDX_HII]*y[IDX_SiC2I] - k[449]*y[IDX_SiC2I] +
        k[450]*y[IDX_SiC3I] + k[590]*y[IDX_SiC3II]*y[IDX_EM] -
        k[1274]*y[IDX_HeII]*y[IDX_SiC2I] + k[1575]*y[IDX_C2H2I]*y[IDX_SiI] -
        k[1918]*y[IDX_OI]*y[IDX_SiC2I] + k[1919]*y[IDX_OI]*y[IDX_SiC3I] -
        k[2078]*y[IDX_SiC2I] + k[2080]*y[IDX_SiC3I] - k[2182]*y[IDX_SiC2I] +
        k[2475]*y[IDX_GSiC2I] + k[2476]*y[IDX_GSiC2I] + k[2477]*y[IDX_GSiC2I] +
        k[2478]*y[IDX_GSiC2I];
    ydot[IDX_SO2II] = 0.0 + k[141]*y[IDX_HII]*y[IDX_SO2I] +
        k[218]*y[IDX_HeII]*y[IDX_SO2I] + k[320]*y[IDX_OII]*y[IDX_SO2I] -
        k[325]*y[IDX_O2I]*y[IDX_SO2II] - k[585]*y[IDX_SO2II]*y[IDX_EM] -
        k[586]*y[IDX_SO2II]*y[IDX_EM] - k[879]*y[IDX_COI]*y[IDX_SO2II] -
        k[962]*y[IDX_H2I]*y[IDX_SO2II] - k[1107]*y[IDX_HI]*y[IDX_SO2II] -
        k[2284]*y[IDX_SO2II];
    ydot[IDX_H2CSI] = 0.0 - k[120]*y[IDX_HII]*y[IDX_H2CSI] -
        k[400]*y[IDX_H2CSI] + k[527]*y[IDX_H3CSII]*y[IDX_EM] -
        k[619]*y[IDX_CII]*y[IDX_H2CSI] - k[1042]*y[IDX_H3II]*y[IDX_H2CSI] -
        k[1142]*y[IDX_HCOII]*y[IDX_H2CSI] - k[1219]*y[IDX_HeII]*y[IDX_H2CSI] -
        k[1220]*y[IDX_HeII]*y[IDX_H2CSI] - k[1221]*y[IDX_HeII]*y[IDX_H2CSI] +
        k[1669]*y[IDX_CH3I]*y[IDX_SI] - k[2015]*y[IDX_H2CSI] +
        k[2137]*y[IDX_H2CSII]*y[IDX_EM] - k[2263]*y[IDX_H2CSI] +
        k[2443]*y[IDX_GH2CSI] + k[2444]*y[IDX_GH2CSI] + k[2445]*y[IDX_GH2CSI] +
        k[2446]*y[IDX_GH2CSI];
    ydot[IDX_H2CSII] = 0.0 + k[120]*y[IDX_HII]*y[IDX_H2CSI] -
        k[506]*y[IDX_H2CSII]*y[IDX_EM] - k[507]*y[IDX_H2CSII]*y[IDX_EM] +
        k[751]*y[IDX_CH2II]*y[IDX_OCSI] + k[780]*y[IDX_CH3II]*y[IDX_HSI] +
        k[790]*y[IDX_CH3I]*y[IDX_SII] + k[1048]*y[IDX_H3II]*y[IDX_HCSI] +
        k[1556]*y[IDX_SOII]*y[IDX_C2H2I] + k[1559]*y[IDX_SOII]*y[IDX_C2H4I] -
        k[2137]*y[IDX_H2CSII]*y[IDX_EM] - k[2279]*y[IDX_H2CSII];
    ydot[IDX_HC3NI] = 0.0 - k[407]*y[IDX_HC3NI] -
        k[623]*y[IDX_CII]*y[IDX_HC3NI] - k[624]*y[IDX_CII]*y[IDX_HC3NI] -
        k[625]*y[IDX_CII]*y[IDX_HC3NI] + k[994]*y[IDX_H2OI]*y[IDX_C4NII] -
        k[1229]*y[IDX_HeII]*y[IDX_HC3NI] - k[1230]*y[IDX_HeII]*y[IDX_HC3NI] +
        k[1578]*y[IDX_C2HI]*y[IDX_HCNI] + k[1579]*y[IDX_C2HI]*y[IDX_HNCI] +
        k[1580]*y[IDX_C2HI]*y[IDX_NCCNI] + k[1701]*y[IDX_CNI]*y[IDX_C2H2I] +
        k[1797]*y[IDX_NI]*y[IDX_C3H2I] - k[2027]*y[IDX_HC3NI] +
        k[2100]*y[IDX_C2HI]*y[IDX_CNI] - k[2145]*y[IDX_HC3NI] +
        k[2467]*y[IDX_GHC3NI] + k[2468]*y[IDX_GHC3NI] + k[2469]*y[IDX_GHC3NI] +
        k[2470]*y[IDX_GHC3NI];
    ydot[IDX_HSiSII] = 0.0 - k[561]*y[IDX_HSiSII]*y[IDX_EM] -
        k[562]*y[IDX_HSiSII]*y[IDX_EM] - k[1009]*y[IDX_H2OI]*y[IDX_HSiSII] -
        k[1020]*y[IDX_H2SI]*y[IDX_HSiSII] + k[1077]*y[IDX_H3II]*y[IDX_SiSI] -
        k[1127]*y[IDX_HCNI]*y[IDX_HSiSII] + k[1157]*y[IDX_HCOII]*y[IDX_SiSI] -
        k[1439]*y[IDX_NH3I]*y[IDX_HSiSII] + k[1568]*y[IDX_SiH2II]*y[IDX_SI] -
        k[2191]*y[IDX_HSiSII];
    ydot[IDX_SiSII] = 0.0 + k[32]*y[IDX_CII]*y[IDX_SiSI] +
        k[152]*y[IDX_HII]*y[IDX_SiSI] + k[344]*y[IDX_SII]*y[IDX_SiSI] -
        k[605]*y[IDX_SiSII]*y[IDX_EM] - k[1109]*y[IDX_HI]*y[IDX_SiSII] -
        k[1487]*y[IDX_O2I]*y[IDX_SiSII] - k[1488]*y[IDX_O2I]*y[IDX_SiSII] +
        k[1565]*y[IDX_SiII]*y[IDX_OCSI] + k[1569]*y[IDX_SiHI]*y[IDX_SII] -
        k[2190]*y[IDX_SiSII];
    ydot[IDX_HS2II] = 0.0 + k[127]*y[IDX_HII]*y[IDX_HS2I] -
        k[555]*y[IDX_HS2II]*y[IDX_EM] - k[556]*y[IDX_HS2II]*y[IDX_EM] -
        k[1019]*y[IDX_H2SI]*y[IDX_HS2II] + k[1067]*y[IDX_H3II]*y[IDX_S2I] +
        k[1092]*y[IDX_H3OII]*y[IDX_S2I] + k[1150]*y[IDX_HCOII]*y[IDX_S2I] +
        k[1176]*y[IDX_HSII]*y[IDX_H2SI] + k[1225]*y[IDX_HeII]*y[IDX_H2S2I] +
        k[1550]*y[IDX_SII]*y[IDX_H2SI] + k[2041]*y[IDX_HS2I] -
        k[2205]*y[IDX_HS2II];
    ydot[IDX_O2HI] = 0.0 - k[437]*y[IDX_O2HI] +
        k[967]*y[IDX_H2COII]*y[IDX_O2I] + k[1577]*y[IDX_C2H3I]*y[IDX_O2I] +
        k[1662]*y[IDX_CH3I]*y[IDX_O2I] - k[1663]*y[IDX_CH3I]*y[IDX_O2HI] +
        k[1671]*y[IDX_CH4I]*y[IDX_O2I] - k[1691]*y[IDX_CHI]*y[IDX_O2HI] -
        k[1692]*y[IDX_CHI]*y[IDX_O2HI] - k[1719]*y[IDX_COI]*y[IDX_O2HI] +
        k[1731]*y[IDX_H2I]*y[IDX_O2I] - k[1769]*y[IDX_HI]*y[IDX_O2HI] -
        k[1770]*y[IDX_HI]*y[IDX_O2HI] - k[1771]*y[IDX_HI]*y[IDX_O2HI] +
        k[1785]*y[IDX_HCOI]*y[IDX_O2I] - k[1786]*y[IDX_HCOI]*y[IDX_O2HI] -
        k[1826]*y[IDX_NI]*y[IDX_O2HI] - k[1909]*y[IDX_OI]*y[IDX_O2HI] -
        k[1946]*y[IDX_OHI]*y[IDX_O2HI] - k[2063]*y[IDX_O2HI] -
        k[2064]*y[IDX_O2HI] - k[2294]*y[IDX_O2HI] + k[2387]*y[IDX_GO2HI] +
        k[2388]*y[IDX_GO2HI] + k[2389]*y[IDX_GO2HI] + k[2390]*y[IDX_GO2HI];
    ydot[IDX_CH3CNHII] = 0.0 - k[484]*y[IDX_CH3CNHII]*y[IDX_EM] -
        k[485]*y[IDX_CH3CNHII]*y[IDX_EM] + k[666]*y[IDX_C2H2II]*y[IDX_CH3CNI] +
        k[791]*y[IDX_CH3CNI]*y[IDX_HCO2II] + k[1030]*y[IDX_H3II]*y[IDX_CH3CNI] +
        k[1083]*y[IDX_H3OII]*y[IDX_CH3CNI] + k[1120]*y[IDX_HCNI]*y[IDX_CH3OH2II]
        + k[1130]*y[IDX_HCNHII]*y[IDX_CH3CNI] +
        k[1131]*y[IDX_HCNHII]*y[IDX_CH3CNI] + k[1138]*y[IDX_HCOII]*y[IDX_CH3CNI]
        + k[1321]*y[IDX_N2HII]*y[IDX_CH3CNI] + k[2108]*y[IDX_CH3II]*y[IDX_HCNI]
        - k[2156]*y[IDX_CH3CNHII];
    ydot[IDX_NO2I] = 0.0 - k[431]*y[IDX_NO2I] -
        k[903]*y[IDX_HII]*y[IDX_NO2I] - k[1059]*y[IDX_H3II]*y[IDX_NO2I] -
        k[1478]*y[IDX_OII]*y[IDX_NO2I] - k[1628]*y[IDX_CH2I]*y[IDX_NO2I] -
        k[1658]*y[IDX_CH3I]*y[IDX_NO2I] - k[1709]*y[IDX_CNI]*y[IDX_NO2I] -
        k[1717]*y[IDX_COI]*y[IDX_NO2I] - k[1763]*y[IDX_HI]*y[IDX_NO2I] -
        k[1820]*y[IDX_NI]*y[IDX_NO2I] - k[1821]*y[IDX_NI]*y[IDX_NO2I] -
        k[1822]*y[IDX_NI]*y[IDX_NO2I] - k[1846]*y[IDX_NHI]*y[IDX_NO2I] +
        k[1859]*y[IDX_NOI]*y[IDX_O2I] + k[1864]*y[IDX_O2I]*y[IDX_OCNI] +
        k[1896]*y[IDX_OI]*y[IDX_HNOI] - k[1905]*y[IDX_OI]*y[IDX_NO2I] +
        k[1945]*y[IDX_OHI]*y[IDX_NOI] - k[2056]*y[IDX_NO2I] -
        k[2270]*y[IDX_NO2I] + k[2447]*y[IDX_GNO2I] + k[2448]*y[IDX_GNO2I] +
        k[2449]*y[IDX_GNO2I] + k[2450]*y[IDX_GNO2I];
    ydot[IDX_S2II] = 0.0 + k[139]*y[IDX_HII]*y[IDX_S2I] -
        k[304]*y[IDX_NOI]*y[IDX_S2II] - k[583]*y[IDX_S2II]*y[IDX_EM] -
        k[793]*y[IDX_CH3OHI]*y[IDX_S2II] + k[1021]*y[IDX_H2SI]*y[IDX_SOII] +
        k[1248]*y[IDX_HeII]*y[IDX_HS2I] + k[1551]*y[IDX_SII]*y[IDX_H2SI] +
        k[1552]*y[IDX_SII]*y[IDX_OCSI] + k[1562]*y[IDX_SOII]*y[IDX_OCSI] +
        k[2071]*y[IDX_S2I] - k[2281]*y[IDX_S2II];
    ydot[IDX_C2H5I] = 0.0 - k[371]*y[IDX_C2H5I] + k[372]*y[IDX_C2H5OHI] +
        k[465]*y[IDX_C2H5OH2II]*y[IDX_EM] + k[466]*y[IDX_C2H5OH2II]*y[IDX_EM] +
        k[672]*y[IDX_C2H2II]*y[IDX_SiH4I] + k[1563]*y[IDX_SiII]*y[IDX_C2H5OHI] -
        k[1583]*y[IDX_CI]*y[IDX_C2H5I] + k[1648]*y[IDX_CH3I]*y[IDX_CH3I] -
        k[1793]*y[IDX_NI]*y[IDX_C2H5I] - k[1794]*y[IDX_NI]*y[IDX_C2H5I] -
        k[1874]*y[IDX_OI]*y[IDX_C2H5I] - k[1931]*y[IDX_OHI]*y[IDX_C2H5I] -
        k[1968]*y[IDX_C2H5I] + k[1969]*y[IDX_C2H5OHI] - k[2301]*y[IDX_C2H5I] +
        k[2359]*y[IDX_GC2H5I] + k[2360]*y[IDX_GC2H5I] + k[2361]*y[IDX_GC2H5I] +
        k[2362]*y[IDX_GC2H5I];
    ydot[IDX_C2H5OHI] = 0.0 - k[372]*y[IDX_C2H5OHI] +
        k[467]*y[IDX_C2H5OH2II]*y[IDX_EM] - k[606]*y[IDX_CII]*y[IDX_C2H5OHI] -
        k[1023]*y[IDX_H3II]*y[IDX_C2H5OHI] - k[1024]*y[IDX_H3II]*y[IDX_C2H5OHI]
        - k[1081]*y[IDX_H3OII]*y[IDX_C2H5OHI] -
        k[1136]*y[IDX_HCOII]*y[IDX_C2H5OHI] -
        k[1164]*y[IDX_HCSII]*y[IDX_C2H5OHI] +
        k[1420]*y[IDX_NH3I]*y[IDX_C2H5OH2II] -
        k[1563]*y[IDX_SiII]*y[IDX_C2H5OHI] - k[1969]*y[IDX_C2H5OHI] -
        k[2150]*y[IDX_C2H5OHI] + k[2439]*y[IDX_GC2H5OHI] +
        k[2440]*y[IDX_GC2H5OHI] + k[2441]*y[IDX_GC2H5OHI] +
        k[2442]*y[IDX_GC2H5OHI];
    ydot[IDX_SiCII] = 0.0 + k[29]*y[IDX_CII]*y[IDX_SiCI] +
        k[146]*y[IDX_HII]*y[IDX_SiCI] + k[343]*y[IDX_SII]*y[IDX_SiCI] -
        k[587]*y[IDX_SiCII]*y[IDX_EM] + k[643]*y[IDX_CII]*y[IDX_SiH2I] +
        k[644]*y[IDX_CII]*y[IDX_SiHI] + k[646]*y[IDX_CII]*y[IDX_SiSI] +
        k[659]*y[IDX_C2I]*y[IDX_SiOII] + k[703]*y[IDX_CI]*y[IDX_SiHII] +
        k[862]*y[IDX_CHI]*y[IDX_SiII] - k[1342]*y[IDX_NI]*y[IDX_SiCII] -
        k[1512]*y[IDX_OI]*y[IDX_SiCII] - k[2183]*y[IDX_SiCII];
    ydot[IDX_C2NI] = 0.0 - k[113]*y[IDX_HII]*y[IDX_C2NI] -
        k[375]*y[IDX_C2NI] - k[376]*y[IDX_C2NI] + k[470]*y[IDX_C2N2II]*y[IDX_EM]
        + k[472]*y[IDX_C2NHII]*y[IDX_EM] + k[475]*y[IDX_C4NII]*y[IDX_EM] -
        k[1026]*y[IDX_H3II]*y[IDX_C2NI] - k[1189]*y[IDX_HeII]*y[IDX_C2NI] -
        k[1584]*y[IDX_CI]*y[IDX_C2NI] + k[1593]*y[IDX_CI]*y[IDX_H2CNI] +
        k[1598]*y[IDX_CI]*y[IDX_NCCNI] + k[1795]*y[IDX_NI]*y[IDX_C2HI] -
        k[1796]*y[IDX_NI]*y[IDX_C2NI] + k[1798]*y[IDX_NI]*y[IDX_C3NI] +
        k[1818]*y[IDX_NI]*y[IDX_NCCNI] - k[1876]*y[IDX_OI]*y[IDX_C2NI] +
        k[1877]*y[IDX_OI]*y[IDX_C3NI] - k[1972]*y[IDX_C2NI] -
        k[1973]*y[IDX_C2NI] - k[2159]*y[IDX_C2NI];
    ydot[IDX_NCCNI] = 0.0 - k[20]*y[IDX_CII]*y[IDX_NCCNI] -
        k[423]*y[IDX_NCCNI] + k[675]*y[IDX_C2H2I]*y[IDX_C2N2II] +
        k[1118]*y[IDX_HCNI]*y[IDX_C2N2II] - k[1251]*y[IDX_HeII]*y[IDX_NCCNI] -
        k[1304]*y[IDX_NII]*y[IDX_NCCNI] - k[1305]*y[IDX_NII]*y[IDX_NCCNI] -
        k[1580]*y[IDX_C2HI]*y[IDX_NCCNI] - k[1598]*y[IDX_CI]*y[IDX_NCCNI] +
        k[1705]*y[IDX_CNI]*y[IDX_HCNI] + k[1707]*y[IDX_CNI]*y[IDX_HNCI] -
        k[1759]*y[IDX_HI]*y[IDX_NCCNI] - k[1818]*y[IDX_NI]*y[IDX_NCCNI] -
        k[1943]*y[IDX_OHI]*y[IDX_NCCNI] - k[2046]*y[IDX_NCCNI] -
        k[2047]*y[IDX_NCCNI] - k[2164]*y[IDX_NCCNI] + k[2471]*y[IDX_GNCCNI] +
        k[2472]*y[IDX_GNCCNI] + k[2473]*y[IDX_GNCCNI] + k[2474]*y[IDX_GNCCNI];
    ydot[IDX_SiCI] = 0.0 - k[29]*y[IDX_CII]*y[IDX_SiCI] -
        k[146]*y[IDX_HII]*y[IDX_SiCI] - k[343]*y[IDX_SII]*y[IDX_SiCI] +
        k[449]*y[IDX_SiC2I] - k[451]*y[IDX_SiCI] +
        k[589]*y[IDX_SiC2II]*y[IDX_EM] + k[591]*y[IDX_SiC3II]*y[IDX_EM] -
        k[642]*y[IDX_CII]*y[IDX_SiCI] - k[1276]*y[IDX_HeII]*y[IDX_SiCI] -
        k[1277]*y[IDX_HeII]*y[IDX_SiCI] + k[1617]*y[IDX_CI]*y[IDX_SiHI] -
        k[1832]*y[IDX_NI]*y[IDX_SiCI] + k[1918]*y[IDX_OI]*y[IDX_SiC2I] -
        k[1920]*y[IDX_OI]*y[IDX_SiCI] - k[1921]*y[IDX_OI]*y[IDX_SiCI] +
        k[2079]*y[IDX_SiC3I] - k[2081]*y[IDX_SiCI] - k[2181]*y[IDX_SiCI] +
        k[2407]*y[IDX_GSiCI] + k[2408]*y[IDX_GSiCI] + k[2409]*y[IDX_GSiCI] +
        k[2410]*y[IDX_GSiCI];
    ydot[IDX_HCSI] = 0.0 - k[412]*y[IDX_HCSI] +
        k[507]*y[IDX_H2CSII]*y[IDX_EM] - k[899]*y[IDX_HII]*y[IDX_HCSI] -
        k[1048]*y[IDX_H3II]*y[IDX_HCSI] - k[1239]*y[IDX_HeII]*y[IDX_HCSI] -
        k[1240]*y[IDX_HeII]*y[IDX_HCSI] + k[1557]*y[IDX_SOII]*y[IDX_C2H2I] +
        k[1560]*y[IDX_SOII]*y[IDX_C2H4I] + k[1645]*y[IDX_CH2I]*y[IDX_SI] -
        k[1753]*y[IDX_HI]*y[IDX_HCSI] - k[1814]*y[IDX_NI]*y[IDX_HCSI] -
        k[1894]*y[IDX_OI]*y[IDX_HCSI] - k[1895]*y[IDX_OI]*y[IDX_HCSI] -
        k[2033]*y[IDX_HCSI] - k[2258]*y[IDX_HCSI];
    ydot[IDX_SiH2I] = 0.0 - k[30]*y[IDX_CII]*y[IDX_SiH2I] -
        k[147]*y[IDX_HII]*y[IDX_SiH2I] - k[452]*y[IDX_SiH2I] +
        k[453]*y[IDX_SiH3I] + k[454]*y[IDX_SiH4I] +
        k[596]*y[IDX_SiH3II]*y[IDX_EM] + k[598]*y[IDX_SiH4II]*y[IDX_EM] -
        k[643]*y[IDX_CII]*y[IDX_SiH2I] - k[905]*y[IDX_HII]*y[IDX_SiH2I] -
        k[1072]*y[IDX_H3II]*y[IDX_SiH2I] - k[1094]*y[IDX_H3OII]*y[IDX_SiH2I] -
        k[1153]*y[IDX_HCOII]*y[IDX_SiH2I] - k[1278]*y[IDX_HeII]*y[IDX_SiH2I] -
        k[1279]*y[IDX_HeII]*y[IDX_SiH2I] - k[1922]*y[IDX_OI]*y[IDX_SiH2I] -
        k[1923]*y[IDX_OI]*y[IDX_SiH2I] - k[2083]*y[IDX_SiH2I] -
        k[2084]*y[IDX_SiH2I] + k[2085]*y[IDX_SiH3I] + k[2088]*y[IDX_SiH4I] -
        k[2175]*y[IDX_SiH2I];
    ydot[IDX_SiH3II] = 0.0 + k[31]*y[IDX_CII]*y[IDX_SiH3I] +
        k[148]*y[IDX_HII]*y[IDX_SiH3I] - k[596]*y[IDX_SiH3II]*y[IDX_EM] -
        k[597]*y[IDX_SiH3II]*y[IDX_EM] + k[674]*y[IDX_C2H2II]*y[IDX_SiH4I] +
        k[789]*y[IDX_CH3II]*y[IDX_SiH4I] + k[836]*y[IDX_CH5II]*y[IDX_SiH4I] +
        k[907]*y[IDX_HII]*y[IDX_SiH4I] + k[1072]*y[IDX_H3II]*y[IDX_SiH2I] +
        k[1094]*y[IDX_H3OII]*y[IDX_SiH2I] + k[1153]*y[IDX_HCOII]*y[IDX_SiH2I] -
        k[1515]*y[IDX_OI]*y[IDX_SiH3II] + k[2086]*y[IDX_SiH3I] +
        k[2119]*y[IDX_H2I]*y[IDX_SiHII] - k[2120]*y[IDX_H2I]*y[IDX_SiH3II] -
        k[2178]*y[IDX_SiH3II];
    ydot[IDX_SiSI] = 0.0 - k[32]*y[IDX_CII]*y[IDX_SiSI] -
        k[152]*y[IDX_HII]*y[IDX_SiSI] - k[344]*y[IDX_SII]*y[IDX_SiSI] -
        k[457]*y[IDX_SiSI] + k[562]*y[IDX_HSiSII]*y[IDX_EM] -
        k[646]*y[IDX_CII]*y[IDX_SiSI] + k[1020]*y[IDX_H2SI]*y[IDX_HSiSII] -
        k[1077]*y[IDX_H3II]*y[IDX_SiSI] + k[1127]*y[IDX_HCNI]*y[IDX_HSiSII] -
        k[1157]*y[IDX_HCOII]*y[IDX_SiSI] - k[1287]*y[IDX_HeII]*y[IDX_SiSI] -
        k[1288]*y[IDX_HeII]*y[IDX_SiSI] + k[1439]*y[IDX_NH3I]*y[IDX_HSiSII] -
        k[2095]*y[IDX_SiSI] - k[2189]*y[IDX_SiSI] + k[2483]*y[IDX_GSiSI] +
        k[2484]*y[IDX_GSiSI] + k[2485]*y[IDX_GSiSI] + k[2486]*y[IDX_GSiSI];
    ydot[IDX_CH3OH2II] = 0.0 - k[486]*y[IDX_CH3OH2II]*y[IDX_EM] -
        k[487]*y[IDX_CH3OH2II]*y[IDX_EM] - k[488]*y[IDX_CH3OH2II]*y[IDX_EM] -
        k[489]*y[IDX_CH3OH2II]*y[IDX_EM] - k[490]*y[IDX_CH3OH2II]*y[IDX_EM] +
        k[708]*y[IDX_CHII]*y[IDX_CH3OHI] - k[792]*y[IDX_CH3OH2II]*y[IDX_NH3I] +
        k[794]*y[IDX_CH4II]*y[IDX_CH3OHI] + k[965]*y[IDX_H2COII]*y[IDX_CH3OHI] -
        k[969]*y[IDX_H2COI]*y[IDX_CH3OH2II] + k[1032]*y[IDX_H3II]*y[IDX_CH3OHI]
        + k[1078]*y[IDX_H3COII]*y[IDX_CH3OHI] +
        k[1084]*y[IDX_H3OII]*y[IDX_CH3OHI] - k[1120]*y[IDX_HCNI]*y[IDX_CH3OH2II]
        + k[1139]*y[IDX_HCOII]*y[IDX_CH3OHI] + k[2107]*y[IDX_CH3II]*y[IDX_H2OI]
        - k[2152]*y[IDX_CH3OH2II];
    ydot[IDX_OCNI] = 0.0 - k[439]*y[IDX_OCNI] -
        k[635]*y[IDX_CII]*y[IDX_OCNI] - k[1262]*y[IDX_HeII]*y[IDX_OCNI] -
        k[1263]*y[IDX_HeII]*y[IDX_OCNI] - k[1609]*y[IDX_CI]*y[IDX_OCNI] +
        k[1686]*y[IDX_CHI]*y[IDX_NOI] + k[1709]*y[IDX_CNI]*y[IDX_NO2I] +
        k[1711]*y[IDX_CNI]*y[IDX_NOI] + k[1713]*y[IDX_CNI]*y[IDX_O2I] -
        k[1772]*y[IDX_HI]*y[IDX_OCNI] - k[1773]*y[IDX_HI]*y[IDX_OCNI] -
        k[1774]*y[IDX_HI]*y[IDX_OCNI] + k[1813]*y[IDX_NI]*y[IDX_HCOI] -
        k[1860]*y[IDX_NOI]*y[IDX_OCNI] - k[1863]*y[IDX_O2I]*y[IDX_OCNI] -
        k[1864]*y[IDX_O2I]*y[IDX_OCNI] + k[1885]*y[IDX_OI]*y[IDX_H2CNI] +
        k[1891]*y[IDX_OI]*y[IDX_HCNI] - k[1910]*y[IDX_OI]*y[IDX_OCNI] -
        k[1911]*y[IDX_OI]*y[IDX_OCNI] + k[1933]*y[IDX_OHI]*y[IDX_CNI] +
        k[1943]*y[IDX_OHI]*y[IDX_NCCNI] - k[2065]*y[IDX_OCNI] -
        k[2158]*y[IDX_OCNI];
    ydot[IDX_S2I] = 0.0 - k[139]*y[IDX_HII]*y[IDX_S2I] +
        k[304]*y[IDX_NOI]*y[IDX_S2II] - k[443]*y[IDX_S2I] +
        k[556]*y[IDX_HS2II]*y[IDX_EM] + k[1019]*y[IDX_H2SI]*y[IDX_HS2II] -
        k[1067]*y[IDX_H3II]*y[IDX_S2I] - k[1092]*y[IDX_H3OII]*y[IDX_S2I] -
        k[1150]*y[IDX_HCOII]*y[IDX_S2I] - k[1269]*y[IDX_HeII]*y[IDX_S2I] -
        k[1613]*y[IDX_CI]*y[IDX_S2I] - k[1777]*y[IDX_HI]*y[IDX_S2I] -
        k[1829]*y[IDX_NI]*y[IDX_S2I] - k[1915]*y[IDX_OI]*y[IDX_S2I] +
        k[1953]*y[IDX_SI]*y[IDX_HSI] + k[1955]*y[IDX_SI]*y[IDX_SOI] -
        k[2071]*y[IDX_S2I] - k[2072]*y[IDX_S2I] - k[2201]*y[IDX_S2I];
    ydot[IDX_SiH3I] = 0.0 - k[31]*y[IDX_CII]*y[IDX_SiH3I] -
        k[148]*y[IDX_HII]*y[IDX_SiH3I] - k[453]*y[IDX_SiH3I] +
        k[599]*y[IDX_SiH4II]*y[IDX_EM] + k[600]*y[IDX_SiH5II]*y[IDX_EM] +
        k[880]*y[IDX_COI]*y[IDX_SiH4II] - k[906]*y[IDX_HII]*y[IDX_SiH3I] +
        k[1015]*y[IDX_H2OI]*y[IDX_SiH4II] - k[1073]*y[IDX_H3II]*y[IDX_SiH3I] -
        k[1280]*y[IDX_HeII]*y[IDX_SiH3I] - k[1281]*y[IDX_HeII]*y[IDX_SiH3I] +
        k[1715]*y[IDX_CNI]*y[IDX_SiH4I] - k[1924]*y[IDX_OI]*y[IDX_SiH3I] +
        k[1925]*y[IDX_OI]*y[IDX_SiH4I] - k[2085]*y[IDX_SiH3I] -
        k[2086]*y[IDX_SiH3I] - k[2087]*y[IDX_SiH3I] + k[2089]*y[IDX_SiH4I] -
        k[2177]*y[IDX_SiH3I];
    ydot[IDX_CH3CNI] = 0.0 - k[387]*y[IDX_CH3CNI] +
        k[484]*y[IDX_CH3CNHII]*y[IDX_EM] - k[665]*y[IDX_C2H2II]*y[IDX_CH3CNI] -
        k[666]*y[IDX_C2H2II]*y[IDX_CH3CNI] - k[775]*y[IDX_CH3II]*y[IDX_CH3CNI] -
        k[791]*y[IDX_CH3CNI]*y[IDX_HCO2II] - k[886]*y[IDX_HII]*y[IDX_CH3CNI] -
        k[1030]*y[IDX_H3II]*y[IDX_CH3CNI] - k[1083]*y[IDX_H3OII]*y[IDX_CH3CNI] -
        k[1130]*y[IDX_HCNHII]*y[IDX_CH3CNI] -
        k[1131]*y[IDX_HCNHII]*y[IDX_CH3CNI] - k[1138]*y[IDX_HCOII]*y[IDX_CH3CNI]
        - k[1198]*y[IDX_HeII]*y[IDX_CH3CNI] - k[1199]*y[IDX_HeII]*y[IDX_CH3CNI]
        - k[1321]*y[IDX_N2HII]*y[IDX_CH3CNI] - k[1989]*y[IDX_CH3CNI] +
        k[2109]*y[IDX_CH3I]*y[IDX_CNI] - k[2299]*y[IDX_CH3CNI] +
        k[2411]*y[IDX_GCH3CNI] + k[2412]*y[IDX_GCH3CNI] + k[2413]*y[IDX_GCH3CNI]
        + k[2414]*y[IDX_GCH3CNI];
    ydot[IDX_NSI] = 0.0 - k[23]*y[IDX_CII]*y[IDX_NSI] -
        k[134]*y[IDX_HII]*y[IDX_NSI] - k[434]*y[IDX_NSI] +
        k[550]*y[IDX_HNSII]*y[IDX_EM] - k[632]*y[IDX_CII]*y[IDX_NSI] -
        k[1061]*y[IDX_H3II]*y[IDX_NSI] - k[1148]*y[IDX_HCOII]*y[IDX_NSI] -
        k[1259]*y[IDX_HeII]*y[IDX_NSI] - k[1260]*y[IDX_HeII]*y[IDX_NSI] -
        k[1606]*y[IDX_CI]*y[IDX_NSI] - k[1607]*y[IDX_CI]*y[IDX_NSI] +
        k[1714]*y[IDX_CNI]*y[IDX_SI] - k[1766]*y[IDX_HI]*y[IDX_NSI] -
        k[1767]*y[IDX_HI]*y[IDX_NSI] + k[1816]*y[IDX_NI]*y[IDX_HSI] -
        k[1824]*y[IDX_NI]*y[IDX_NSI] + k[1829]*y[IDX_NI]*y[IDX_S2I] +
        k[1830]*y[IDX_NI]*y[IDX_SOI] + k[1857]*y[IDX_NHI]*y[IDX_SI] +
        k[1861]*y[IDX_NOI]*y[IDX_SI] - k[1907]*y[IDX_OI]*y[IDX_NSI] -
        k[1908]*y[IDX_OI]*y[IDX_NSI] - k[2059]*y[IDX_NSI] - k[2262]*y[IDX_NSI] +
        k[2451]*y[IDX_GNSI] + k[2452]*y[IDX_GNSI] + k[2453]*y[IDX_GNSI] +
        k[2454]*y[IDX_GNSI];
    ydot[IDX_OCSII] = 0.0 + k[24]*y[IDX_CII]*y[IDX_OCSI] +
        k[80]*y[IDX_CH4II]*y[IDX_OCSI] + k[137]*y[IDX_HII]*y[IDX_OCSI] +
        k[184]*y[IDX_H2OII]*y[IDX_OCSI] - k[190]*y[IDX_H2SI]*y[IDX_OCSII] +
        k[250]*y[IDX_NII]*y[IDX_OCSI] + k[257]*y[IDX_N2II]*y[IDX_OCSI] -
        k[291]*y[IDX_NH3I]*y[IDX_OCSII] + k[318]*y[IDX_OII]*y[IDX_OCSI] +
        k[440]*y[IDX_OCSI] - k[579]*y[IDX_OCSII]*y[IDX_EM] -
        k[580]*y[IDX_OCSII]*y[IDX_EM] - k[581]*y[IDX_OCSII]*y[IDX_EM] +
        k[1485]*y[IDX_O2I]*y[IDX_CSII] + k[1501]*y[IDX_OI]*y[IDX_HCSII] +
        k[2066]*y[IDX_OCSI] - k[2269]*y[IDX_OCSII];
    ydot[IDX_SiH2II] = 0.0 + k[30]*y[IDX_CII]*y[IDX_SiH2I] +
        k[147]*y[IDX_HII]*y[IDX_SiH2I] - k[593]*y[IDX_SiH2II]*y[IDX_EM] -
        k[594]*y[IDX_SiH2II]*y[IDX_EM] - k[595]*y[IDX_SiH2II]*y[IDX_EM] +
        k[673]*y[IDX_C2H2II]*y[IDX_SiH4I] + k[906]*y[IDX_HII]*y[IDX_SiH3I] +
        k[1075]*y[IDX_H3II]*y[IDX_SiHI] + k[1095]*y[IDX_H3OII]*y[IDX_SiHI] +
        k[1155]*y[IDX_HCOII]*y[IDX_SiHI] + k[1281]*y[IDX_HeII]*y[IDX_SiH3I] -
        k[1514]*y[IDX_OI]*y[IDX_SiH2II] + k[1536]*y[IDX_OHII]*y[IDX_SiHI] -
        k[1567]*y[IDX_SiH2II]*y[IDX_O2I] - k[1568]*y[IDX_SiH2II]*y[IDX_SI] +
        k[2083]*y[IDX_SiH2I] + k[2118]*y[IDX_H2I]*y[IDX_SiII] -
        k[2176]*y[IDX_SiH2II];
    ydot[IDX_SO2I] = 0.0 - k[141]*y[IDX_HII]*y[IDX_SO2I] -
        k[218]*y[IDX_HeII]*y[IDX_SO2I] - k[320]*y[IDX_OII]*y[IDX_SO2I] +
        k[325]*y[IDX_O2I]*y[IDX_SO2II] - k[445]*y[IDX_SO2I] +
        k[558]*y[IDX_HSO2II]*y[IDX_EM] - k[638]*y[IDX_CII]*y[IDX_SO2I] -
        k[873]*y[IDX_COII]*y[IDX_SO2I] - k[989]*y[IDX_H2OII]*y[IDX_SO2I] +
        k[1008]*y[IDX_H2OI]*y[IDX_HSO2II] - k[1069]*y[IDX_H3II]*y[IDX_SO2I] -
        k[1270]*y[IDX_HeII]*y[IDX_SO2I] - k[1271]*y[IDX_HeII]*y[IDX_SO2I] +
        k[1438]*y[IDX_NH3I]*y[IDX_HSO2II] - k[1481]*y[IDX_OII]*y[IDX_SO2I] -
        k[1614]*y[IDX_CI]*y[IDX_SO2I] + k[1866]*y[IDX_O2I]*y[IDX_SOI] -
        k[1916]*y[IDX_OI]*y[IDX_SO2I] + k[1949]*y[IDX_OHI]*y[IDX_SOI] -
        k[1954]*y[IDX_SI]*y[IDX_SO2I] - k[2074]*y[IDX_SO2I] +
        k[2129]*y[IDX_OI]*y[IDX_SOI] - k[2260]*y[IDX_SO2I] +
        k[2499]*y[IDX_GSO2I] + k[2500]*y[IDX_GSO2I] + k[2501]*y[IDX_GSO2I] +
        k[2502]*y[IDX_GSO2I];
    ydot[IDX_HNOI] = 0.0 + k[300]*y[IDX_NOI]*y[IDX_HNOII] -
        k[416]*y[IDX_HNOI] + k[510]*y[IDX_H2NOII]*y[IDX_EM] -
        k[901]*y[IDX_HII]*y[IDX_HNOI] - k[1051]*y[IDX_H3II]*y[IDX_HNOI] -
        k[1245]*y[IDX_HeII]*y[IDX_HNOI] - k[1246]*y[IDX_HeII]*y[IDX_HNOI] -
        k[1626]*y[IDX_CH2I]*y[IDX_HNOI] - k[1655]*y[IDX_CH3I]*y[IDX_HNOI] +
        k[1658]*y[IDX_CH3I]*y[IDX_NO2I] - k[1680]*y[IDX_CHI]*y[IDX_HNOI] -
        k[1708]*y[IDX_CNI]*y[IDX_HNOI] - k[1716]*y[IDX_COI]*y[IDX_HNOI] -
        k[1755]*y[IDX_HI]*y[IDX_HNOI] - k[1756]*y[IDX_HI]*y[IDX_HNOI] -
        k[1757]*y[IDX_HI]*y[IDX_HNOI] - k[1782]*y[IDX_HCOI]*y[IDX_HNOI] +
        k[1783]*y[IDX_HCOI]*y[IDX_NOI] - k[1815]*y[IDX_NI]*y[IDX_HNOI] +
        k[1846]*y[IDX_NHI]*y[IDX_NO2I] + k[1849]*y[IDX_NHI]*y[IDX_O2I] +
        k[1854]*y[IDX_NHI]*y[IDX_OHI] - k[1896]*y[IDX_OI]*y[IDX_HNOI] -
        k[1897]*y[IDX_OI]*y[IDX_HNOI] - k[1898]*y[IDX_OI]*y[IDX_HNOI] +
        k[1902]*y[IDX_OI]*y[IDX_NH2I] - k[1942]*y[IDX_OHI]*y[IDX_HNOI] -
        k[2038]*y[IDX_HNOI] - k[2271]*y[IDX_HNOI] + k[2371]*y[IDX_GHNOI] +
        k[2372]*y[IDX_GHNOI] + k[2373]*y[IDX_GHNOI] + k[2374]*y[IDX_GHNOI];
    ydot[IDX_SiHI] = 0.0 - k[150]*y[IDX_HII]*y[IDX_SiHI] -
        k[355]*y[IDX_SiHI]*y[IDX_SII] + k[452]*y[IDX_SiH2I] - k[455]*y[IDX_SiHI]
        + k[595]*y[IDX_SiH2II]*y[IDX_EM] + k[597]*y[IDX_SiH3II]*y[IDX_EM] -
        k[644]*y[IDX_CII]*y[IDX_SiHI] - k[908]*y[IDX_HII]*y[IDX_SiHI] -
        k[1075]*y[IDX_H3II]*y[IDX_SiHI] - k[1095]*y[IDX_H3OII]*y[IDX_SiHI] -
        k[1155]*y[IDX_HCOII]*y[IDX_SiHI] - k[1284]*y[IDX_HeII]*y[IDX_SiHI] -
        k[1536]*y[IDX_OHII]*y[IDX_SiHI] - k[1569]*y[IDX_SiHI]*y[IDX_SII] -
        k[1617]*y[IDX_CI]*y[IDX_SiHI] - k[1926]*y[IDX_OI]*y[IDX_SiHI] +
        k[2084]*y[IDX_SiH2I] + k[2087]*y[IDX_SiH3I] + k[2090]*y[IDX_SiH4I] -
        k[2091]*y[IDX_SiHI] - k[2172]*y[IDX_SiHI];
    ydot[IDX_SiOII] = 0.0 + k[151]*y[IDX_HII]*y[IDX_SiOI] -
        k[206]*y[IDX_HCOI]*y[IDX_SiOII] - k[232]*y[IDX_MgI]*y[IDX_SiOII] -
        k[305]*y[IDX_NOI]*y[IDX_SiOII] - k[602]*y[IDX_SiOII]*y[IDX_EM] -
        k[659]*y[IDX_C2I]*y[IDX_SiOII] - k[704]*y[IDX_CI]*y[IDX_SiOII] -
        k[773]*y[IDX_CH2I]*y[IDX_SiOII] - k[864]*y[IDX_CHI]*y[IDX_SiOII] -
        k[881]*y[IDX_COI]*y[IDX_SiOII] - k[964]*y[IDX_H2I]*y[IDX_SiOII] -
        k[1343]*y[IDX_NI]*y[IDX_SiOII] - k[1344]*y[IDX_NI]*y[IDX_SiOII] +
        k[1488]*y[IDX_O2I]*y[IDX_SiSII] + k[1512]*y[IDX_OI]*y[IDX_SiCII] +
        k[1513]*y[IDX_OI]*y[IDX_SiHII] - k[1516]*y[IDX_OI]*y[IDX_SiOII] +
        k[1549]*y[IDX_OHI]*y[IDX_SiII] - k[1555]*y[IDX_SI]*y[IDX_SiOII] -
        k[2092]*y[IDX_SiOII] + k[2094]*y[IDX_SiOI] +
        k[2130]*y[IDX_OI]*y[IDX_SiII] - k[2187]*y[IDX_SiOII];
    ydot[IDX_SiOHII] = 0.0 - k[603]*y[IDX_SiOHII]*y[IDX_EM] -
        k[604]*y[IDX_SiOHII]*y[IDX_EM] + k[896]*y[IDX_HII]*y[IDX_H2SiOI] +
        k[964]*y[IDX_H2I]*y[IDX_SiOII] + k[1009]*y[IDX_H2OI]*y[IDX_HSiSII] +
        k[1013]*y[IDX_H2OI]*y[IDX_SiII] + k[1076]*y[IDX_H3II]*y[IDX_SiOI] +
        k[1096]*y[IDX_H3OII]*y[IDX_SiOI] + k[1156]*y[IDX_HCOII]*y[IDX_SiOI] +
        k[1228]*y[IDX_HeII]*y[IDX_H2SiOI] - k[1443]*y[IDX_NH3I]*y[IDX_SiOHII] +
        k[1514]*y[IDX_OI]*y[IDX_SiH2II] + k[1515]*y[IDX_OI]*y[IDX_SiH3II] +
        k[1537]*y[IDX_OHII]*y[IDX_SiOI] + k[1563]*y[IDX_SiII]*y[IDX_C2H5OHI] +
        k[1564]*y[IDX_SiII]*y[IDX_CH3OHI] + k[1567]*y[IDX_SiH2II]*y[IDX_O2I] -
        k[2188]*y[IDX_SiOHII];
    ydot[IDX_SiHII] = 0.0 + k[150]*y[IDX_HII]*y[IDX_SiHI] +
        k[355]*y[IDX_SiHI]*y[IDX_SII] - k[592]*y[IDX_SiHII]*y[IDX_EM] +
        k[672]*y[IDX_C2H2II]*y[IDX_SiH4I] - k[703]*y[IDX_CI]*y[IDX_SiHII] -
        k[863]*y[IDX_CHI]*y[IDX_SiHII] + k[905]*y[IDX_HII]*y[IDX_SiH2I] -
        k[1014]*y[IDX_H2OI]*y[IDX_SiHII] + k[1071]*y[IDX_H3II]*y[IDX_SiI] +
        k[1093]*y[IDX_H3OII]*y[IDX_SiI] - k[1108]*y[IDX_HI]*y[IDX_SiHII] +
        k[1279]*y[IDX_HeII]*y[IDX_SiH2I] + k[1280]*y[IDX_HeII]*y[IDX_SiH3I] +
        k[1283]*y[IDX_HeII]*y[IDX_SiH4I] - k[1442]*y[IDX_NH3I]*y[IDX_SiHII] -
        k[1513]*y[IDX_OI]*y[IDX_SiHII] + k[1535]*y[IDX_OHII]*y[IDX_SiI] +
        k[1566]*y[IDX_SiI]*y[IDX_HCOII] - k[2082]*y[IDX_SiHII] -
        k[2119]*y[IDX_H2I]*y[IDX_SiHII] + k[2126]*y[IDX_HI]*y[IDX_SiII] -
        k[2174]*y[IDX_SiHII];
    ydot[IDX_SiH4I] = 0.0 - k[149]*y[IDX_HII]*y[IDX_SiH4I] -
        k[454]*y[IDX_SiH4I] + k[601]*y[IDX_SiH5II]*y[IDX_EM] -
        k[671]*y[IDX_C2H2II]*y[IDX_SiH4I] - k[672]*y[IDX_C2H2II]*y[IDX_SiH4I] -
        k[673]*y[IDX_C2H2II]*y[IDX_SiH4I] - k[674]*y[IDX_C2H2II]*y[IDX_SiH4I] -
        k[789]*y[IDX_CH3II]*y[IDX_SiH4I] - k[836]*y[IDX_CH5II]*y[IDX_SiH4I] -
        k[907]*y[IDX_HII]*y[IDX_SiH4I] + k[1016]*y[IDX_H2OI]*y[IDX_SiH5II] -
        k[1074]*y[IDX_H3II]*y[IDX_SiH4I] - k[1154]*y[IDX_HCOII]*y[IDX_SiH4I] -
        k[1282]*y[IDX_HeII]*y[IDX_SiH4I] - k[1283]*y[IDX_HeII]*y[IDX_SiH4I] -
        k[1715]*y[IDX_CNI]*y[IDX_SiH4I] - k[1925]*y[IDX_OI]*y[IDX_SiH4I] -
        k[2088]*y[IDX_SiH4I] - k[2089]*y[IDX_SiH4I] - k[2090]*y[IDX_SiH4I] -
        k[2179]*y[IDX_SiH4I] + k[2383]*y[IDX_GSiH4I] + k[2384]*y[IDX_GSiH4I] +
        k[2385]*y[IDX_GSiH4I] + k[2386]*y[IDX_GSiH4I];
    ydot[IDX_C2H3I] = 0.0 - k[369]*y[IDX_C2H3I] + k[371]*y[IDX_C2H5I] +
        k[606]*y[IDX_CII]*y[IDX_C2H5OHI] + k[674]*y[IDX_C2H2II]*y[IDX_SiH4I] -
        k[882]*y[IDX_HII]*y[IDX_C2H3I] - k[1181]*y[IDX_HeII]*y[IDX_C2H3I] -
        k[1182]*y[IDX_HeII]*y[IDX_C2H3I] - k[1576]*y[IDX_C2H3I]*y[IDX_O2I] -
        k[1577]*y[IDX_C2H3I]*y[IDX_O2I] - k[1582]*y[IDX_CI]*y[IDX_C2H3I] +
        k[1620]*y[IDX_CH2I]*y[IDX_CH2I] - k[1646]*y[IDX_CH3I]*y[IDX_C2H3I] +
        k[1702]*y[IDX_CNI]*y[IDX_C2H4I] - k[1738]*y[IDX_HI]*y[IDX_C2H3I] -
        k[1791]*y[IDX_NI]*y[IDX_C2H3I] - k[1869]*y[IDX_OI]*y[IDX_C2H3I] +
        k[1870]*y[IDX_OI]*y[IDX_C2H4I] - k[1930]*y[IDX_OHI]*y[IDX_C2H3I] -
        k[1966]*y[IDX_C2H3I] + k[1968]*y[IDX_C2H5I] - k[2146]*y[IDX_C2H3I] +
        k[2339]*y[IDX_GC2H3I] + k[2340]*y[IDX_GC2H3I] + k[2341]*y[IDX_GC2H3I] +
        k[2342]*y[IDX_GC2H3I];
    ydot[IDX_C2NII] = 0.0 + k[20]*y[IDX_CII]*y[IDX_NCCNI] +
        k[113]*y[IDX_HII]*y[IDX_C2NI] - k[468]*y[IDX_C2NII]*y[IDX_EM] -
        k[469]*y[IDX_C2NII]*y[IDX_EM] + k[623]*y[IDX_CII]*y[IDX_HC3NI] +
        k[627]*y[IDX_CII]*y[IDX_HNCI] + k[713]*y[IDX_CHII]*y[IDX_CNI] +
        k[723]*y[IDX_CHII]*y[IDX_HCNI] - k[992]*y[IDX_H2OI]*y[IDX_C2NII] -
        k[993]*y[IDX_H2OI]*y[IDX_C2NII] - k[1018]*y[IDX_H2SI]*y[IDX_C2NII] +
        k[1229]*y[IDX_HeII]*y[IDX_HC3NI] + k[1304]*y[IDX_NII]*y[IDX_NCCNI] +
        k[1326]*y[IDX_NI]*y[IDX_C2HII] + k[1328]*y[IDX_NI]*y[IDX_C2H2II] +
        k[1346]*y[IDX_NHII]*y[IDX_C2I] - k[1421]*y[IDX_NH3I]*y[IDX_C2NII] +
        k[1445]*y[IDX_NHI]*y[IDX_C2II] - k[2160]*y[IDX_C2NII];
    ydot[IDX_HCO2II] = 0.0 - k[543]*y[IDX_HCO2II]*y[IDX_EM] -
        k[544]*y[IDX_HCO2II]*y[IDX_EM] - k[545]*y[IDX_HCO2II]*y[IDX_EM] -
        k[695]*y[IDX_CI]*y[IDX_HCO2II] - k[791]*y[IDX_CH3CNI]*y[IDX_HCO2II] +
        k[796]*y[IDX_CH4II]*y[IDX_CO2I] - k[812]*y[IDX_CH4I]*y[IDX_HCO2II] +
        k[825]*y[IDX_CH5II]*y[IDX_CO2I] - k[875]*y[IDX_COI]*y[IDX_HCO2II] +
        k[918]*y[IDX_H2II]*y[IDX_CO2I] - k[1004]*y[IDX_H2OI]*y[IDX_HCO2II] +
        k[1036]*y[IDX_H3II]*y[IDX_CO2I] + k[1110]*y[IDX_HCNII]*y[IDX_CO2I] +
        k[1173]*y[IDX_HNOII]*y[IDX_CO2I] + k[1238]*y[IDX_HeII]*y[IDX_HCOOCH3I] +
        k[1322]*y[IDX_N2HII]*y[IDX_CO2I] + k[1350]*y[IDX_NHII]*y[IDX_CO2I] -
        k[1434]*y[IDX_NH3I]*y[IDX_HCO2II] + k[1489]*y[IDX_O2HII]*y[IDX_CO2I] -
        k[1500]*y[IDX_OI]*y[IDX_HCO2II] + k[1520]*y[IDX_OHII]*y[IDX_CO2I] +
        k[1543]*y[IDX_OHI]*y[IDX_HCOII] - k[2247]*y[IDX_HCO2II];
    ydot[IDX_CSII] = 0.0 + k[118]*y[IDX_HII]*y[IDX_CSI] -
        k[221]*y[IDX_MgI]*y[IDX_CSII] - k[348]*y[IDX_SiI]*y[IDX_CSII] +
        k[395]*y[IDX_CSI] - k[500]*y[IDX_CSII]*y[IDX_EM] +
        k[628]*y[IDX_CII]*y[IDX_HSI] + k[632]*y[IDX_CII]*y[IDX_NSI] +
        k[636]*y[IDX_CII]*y[IDX_OCSI] + k[639]*y[IDX_CII]*y[IDX_SOI] +
        k[650]*y[IDX_C2II]*y[IDX_SI] + k[658]*y[IDX_C2I]*y[IDX_SII] +
        k[697]*y[IDX_CI]*y[IDX_HSII] + k[739]*y[IDX_CHII]*y[IDX_SI] -
        k[808]*y[IDX_CH4I]*y[IDX_CSII] + k[861]*y[IDX_CHI]*y[IDX_SII] +
        k[899]*y[IDX_HII]*y[IDX_HCSI] - k[943]*y[IDX_H2I]*y[IDX_CSII] +
        k[1219]*y[IDX_HeII]*y[IDX_H2CSI] + k[1239]*y[IDX_HeII]*y[IDX_HCSI] +
        k[1264]*y[IDX_HeII]*y[IDX_OCSI] + k[1312]*y[IDX_NII]*y[IDX_OCSI] -
        k[1485]*y[IDX_O2I]*y[IDX_CSII] - k[1496]*y[IDX_OI]*y[IDX_CSII] -
        k[2005]*y[IDX_CSII] + k[2006]*y[IDX_CSI] + k[2099]*y[IDX_CII]*y[IDX_SI]
        + k[2105]*y[IDX_CI]*y[IDX_SII] - k[2266]*y[IDX_CSII];
    ydot[IDX_HCSII] = 0.0 + k[412]*y[IDX_HCSI] -
        k[546]*y[IDX_HCSII]*y[IDX_EM] - k[547]*y[IDX_HCSII]*y[IDX_EM] +
        k[622]*y[IDX_CII]*y[IDX_H2SI] + k[676]*y[IDX_C2H4I]*y[IDX_SII] +
        k[691]*y[IDX_CI]*y[IDX_H2SII] + k[722]*y[IDX_CHII]*y[IDX_H2SI] +
        k[736]*y[IDX_CHII]*y[IDX_OCSI] + k[746]*y[IDX_CH2II]*y[IDX_H2SI] +
        k[752]*y[IDX_CH2II]*y[IDX_OCSI] + k[753]*y[IDX_CH2II]*y[IDX_SI] +
        k[772]*y[IDX_CH2I]*y[IDX_SII] + k[787]*y[IDX_CH3II]*y[IDX_SI] +
        k[808]*y[IDX_CH4I]*y[IDX_CSII] + k[822]*y[IDX_CH4I]*y[IDX_SII] +
        k[943]*y[IDX_H2I]*y[IDX_CSII] + k[1018]*y[IDX_H2SI]*y[IDX_C2NII] +
        k[1039]*y[IDX_H3II]*y[IDX_CSI] + k[1085]*y[IDX_H3OII]*y[IDX_CSI] +
        k[1140]*y[IDX_HCOII]*y[IDX_CSI] - k[1164]*y[IDX_HCSII]*y[IDX_C2H5OHI] -
        k[1435]*y[IDX_NH3I]*y[IDX_HCSII] - k[1501]*y[IDX_OI]*y[IDX_HCSII] -
        k[1502]*y[IDX_OI]*y[IDX_HCSII] + k[1558]*y[IDX_SOII]*y[IDX_C2H2I] +
        k[2033]*y[IDX_HCSI] - k[2268]*y[IDX_HCSII];
    ydot[IDX_SiOI] = 0.0 - k[151]*y[IDX_HII]*y[IDX_SiOI] +
        k[206]*y[IDX_HCOI]*y[IDX_SiOII] + k[232]*y[IDX_MgI]*y[IDX_SiOII] +
        k[305]*y[IDX_NOI]*y[IDX_SiOII] + k[405]*y[IDX_H2SiOI] -
        k[456]*y[IDX_SiOI] + k[604]*y[IDX_SiOHII]*y[IDX_EM] -
        k[645]*y[IDX_CII]*y[IDX_SiOI] - k[1076]*y[IDX_H3II]*y[IDX_SiOI] -
        k[1096]*y[IDX_H3OII]*y[IDX_SiOI] - k[1156]*y[IDX_HCOII]*y[IDX_SiOI] -
        k[1285]*y[IDX_HeII]*y[IDX_SiOI] - k[1286]*y[IDX_HeII]*y[IDX_SiOI] +
        k[1443]*y[IDX_NH3I]*y[IDX_SiOHII] + k[1487]*y[IDX_O2I]*y[IDX_SiSII] -
        k[1537]*y[IDX_OHII]*y[IDX_SiOI] + k[1921]*y[IDX_OI]*y[IDX_SiCI] +
        k[1922]*y[IDX_OI]*y[IDX_SiH2I] + k[1923]*y[IDX_OI]*y[IDX_SiH2I] +
        k[1926]*y[IDX_OI]*y[IDX_SiHI] + k[1950]*y[IDX_OHI]*y[IDX_SiI] +
        k[1956]*y[IDX_SiI]*y[IDX_CO2I] + k[1957]*y[IDX_SiI]*y[IDX_COI] +
        k[1958]*y[IDX_SiI]*y[IDX_NOI] + k[1959]*y[IDX_SiI]*y[IDX_O2I] +
        k[2023]*y[IDX_H2SiOI] + k[2024]*y[IDX_H2SiOI] - k[2093]*y[IDX_SiOI] -
        k[2094]*y[IDX_SiOI] + k[2131]*y[IDX_OI]*y[IDX_SiI] - k[2171]*y[IDX_SiOI]
        + k[2427]*y[IDX_GSiOI] + k[2428]*y[IDX_GSiOI] + k[2429]*y[IDX_GSiOI] +
        k[2430]*y[IDX_GSiOI];
    ydot[IDX_CH3OHI] = 0.0 - k[388]*y[IDX_CH3OHI] - k[389]*y[IDX_CH3OHI] +
        k[489]*y[IDX_CH3OH2II]*y[IDX_EM] + k[536]*y[IDX_H5C2O2II]*y[IDX_EM] -
        k[612]*y[IDX_CII]*y[IDX_CH3OHI] - k[613]*y[IDX_CII]*y[IDX_CH3OHI] -
        k[708]*y[IDX_CHII]*y[IDX_CH3OHI] - k[709]*y[IDX_CHII]*y[IDX_CH3OHI] -
        k[710]*y[IDX_CHII]*y[IDX_CH3OHI] - k[776]*y[IDX_CH3II]*y[IDX_CH3OHI] +
        k[792]*y[IDX_CH3OH2II]*y[IDX_NH3I] - k[793]*y[IDX_CH3OHI]*y[IDX_S2II] -
        k[794]*y[IDX_CH4II]*y[IDX_CH3OHI] - k[887]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[888]*y[IDX_HII]*y[IDX_CH3OHI] - k[889]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[965]*y[IDX_H2COII]*y[IDX_CH3OHI] - k[1031]*y[IDX_H3II]*y[IDX_CH3OHI] -
        k[1032]*y[IDX_H3II]*y[IDX_CH3OHI] - k[1078]*y[IDX_H3COII]*y[IDX_CH3OHI]
        - k[1084]*y[IDX_H3OII]*y[IDX_CH3OHI] -
        k[1139]*y[IDX_HCOII]*y[IDX_CH3OHI] - k[1200]*y[IDX_HeII]*y[IDX_CH3OHI] -
        k[1201]*y[IDX_HeII]*y[IDX_CH3OHI] - k[1289]*y[IDX_NII]*y[IDX_CH3OHI] -
        k[1290]*y[IDX_NII]*y[IDX_CH3OHI] - k[1291]*y[IDX_NII]*y[IDX_CH3OHI] -
        k[1292]*y[IDX_NII]*y[IDX_CH3OHI] - k[1466]*y[IDX_OII]*y[IDX_CH3OHI] -
        k[1467]*y[IDX_OII]*y[IDX_CH3OHI] - k[1483]*y[IDX_O2II]*y[IDX_CH3OHI] -
        k[1564]*y[IDX_SiII]*y[IDX_CH3OHI] - k[1990]*y[IDX_CH3OHI] -
        k[1991]*y[IDX_CH3OHI] - k[1992]*y[IDX_CH3OHI] - k[2154]*y[IDX_CH3OHI] +
        k[2379]*y[IDX_GCH3OHI] + k[2380]*y[IDX_GCH3OHI] + k[2381]*y[IDX_GCH3OHI]
        + k[2382]*y[IDX_GCH3OHI];
    ydot[IDX_H3SII] = 0.0 - k[532]*y[IDX_H3SII]*y[IDX_EM] -
        k[533]*y[IDX_H3SII]*y[IDX_EM] - k[534]*y[IDX_H3SII]*y[IDX_EM] -
        k[535]*y[IDX_H3SII]*y[IDX_EM] + k[667]*y[IDX_C2H2II]*y[IDX_H2SI] +
        k[721]*y[IDX_CHII]*y[IDX_H2SI] + k[745]*y[IDX_CH2II]*y[IDX_H2SI] +
        k[800]*y[IDX_CH4II]*y[IDX_H2SI] + k[829]*y[IDX_CH5II]*y[IDX_H2SI] +
        k[946]*y[IDX_H2I]*y[IDX_H2SII] - k[970]*y[IDX_H2COI]*y[IDX_H3SII] +
        k[981]*y[IDX_H2OII]*y[IDX_H2SI] + k[1017]*y[IDX_H2SII]*y[IDX_H2SI] +
        k[1019]*y[IDX_H2SI]*y[IDX_HS2II] + k[1020]*y[IDX_H2SI]*y[IDX_HSiSII] +
        k[1044]*y[IDX_H3II]*y[IDX_H2SI] + k[1079]*y[IDX_H3COII]*y[IDX_H2SI] +
        k[1087]*y[IDX_H3OII]*y[IDX_H2SI] - k[1104]*y[IDX_HI]*y[IDX_H3SII] -
        k[1123]*y[IDX_HCNI]*y[IDX_H3SII] + k[1134]*y[IDX_HCNHII]*y[IDX_H2SI] +
        k[1135]*y[IDX_HCNHII]*y[IDX_H2SI] + k[1143]*y[IDX_HCOII]*y[IDX_H2SI] -
        k[1167]*y[IDX_HNCI]*y[IDX_H3SII] + k[1175]*y[IDX_HSII]*y[IDX_H2SI] +
        k[1381]*y[IDX_NH2II]*y[IDX_H2SI] - k[1429]*y[IDX_NH3I]*y[IDX_H3SII] +
        k[1524]*y[IDX_OHII]*y[IDX_H2SI] - k[1553]*y[IDX_SI]*y[IDX_H3SII] +
        k[2116]*y[IDX_H2I]*y[IDX_HSII] - k[2278]*y[IDX_H3SII];
    ydot[IDX_SOI] = 0.0 - k[25]*y[IDX_CII]*y[IDX_SOI] -
        k[142]*y[IDX_HII]*y[IDX_SOI] + k[230]*y[IDX_MgI]*y[IDX_SOII] +
        k[293]*y[IDX_NH3I]*y[IDX_SOII] + k[445]*y[IDX_SO2I] - k[446]*y[IDX_SOI]
        - k[447]*y[IDX_SOI] + k[557]*y[IDX_HSOII]*y[IDX_EM] +
        k[559]*y[IDX_HSO2II]*y[IDX_EM] + k[560]*y[IDX_HSO2II]*y[IDX_EM] +
        k[579]*y[IDX_OCSII]*y[IDX_EM] + k[586]*y[IDX_SO2II]*y[IDX_EM] -
        k[639]*y[IDX_CII]*y[IDX_SOI] - k[640]*y[IDX_CII]*y[IDX_SOI] -
        k[641]*y[IDX_CII]*y[IDX_SOI] - k[788]*y[IDX_CH3II]*y[IDX_SOI] -
        k[1070]*y[IDX_H3II]*y[IDX_SOI] - k[1152]*y[IDX_HCOII]*y[IDX_SOI] -
        k[1272]*y[IDX_HeII]*y[IDX_SOI] - k[1273]*y[IDX_HeII]*y[IDX_SOI] +
        k[1488]*y[IDX_O2I]*y[IDX_SiSII] + k[1555]*y[IDX_SI]*y[IDX_SiOII] +
        k[1614]*y[IDX_CI]*y[IDX_SO2I] - k[1615]*y[IDX_CI]*y[IDX_SOI] -
        k[1616]*y[IDX_CI]*y[IDX_SOI] - k[1699]*y[IDX_CHI]*y[IDX_SOI] -
        k[1700]*y[IDX_CHI]*y[IDX_SOI] - k[1778]*y[IDX_HI]*y[IDX_SOI] -
        k[1779]*y[IDX_HI]*y[IDX_SOI] - k[1830]*y[IDX_NI]*y[IDX_SOI] -
        k[1831]*y[IDX_NI]*y[IDX_SOI] + k[1862]*y[IDX_NOI]*y[IDX_SI] +
        k[1865]*y[IDX_O2I]*y[IDX_SI] - k[1866]*y[IDX_O2I]*y[IDX_SOI] +
        k[1884]*y[IDX_OI]*y[IDX_CSI] + k[1900]*y[IDX_OI]*y[IDX_HSI] +
        k[1908]*y[IDX_OI]*y[IDX_NSI] + k[1913]*y[IDX_OI]*y[IDX_OCSI] +
        k[1915]*y[IDX_OI]*y[IDX_S2I] + k[1916]*y[IDX_OI]*y[IDX_SO2I] -
        k[1917]*y[IDX_OI]*y[IDX_SOI] + k[1948]*y[IDX_OHI]*y[IDX_SI] -
        k[1949]*y[IDX_OHI]*y[IDX_SOI] + k[1954]*y[IDX_SI]*y[IDX_SO2I] +
        k[1954]*y[IDX_SI]*y[IDX_SO2I] - k[1955]*y[IDX_SI]*y[IDX_SOI] +
        k[2074]*y[IDX_SO2I] - k[2075]*y[IDX_SOI] - k[2076]*y[IDX_SOI] -
        k[2129]*y[IDX_OI]*y[IDX_SOI] - k[2256]*y[IDX_SOI] + k[2459]*y[IDX_GSOI]
        + k[2460]*y[IDX_GSOI] + k[2461]*y[IDX_GSOI] + k[2462]*y[IDX_GSOI];
    ydot[IDX_C2H4I] = 0.0 - k[370]*y[IDX_C2H4I] +
        k[464]*y[IDX_C2H5OH2II]*y[IDX_EM] + k[671]*y[IDX_C2H2II]*y[IDX_SiH4I] +
        k[673]*y[IDX_C2H2II]*y[IDX_SiH4I] - k[676]*y[IDX_C2H4I]*y[IDX_SII] -
        k[774]*y[IDX_CH3II]*y[IDX_C2H4I] + k[775]*y[IDX_CH3II]*y[IDX_CH3CNI] -
        k[883]*y[IDX_HII]*y[IDX_C2H4I] - k[910]*y[IDX_H2II]*y[IDX_C2H4I] -
        k[1183]*y[IDX_HeII]*y[IDX_C2H4I] - k[1184]*y[IDX_HeII]*y[IDX_C2H4I] -
        k[1185]*y[IDX_HeII]*y[IDX_C2H4I] - k[1464]*y[IDX_OII]*y[IDX_C2H4I] -
        k[1559]*y[IDX_SOII]*y[IDX_C2H4I] - k[1560]*y[IDX_SOII]*y[IDX_C2H4I] -
        k[1561]*y[IDX_SOII]*y[IDX_C2H4I] + k[1647]*y[IDX_CH3I]*y[IDX_CH3I] -
        k[1675]*y[IDX_CHI]*y[IDX_C2H4I] + k[1676]*y[IDX_CHI]*y[IDX_CH4I] -
        k[1702]*y[IDX_CNI]*y[IDX_C2H4I] - k[1792]*y[IDX_NI]*y[IDX_C2H4I] +
        k[1793]*y[IDX_NI]*y[IDX_C2H5I] - k[1870]*y[IDX_OI]*y[IDX_C2H4I] -
        k[1871]*y[IDX_OI]*y[IDX_C2H4I] - k[1872]*y[IDX_OI]*y[IDX_C2H4I] -
        k[1873]*y[IDX_OI]*y[IDX_C2H4I] + k[1931]*y[IDX_OHI]*y[IDX_C2H5I] -
        k[1967]*y[IDX_C2H4I] - k[2121]*y[IDX_H3OII]*y[IDX_C2H4I] -
        k[2300]*y[IDX_C2H4I] + k[2351]*y[IDX_GC2H4I] + k[2352]*y[IDX_GC2H4I] +
        k[2353]*y[IDX_GC2H4I] + k[2354]*y[IDX_GC2H4I];
    ydot[IDX_CH4II] = 0.0 - k[75]*y[IDX_CH4II]*y[IDX_C2H2I] -
        k[76]*y[IDX_CH4II]*y[IDX_H2COI] - k[77]*y[IDX_CH4II]*y[IDX_H2SI] -
        k[78]*y[IDX_CH4II]*y[IDX_NH3I] - k[79]*y[IDX_CH4II]*y[IDX_O2I] -
        k[80]*y[IDX_CH4II]*y[IDX_OCSI] + k[81]*y[IDX_CH4I]*y[IDX_COII] +
        k[116]*y[IDX_HII]*y[IDX_CH4I] + k[157]*y[IDX_H2II]*y[IDX_CH4I] +
        k[210]*y[IDX_HeII]*y[IDX_CH4I] + k[236]*y[IDX_NII]*y[IDX_CH4I] +
        k[309]*y[IDX_OII]*y[IDX_CH4I] - k[491]*y[IDX_CH4II]*y[IDX_EM] -
        k[492]*y[IDX_CH4II]*y[IDX_EM] + k[779]*y[IDX_CH3II]*y[IDX_HCOI] -
        k[794]*y[IDX_CH4II]*y[IDX_CH3OHI] - k[795]*y[IDX_CH4II]*y[IDX_CH4I] -
        k[796]*y[IDX_CH4II]*y[IDX_CO2I] - k[797]*y[IDX_CH4II]*y[IDX_COI] -
        k[798]*y[IDX_CH4II]*y[IDX_H2COI] - k[799]*y[IDX_CH4II]*y[IDX_H2OI] -
        k[800]*y[IDX_CH4II]*y[IDX_H2SI] - k[801]*y[IDX_CH4II]*y[IDX_NH3I] -
        k[802]*y[IDX_CH4II]*y[IDX_OCSI] - k[939]*y[IDX_H2I]*y[IDX_CH4II] +
        k[1029]*y[IDX_H3II]*y[IDX_CH3I] - k[1101]*y[IDX_HI]*y[IDX_CH4II] +
        k[1102]*y[IDX_HI]*y[IDX_CH5II] - k[1493]*y[IDX_OI]*y[IDX_CH4II] -
        k[1993]*y[IDX_CH4II] - k[1994]*y[IDX_CH4II] + k[1997]*y[IDX_CH4I] -
        k[2249]*y[IDX_CH4II];
    ydot[IDX_CSI] = 0.0 - k[118]*y[IDX_HII]*y[IDX_CSI] +
        k[221]*y[IDX_MgI]*y[IDX_CSII] + k[348]*y[IDX_SiI]*y[IDX_CSII] -
        k[395]*y[IDX_CSI] - k[396]*y[IDX_CSI] + k[400]*y[IDX_H2CSI] +
        k[506]*y[IDX_H2CSII]*y[IDX_EM] + k[526]*y[IDX_H3CSII]*y[IDX_EM] +
        k[547]*y[IDX_HCSII]*y[IDX_EM] + k[552]*y[IDX_HOCSII]*y[IDX_EM] +
        k[580]*y[IDX_OCSII]*y[IDX_EM] + k[619]*y[IDX_CII]*y[IDX_H2CSI] -
        k[1039]*y[IDX_H3II]*y[IDX_CSI] - k[1085]*y[IDX_H3OII]*y[IDX_CSI] -
        k[1140]*y[IDX_HCOII]*y[IDX_CSI] + k[1164]*y[IDX_HCSII]*y[IDX_C2H5OHI] -
        k[1214]*y[IDX_HeII]*y[IDX_CSI] - k[1215]*y[IDX_HeII]*y[IDX_CSI] +
        k[1240]*y[IDX_HeII]*y[IDX_HCSI] + k[1265]*y[IDX_HeII]*y[IDX_OCSI] +
        k[1435]*y[IDX_NH3I]*y[IDX_HCSII] + k[1573]*y[IDX_C2I]*y[IDX_SI] -
        k[1592]*y[IDX_CI]*y[IDX_CSI] + k[1595]*y[IDX_CI]*y[IDX_HSI] +
        k[1606]*y[IDX_CI]*y[IDX_NSI] + k[1610]*y[IDX_CI]*y[IDX_OCSI] +
        k[1613]*y[IDX_CI]*y[IDX_S2I] + k[1615]*y[IDX_CI]*y[IDX_SOI] +
        k[1644]*y[IDX_CH2I]*y[IDX_SI] + k[1695]*y[IDX_CHI]*y[IDX_OCSI] +
        k[1697]*y[IDX_CHI]*y[IDX_SI] + k[1753]*y[IDX_HI]*y[IDX_HCSI] -
        k[1809]*y[IDX_NI]*y[IDX_CSI] - k[1883]*y[IDX_OI]*y[IDX_CSI] -
        k[1884]*y[IDX_OI]*y[IDX_CSI] - k[1935]*y[IDX_OHI]*y[IDX_CSI] -
        k[1936]*y[IDX_OHI]*y[IDX_CSI] - k[2006]*y[IDX_CSI] - k[2007]*y[IDX_CSI]
        + k[2015]*y[IDX_H2CSI] + k[2106]*y[IDX_CI]*y[IDX_SI] -
        k[2255]*y[IDX_CSI] + k[2431]*y[IDX_GCSI] + k[2432]*y[IDX_GCSI] +
        k[2433]*y[IDX_GCSI] + k[2434]*y[IDX_GCSI];
    ydot[IDX_MgI] = 0.0 - k[19]*y[IDX_CII]*y[IDX_MgI] -
        k[56]*y[IDX_CHII]*y[IDX_MgI] - k[73]*y[IDX_CH3II]*y[IDX_MgI] -
        k[129]*y[IDX_HII]*y[IDX_MgI] - k[181]*y[IDX_H2OII]*y[IDX_MgI] -
        k[220]*y[IDX_MgI]*y[IDX_C2H2II] - k[221]*y[IDX_MgI]*y[IDX_CSII] -
        k[222]*y[IDX_MgI]*y[IDX_H2COII] - k[223]*y[IDX_MgI]*y[IDX_H2SII] -
        k[224]*y[IDX_MgI]*y[IDX_HCOII] - k[225]*y[IDX_MgI]*y[IDX_HSII] -
        k[226]*y[IDX_MgI]*y[IDX_N2II] - k[227]*y[IDX_MgI]*y[IDX_NOII] -
        k[228]*y[IDX_MgI]*y[IDX_O2II] - k[229]*y[IDX_MgI]*y[IDX_SII] -
        k[230]*y[IDX_MgI]*y[IDX_SOII] - k[231]*y[IDX_MgI]*y[IDX_SiII] -
        k[232]*y[IDX_MgI]*y[IDX_SiOII] - k[244]*y[IDX_NII]*y[IDX_MgI] -
        k[279]*y[IDX_NH3II]*y[IDX_MgI] - k[420]*y[IDX_MgI] -
        k[834]*y[IDX_CH5II]*y[IDX_MgI] - k[1054]*y[IDX_H3II]*y[IDX_MgI] -
        k[2044]*y[IDX_MgI] + k[2140]*y[IDX_MgII]*y[IDX_EM] - k[2287]*y[IDX_MgI]
        + k[2319]*y[IDX_GMgI] + k[2320]*y[IDX_GMgI] + k[2321]*y[IDX_GMgI] +
        k[2322]*y[IDX_GMgI];
    ydot[IDX_MgII] = 0.0 + k[19]*y[IDX_CII]*y[IDX_MgI] +
        k[56]*y[IDX_CHII]*y[IDX_MgI] + k[73]*y[IDX_CH3II]*y[IDX_MgI] +
        k[129]*y[IDX_HII]*y[IDX_MgI] + k[181]*y[IDX_H2OII]*y[IDX_MgI] +
        k[220]*y[IDX_MgI]*y[IDX_C2H2II] + k[221]*y[IDX_MgI]*y[IDX_CSII] +
        k[222]*y[IDX_MgI]*y[IDX_H2COII] + k[223]*y[IDX_MgI]*y[IDX_H2SII] +
        k[224]*y[IDX_MgI]*y[IDX_HCOII] + k[225]*y[IDX_MgI]*y[IDX_HSII] +
        k[226]*y[IDX_MgI]*y[IDX_N2II] + k[227]*y[IDX_MgI]*y[IDX_NOII] +
        k[228]*y[IDX_MgI]*y[IDX_O2II] + k[229]*y[IDX_MgI]*y[IDX_SII] +
        k[230]*y[IDX_MgI]*y[IDX_SOII] + k[231]*y[IDX_MgI]*y[IDX_SiII] +
        k[232]*y[IDX_MgI]*y[IDX_SiOII] + k[244]*y[IDX_NII]*y[IDX_MgI] +
        k[279]*y[IDX_NH3II]*y[IDX_MgI] + k[420]*y[IDX_MgI] +
        k[834]*y[IDX_CH5II]*y[IDX_MgI] + k[1054]*y[IDX_H3II]*y[IDX_MgI] +
        k[2044]*y[IDX_MgI] - k[2140]*y[IDX_MgII]*y[IDX_EM] -
        k[2286]*y[IDX_MgII];
    ydot[IDX_SOII] = 0.0 + k[25]*y[IDX_CII]*y[IDX_SOI] +
        k[142]*y[IDX_HII]*y[IDX_SOI] - k[230]*y[IDX_MgI]*y[IDX_SOII] -
        k[293]*y[IDX_NH3I]*y[IDX_SOII] + k[447]*y[IDX_SOI] -
        k[584]*y[IDX_SOII]*y[IDX_EM] + k[638]*y[IDX_CII]*y[IDX_SO2I] +
        k[873]*y[IDX_COII]*y[IDX_SO2I] + k[879]*y[IDX_COI]*y[IDX_SO2II] -
        k[1021]*y[IDX_H2SI]*y[IDX_SOII] + k[1107]*y[IDX_HI]*y[IDX_SO2II] +
        k[1271]*y[IDX_HeII]*y[IDX_SO2I] - k[1341]*y[IDX_NI]*y[IDX_SOII] +
        k[1481]*y[IDX_OII]*y[IDX_SO2I] + k[1484]*y[IDX_O2II]*y[IDX_SI] +
        k[1486]*y[IDX_O2I]*y[IDX_SII] + k[1487]*y[IDX_O2I]*y[IDX_SiSII] +
        k[1499]*y[IDX_OI]*y[IDX_H2SII] + k[1504]*y[IDX_OI]*y[IDX_HSII] +
        k[1534]*y[IDX_OHII]*y[IDX_SI] + k[1548]*y[IDX_OHI]*y[IDX_SII] -
        k[1556]*y[IDX_SOII]*y[IDX_C2H2I] - k[1557]*y[IDX_SOII]*y[IDX_C2H2I] -
        k[1558]*y[IDX_SOII]*y[IDX_C2H2I] - k[1559]*y[IDX_SOII]*y[IDX_C2H4I] -
        k[1560]*y[IDX_SOII]*y[IDX_C2H4I] - k[1561]*y[IDX_SOII]*y[IDX_C2H4I] -
        k[1562]*y[IDX_SOII]*y[IDX_OCSI] + k[2076]*y[IDX_SOI] -
        k[2267]*y[IDX_SOII];
    ydot[IDX_C2II] = 0.0 - k[33]*y[IDX_C2II]*y[IDX_HCOI] -
        k[34]*y[IDX_C2II]*y[IDX_NOI] - k[35]*y[IDX_C2II]*y[IDX_SI] +
        k[36]*y[IDX_C2I]*y[IDX_CNII] + k[37]*y[IDX_C2I]*y[IDX_COII] +
        k[38]*y[IDX_C2I]*y[IDX_N2II] + k[39]*y[IDX_C2I]*y[IDX_O2II] -
        k[50]*y[IDX_CI]*y[IDX_C2II] - k[62]*y[IDX_CH2I]*y[IDX_C2II] -
        k[82]*y[IDX_CHI]*y[IDX_C2II] + k[110]*y[IDX_HII]*y[IDX_C2I] +
        k[153]*y[IDX_H2II]*y[IDX_C2I] + k[175]*y[IDX_H2OII]*y[IDX_C2I] +
        k[207]*y[IDX_HeII]*y[IDX_C2I] + k[233]*y[IDX_NII]*y[IDX_C2I] -
        k[271]*y[IDX_NH2I]*y[IDX_C2II] + k[306]*y[IDX_OII]*y[IDX_C2I] +
        k[329]*y[IDX_OHII]*y[IDX_C2I] - k[339]*y[IDX_OHI]*y[IDX_C2II] -
        k[458]*y[IDX_C2II]*y[IDX_EM] + k[615]*y[IDX_CII]*y[IDX_CHI] -
        k[647]*y[IDX_C2II]*y[IDX_C2I] - k[648]*y[IDX_C2II]*y[IDX_HCOI] -
        k[649]*y[IDX_C2II]*y[IDX_O2I] - k[650]*y[IDX_C2II]*y[IDX_SI] +
        k[686]*y[IDX_CI]*y[IDX_CHII] + k[712]*y[IDX_CHII]*y[IDX_CHI] -
        k[803]*y[IDX_CH4I]*y[IDX_C2II] - k[804]*y[IDX_CH4I]*y[IDX_C2II] -
        k[837]*y[IDX_CHI]*y[IDX_C2II] + k[884]*y[IDX_HII]*y[IDX_C2HI] -
        k[935]*y[IDX_H2I]*y[IDX_C2II] - k[990]*y[IDX_H2OI]*y[IDX_C2II] +
        k[1178]*y[IDX_HeII]*y[IDX_C2H2I] + k[1186]*y[IDX_HeII]*y[IDX_C2HI] +
        k[1190]*y[IDX_HeII]*y[IDX_C3NI] - k[1325]*y[IDX_NI]*y[IDX_C2II] -
        k[1394]*y[IDX_NH2I]*y[IDX_C2II] - k[1444]*y[IDX_NHI]*y[IDX_C2II] -
        k[1445]*y[IDX_NHI]*y[IDX_C2II] - k[1490]*y[IDX_OI]*y[IDX_C2II] -
        k[1960]*y[IDX_C2II] + k[1961]*y[IDX_C2I] + k[1963]*y[IDX_C2HII] +
        k[2096]*y[IDX_CII]*y[IDX_CI] - k[2228]*y[IDX_C2II];
    ydot[IDX_O2HII] = 0.0 - k[578]*y[IDX_O2HII]*y[IDX_EM] -
        k[657]*y[IDX_C2I]*y[IDX_O2HII] - k[683]*y[IDX_C2HI]*y[IDX_O2HII] -
        k[701]*y[IDX_CI]*y[IDX_O2HII] - k[770]*y[IDX_CH2I]*y[IDX_O2HII] -
        k[859]*y[IDX_CHI]*y[IDX_O2HII] - k[870]*y[IDX_CNI]*y[IDX_O2HII] -
        k[878]*y[IDX_COI]*y[IDX_O2HII] + k[931]*y[IDX_H2II]*y[IDX_O2I] -
        k[959]*y[IDX_H2I]*y[IDX_O2HII] - k[973]*y[IDX_H2COI]*y[IDX_O2HII] -
        k[1012]*y[IDX_H2OI]*y[IDX_O2HII] + k[1062]*y[IDX_H3II]*y[IDX_O2I] -
        k[1129]*y[IDX_HCNI]*y[IDX_O2HII] + k[1161]*y[IDX_HCOI]*y[IDX_O2II] -
        k[1162]*y[IDX_HCOI]*y[IDX_O2HII] - k[1172]*y[IDX_HNCI]*y[IDX_O2HII] -
        k[1320]*y[IDX_N2I]*y[IDX_O2HII] + k[1369]*y[IDX_NHII]*y[IDX_O2I] -
        k[1410]*y[IDX_NH2I]*y[IDX_O2HII] - k[1441]*y[IDX_NH3I]*y[IDX_O2HII] -
        k[1459]*y[IDX_NHI]*y[IDX_O2HII] - k[1462]*y[IDX_NOI]*y[IDX_O2HII] -
        k[1489]*y[IDX_O2HII]*y[IDX_CO2I] - k[1510]*y[IDX_OI]*y[IDX_O2HII] -
        k[1547]*y[IDX_OHI]*y[IDX_O2HII] - k[1554]*y[IDX_SI]*y[IDX_O2HII] -
        k[2274]*y[IDX_O2HII];
    ydot[IDX_CNII] = 0.0 - k[36]*y[IDX_C2I]*y[IDX_CNII] -
        k[47]*y[IDX_C2HI]*y[IDX_CNII] - k[51]*y[IDX_CI]*y[IDX_CNII] -
        k[63]*y[IDX_CH2I]*y[IDX_CNII] - k[83]*y[IDX_CHI]*y[IDX_CNII] -
        k[93]*y[IDX_CNII]*y[IDX_COI] - k[94]*y[IDX_CNII]*y[IDX_H2COI] -
        k[95]*y[IDX_CNII]*y[IDX_HCNI] - k[96]*y[IDX_CNII]*y[IDX_HCOI] -
        k[97]*y[IDX_CNII]*y[IDX_NOI] - k[98]*y[IDX_CNII]*y[IDX_O2I] -
        k[99]*y[IDX_CNII]*y[IDX_SI] + k[100]*y[IDX_CNI]*y[IDX_N2II] +
        k[159]*y[IDX_H2II]*y[IDX_CNI] - k[191]*y[IDX_HI]*y[IDX_CNII] +
        k[237]*y[IDX_NII]*y[IDX_CNI] - k[272]*y[IDX_NH2I]*y[IDX_CNII] -
        k[294]*y[IDX_NHI]*y[IDX_CNII] - k[326]*y[IDX_OI]*y[IDX_CNII] -
        k[340]*y[IDX_OHI]*y[IDX_CNII] - k[498]*y[IDX_CNII]*y[IDX_EM] +
        k[631]*y[IDX_CII]*y[IDX_NHI] + k[728]*y[IDX_CHII]*y[IDX_NI] +
        k[731]*y[IDX_CHII]*y[IDX_NHI] + k[852]*y[IDX_CHI]*y[IDX_NII] -
        k[865]*y[IDX_CNII]*y[IDX_H2COI] - k[866]*y[IDX_CNII]*y[IDX_HCNI] -
        k[867]*y[IDX_CNII]*y[IDX_HCOI] - k[868]*y[IDX_CNII]*y[IDX_O2I] -
        k[940]*y[IDX_H2I]*y[IDX_CNII] - k[995]*y[IDX_H2OI]*y[IDX_CNII] -
        k[996]*y[IDX_H2OI]*y[IDX_CNII] + k[1198]*y[IDX_HeII]*y[IDX_CH3CNI] +
        k[1231]*y[IDX_HeII]*y[IDX_HCNI] + k[1242]*y[IDX_HeII]*y[IDX_HNCI] +
        k[1251]*y[IDX_HeII]*y[IDX_NCCNI] + k[1262]*y[IDX_HeII]*y[IDX_OCNI] -
        k[1332]*y[IDX_NI]*y[IDX_CNII] + k[2097]*y[IDX_CII]*y[IDX_NI] -
        k[2235]*y[IDX_CNII];
    ydot[IDX_N2HII] = 0.0 - k[565]*y[IDX_N2HII]*y[IDX_EM] -
        k[566]*y[IDX_N2HII]*y[IDX_EM] - k[655]*y[IDX_C2I]*y[IDX_N2HII] -
        k[682]*y[IDX_C2HI]*y[IDX_N2HII] - k[698]*y[IDX_CI]*y[IDX_N2HII] -
        k[765]*y[IDX_CH2I]*y[IDX_N2HII] - k[817]*y[IDX_CH4I]*y[IDX_N2HII] -
        k[853]*y[IDX_CHI]*y[IDX_N2HII] - k[877]*y[IDX_COI]*y[IDX_N2HII] +
        k[927]*y[IDX_H2II]*y[IDX_N2I] + k[953]*y[IDX_H2I]*y[IDX_N2II] +
        k[1010]*y[IDX_H2OI]*y[IDX_N2II] - k[1011]*y[IDX_H2OI]*y[IDX_N2HII] +
        k[1055]*y[IDX_H3II]*y[IDX_N2I] - k[1128]*y[IDX_HCNI]*y[IDX_N2HII] -
        k[1160]*y[IDX_HCOI]*y[IDX_N2HII] - k[1171]*y[IDX_HNCI]*y[IDX_N2HII] +
        k[1306]*y[IDX_NII]*y[IDX_NH3I] + k[1317]*y[IDX_N2II]*y[IDX_HCOI] +
        k[1319]*y[IDX_N2I]*y[IDX_HNOII] + k[1320]*y[IDX_N2I]*y[IDX_O2HII] -
        k[1321]*y[IDX_N2HII]*y[IDX_CH3CNI] - k[1322]*y[IDX_N2HII]*y[IDX_CO2I] -
        k[1323]*y[IDX_N2HII]*y[IDX_H2COI] - k[1324]*y[IDX_N2HII]*y[IDX_SI] +
        k[1338]*y[IDX_NI]*y[IDX_NH2II] + k[1363]*y[IDX_NHII]*y[IDX_N2I] +
        k[1367]*y[IDX_NHII]*y[IDX_NOI] - k[1408]*y[IDX_NH2I]*y[IDX_N2HII] -
        k[1440]*y[IDX_NH3I]*y[IDX_N2HII] - k[1454]*y[IDX_NHI]*y[IDX_N2HII] -
        k[1506]*y[IDX_OI]*y[IDX_N2HII] + k[1529]*y[IDX_OHII]*y[IDX_N2I] -
        k[1545]*y[IDX_OHI]*y[IDX_N2HII] - k[2290]*y[IDX_N2HII];
    ydot[IDX_OCSI] = 0.0 - k[24]*y[IDX_CII]*y[IDX_OCSI] -
        k[80]*y[IDX_CH4II]*y[IDX_OCSI] - k[137]*y[IDX_HII]*y[IDX_OCSI] -
        k[184]*y[IDX_H2OII]*y[IDX_OCSI] + k[190]*y[IDX_H2SI]*y[IDX_OCSII] -
        k[250]*y[IDX_NII]*y[IDX_OCSI] - k[257]*y[IDX_N2II]*y[IDX_OCSI] +
        k[291]*y[IDX_NH3I]*y[IDX_OCSII] - k[318]*y[IDX_OII]*y[IDX_OCSI] -
        k[440]*y[IDX_OCSI] - k[441]*y[IDX_OCSI] + k[553]*y[IDX_HOCSII]*y[IDX_EM]
        - k[636]*y[IDX_CII]*y[IDX_OCSI] - k[736]*y[IDX_CHII]*y[IDX_OCSI] -
        k[737]*y[IDX_CHII]*y[IDX_OCSI] - k[751]*y[IDX_CH2II]*y[IDX_OCSI] -
        k[752]*y[IDX_CH2II]*y[IDX_OCSI] - k[785]*y[IDX_CH3II]*y[IDX_OCSI] -
        k[802]*y[IDX_CH4II]*y[IDX_OCSI] - k[904]*y[IDX_HII]*y[IDX_OCSI] +
        k[1006]*y[IDX_H2OI]*y[IDX_HOCSII] - k[1065]*y[IDX_H3II]*y[IDX_OCSI] -
        k[1149]*y[IDX_HCOII]*y[IDX_OCSI] - k[1264]*y[IDX_HeII]*y[IDX_OCSI] -
        k[1265]*y[IDX_HeII]*y[IDX_OCSI] - k[1266]*y[IDX_HeII]*y[IDX_OCSI] -
        k[1267]*y[IDX_HeII]*y[IDX_OCSI] - k[1312]*y[IDX_NII]*y[IDX_OCSI] -
        k[1313]*y[IDX_NII]*y[IDX_OCSI] - k[1318]*y[IDX_N2II]*y[IDX_OCSI] -
        k[1479]*y[IDX_OII]*y[IDX_OCSI] - k[1552]*y[IDX_SII]*y[IDX_OCSI] -
        k[1562]*y[IDX_SOII]*y[IDX_OCSI] - k[1565]*y[IDX_SiII]*y[IDX_OCSI] -
        k[1610]*y[IDX_CI]*y[IDX_OCSI] - k[1695]*y[IDX_CHI]*y[IDX_OCSI] +
        k[1700]*y[IDX_CHI]*y[IDX_SOI] - k[1775]*y[IDX_HI]*y[IDX_OCSI] +
        k[1895]*y[IDX_OI]*y[IDX_HCSI] - k[1912]*y[IDX_OI]*y[IDX_OCSI] -
        k[1913]*y[IDX_OI]*y[IDX_OCSI] + k[1936]*y[IDX_OHI]*y[IDX_CSI] +
        k[1951]*y[IDX_SI]*y[IDX_HCOI] - k[2066]*y[IDX_OCSI] -
        k[2067]*y[IDX_OCSI] - k[2259]*y[IDX_OCSI] + k[2487]*y[IDX_GOCSI] +
        k[2488]*y[IDX_GOCSI] + k[2489]*y[IDX_GOCSI] + k[2490]*y[IDX_GOCSI];
    ydot[IDX_H2SII] = 0.0 + k[17]*y[IDX_CII]*y[IDX_H2SI] +
        k[43]*y[IDX_C2H2II]*y[IDX_H2SI] + k[77]*y[IDX_CH4II]*y[IDX_H2SI] +
        k[102]*y[IDX_COII]*y[IDX_H2SI] + k[123]*y[IDX_HII]*y[IDX_H2SI] +
        k[163]*y[IDX_H2II]*y[IDX_H2SI] + k[179]*y[IDX_H2OII]*y[IDX_H2SI] +
        k[190]*y[IDX_H2SI]*y[IDX_OCSII] - k[203]*y[IDX_HCOI]*y[IDX_H2SII] +
        k[214]*y[IDX_HeII]*y[IDX_H2SI] - k[223]*y[IDX_MgI]*y[IDX_H2SII] +
        k[241]*y[IDX_NII]*y[IDX_H2SI] + k[253]*y[IDX_N2II]*y[IDX_H2SI] +
        k[266]*y[IDX_NH2II]*y[IDX_H2SI] - k[286]*y[IDX_NH3I]*y[IDX_H2SII] -
        k[299]*y[IDX_NOI]*y[IDX_H2SII] + k[313]*y[IDX_OII]*y[IDX_H2SI] +
        k[322]*y[IDX_O2II]*y[IDX_H2SI] + k[333]*y[IDX_OHII]*y[IDX_H2SI] -
        k[346]*y[IDX_SI]*y[IDX_H2SII] - k[350]*y[IDX_SiI]*y[IDX_H2SII] +
        k[403]*y[IDX_H2SI] - k[515]*y[IDX_H2SII]*y[IDX_EM] -
        k[516]*y[IDX_H2SII]*y[IDX_EM] - k[691]*y[IDX_CI]*y[IDX_H2SII] -
        k[946]*y[IDX_H2I]*y[IDX_H2SII] + k[949]*y[IDX_H2I]*y[IDX_HSII] +
        k[974]*y[IDX_H2COI]*y[IDX_SII] - k[1000]*y[IDX_H2OI]*y[IDX_H2SII] -
        k[1017]*y[IDX_H2SII]*y[IDX_H2SI] + k[1053]*y[IDX_H3II]*y[IDX_HSI] -
        k[1103]*y[IDX_HI]*y[IDX_H2SII] + k[1104]*y[IDX_HI]*y[IDX_H3SII] +
        k[1147]*y[IDX_HCOII]*y[IDX_HSI] - k[1335]*y[IDX_NI]*y[IDX_H2SII] -
        k[1426]*y[IDX_NH3I]*y[IDX_H2SII] - k[1498]*y[IDX_OI]*y[IDX_H2SII] -
        k[1499]*y[IDX_OI]*y[IDX_H2SII] + k[2020]*y[IDX_H2SI] +
        k[2117]*y[IDX_H2I]*y[IDX_SII] - k[2138]*y[IDX_H2SII]*y[IDX_EM] -
        k[2293]*y[IDX_H2SII];
    ydot[IDX_HNOII] = 0.0 - k[300]*y[IDX_NOI]*y[IDX_HNOII] -
        k[549]*y[IDX_HNOII]*y[IDX_EM] - k[654]*y[IDX_C2I]*y[IDX_HNOII] -
        k[681]*y[IDX_C2HI]*y[IDX_HNOII] - k[696]*y[IDX_CI]*y[IDX_HNOII] -
        k[764]*y[IDX_CH2I]*y[IDX_HNOII] - k[813]*y[IDX_CH4I]*y[IDX_HNOII] -
        k[850]*y[IDX_CHI]*y[IDX_HNOII] - k[869]*y[IDX_CNI]*y[IDX_HNOII] -
        k[876]*y[IDX_COI]*y[IDX_HNOII] + k[930]*y[IDX_H2II]*y[IDX_NOI] -
        k[971]*y[IDX_H2COI]*y[IDX_HNOII] - k[1005]*y[IDX_H2OI]*y[IDX_HNOII] +
        k[1060]*y[IDX_H3II]*y[IDX_NOI] - k[1125]*y[IDX_HCNI]*y[IDX_HNOII] -
        k[1159]*y[IDX_HCOI]*y[IDX_HNOII] - k[1169]*y[IDX_HNCI]*y[IDX_HNOII] -
        k[1173]*y[IDX_HNOII]*y[IDX_CO2I] - k[1174]*y[IDX_HNOII]*y[IDX_SI] -
        k[1319]*y[IDX_N2I]*y[IDX_HNOII] + k[1333]*y[IDX_NI]*y[IDX_H2OII] +
        k[1351]*y[IDX_NHII]*y[IDX_CO2I] + k[1357]*y[IDX_NHII]*y[IDX_H2OI] +
        k[1391]*y[IDX_NH2II]*y[IDX_O2I] - k[1407]*y[IDX_NH2I]*y[IDX_HNOII] -
        k[1436]*y[IDX_NH3I]*y[IDX_HNOII] - k[1453]*y[IDX_NHI]*y[IDX_HNOII] +
        k[1458]*y[IDX_NHI]*y[IDX_O2II] + k[1462]*y[IDX_NOI]*y[IDX_O2HII] +
        k[1507]*y[IDX_OI]*y[IDX_NH2II] + k[1508]*y[IDX_OI]*y[IDX_NH3II] +
        k[1531]*y[IDX_OHII]*y[IDX_NOI] - k[1544]*y[IDX_OHI]*y[IDX_HNOII] -
        k[2272]*y[IDX_HNOII];
    ydot[IDX_CH5II] = 0.0 - k[493]*y[IDX_CH5II]*y[IDX_EM] -
        k[494]*y[IDX_CH5II]*y[IDX_EM] - k[495]*y[IDX_CH5II]*y[IDX_EM] -
        k[496]*y[IDX_CH5II]*y[IDX_EM] - k[497]*y[IDX_CH5II]*y[IDX_EM] -
        k[689]*y[IDX_CI]*y[IDX_CH5II] - k[755]*y[IDX_CH2I]*y[IDX_CH5II] +
        k[795]*y[IDX_CH4II]*y[IDX_CH4I] + k[812]*y[IDX_CH4I]*y[IDX_HCO2II] +
        k[813]*y[IDX_CH4I]*y[IDX_HNOII] + k[817]*y[IDX_CH4I]*y[IDX_N2HII] +
        k[819]*y[IDX_CH4I]*y[IDX_OHII] - k[823]*y[IDX_CH5II]*y[IDX_C2I] -
        k[824]*y[IDX_CH5II]*y[IDX_C2HI] - k[825]*y[IDX_CH5II]*y[IDX_CO2I] -
        k[826]*y[IDX_CH5II]*y[IDX_COI] - k[827]*y[IDX_CH5II]*y[IDX_H2COI] -
        k[828]*y[IDX_CH5II]*y[IDX_H2OI] - k[829]*y[IDX_CH5II]*y[IDX_H2SI] -
        k[830]*y[IDX_CH5II]*y[IDX_HCNI] - k[831]*y[IDX_CH5II]*y[IDX_HCOI] -
        k[832]*y[IDX_CH5II]*y[IDX_HClI] - k[833]*y[IDX_CH5II]*y[IDX_HNCI] -
        k[834]*y[IDX_CH5II]*y[IDX_MgI] - k[835]*y[IDX_CH5II]*y[IDX_SI] -
        k[836]*y[IDX_CH5II]*y[IDX_SiH4I] - k[840]*y[IDX_CHI]*y[IDX_CH5II] +
        k[915]*y[IDX_H2II]*y[IDX_CH4I] + k[939]*y[IDX_H2I]*y[IDX_CH4II] +
        k[1033]*y[IDX_H3II]*y[IDX_CH4I] - k[1102]*y[IDX_HI]*y[IDX_CH5II] -
        k[1397]*y[IDX_NH2I]*y[IDX_CH5II] - k[1422]*y[IDX_NH3I]*y[IDX_CH5II] -
        k[1447]*y[IDX_NHI]*y[IDX_CH5II] - k[1494]*y[IDX_OI]*y[IDX_CH5II] -
        k[1495]*y[IDX_OI]*y[IDX_CH5II] - k[1538]*y[IDX_OHI]*y[IDX_CH5II] +
        k[2114]*y[IDX_H2I]*y[IDX_CH3II] - k[2248]*y[IDX_CH5II];
    ydot[IDX_N2II] = 0.0 - k[38]*y[IDX_C2I]*y[IDX_N2II] -
        k[49]*y[IDX_C2HI]*y[IDX_N2II] - k[53]*y[IDX_CI]*y[IDX_N2II] -
        k[67]*y[IDX_CH2I]*y[IDX_N2II] - k[88]*y[IDX_CHI]*y[IDX_N2II] -
        k[100]*y[IDX_CNI]*y[IDX_N2II] - k[107]*y[IDX_COI]*y[IDX_N2II] -
        k[189]*y[IDX_H2OI]*y[IDX_N2II] - k[201]*y[IDX_HCNI]*y[IDX_N2II] +
        k[215]*y[IDX_HeII]*y[IDX_N2I] - k[226]*y[IDX_MgI]*y[IDX_N2II] -
        k[252]*y[IDX_N2II]*y[IDX_H2COI] - k[253]*y[IDX_N2II]*y[IDX_H2SI] -
        k[254]*y[IDX_N2II]*y[IDX_HCOI] - k[255]*y[IDX_N2II]*y[IDX_NOI] -
        k[256]*y[IDX_N2II]*y[IDX_O2I] - k[257]*y[IDX_N2II]*y[IDX_OCSI] -
        k[258]*y[IDX_N2II]*y[IDX_SI] - k[259]*y[IDX_NI]*y[IDX_N2II] -
        k[275]*y[IDX_NH2I]*y[IDX_N2II] - k[289]*y[IDX_NH3I]*y[IDX_N2II] -
        k[296]*y[IDX_NHI]*y[IDX_N2II] - k[328]*y[IDX_OI]*y[IDX_N2II] -
        k[342]*y[IDX_OHI]*y[IDX_N2II] - k[564]*y[IDX_N2II]*y[IDX_EM] -
        k[815]*y[IDX_CH4I]*y[IDX_N2II] - k[816]*y[IDX_CH4I]*y[IDX_N2II] -
        k[953]*y[IDX_H2I]*y[IDX_N2II] - k[1010]*y[IDX_H2OI]*y[IDX_N2II] +
        k[1308]*y[IDX_NII]*y[IDX_NHI] + k[1309]*y[IDX_NII]*y[IDX_NOI] -
        k[1314]*y[IDX_N2II]*y[IDX_H2COI] - k[1315]*y[IDX_N2II]*y[IDX_H2SI] -
        k[1316]*y[IDX_N2II]*y[IDX_H2SI] - k[1317]*y[IDX_N2II]*y[IDX_HCOI] -
        k[1318]*y[IDX_N2II]*y[IDX_OCSI] + k[1332]*y[IDX_NI]*y[IDX_CNII] +
        k[1337]*y[IDX_NI]*y[IDX_NHII] - k[1505]*y[IDX_OI]*y[IDX_N2II] +
        k[2127]*y[IDX_NII]*y[IDX_NI] - k[2230]*y[IDX_N2II];
    ydot[IDX_HSI] = 0.0 - k[128]*y[IDX_HII]*y[IDX_HSI] +
        k[225]*y[IDX_MgI]*y[IDX_HSII] + k[288]*y[IDX_NH3I]*y[IDX_HSII] +
        k[301]*y[IDX_NOI]*y[IDX_HSII] + k[347]*y[IDX_SI]*y[IDX_HSII] +
        k[351]*y[IDX_SiI]*y[IDX_HSII] + k[402]*y[IDX_H2S2I] +
        k[402]*y[IDX_H2S2I] + k[417]*y[IDX_HS2I] - k[418]*y[IDX_HSI] +
        k[515]*y[IDX_H2SII]*y[IDX_EM] + k[518]*y[IDX_H2S2II]*y[IDX_EM] +
        k[518]*y[IDX_H2S2II]*y[IDX_EM] + k[533]*y[IDX_H3SII]*y[IDX_EM] +
        k[534]*y[IDX_H3SII]*y[IDX_EM] + k[555]*y[IDX_HS2II]*y[IDX_EM] +
        k[561]*y[IDX_HSiSII]*y[IDX_EM] - k[628]*y[IDX_CII]*y[IDX_HSI] -
        k[780]*y[IDX_CH3II]*y[IDX_HSI] + k[872]*y[IDX_COII]*y[IDX_H2SI] -
        k[902]*y[IDX_HII]*y[IDX_HSI] + k[975]*y[IDX_H2COI]*y[IDX_SII] +
        k[982]*y[IDX_H2OII]*y[IDX_H2SI] + k[1000]*y[IDX_H2OI]*y[IDX_H2SII] +
        k[1017]*y[IDX_H2SII]*y[IDX_H2SI] - k[1053]*y[IDX_H3II]*y[IDX_HSI] +
        k[1109]*y[IDX_HI]*y[IDX_SiSII] - k[1147]*y[IDX_HCOII]*y[IDX_HSI] +
        k[1224]*y[IDX_HeII]*y[IDX_H2S2I] + k[1247]*y[IDX_HeII]*y[IDX_HS2I] -
        k[1249]*y[IDX_HeII]*y[IDX_HSI] + k[1301]*y[IDX_NII]*y[IDX_H2SI] +
        k[1383]*y[IDX_NH2II]*y[IDX_H2SI] + k[1415]*y[IDX_NH3II]*y[IDX_H2SI] +
        k[1426]*y[IDX_NH3I]*y[IDX_H2SII] - k[1595]*y[IDX_CI]*y[IDX_HSI] -
        k[1596]*y[IDX_CI]*y[IDX_HSI] + k[1653]*y[IDX_CH3I]*y[IDX_H2SI] +
        k[1673]*y[IDX_CH4I]*y[IDX_SI] + k[1698]*y[IDX_CHI]*y[IDX_SI] +
        k[1699]*y[IDX_CHI]*y[IDX_SOI] - k[1727]*y[IDX_H2I]*y[IDX_HSI] +
        k[1735]*y[IDX_H2I]*y[IDX_SI] + k[1749]*y[IDX_HI]*y[IDX_H2SI] -
        k[1758]*y[IDX_HI]*y[IDX_HSI] + k[1766]*y[IDX_HI]*y[IDX_NSI] +
        k[1775]*y[IDX_HI]*y[IDX_OCSI] + k[1777]*y[IDX_HI]*y[IDX_S2I] +
        k[1778]*y[IDX_HI]*y[IDX_SOI] - k[1789]*y[IDX_HSI]*y[IDX_HSI] -
        k[1789]*y[IDX_HSI]*y[IDX_HSI] - k[1816]*y[IDX_NI]*y[IDX_HSI] -
        k[1817]*y[IDX_NI]*y[IDX_HSI] + k[1856]*y[IDX_NHI]*y[IDX_SI] +
        k[1888]*y[IDX_OI]*y[IDX_H2SI] + k[1894]*y[IDX_OI]*y[IDX_HCSI] -
        k[1899]*y[IDX_OI]*y[IDX_HSI] - k[1900]*y[IDX_OI]*y[IDX_HSI] +
        k[1935]*y[IDX_OHI]*y[IDX_CSI] + k[1938]*y[IDX_OHI]*y[IDX_H2SI] +
        k[1952]*y[IDX_SI]*y[IDX_HCOI] - k[1953]*y[IDX_SI]*y[IDX_HSI] +
        k[2019]*y[IDX_H2S2I] + k[2019]*y[IDX_H2S2I] + k[2021]*y[IDX_H2SI] +
        k[2042]*y[IDX_HS2I] - k[2043]*y[IDX_HSI] - k[2254]*y[IDX_HSI];
    ydot[IDX_HCNHII] = 0.0 - k[539]*y[IDX_HCNHII]*y[IDX_EM] -
        k[540]*y[IDX_HCNHII]*y[IDX_EM] - k[541]*y[IDX_HCNHII]*y[IDX_EM] +
        k[662]*y[IDX_C2HII]*y[IDX_HCNI] + k[664]*y[IDX_C2HII]*y[IDX_HNCI] +
        k[668]*y[IDX_C2H2II]*y[IDX_HCNI] + k[669]*y[IDX_C2H2II]*y[IDX_HNCI] +
        k[725]*y[IDX_CHII]*y[IDX_HCNI] + k[727]*y[IDX_CHII]*y[IDX_HNCI] -
        k[761]*y[IDX_CH2I]*y[IDX_HCNHII] - k[762]*y[IDX_CH2I]*y[IDX_HCNHII] +
        k[775]*y[IDX_CH3II]*y[IDX_CH3CNI] + k[811]*y[IDX_CH4I]*y[IDX_HCNII] +
        k[830]*y[IDX_CH5II]*y[IDX_HCNI] + k[833]*y[IDX_CH5II]*y[IDX_HNCI] -
        k[847]*y[IDX_CHI]*y[IDX_HCNHII] - k[848]*y[IDX_CHI]*y[IDX_HCNHII] +
        k[947]*y[IDX_H2I]*y[IDX_HCNII] + k[983]*y[IDX_H2OII]*y[IDX_HCNI] +
        k[986]*y[IDX_H2OII]*y[IDX_HNCI] + k[1045]*y[IDX_H3II]*y[IDX_HCNI] +
        k[1050]*y[IDX_H3II]*y[IDX_HNCI] + k[1088]*y[IDX_H3OII]*y[IDX_HCNI] +
        k[1090]*y[IDX_H3OII]*y[IDX_HNCI] + k[1113]*y[IDX_HCNII]*y[IDX_HCNI] +
        k[1115]*y[IDX_HCNII]*y[IDX_HCOI] + k[1116]*y[IDX_HCNII]*y[IDX_HNCI] +
        k[1121]*y[IDX_HCNI]*y[IDX_H2COII] + k[1122]*y[IDX_HCNI]*y[IDX_H3COII] +
        k[1123]*y[IDX_HCNI]*y[IDX_H3SII] + k[1124]*y[IDX_HCNI]*y[IDX_HCOII] +
        k[1125]*y[IDX_HCNI]*y[IDX_HNOII] + k[1126]*y[IDX_HCNI]*y[IDX_HSII] +
        k[1127]*y[IDX_HCNI]*y[IDX_HSiSII] + k[1128]*y[IDX_HCNI]*y[IDX_N2HII] +
        k[1129]*y[IDX_HCNI]*y[IDX_O2HII] - k[1130]*y[IDX_HCNHII]*y[IDX_CH3CNI] -
        k[1131]*y[IDX_HCNHII]*y[IDX_CH3CNI] - k[1132]*y[IDX_HCNHII]*y[IDX_H2COI]
        - k[1133]*y[IDX_HCNHII]*y[IDX_H2COI] - k[1134]*y[IDX_HCNHII]*y[IDX_H2SI]
        - k[1135]*y[IDX_HCNHII]*y[IDX_H2SI] + k[1165]*y[IDX_HNCI]*y[IDX_H2COII]
        + k[1166]*y[IDX_HNCI]*y[IDX_H3COII] + k[1167]*y[IDX_HNCI]*y[IDX_H3SII] +
        k[1168]*y[IDX_HNCI]*y[IDX_HCOII] + k[1169]*y[IDX_HNCI]*y[IDX_HNOII] +
        k[1170]*y[IDX_HNCI]*y[IDX_HSII] + k[1171]*y[IDX_HNCI]*y[IDX_N2HII] +
        k[1172]*y[IDX_HNCI]*y[IDX_O2HII] + k[1295]*y[IDX_NII]*y[IDX_CH4I] +
        k[1360]*y[IDX_NHII]*y[IDX_HCNI] + k[1362]*y[IDX_NHII]*y[IDX_HNCI] +
        k[1385]*y[IDX_NH2II]*y[IDX_HCNI] + k[1387]*y[IDX_NH2II]*y[IDX_HNCI] -
        k[1404]*y[IDX_NH2I]*y[IDX_HCNHII] - k[1405]*y[IDX_NH2I]*y[IDX_HCNHII] +
        k[1421]*y[IDX_NH3I]*y[IDX_C2NII] + k[1430]*y[IDX_NH3I]*y[IDX_HCNII] -
        k[1431]*y[IDX_NH3I]*y[IDX_HCNHII] - k[1432]*y[IDX_NH3I]*y[IDX_HCNHII] +
        k[1446]*y[IDX_NHI]*y[IDX_CH3II] + k[1525]*y[IDX_OHII]*y[IDX_HCNI] +
        k[1528]*y[IDX_OHII]*y[IDX_HNCI] - k[2289]*y[IDX_HCNHII];
    ydot[IDX_H2II] = 0.0 - k[153]*y[IDX_H2II]*y[IDX_C2I] -
        k[154]*y[IDX_H2II]*y[IDX_C2H2I] - k[155]*y[IDX_H2II]*y[IDX_C2HI] -
        k[156]*y[IDX_H2II]*y[IDX_CH2I] - k[157]*y[IDX_H2II]*y[IDX_CH4I] -
        k[158]*y[IDX_H2II]*y[IDX_CHI] - k[159]*y[IDX_H2II]*y[IDX_CNI] -
        k[160]*y[IDX_H2II]*y[IDX_COI] - k[161]*y[IDX_H2II]*y[IDX_H2COI] -
        k[162]*y[IDX_H2II]*y[IDX_H2OI] - k[163]*y[IDX_H2II]*y[IDX_H2SI] -
        k[164]*y[IDX_H2II]*y[IDX_HCNI] - k[165]*y[IDX_H2II]*y[IDX_HCOI] -
        k[166]*y[IDX_H2II]*y[IDX_NH2I] - k[167]*y[IDX_H2II]*y[IDX_NH3I] -
        k[168]*y[IDX_H2II]*y[IDX_NHI] - k[169]*y[IDX_H2II]*y[IDX_NOI] -
        k[170]*y[IDX_H2II]*y[IDX_O2I] - k[171]*y[IDX_H2II]*y[IDX_OHI] +
        k[172]*y[IDX_H2I]*y[IDX_HeII] - k[193]*y[IDX_HI]*y[IDX_H2II] +
        k[360]*y[IDX_H2I] - k[501]*y[IDX_H2II]*y[IDX_EM] +
        k[898]*y[IDX_HII]*y[IDX_HCOI] - k[909]*y[IDX_H2II]*y[IDX_C2I] -
        k[910]*y[IDX_H2II]*y[IDX_C2H4I] - k[911]*y[IDX_H2II]*y[IDX_C2HI] -
        k[912]*y[IDX_H2II]*y[IDX_CI] - k[913]*y[IDX_H2II]*y[IDX_CH2I] -
        k[914]*y[IDX_H2II]*y[IDX_CH4I] - k[915]*y[IDX_H2II]*y[IDX_CH4I] -
        k[916]*y[IDX_H2II]*y[IDX_CHI] - k[917]*y[IDX_H2II]*y[IDX_CNI] -
        k[918]*y[IDX_H2II]*y[IDX_CO2I] - k[919]*y[IDX_H2II]*y[IDX_COI] -
        k[920]*y[IDX_H2II]*y[IDX_H2I] - k[921]*y[IDX_H2II]*y[IDX_H2COI] -
        k[922]*y[IDX_H2II]*y[IDX_H2OI] - k[923]*y[IDX_H2II]*y[IDX_H2SI] -
        k[924]*y[IDX_H2II]*y[IDX_H2SI] - k[925]*y[IDX_H2II]*y[IDX_HCOI] -
        k[926]*y[IDX_H2II]*y[IDX_HeI] - k[927]*y[IDX_H2II]*y[IDX_N2I] -
        k[928]*y[IDX_H2II]*y[IDX_NI] - k[929]*y[IDX_H2II]*y[IDX_NHI] -
        k[930]*y[IDX_H2II]*y[IDX_NOI] - k[931]*y[IDX_H2II]*y[IDX_O2I] -
        k[932]*y[IDX_H2II]*y[IDX_OI] - k[933]*y[IDX_H2II]*y[IDX_OHI] +
        k[1106]*y[IDX_HI]*y[IDX_HeHII] - k[2009]*y[IDX_H2II] +
        k[2025]*y[IDX_H3II] + k[2110]*y[IDX_HII]*y[IDX_HI];
    ydot[IDX_H3COII] = 0.0 - k[521]*y[IDX_H3COII]*y[IDX_EM] -
        k[522]*y[IDX_H3COII]*y[IDX_EM] - k[523]*y[IDX_H3COII]*y[IDX_EM] -
        k[524]*y[IDX_H3COII]*y[IDX_EM] - k[525]*y[IDX_H3COII]*y[IDX_EM] +
        k[606]*y[IDX_CII]*y[IDX_C2H5OHI] + k[612]*y[IDX_CII]*y[IDX_CH3OHI] +
        k[660]*y[IDX_C2HII]*y[IDX_H2COI] + k[710]*y[IDX_CHII]*y[IDX_CH3OHI] +
        k[716]*y[IDX_CHII]*y[IDX_H2COI] + k[743]*y[IDX_CH2II]*y[IDX_H2OI] +
        k[776]*y[IDX_CH3II]*y[IDX_CH3OHI] + k[782]*y[IDX_CH3II]*y[IDX_O2I] +
        k[798]*y[IDX_CH4II]*y[IDX_H2COI] + k[809]*y[IDX_CH4I]*y[IDX_H2COII] +
        k[827]*y[IDX_CH5II]*y[IDX_H2COI] - k[844]*y[IDX_CHI]*y[IDX_H3COII] +
        k[888]*y[IDX_HII]*y[IDX_CH3OHI] + k[966]*y[IDX_H2COII]*y[IDX_H2COI] +
        k[970]*y[IDX_H2COI]*y[IDX_H3SII] + k[971]*y[IDX_H2COI]*y[IDX_HNOII] +
        k[973]*y[IDX_H2COI]*y[IDX_O2HII] + k[979]*y[IDX_H2OII]*y[IDX_H2COI] -
        k[1001]*y[IDX_H2OI]*y[IDX_H3COII] + k[1024]*y[IDX_H3II]*y[IDX_C2H5OHI] +
        k[1041]*y[IDX_H3II]*y[IDX_H2COI] - k[1078]*y[IDX_H3COII]*y[IDX_CH3OHI] -
        k[1079]*y[IDX_H3COII]*y[IDX_H2SI] + k[1086]*y[IDX_H3OII]*y[IDX_H2COI] +
        k[1112]*y[IDX_HCNII]*y[IDX_H2COI] - k[1122]*y[IDX_HCNI]*y[IDX_H3COII] +
        k[1132]*y[IDX_HCNHII]*y[IDX_H2COI] + k[1133]*y[IDX_HCNHII]*y[IDX_H2COI]
        + k[1141]*y[IDX_HCOII]*y[IDX_H2COI] + k[1158]*y[IDX_HCOI]*y[IDX_H2COII]
        - k[1166]*y[IDX_HNCI]*y[IDX_H3COII] + k[1290]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[1323]*y[IDX_N2HII]*y[IDX_H2COI] + k[1354]*y[IDX_NHII]*y[IDX_H2COI] +
        k[1376]*y[IDX_NH2II]*y[IDX_H2COI] - k[1401]*y[IDX_NH2I]*y[IDX_H3COII] -
        k[1427]*y[IDX_NH3I]*y[IDX_H3COII] + k[1449]*y[IDX_NHI]*y[IDX_H2COII] +
        k[1467]*y[IDX_OII]*y[IDX_CH3OHI] + k[1483]*y[IDX_O2II]*y[IDX_CH3OHI] +
        k[1494]*y[IDX_OI]*y[IDX_CH5II] + k[1522]*y[IDX_OHII]*y[IDX_H2COI] +
        k[1560]*y[IDX_SOII]*y[IDX_C2H4I] + k[1991]*y[IDX_CH3OHI] -
        k[2202]*y[IDX_H3COII];
    ydot[IDX_SiI] = 0.0 - k[26]*y[IDX_CII]*y[IDX_SiI] -
        k[60]*y[IDX_CHII]*y[IDX_SiI] - k[143]*y[IDX_HII]*y[IDX_SiI] -
        k[186]*y[IDX_H2OII]*y[IDX_SiI] - k[219]*y[IDX_HeII]*y[IDX_SiI] +
        k[231]*y[IDX_MgI]*y[IDX_SiII] - k[281]*y[IDX_NH3II]*y[IDX_SiI] -
        k[348]*y[IDX_SiI]*y[IDX_CSII] - k[349]*y[IDX_SiI]*y[IDX_H2COII] -
        k[350]*y[IDX_SiI]*y[IDX_H2SII] - k[351]*y[IDX_SiI]*y[IDX_HSII] -
        k[352]*y[IDX_SiI]*y[IDX_NOII] - k[353]*y[IDX_SiI]*y[IDX_O2II] -
        k[354]*y[IDX_SiI]*y[IDX_SII] - k[448]*y[IDX_SiI] + k[451]*y[IDX_SiCI] +
        k[455]*y[IDX_SiHI] + k[456]*y[IDX_SiOI] + k[457]*y[IDX_SiSI] +
        k[561]*y[IDX_HSiSII]*y[IDX_EM] + k[587]*y[IDX_SiCII]*y[IDX_EM] +
        k[588]*y[IDX_SiC2II]*y[IDX_EM] + k[592]*y[IDX_SiHII]*y[IDX_EM] +
        k[593]*y[IDX_SiH2II]*y[IDX_EM] + k[594]*y[IDX_SiH2II]*y[IDX_EM] +
        k[602]*y[IDX_SiOII]*y[IDX_EM] + k[603]*y[IDX_SiOHII]*y[IDX_EM] +
        k[605]*y[IDX_SiSII]*y[IDX_EM] - k[670]*y[IDX_C2H2II]*y[IDX_SiI] +
        k[863]*y[IDX_CHI]*y[IDX_SiHII] + k[864]*y[IDX_CHI]*y[IDX_SiOII] +
        k[1014]*y[IDX_H2OI]*y[IDX_SiHII] - k[1071]*y[IDX_H3II]*y[IDX_SiI] -
        k[1093]*y[IDX_H3OII]*y[IDX_SiI] + k[1277]*y[IDX_HeII]*y[IDX_SiCI] +
        k[1286]*y[IDX_HeII]*y[IDX_SiOI] + k[1287]*y[IDX_HeII]*y[IDX_SiSI] +
        k[1343]*y[IDX_NI]*y[IDX_SiOII] + k[1442]*y[IDX_NH3I]*y[IDX_SiHII] -
        k[1535]*y[IDX_OHII]*y[IDX_SiI] - k[1566]*y[IDX_SiI]*y[IDX_HCOII] -
        k[1575]*y[IDX_C2H2I]*y[IDX_SiI] + k[1832]*y[IDX_NI]*y[IDX_SiCI] +
        k[1920]*y[IDX_OI]*y[IDX_SiCI] - k[1950]*y[IDX_OHI]*y[IDX_SiI] -
        k[1956]*y[IDX_SiI]*y[IDX_CO2I] - k[1957]*y[IDX_SiI]*y[IDX_COI] -
        k[1958]*y[IDX_SiI]*y[IDX_NOI] - k[1959]*y[IDX_SiI]*y[IDX_O2I] -
        k[2077]*y[IDX_SiI] + k[2078]*y[IDX_SiC2I] + k[2081]*y[IDX_SiCI] +
        k[2091]*y[IDX_SiHI] + k[2093]*y[IDX_SiOI] + k[2095]*y[IDX_SiSI] -
        k[2131]*y[IDX_OI]*y[IDX_SiI] + k[2144]*y[IDX_SiII]*y[IDX_EM] -
        k[2170]*y[IDX_SiI];
    ydot[IDX_C2H2I] = 0.0 + k[42]*y[IDX_C2H2II]*y[IDX_H2COI] +
        k[43]*y[IDX_C2H2II]*y[IDX_H2SI] + k[44]*y[IDX_C2H2II]*y[IDX_HCOI] +
        k[45]*y[IDX_C2H2II]*y[IDX_NOI] - k[46]*y[IDX_C2H2I]*y[IDX_HCNII] -
        k[75]*y[IDX_CH4II]*y[IDX_C2H2I] - k[111]*y[IDX_HII]*y[IDX_C2H2I] -
        k[154]*y[IDX_H2II]*y[IDX_C2H2I] - k[176]*y[IDX_H2OII]*y[IDX_C2H2I] -
        k[208]*y[IDX_HeII]*y[IDX_C2H2I] + k[220]*y[IDX_MgI]*y[IDX_C2H2II] +
        k[282]*y[IDX_NH3I]*y[IDX_C2H2II] - k[307]*y[IDX_OII]*y[IDX_C2H2I] -
        k[321]*y[IDX_O2II]*y[IDX_C2H2I] - k[367]*y[IDX_C2H2I] -
        k[368]*y[IDX_C2H2I] + k[369]*y[IDX_C2H3I] + k[370]*y[IDX_C2H4I] +
        k[611]*y[IDX_CII]*y[IDX_CH3CCHI] - k[675]*y[IDX_C2H2I]*y[IDX_C2N2II] -
        k[1178]*y[IDX_HeII]*y[IDX_C2H2I] - k[1179]*y[IDX_HeII]*y[IDX_C2H2I] -
        k[1180]*y[IDX_HeII]*y[IDX_C2H2I] - k[1482]*y[IDX_O2II]*y[IDX_C2H2I] -
        k[1556]*y[IDX_SOII]*y[IDX_C2H2I] - k[1557]*y[IDX_SOII]*y[IDX_C2H2I] -
        k[1558]*y[IDX_SOII]*y[IDX_C2H2I] - k[1570]*y[IDX_C2I]*y[IDX_C2H2I] -
        k[1574]*y[IDX_C2H2I]*y[IDX_NOI] - k[1575]*y[IDX_C2H2I]*y[IDX_SiI] +
        k[1577]*y[IDX_C2H3I]*y[IDX_O2I] + k[1588]*y[IDX_CI]*y[IDX_CH3I] +
        k[1618]*y[IDX_CH2I]*y[IDX_CH2I] + k[1619]*y[IDX_CH2I]*y[IDX_CH2I] +
        k[1646]*y[IDX_CH3I]*y[IDX_C2H3I] - k[1674]*y[IDX_CHI]*y[IDX_C2H2I] -
        k[1701]*y[IDX_CNI]*y[IDX_C2H2I] + k[1721]*y[IDX_H2I]*y[IDX_C2HI] -
        k[1737]*y[IDX_HI]*y[IDX_C2H2I] + k[1738]*y[IDX_HI]*y[IDX_C2H3I] +
        k[1791]*y[IDX_NI]*y[IDX_C2H3I] - k[1868]*y[IDX_OI]*y[IDX_C2H2I] -
        k[1927]*y[IDX_OHI]*y[IDX_C2H2I] - k[1928]*y[IDX_OHI]*y[IDX_C2H2I] -
        k[1929]*y[IDX_OHI]*y[IDX_C2H2I] + k[1930]*y[IDX_OHI]*y[IDX_C2H3I] -
        k[1964]*y[IDX_C2H2I] - k[1965]*y[IDX_C2H2I] + k[1966]*y[IDX_C2H3I] +
        k[1967]*y[IDX_C2H4I] - k[2298]*y[IDX_C2H2I] + k[2327]*y[IDX_GC2H2I] +
        k[2328]*y[IDX_GC2H2I] + k[2329]*y[IDX_GC2H2I] + k[2330]*y[IDX_GC2H2I];
    ydot[IDX_HCNII] = 0.0 - k[46]*y[IDX_C2H2I]*y[IDX_HCNII] +
        k[95]*y[IDX_CNII]*y[IDX_HCNI] + k[124]*y[IDX_HII]*y[IDX_HCNI] +
        k[164]*y[IDX_H2II]*y[IDX_HCNI] - k[188]*y[IDX_H2OI]*y[IDX_HCNII] -
        k[194]*y[IDX_HI]*y[IDX_HCNII] - k[197]*y[IDX_HCNII]*y[IDX_NOI] -
        k[198]*y[IDX_HCNII]*y[IDX_O2I] - k[199]*y[IDX_HCNII]*y[IDX_SI] +
        k[200]*y[IDX_HCNI]*y[IDX_COII] + k[201]*y[IDX_HCNI]*y[IDX_N2II] +
        k[242]*y[IDX_NII]*y[IDX_HCNI] - k[287]*y[IDX_NH3I]*y[IDX_HCNII] -
        k[538]*y[IDX_HCNII]*y[IDX_EM] + k[629]*y[IDX_CII]*y[IDX_NH2I] +
        k[630]*y[IDX_CII]*y[IDX_NH3I] - k[652]*y[IDX_C2I]*y[IDX_HCNII] -
        k[679]*y[IDX_C2HI]*y[IDX_HCNII] - k[693]*y[IDX_CI]*y[IDX_HCNII] +
        k[729]*y[IDX_CHII]*y[IDX_NH2I] - k[760]*y[IDX_CH2I]*y[IDX_HCNII] -
        k[811]*y[IDX_CH4I]*y[IDX_HCNII] - k[846]*y[IDX_CHI]*y[IDX_HCNII] +
        k[867]*y[IDX_CNII]*y[IDX_HCOI] + k[869]*y[IDX_CNI]*y[IDX_HNOII] +
        k[870]*y[IDX_CNI]*y[IDX_O2HII] + k[917]*y[IDX_H2II]*y[IDX_CNI] +
        k[940]*y[IDX_H2I]*y[IDX_CNII] - k[947]*y[IDX_H2I]*y[IDX_HCNII] +
        k[995]*y[IDX_H2OI]*y[IDX_CNII] - k[1002]*y[IDX_H2OI]*y[IDX_HCNII] +
        k[1035]*y[IDX_H3II]*y[IDX_CNI] + k[1097]*y[IDX_HI]*y[IDX_C2N2II] -
        k[1110]*y[IDX_HCNII]*y[IDX_CO2I] - k[1111]*y[IDX_HCNII]*y[IDX_COI] -
        k[1112]*y[IDX_HCNII]*y[IDX_H2COI] - k[1113]*y[IDX_HCNII]*y[IDX_HCNI] -
        k[1114]*y[IDX_HCNII]*y[IDX_HCOI] - k[1115]*y[IDX_HCNII]*y[IDX_HCOI] -
        k[1116]*y[IDX_HCNII]*y[IDX_HNCI] - k[1117]*y[IDX_HCNII]*y[IDX_SI] +
        k[1118]*y[IDX_HCNI]*y[IDX_C2N2II] + k[1294]*y[IDX_NII]*y[IDX_CH4I] +
        k[1331]*y[IDX_NI]*y[IDX_CH2II] + k[1347]*y[IDX_NHII]*y[IDX_C2I] +
        k[1349]*y[IDX_NHII]*y[IDX_CNI] - k[1403]*y[IDX_NH2I]*y[IDX_HCNII] -
        k[1430]*y[IDX_NH3I]*y[IDX_HCNII] - k[1451]*y[IDX_NHI]*y[IDX_HCNII] +
        k[1519]*y[IDX_OHII]*y[IDX_CNI] - k[1541]*y[IDX_OHI]*y[IDX_HCNII] -
        k[2241]*y[IDX_HCNII];
    ydot[IDX_HNCI] = 0.0 - k[1]*y[IDX_HII]*y[IDX_HNCI] - k[414]*y[IDX_HNCI]
        + k[485]*y[IDX_CH3CNHII]*y[IDX_EM] + k[541]*y[IDX_HCNHII]*y[IDX_EM] -
        k[627]*y[IDX_CII]*y[IDX_HNCI] - k[664]*y[IDX_C2HII]*y[IDX_HNCI] -
        k[669]*y[IDX_C2H2II]*y[IDX_HNCI] - k[727]*y[IDX_CHII]*y[IDX_HNCI] +
        k[762]*y[IDX_CH2I]*y[IDX_HCNHII] - k[833]*y[IDX_CH5II]*y[IDX_HNCI] +
        k[848]*y[IDX_CHI]*y[IDX_HCNHII] - k[986]*y[IDX_H2OII]*y[IDX_HNCI] -
        k[1050]*y[IDX_H3II]*y[IDX_HNCI] - k[1090]*y[IDX_H3OII]*y[IDX_HNCI] -
        k[1116]*y[IDX_HCNII]*y[IDX_HNCI] + k[1131]*y[IDX_HCNHII]*y[IDX_CH3CNI] +
        k[1133]*y[IDX_HCNHII]*y[IDX_H2COI] + k[1135]*y[IDX_HCNHII]*y[IDX_H2SI] -
        k[1165]*y[IDX_HNCI]*y[IDX_H2COII] - k[1166]*y[IDX_HNCI]*y[IDX_H3COII] -
        k[1167]*y[IDX_HNCI]*y[IDX_H3SII] - k[1168]*y[IDX_HNCI]*y[IDX_HCOII] -
        k[1169]*y[IDX_HNCI]*y[IDX_HNOII] - k[1170]*y[IDX_HNCI]*y[IDX_HSII] -
        k[1171]*y[IDX_HNCI]*y[IDX_N2HII] - k[1172]*y[IDX_HNCI]*y[IDX_O2HII] -
        k[1242]*y[IDX_HeII]*y[IDX_HNCI] - k[1243]*y[IDX_HeII]*y[IDX_HNCI] -
        k[1244]*y[IDX_HeII]*y[IDX_HNCI] - k[1362]*y[IDX_NHII]*y[IDX_HNCI] -
        k[1387]*y[IDX_NH2II]*y[IDX_HNCI] + k[1405]*y[IDX_NH2I]*y[IDX_HCNHII] +
        k[1432]*y[IDX_NH3I]*y[IDX_HCNHII] - k[1528]*y[IDX_OHII]*y[IDX_HNCI] -
        k[1579]*y[IDX_C2HI]*y[IDX_HNCI] + k[1600]*y[IDX_CI]*y[IDX_NH2I] -
        k[1707]*y[IDX_CNI]*y[IDX_HNCI] - k[1754]*y[IDX_HI]*y[IDX_HNCI] +
        k[1788]*y[IDX_HNCOI]*y[IDX_CI] + k[1802]*y[IDX_NI]*y[IDX_CH2I] -
        k[2036]*y[IDX_HNCI] - k[2297]*y[IDX_HNCI] + k[2335]*y[IDX_GHNCI] +
        k[2336]*y[IDX_GHNCI] + k[2337]*y[IDX_GHNCI] + k[2338]*y[IDX_GHNCI];
    ydot[IDX_SiII] = 0.0 + k[26]*y[IDX_CII]*y[IDX_SiI] +
        k[60]*y[IDX_CHII]*y[IDX_SiI] + k[143]*y[IDX_HII]*y[IDX_SiI] +
        k[186]*y[IDX_H2OII]*y[IDX_SiI] + k[219]*y[IDX_HeII]*y[IDX_SiI] -
        k[231]*y[IDX_MgI]*y[IDX_SiII] + k[281]*y[IDX_NH3II]*y[IDX_SiI] +
        k[348]*y[IDX_SiI]*y[IDX_CSII] + k[349]*y[IDX_SiI]*y[IDX_H2COII] +
        k[350]*y[IDX_SiI]*y[IDX_H2SII] + k[351]*y[IDX_SiI]*y[IDX_HSII] +
        k[352]*y[IDX_SiI]*y[IDX_NOII] + k[353]*y[IDX_SiI]*y[IDX_O2II] +
        k[354]*y[IDX_SiI]*y[IDX_SII] + k[448]*y[IDX_SiI] +
        k[642]*y[IDX_CII]*y[IDX_SiCI] + k[645]*y[IDX_CII]*y[IDX_SiOI] +
        k[671]*y[IDX_C2H2II]*y[IDX_SiH4I] - k[684]*y[IDX_C2HI]*y[IDX_SiII] +
        k[704]*y[IDX_CI]*y[IDX_SiOII] + k[773]*y[IDX_CH2I]*y[IDX_SiOII] -
        k[862]*y[IDX_CHI]*y[IDX_SiII] + k[881]*y[IDX_COI]*y[IDX_SiOII] +
        k[908]*y[IDX_HII]*y[IDX_SiHI] - k[1013]*y[IDX_H2OI]*y[IDX_SiII] +
        k[1108]*y[IDX_HI]*y[IDX_SiHII] + k[1109]*y[IDX_HI]*y[IDX_SiSII] +
        k[1274]*y[IDX_HeII]*y[IDX_SiC2I] + k[1276]*y[IDX_HeII]*y[IDX_SiCI] +
        k[1278]*y[IDX_HeII]*y[IDX_SiH2I] + k[1282]*y[IDX_HeII]*y[IDX_SiH4I] +
        k[1284]*y[IDX_HeII]*y[IDX_SiHI] + k[1285]*y[IDX_HeII]*y[IDX_SiOI] +
        k[1288]*y[IDX_HeII]*y[IDX_SiSI] + k[1342]*y[IDX_NI]*y[IDX_SiCII] +
        k[1344]*y[IDX_NI]*y[IDX_SiOII] + k[1516]*y[IDX_OI]*y[IDX_SiOII] -
        k[1549]*y[IDX_OHI]*y[IDX_SiII] + k[1555]*y[IDX_SI]*y[IDX_SiOII] -
        k[1563]*y[IDX_SiII]*y[IDX_C2H5OHI] - k[1564]*y[IDX_SiII]*y[IDX_CH3OHI] -
        k[1565]*y[IDX_SiII]*y[IDX_OCSI] + k[2077]*y[IDX_SiI] +
        k[2082]*y[IDX_SiHII] + k[2092]*y[IDX_SiOII] -
        k[2118]*y[IDX_H2I]*y[IDX_SiII] - k[2126]*y[IDX_HI]*y[IDX_SiII] -
        k[2130]*y[IDX_OI]*y[IDX_SiII] - k[2144]*y[IDX_SiII]*y[IDX_EM] -
        k[2173]*y[IDX_SiII];
    ydot[IDX_COII] = 0.0 - k[37]*y[IDX_C2I]*y[IDX_COII] -
        k[48]*y[IDX_C2HI]*y[IDX_COII] - k[52]*y[IDX_CI]*y[IDX_COII] -
        k[64]*y[IDX_CH2I]*y[IDX_COII] - k[81]*y[IDX_CH4I]*y[IDX_COII] -
        k[84]*y[IDX_CHI]*y[IDX_COII] + k[93]*y[IDX_CNII]*y[IDX_COI] -
        k[101]*y[IDX_COII]*y[IDX_H2COI] - k[102]*y[IDX_COII]*y[IDX_H2SI] -
        k[103]*y[IDX_COII]*y[IDX_HCOI] - k[104]*y[IDX_COII]*y[IDX_NOI] -
        k[105]*y[IDX_COII]*y[IDX_O2I] - k[106]*y[IDX_COII]*y[IDX_SI] +
        k[107]*y[IDX_COI]*y[IDX_N2II] + k[160]*y[IDX_H2II]*y[IDX_COI] -
        k[187]*y[IDX_H2OI]*y[IDX_COII] - k[192]*y[IDX_HI]*y[IDX_COII] -
        k[200]*y[IDX_HCNI]*y[IDX_COII] + k[238]*y[IDX_NII]*y[IDX_COI] -
        k[273]*y[IDX_NH2I]*y[IDX_COII] - k[283]*y[IDX_NH3I]*y[IDX_COII] -
        k[295]*y[IDX_NHI]*y[IDX_COII] + k[310]*y[IDX_OII]*y[IDX_COI] -
        k[327]*y[IDX_OI]*y[IDX_COII] - k[341]*y[IDX_OHI]*y[IDX_COII] +
        k[357]*y[IDX_COI] - k[499]*y[IDX_COII]*y[IDX_EM] +
        k[616]*y[IDX_CII]*y[IDX_CO2I] + k[633]*y[IDX_CII]*y[IDX_O2I] +
        k[635]*y[IDX_CII]*y[IDX_OCNI] + k[637]*y[IDX_CII]*y[IDX_OHI] +
        k[641]*y[IDX_CII]*y[IDX_SOI] + k[649]*y[IDX_C2II]*y[IDX_O2I] +
        k[656]*y[IDX_C2I]*y[IDX_O2II] - k[677]*y[IDX_C2HI]*y[IDX_COII] +
        k[700]*y[IDX_CI]*y[IDX_O2II] + k[732]*y[IDX_CHII]*y[IDX_O2I] +
        k[735]*y[IDX_CHII]*y[IDX_OI] + k[738]*y[IDX_CHII]*y[IDX_OHI] -
        k[756]*y[IDX_CH2I]*y[IDX_COII] - k[807]*y[IDX_CH4I]*y[IDX_COII] -
        k[841]*y[IDX_CHI]*y[IDX_COII] + k[857]*y[IDX_CHI]*y[IDX_OII] -
        k[871]*y[IDX_COII]*y[IDX_H2COI] - k[872]*y[IDX_COII]*y[IDX_H2SI] -
        k[873]*y[IDX_COII]*y[IDX_SO2I] + k[892]*y[IDX_HII]*y[IDX_H2COI] +
        k[897]*y[IDX_HII]*y[IDX_HCOI] - k[941]*y[IDX_H2I]*y[IDX_COII] -
        k[942]*y[IDX_H2I]*y[IDX_COII] - k[997]*y[IDX_H2OI]*y[IDX_COII] +
        k[1194]*y[IDX_HeII]*y[IDX_CH2COI] + k[1209]*y[IDX_HeII]*y[IDX_CO2I] +
        k[1216]*y[IDX_HeII]*y[IDX_H2COI] + k[1235]*y[IDX_HeII]*y[IDX_HCOI] +
        k[1267]*y[IDX_HeII]*y[IDX_OCSI] + k[1296]*y[IDX_NII]*y[IDX_CO2I] -
        k[1398]*y[IDX_NH2I]*y[IDX_COII] - k[1423]*y[IDX_NH3I]*y[IDX_COII] -
        k[1448]*y[IDX_NHI]*y[IDX_COII] + k[1463]*y[IDX_OII]*y[IDX_C2I] +
        k[1465]*y[IDX_OII]*y[IDX_C2HI] + k[1490]*y[IDX_OI]*y[IDX_C2II] +
        k[1496]*y[IDX_OI]*y[IDX_CSII] - k[1539]*y[IDX_OHI]*y[IDX_COII] -
        k[2002]*y[IDX_COII] + k[2029]*y[IDX_HCOII] +
        k[2098]*y[IDX_CII]*y[IDX_OI] + k[2103]*y[IDX_CI]*y[IDX_OII] -
        k[2234]*y[IDX_COII];
    ydot[IDX_HSII] = 0.0 + k[128]*y[IDX_HII]*y[IDX_HSI] -
        k[225]*y[IDX_MgI]*y[IDX_HSII] - k[288]*y[IDX_NH3I]*y[IDX_HSII] -
        k[301]*y[IDX_NOI]*y[IDX_HSII] - k[347]*y[IDX_SI]*y[IDX_HSII] -
        k[351]*y[IDX_SiI]*y[IDX_HSII] - k[554]*y[IDX_HSII]*y[IDX_EM] -
        k[697]*y[IDX_CI]*y[IDX_HSII] + k[740]*y[IDX_CHII]*y[IDX_SI] -
        k[814]*y[IDX_CH4I]*y[IDX_HSII] + k[835]*y[IDX_CH5II]*y[IDX_SI] -
        k[851]*y[IDX_CHI]*y[IDX_HSII] + k[894]*y[IDX_HII]*y[IDX_H2SI] +
        k[904]*y[IDX_HII]*y[IDX_OCSI] + k[923]*y[IDX_H2II]*y[IDX_H2SI] -
        k[949]*y[IDX_H2I]*y[IDX_HSII] + k[961]*y[IDX_H2I]*y[IDX_SII] +
        k[968]*y[IDX_H2COII]*y[IDX_SI] + k[987]*y[IDX_H2OII]*y[IDX_SI] -
        k[1007]*y[IDX_H2OI]*y[IDX_HSII] + k[1068]*y[IDX_H3II]*y[IDX_SI] +
        k[1103]*y[IDX_HI]*y[IDX_H2SII] - k[1105]*y[IDX_HI]*y[IDX_HSII] +
        k[1117]*y[IDX_HCNII]*y[IDX_SI] - k[1126]*y[IDX_HCNI]*y[IDX_HSII] +
        k[1151]*y[IDX_HCOII]*y[IDX_SI] + k[1163]*y[IDX_HCOI]*y[IDX_SII] -
        k[1170]*y[IDX_HNCI]*y[IDX_HSII] + k[1174]*y[IDX_HNOII]*y[IDX_SI] -
        k[1175]*y[IDX_HSII]*y[IDX_H2SI] - k[1176]*y[IDX_HSII]*y[IDX_H2SI] +
        k[1224]*y[IDX_HeII]*y[IDX_H2S2I] + k[1226]*y[IDX_HeII]*y[IDX_H2SI] +
        k[1300]*y[IDX_NII]*y[IDX_H2SI] + k[1315]*y[IDX_N2II]*y[IDX_H2SI] +
        k[1324]*y[IDX_N2HII]*y[IDX_SI] - k[1336]*y[IDX_NI]*y[IDX_HSII] +
        k[1372]*y[IDX_NHII]*y[IDX_SI] + k[1382]*y[IDX_NH2II]*y[IDX_H2SI] +
        k[1393]*y[IDX_NH2II]*y[IDX_SI] - k[1437]*y[IDX_NH3I]*y[IDX_HSII] +
        k[1472]*y[IDX_OII]*y[IDX_H2SI] + k[1498]*y[IDX_OI]*y[IDX_H2SII] -
        k[1503]*y[IDX_OI]*y[IDX_HSII] - k[1504]*y[IDX_OI]*y[IDX_HSII] +
        k[1533]*y[IDX_OHII]*y[IDX_SI] + k[1554]*y[IDX_SI]*y[IDX_O2HII] -
        k[2039]*y[IDX_HSII] - k[2040]*y[IDX_HSII] -
        k[2116]*y[IDX_H2I]*y[IDX_HSII] - k[2265]*y[IDX_HSII];
    ydot[IDX_CO2I] = 0.0 - k[393]*y[IDX_CO2I] + k[411]*y[IDX_HCOOCH3I] +
        k[543]*y[IDX_HCO2II]*y[IDX_EM] - k[616]*y[IDX_CII]*y[IDX_CO2I] +
        k[695]*y[IDX_CI]*y[IDX_HCO2II] - k[714]*y[IDX_CHII]*y[IDX_CO2I] -
        k[741]*y[IDX_CH2II]*y[IDX_CO2I] + k[791]*y[IDX_CH3CNI]*y[IDX_HCO2II] -
        k[796]*y[IDX_CH4II]*y[IDX_CO2I] + k[812]*y[IDX_CH4I]*y[IDX_HCO2II] -
        k[825]*y[IDX_CH5II]*y[IDX_CO2I] + k[873]*y[IDX_COII]*y[IDX_SO2I] +
        k[875]*y[IDX_COI]*y[IDX_HCO2II] + k[879]*y[IDX_COI]*y[IDX_SO2II] +
        k[881]*y[IDX_COI]*y[IDX_SiOII] - k[891]*y[IDX_HII]*y[IDX_CO2I] -
        k[918]*y[IDX_H2II]*y[IDX_CO2I] + k[1004]*y[IDX_H2OI]*y[IDX_HCO2II] -
        k[1036]*y[IDX_H3II]*y[IDX_CO2I] - k[1110]*y[IDX_HCNII]*y[IDX_CO2I] -
        k[1173]*y[IDX_HNOII]*y[IDX_CO2I] - k[1209]*y[IDX_HeII]*y[IDX_CO2I] -
        k[1210]*y[IDX_HeII]*y[IDX_CO2I] - k[1211]*y[IDX_HeII]*y[IDX_CO2I] -
        k[1212]*y[IDX_HeII]*y[IDX_CO2I] - k[1296]*y[IDX_NII]*y[IDX_CO2I] -
        k[1322]*y[IDX_N2HII]*y[IDX_CO2I] - k[1350]*y[IDX_NHII]*y[IDX_CO2I] -
        k[1351]*y[IDX_NHII]*y[IDX_CO2I] - k[1352]*y[IDX_NHII]*y[IDX_CO2I] +
        k[1434]*y[IDX_NH3I]*y[IDX_HCO2II] - k[1470]*y[IDX_OII]*y[IDX_CO2I] +
        k[1479]*y[IDX_OII]*y[IDX_OCSI] - k[1489]*y[IDX_O2HII]*y[IDX_CO2I] -
        k[1520]*y[IDX_OHII]*y[IDX_CO2I] + k[1562]*y[IDX_SOII]*y[IDX_OCSI] +
        k[1632]*y[IDX_CH2I]*y[IDX_O2I] + k[1633]*y[IDX_CH2I]*y[IDX_O2I] -
        k[1677]*y[IDX_CHI]*y[IDX_CO2I] + k[1687]*y[IDX_CHI]*y[IDX_O2I] +
        k[1716]*y[IDX_COI]*y[IDX_HNOI] + k[1717]*y[IDX_COI]*y[IDX_NO2I] +
        k[1718]*y[IDX_COI]*y[IDX_O2I] + k[1719]*y[IDX_COI]*y[IDX_O2HI] -
        k[1744]*y[IDX_HI]*y[IDX_CO2I] + k[1784]*y[IDX_HCOI]*y[IDX_O2I] -
        k[1808]*y[IDX_NI]*y[IDX_CO2I] + k[1860]*y[IDX_NOI]*y[IDX_OCNI] +
        k[1863]*y[IDX_O2I]*y[IDX_OCNI] - k[1882]*y[IDX_OI]*y[IDX_CO2I] +
        k[1892]*y[IDX_OI]*y[IDX_HCOI] + k[1912]*y[IDX_OI]*y[IDX_OCSI] +
        k[1934]*y[IDX_OHI]*y[IDX_COI] - k[1956]*y[IDX_SiI]*y[IDX_CO2I] -
        k[2003]*y[IDX_CO2I] - k[2215]*y[IDX_CO2I] + k[2435]*y[IDX_GCO2I] +
        k[2436]*y[IDX_GCO2I] + k[2437]*y[IDX_GCO2I] + k[2438]*y[IDX_GCO2I];
    ydot[IDX_NHII] = 0.0 + k[132]*y[IDX_HII]*y[IDX_NHI] +
        k[168]*y[IDX_H2II]*y[IDX_NHI] + k[247]*y[IDX_NII]*y[IDX_NHI] -
        k[260]*y[IDX_NHII]*y[IDX_H2COI] - k[261]*y[IDX_NHII]*y[IDX_H2OI] -
        k[262]*y[IDX_NHII]*y[IDX_NH3I] - k[263]*y[IDX_NHII]*y[IDX_NOI] -
        k[264]*y[IDX_NHII]*y[IDX_O2I] - k[265]*y[IDX_NHII]*y[IDX_SI] +
        k[294]*y[IDX_NHI]*y[IDX_CNII] + k[295]*y[IDX_NHI]*y[IDX_COII] +
        k[296]*y[IDX_NHI]*y[IDX_N2II] + k[297]*y[IDX_NHI]*y[IDX_OII] +
        k[430]*y[IDX_NHI] - k[567]*y[IDX_NHII]*y[IDX_EM] -
        k[699]*y[IDX_CI]*y[IDX_NHII] - k[766]*y[IDX_CH2I]*y[IDX_NHII] -
        k[854]*y[IDX_CHI]*y[IDX_NHII] + k[928]*y[IDX_H2II]*y[IDX_NI] +
        k[952]*y[IDX_H2I]*y[IDX_NII] - k[954]*y[IDX_H2I]*y[IDX_NHII] -
        k[955]*y[IDX_H2I]*y[IDX_NHII] + k[1244]*y[IDX_HeII]*y[IDX_HNCI] +
        k[1253]*y[IDX_HeII]*y[IDX_NH2I] + k[1254]*y[IDX_HeII]*y[IDX_NH3I] +
        k[1301]*y[IDX_NII]*y[IDX_H2SI] + k[1303]*y[IDX_NII]*y[IDX_HCOI] -
        k[1337]*y[IDX_NI]*y[IDX_NHII] - k[1345]*y[IDX_NHII]*y[IDX_C2I] -
        k[1346]*y[IDX_NHII]*y[IDX_C2I] - k[1347]*y[IDX_NHII]*y[IDX_C2I] -
        k[1348]*y[IDX_NHII]*y[IDX_C2HI] - k[1349]*y[IDX_NHII]*y[IDX_CNI] -
        k[1350]*y[IDX_NHII]*y[IDX_CO2I] - k[1351]*y[IDX_NHII]*y[IDX_CO2I] -
        k[1352]*y[IDX_NHII]*y[IDX_CO2I] - k[1353]*y[IDX_NHII]*y[IDX_COI] -
        k[1354]*y[IDX_NHII]*y[IDX_H2COI] - k[1355]*y[IDX_NHII]*y[IDX_H2COI] -
        k[1356]*y[IDX_NHII]*y[IDX_H2OI] - k[1357]*y[IDX_NHII]*y[IDX_H2OI] -
        k[1358]*y[IDX_NHII]*y[IDX_H2OI] - k[1359]*y[IDX_NHII]*y[IDX_H2OI] -
        k[1360]*y[IDX_NHII]*y[IDX_HCNI] - k[1361]*y[IDX_NHII]*y[IDX_HCOI] -
        k[1362]*y[IDX_NHII]*y[IDX_HNCI] - k[1363]*y[IDX_NHII]*y[IDX_N2I] -
        k[1364]*y[IDX_NHII]*y[IDX_NH2I] - k[1365]*y[IDX_NHII]*y[IDX_NH3I] -
        k[1366]*y[IDX_NHII]*y[IDX_NHI] - k[1367]*y[IDX_NHII]*y[IDX_NOI] -
        k[1368]*y[IDX_NHII]*y[IDX_O2I] - k[1369]*y[IDX_NHII]*y[IDX_O2I] -
        k[1370]*y[IDX_NHII]*y[IDX_OI] - k[1371]*y[IDX_NHII]*y[IDX_OHI] -
        k[1372]*y[IDX_NHII]*y[IDX_SI] - k[1373]*y[IDX_NHII]*y[IDX_SI] -
        k[2048]*y[IDX_NHII] + k[2055]*y[IDX_NHI] - k[2232]*y[IDX_NHII];
    ydot[IDX_NH2I] = 0.0 + k[68]*y[IDX_CH2I]*y[IDX_NH2II] +
        k[89]*y[IDX_CHI]*y[IDX_NH2II] - k[130]*y[IDX_HII]*y[IDX_NH2I] -
        k[166]*y[IDX_H2II]*y[IDX_NH2I] - k[245]*y[IDX_NII]*y[IDX_NH2I] +
        k[266]*y[IDX_NH2II]*y[IDX_H2SI] + k[267]*y[IDX_NH2II]*y[IDX_HCOI] +
        k[268]*y[IDX_NH2II]*y[IDX_NH3I] + k[269]*y[IDX_NH2II]*y[IDX_NOI] +
        k[270]*y[IDX_NH2II]*y[IDX_SI] - k[271]*y[IDX_NH2I]*y[IDX_C2II] -
        k[272]*y[IDX_NH2I]*y[IDX_CNII] - k[273]*y[IDX_NH2I]*y[IDX_COII] -
        k[274]*y[IDX_NH2I]*y[IDX_H2OII] - k[275]*y[IDX_NH2I]*y[IDX_N2II] -
        k[276]*y[IDX_NH2I]*y[IDX_O2II] - k[277]*y[IDX_NH2I]*y[IDX_OHII] -
        k[315]*y[IDX_OII]*y[IDX_NH2I] - k[424]*y[IDX_NH2I] - k[425]*y[IDX_NH2I]
        + k[426]*y[IDX_NH3I] + k[570]*y[IDX_NH3II]*y[IDX_EM] +
        k[572]*y[IDX_NH4II]*y[IDX_EM] + k[573]*y[IDX_NH4II]*y[IDX_EM] -
        k[629]*y[IDX_CII]*y[IDX_NH2I] - k[729]*y[IDX_CHII]*y[IDX_NH2I] +
        k[768]*y[IDX_CH2I]*y[IDX_NH3II] - k[1056]*y[IDX_H3II]*y[IDX_NH2I] -
        k[1252]*y[IDX_HeII]*y[IDX_NH2I] - k[1253]*y[IDX_HeII]*y[IDX_NH2I] +
        k[1355]*y[IDX_NHII]*y[IDX_H2COI] - k[1364]*y[IDX_NHII]*y[IDX_NH2I] -
        k[1388]*y[IDX_NH2II]*y[IDX_NH2I] - k[1394]*y[IDX_NH2I]*y[IDX_C2II] -
        k[1395]*y[IDX_NH2I]*y[IDX_C2HII] - k[1396]*y[IDX_NH2I]*y[IDX_C2H2II] -
        k[1397]*y[IDX_NH2I]*y[IDX_CH5II] - k[1398]*y[IDX_NH2I]*y[IDX_COII] -
        k[1399]*y[IDX_NH2I]*y[IDX_H2COII] - k[1400]*y[IDX_NH2I]*y[IDX_H2OII] -
        k[1401]*y[IDX_NH2I]*y[IDX_H3COII] - k[1402]*y[IDX_NH2I]*y[IDX_H3OII] -
        k[1403]*y[IDX_NH2I]*y[IDX_HCNII] - k[1404]*y[IDX_NH2I]*y[IDX_HCNHII] -
        k[1405]*y[IDX_NH2I]*y[IDX_HCNHII] - k[1406]*y[IDX_NH2I]*y[IDX_HCOII] -
        k[1407]*y[IDX_NH2I]*y[IDX_HNOII] - k[1408]*y[IDX_NH2I]*y[IDX_N2HII] -
        k[1409]*y[IDX_NH2I]*y[IDX_NH3II] - k[1410]*y[IDX_NH2I]*y[IDX_O2HII] -
        k[1411]*y[IDX_NH2I]*y[IDX_OHII] + k[1417]*y[IDX_NH3II]*y[IDX_NH3I] +
        k[1423]*y[IDX_NH3I]*y[IDX_COII] + k[1430]*y[IDX_NH3I]*y[IDX_HCNII] -
        k[1599]*y[IDX_CI]*y[IDX_NH2I] - k[1600]*y[IDX_CI]*y[IDX_NH2I] -
        k[1601]*y[IDX_CI]*y[IDX_NH2I] - k[1656]*y[IDX_CH3I]*y[IDX_NH2I] +
        k[1657]*y[IDX_CH3I]*y[IDX_NH3I] - k[1729]*y[IDX_H2I]*y[IDX_NH2I] +
        k[1730]*y[IDX_H2I]*y[IDX_NHI] + k[1755]*y[IDX_HI]*y[IDX_HNOI] -
        k[1760]*y[IDX_HI]*y[IDX_NH2I] + k[1761]*y[IDX_HI]*y[IDX_NH3I] -
        k[1833]*y[IDX_NH2I]*y[IDX_CH4I] - k[1834]*y[IDX_NH2I]*y[IDX_NOI] -
        k[1835]*y[IDX_NH2I]*y[IDX_NOI] - k[1836]*y[IDX_NH2I]*y[IDX_OHI] -
        k[1837]*y[IDX_NH2I]*y[IDX_OHI] + k[1838]*y[IDX_NH3I]*y[IDX_CNI] +
        k[1839]*y[IDX_NHI]*y[IDX_CH4I] + k[1841]*y[IDX_NHI]*y[IDX_H2OI] +
        k[1842]*y[IDX_NHI]*y[IDX_NH3I] + k[1842]*y[IDX_NHI]*y[IDX_NH3I] +
        k[1845]*y[IDX_NHI]*y[IDX_NHI] + k[1855]*y[IDX_NHI]*y[IDX_OHI] -
        k[1902]*y[IDX_OI]*y[IDX_NH2I] - k[1903]*y[IDX_OI]*y[IDX_NH2I] +
        k[1904]*y[IDX_OI]*y[IDX_NH3I] + k[1940]*y[IDX_OHI]*y[IDX_HCNI] +
        k[1944]*y[IDX_OHI]*y[IDX_NH3I] - k[2049]*y[IDX_NH2I] -
        k[2050]*y[IDX_NH2I] + k[2051]*y[IDX_NH3I] - k[2250]*y[IDX_NH2I];
    ydot[IDX_O2II] = 0.0 - k[39]*y[IDX_C2I]*y[IDX_O2II] -
        k[54]*y[IDX_CI]*y[IDX_O2II] - k[70]*y[IDX_CH2I]*y[IDX_O2II] +
        k[79]*y[IDX_CH4II]*y[IDX_O2I] - k[91]*y[IDX_CHI]*y[IDX_O2II] +
        k[98]*y[IDX_CNII]*y[IDX_O2I] + k[105]*y[IDX_COII]*y[IDX_O2I] +
        k[135]*y[IDX_HII]*y[IDX_O2I] + k[170]*y[IDX_H2II]*y[IDX_O2I] -
        k[174]*y[IDX_H2COI]*y[IDX_O2II] + k[183]*y[IDX_H2OII]*y[IDX_O2I] +
        k[198]*y[IDX_HCNII]*y[IDX_O2I] - k[204]*y[IDX_HCOI]*y[IDX_O2II] +
        k[217]*y[IDX_HeII]*y[IDX_O2I] - k[228]*y[IDX_MgI]*y[IDX_O2II] +
        k[249]*y[IDX_NII]*y[IDX_O2I] + k[256]*y[IDX_N2II]*y[IDX_O2I] +
        k[264]*y[IDX_NHII]*y[IDX_O2I] - k[276]*y[IDX_NH2I]*y[IDX_O2II] -
        k[290]*y[IDX_NH3I]*y[IDX_O2II] - k[302]*y[IDX_NOI]*y[IDX_O2II] +
        k[317]*y[IDX_OII]*y[IDX_O2I] - k[321]*y[IDX_O2II]*y[IDX_C2H2I] -
        k[322]*y[IDX_O2II]*y[IDX_H2SI] - k[323]*y[IDX_O2II]*y[IDX_SI] +
        k[324]*y[IDX_O2I]*y[IDX_ClII] + k[325]*y[IDX_O2I]*y[IDX_SO2II] +
        k[337]*y[IDX_OHII]*y[IDX_O2I] - k[353]*y[IDX_SiI]*y[IDX_O2II] +
        k[435]*y[IDX_O2I] - k[577]*y[IDX_O2II]*y[IDX_EM] -
        k[656]*y[IDX_C2I]*y[IDX_O2II] - k[700]*y[IDX_CI]*y[IDX_O2II] -
        k[769]*y[IDX_CH2I]*y[IDX_O2II] - k[858]*y[IDX_CHI]*y[IDX_O2II] -
        k[972]*y[IDX_H2COI]*y[IDX_O2II] - k[1161]*y[IDX_HCOI]*y[IDX_O2II] +
        k[1211]*y[IDX_HeII]*y[IDX_CO2I] - k[1339]*y[IDX_NI]*y[IDX_O2II] -
        k[1458]*y[IDX_NHI]*y[IDX_O2II] + k[1470]*y[IDX_OII]*y[IDX_CO2I] +
        k[1480]*y[IDX_OII]*y[IDX_OHI] - k[1482]*y[IDX_O2II]*y[IDX_C2H2I] -
        k[1483]*y[IDX_O2II]*y[IDX_CH3OHI] - k[1484]*y[IDX_O2II]*y[IDX_SI] +
        k[1497]*y[IDX_OI]*y[IDX_H2OII] + k[1511]*y[IDX_OI]*y[IDX_OHII] -
        k[2060]*y[IDX_O2II] + k[2061]*y[IDX_O2I] - k[2229]*y[IDX_O2II];
    ydot[IDX_NII] = 0.0 - k[87]*y[IDX_CHI]*y[IDX_NII] -
        k[233]*y[IDX_NII]*y[IDX_C2I] - k[234]*y[IDX_NII]*y[IDX_C2HI] -
        k[235]*y[IDX_NII]*y[IDX_CH2I] - k[236]*y[IDX_NII]*y[IDX_CH4I] -
        k[237]*y[IDX_NII]*y[IDX_CNI] - k[238]*y[IDX_NII]*y[IDX_COI] -
        k[239]*y[IDX_NII]*y[IDX_H2COI] - k[240]*y[IDX_NII]*y[IDX_H2OI] -
        k[241]*y[IDX_NII]*y[IDX_H2SI] - k[242]*y[IDX_NII]*y[IDX_HCNI] -
        k[243]*y[IDX_NII]*y[IDX_HCOI] - k[244]*y[IDX_NII]*y[IDX_MgI] -
        k[245]*y[IDX_NII]*y[IDX_NH2I] - k[246]*y[IDX_NII]*y[IDX_NH3I] -
        k[247]*y[IDX_NII]*y[IDX_NHI] - k[248]*y[IDX_NII]*y[IDX_NOI] -
        k[249]*y[IDX_NII]*y[IDX_O2I] - k[250]*y[IDX_NII]*y[IDX_OCSI] -
        k[251]*y[IDX_NII]*y[IDX_OHI] + k[259]*y[IDX_NI]*y[IDX_N2II] +
        k[364]*y[IDX_NI] + k[422]*y[IDX_NI] - k[852]*y[IDX_CHI]*y[IDX_NII] -
        k[952]*y[IDX_H2I]*y[IDX_NII] + k[1207]*y[IDX_HeII]*y[IDX_CNI] +
        k[1232]*y[IDX_HeII]*y[IDX_HCNI] + k[1250]*y[IDX_HeII]*y[IDX_N2I] +
        k[1252]*y[IDX_HeII]*y[IDX_NH2I] + k[1256]*y[IDX_HeII]*y[IDX_NHI] +
        k[1258]*y[IDX_HeII]*y[IDX_NOI] + k[1260]*y[IDX_HeII]*y[IDX_NSI] -
        k[1289]*y[IDX_NII]*y[IDX_CH3OHI] - k[1290]*y[IDX_NII]*y[IDX_CH3OHI] -
        k[1291]*y[IDX_NII]*y[IDX_CH3OHI] - k[1292]*y[IDX_NII]*y[IDX_CH3OHI] -
        k[1293]*y[IDX_NII]*y[IDX_CH4I] - k[1294]*y[IDX_NII]*y[IDX_CH4I] -
        k[1295]*y[IDX_NII]*y[IDX_CH4I] - k[1296]*y[IDX_NII]*y[IDX_CO2I] -
        k[1297]*y[IDX_NII]*y[IDX_COI] - k[1298]*y[IDX_NII]*y[IDX_H2COI] -
        k[1299]*y[IDX_NII]*y[IDX_H2COI] - k[1300]*y[IDX_NII]*y[IDX_H2SI] -
        k[1301]*y[IDX_NII]*y[IDX_H2SI] - k[1302]*y[IDX_NII]*y[IDX_H2SI] -
        k[1303]*y[IDX_NII]*y[IDX_HCOI] - k[1304]*y[IDX_NII]*y[IDX_NCCNI] -
        k[1305]*y[IDX_NII]*y[IDX_NCCNI] - k[1306]*y[IDX_NII]*y[IDX_NH3I] -
        k[1307]*y[IDX_NII]*y[IDX_NH3I] - k[1308]*y[IDX_NII]*y[IDX_NHI] -
        k[1309]*y[IDX_NII]*y[IDX_NOI] - k[1310]*y[IDX_NII]*y[IDX_O2I] -
        k[1311]*y[IDX_NII]*y[IDX_O2I] - k[1312]*y[IDX_NII]*y[IDX_OCSI] -
        k[1313]*y[IDX_NII]*y[IDX_OCSI] - k[2127]*y[IDX_NII]*y[IDX_NI] -
        k[2141]*y[IDX_NII]*y[IDX_EM] - k[2226]*y[IDX_NII];
    ydot[IDX_NH2II] = 0.0 - k[68]*y[IDX_CH2I]*y[IDX_NH2II] -
        k[89]*y[IDX_CHI]*y[IDX_NH2II] + k[130]*y[IDX_HII]*y[IDX_NH2I] +
        k[166]*y[IDX_H2II]*y[IDX_NH2I] + k[245]*y[IDX_NII]*y[IDX_NH2I] -
        k[266]*y[IDX_NH2II]*y[IDX_H2SI] - k[267]*y[IDX_NH2II]*y[IDX_HCOI] -
        k[268]*y[IDX_NH2II]*y[IDX_NH3I] - k[269]*y[IDX_NH2II]*y[IDX_NOI] -
        k[270]*y[IDX_NH2II]*y[IDX_SI] + k[271]*y[IDX_NH2I]*y[IDX_C2II] +
        k[272]*y[IDX_NH2I]*y[IDX_CNII] + k[273]*y[IDX_NH2I]*y[IDX_COII] +
        k[274]*y[IDX_NH2I]*y[IDX_H2OII] + k[275]*y[IDX_NH2I]*y[IDX_N2II] +
        k[276]*y[IDX_NH2I]*y[IDX_O2II] + k[277]*y[IDX_NH2I]*y[IDX_OHII] +
        k[315]*y[IDX_OII]*y[IDX_NH2I] + k[424]*y[IDX_NH2I] -
        k[568]*y[IDX_NH2II]*y[IDX_EM] - k[569]*y[IDX_NH2II]*y[IDX_EM] -
        k[767]*y[IDX_CH2I]*y[IDX_NH2II] - k[855]*y[IDX_CHI]*y[IDX_NH2II] +
        k[900]*y[IDX_HII]*y[IDX_HNCOI] + k[929]*y[IDX_H2II]*y[IDX_NHI] +
        k[955]*y[IDX_H2I]*y[IDX_NHII] - k[956]*y[IDX_H2I]*y[IDX_NH2II] +
        k[1058]*y[IDX_H3II]*y[IDX_NHI] + k[1255]*y[IDX_HeII]*y[IDX_NH3I] +
        k[1307]*y[IDX_NII]*y[IDX_NH3I] - k[1338]*y[IDX_NI]*y[IDX_NH2II] +
        k[1359]*y[IDX_NHII]*y[IDX_H2OI] + k[1366]*y[IDX_NHII]*y[IDX_NHI] -
        k[1374]*y[IDX_NH2II]*y[IDX_C2I] - k[1375]*y[IDX_NH2II]*y[IDX_C2HI] -
        k[1376]*y[IDX_NH2II]*y[IDX_H2COI] - k[1377]*y[IDX_NH2II]*y[IDX_H2COI] -
        k[1378]*y[IDX_NH2II]*y[IDX_H2OI] - k[1379]*y[IDX_NH2II]*y[IDX_H2OI] -
        k[1380]*y[IDX_NH2II]*y[IDX_H2OI] - k[1381]*y[IDX_NH2II]*y[IDX_H2SI] -
        k[1382]*y[IDX_NH2II]*y[IDX_H2SI] - k[1383]*y[IDX_NH2II]*y[IDX_H2SI] -
        k[1384]*y[IDX_NH2II]*y[IDX_H2SI] - k[1385]*y[IDX_NH2II]*y[IDX_HCNI] -
        k[1386]*y[IDX_NH2II]*y[IDX_HCOI] - k[1387]*y[IDX_NH2II]*y[IDX_HNCI] -
        k[1388]*y[IDX_NH2II]*y[IDX_NH2I] - k[1389]*y[IDX_NH2II]*y[IDX_NH3I] -
        k[1390]*y[IDX_NH2II]*y[IDX_O2I] - k[1391]*y[IDX_NH2II]*y[IDX_O2I] -
        k[1392]*y[IDX_NH2II]*y[IDX_SI] - k[1393]*y[IDX_NH2II]*y[IDX_SI] +
        k[1447]*y[IDX_NHI]*y[IDX_CH5II] + k[1451]*y[IDX_NHI]*y[IDX_HCNII] +
        k[1452]*y[IDX_NHI]*y[IDX_HCOII] + k[1453]*y[IDX_NHI]*y[IDX_HNOII] +
        k[1454]*y[IDX_NHI]*y[IDX_N2HII] - k[1455]*y[IDX_NHI]*y[IDX_NH2II] +
        k[1459]*y[IDX_NHI]*y[IDX_O2HII] + k[1460]*y[IDX_NHI]*y[IDX_OHII] -
        k[1507]*y[IDX_OI]*y[IDX_NH2II] + k[2049]*y[IDX_NH2I] -
        k[2238]*y[IDX_NH2II];
    ydot[IDX_C2HII] = 0.0 - k[40]*y[IDX_C2HII]*y[IDX_NOI] -
        k[41]*y[IDX_C2HII]*y[IDX_SI] + k[47]*y[IDX_C2HI]*y[IDX_CNII] +
        k[48]*y[IDX_C2HI]*y[IDX_COII] + k[49]*y[IDX_C2HI]*y[IDX_N2II] +
        k[112]*y[IDX_HII]*y[IDX_C2HI] + k[155]*y[IDX_H2II]*y[IDX_C2HI] +
        k[177]*y[IDX_H2OII]*y[IDX_C2HI] + k[234]*y[IDX_NII]*y[IDX_C2HI] +
        k[308]*y[IDX_OII]*y[IDX_C2HI] + k[330]*y[IDX_OHII]*y[IDX_C2HI] +
        k[374]*y[IDX_C2HI] - k[459]*y[IDX_C2HII]*y[IDX_EM] -
        k[460]*y[IDX_C2HII]*y[IDX_EM] + k[608]*y[IDX_CII]*y[IDX_CH2I] +
        k[609]*y[IDX_CII]*y[IDX_CH3I] + k[648]*y[IDX_C2II]*y[IDX_HCOI] +
        k[651]*y[IDX_C2I]*y[IDX_H2COII] + k[652]*y[IDX_C2I]*y[IDX_HCNII] +
        k[653]*y[IDX_C2I]*y[IDX_HCOII] + k[654]*y[IDX_C2I]*y[IDX_HNOII] +
        k[655]*y[IDX_C2I]*y[IDX_N2HII] + k[657]*y[IDX_C2I]*y[IDX_O2HII] -
        k[660]*y[IDX_C2HII]*y[IDX_H2COI] - k[661]*y[IDX_C2HII]*y[IDX_HCNI] -
        k[662]*y[IDX_C2HII]*y[IDX_HCNI] - k[663]*y[IDX_C2HII]*y[IDX_HCOI] -
        k[664]*y[IDX_C2HII]*y[IDX_HNCI] - k[685]*y[IDX_CI]*y[IDX_C2HII] +
        k[687]*y[IDX_CI]*y[IDX_CH2II] + k[688]*y[IDX_CI]*y[IDX_CH3II] +
        k[707]*y[IDX_CHII]*y[IDX_CH2I] - k[754]*y[IDX_CH2I]*y[IDX_C2HII] +
        k[803]*y[IDX_CH4I]*y[IDX_C2II] - k[805]*y[IDX_CH4I]*y[IDX_C2HII] +
        k[823]*y[IDX_CH5II]*y[IDX_C2I] - k[838]*y[IDX_CHI]*y[IDX_C2HII] +
        k[909]*y[IDX_H2II]*y[IDX_C2I] + k[935]*y[IDX_H2I]*y[IDX_C2II] -
        k[936]*y[IDX_H2I]*y[IDX_C2HII] + k[976]*y[IDX_H2OII]*y[IDX_C2I] +
        k[990]*y[IDX_H2OI]*y[IDX_C2II] + k[1022]*y[IDX_H3II]*y[IDX_C2I] +
        k[1080]*y[IDX_H3OII]*y[IDX_C2I] + k[1179]*y[IDX_HeII]*y[IDX_C2H2I] +
        k[1181]*y[IDX_HeII]*y[IDX_C2H3I] + k[1183]*y[IDX_HeII]*y[IDX_C2H4I] +
        k[1191]*y[IDX_HeII]*y[IDX_C4HI] + k[1230]*y[IDX_HeII]*y[IDX_HC3NI] -
        k[1326]*y[IDX_NI]*y[IDX_C2HII] - k[1327]*y[IDX_NI]*y[IDX_C2HII] +
        k[1345]*y[IDX_NHII]*y[IDX_C2I] + k[1374]*y[IDX_NH2II]*y[IDX_C2I] -
        k[1395]*y[IDX_NH2I]*y[IDX_C2HII] - k[1418]*y[IDX_NH3I]*y[IDX_C2HII] +
        k[1444]*y[IDX_NHI]*y[IDX_C2II] - k[1491]*y[IDX_OI]*y[IDX_C2HII] +
        k[1517]*y[IDX_OHII]*y[IDX_C2I] - k[1963]*y[IDX_C2HII] +
        k[1971]*y[IDX_C2HI] - k[2242]*y[IDX_C2HII];
    ydot[IDX_NH4II] = 0.0 - k[572]*y[IDX_NH4II]*y[IDX_EM] -
        k[573]*y[IDX_NH4II]*y[IDX_EM] - k[574]*y[IDX_NH4II]*y[IDX_EM] +
        k[730]*y[IDX_CHII]*y[IDX_NH3I] + k[748]*y[IDX_CH2II]*y[IDX_NH3I] +
        k[781]*y[IDX_CH3II]*y[IDX_NH3I] + k[792]*y[IDX_CH3OH2II]*y[IDX_NH3I] +
        k[801]*y[IDX_CH4II]*y[IDX_NH3I] + k[818]*y[IDX_CH4I]*y[IDX_NH3II] +
        k[856]*y[IDX_CHI]*y[IDX_NH3II] + k[957]*y[IDX_H2I]*y[IDX_NH3II] +
        k[1057]*y[IDX_H3II]*y[IDX_NH3I] + k[1365]*y[IDX_NHII]*y[IDX_NH3I] +
        k[1380]*y[IDX_NH2II]*y[IDX_H2OI] + k[1384]*y[IDX_NH2II]*y[IDX_H2SI] +
        k[1389]*y[IDX_NH2II]*y[IDX_NH3I] + k[1409]*y[IDX_NH2I]*y[IDX_NH3II] +
        k[1413]*y[IDX_NH3II]*y[IDX_H2COI] + k[1414]*y[IDX_NH3II]*y[IDX_H2OI] +
        k[1415]*y[IDX_NH3II]*y[IDX_H2SI] + k[1416]*y[IDX_NH3II]*y[IDX_HCOI] +
        k[1417]*y[IDX_NH3II]*y[IDX_NH3I] + k[1418]*y[IDX_NH3I]*y[IDX_C2HII] +
        k[1419]*y[IDX_NH3I]*y[IDX_C2H2II] + k[1420]*y[IDX_NH3I]*y[IDX_C2H5OH2II]
        + k[1422]*y[IDX_NH3I]*y[IDX_CH5II] + k[1424]*y[IDX_NH3I]*y[IDX_H2COII] +
        k[1425]*y[IDX_NH3I]*y[IDX_H2OII] + k[1426]*y[IDX_NH3I]*y[IDX_H2SII] +
        k[1427]*y[IDX_NH3I]*y[IDX_H3COII] + k[1428]*y[IDX_NH3I]*y[IDX_H3OII] +
        k[1429]*y[IDX_NH3I]*y[IDX_H3SII] + k[1431]*y[IDX_NH3I]*y[IDX_HCNHII] +
        k[1432]*y[IDX_NH3I]*y[IDX_HCNHII] + k[1433]*y[IDX_NH3I]*y[IDX_HCOII] +
        k[1434]*y[IDX_NH3I]*y[IDX_HCO2II] + k[1435]*y[IDX_NH3I]*y[IDX_HCSII] +
        k[1436]*y[IDX_NH3I]*y[IDX_HNOII] + k[1437]*y[IDX_NH3I]*y[IDX_HSII] +
        k[1438]*y[IDX_NH3I]*y[IDX_HSO2II] + k[1439]*y[IDX_NH3I]*y[IDX_HSiSII] +
        k[1440]*y[IDX_NH3I]*y[IDX_N2HII] + k[1441]*y[IDX_NH3I]*y[IDX_O2HII] +
        k[1442]*y[IDX_NH3I]*y[IDX_SiHII] + k[1443]*y[IDX_NH3I]*y[IDX_SiOHII] +
        k[1456]*y[IDX_NHI]*y[IDX_NH3II] + k[1530]*y[IDX_OHII]*y[IDX_NH3I] +
        k[1546]*y[IDX_OHI]*y[IDX_NH3II] - k[2288]*y[IDX_NH4II];
    ydot[IDX_C2HI] = 0.0 + k[40]*y[IDX_C2HII]*y[IDX_NOI] +
        k[41]*y[IDX_C2HII]*y[IDX_SI] - k[47]*y[IDX_C2HI]*y[IDX_CNII] -
        k[48]*y[IDX_C2HI]*y[IDX_COII] - k[49]*y[IDX_C2HI]*y[IDX_N2II] -
        k[112]*y[IDX_HII]*y[IDX_C2HI] - k[155]*y[IDX_H2II]*y[IDX_C2HI] -
        k[177]*y[IDX_H2OII]*y[IDX_C2HI] - k[234]*y[IDX_NII]*y[IDX_C2HI] -
        k[308]*y[IDX_OII]*y[IDX_C2HI] - k[330]*y[IDX_OHII]*y[IDX_C2HI] +
        k[368]*y[IDX_C2H2I] - k[373]*y[IDX_C2HI] - k[374]*y[IDX_C2HI] +
        k[378]*y[IDX_C4HI] + k[407]*y[IDX_HC3NI] +
        k[462]*y[IDX_C2H2II]*y[IDX_EM] - k[607]*y[IDX_CII]*y[IDX_C2HI] +
        k[623]*y[IDX_CII]*y[IDX_HC3NI] + k[666]*y[IDX_C2H2II]*y[IDX_CH3CNI] +
        k[667]*y[IDX_C2H2II]*y[IDX_H2SI] + k[668]*y[IDX_C2H2II]*y[IDX_HCNI] +
        k[669]*y[IDX_C2H2II]*y[IDX_HNCI] - k[677]*y[IDX_C2HI]*y[IDX_COII] -
        k[678]*y[IDX_C2HI]*y[IDX_H2COII] - k[679]*y[IDX_C2HI]*y[IDX_HCNII] -
        k[680]*y[IDX_C2HI]*y[IDX_HCOII] - k[681]*y[IDX_C2HI]*y[IDX_HNOII] -
        k[682]*y[IDX_C2HI]*y[IDX_N2HII] - k[683]*y[IDX_C2HI]*y[IDX_O2HII] -
        k[684]*y[IDX_C2HI]*y[IDX_SiII] - k[706]*y[IDX_CHII]*y[IDX_C2HI] -
        k[824]*y[IDX_CH5II]*y[IDX_C2HI] - k[884]*y[IDX_HII]*y[IDX_C2HI] -
        k[911]*y[IDX_H2II]*y[IDX_C2HI] - k[977]*y[IDX_H2OII]*y[IDX_C2HI] +
        k[991]*y[IDX_H2OI]*y[IDX_C2H2II] - k[1025]*y[IDX_H3II]*y[IDX_C2HI] -
        k[1186]*y[IDX_HeII]*y[IDX_C2HI] - k[1187]*y[IDX_HeII]*y[IDX_C2HI] -
        k[1188]*y[IDX_HeII]*y[IDX_C2HI] - k[1348]*y[IDX_NHII]*y[IDX_C2HI] -
        k[1375]*y[IDX_NH2II]*y[IDX_C2HI] + k[1396]*y[IDX_NH2I]*y[IDX_C2H2II] +
        k[1419]*y[IDX_NH3I]*y[IDX_C2H2II] - k[1465]*y[IDX_OII]*y[IDX_C2HI] -
        k[1518]*y[IDX_OHII]*y[IDX_C2HI] - k[1578]*y[IDX_C2HI]*y[IDX_HCNI] -
        k[1579]*y[IDX_C2HI]*y[IDX_HNCI] - k[1580]*y[IDX_C2HI]*y[IDX_NCCNI] -
        k[1581]*y[IDX_C2HI]*y[IDX_O2I] + k[1586]*y[IDX_CI]*y[IDX_CH2I] -
        k[1721]*y[IDX_H2I]*y[IDX_C2HI] + k[1737]*y[IDX_HI]*y[IDX_C2H2I] -
        k[1795]*y[IDX_NI]*y[IDX_C2HI] - k[1875]*y[IDX_OI]*y[IDX_C2HI] +
        k[1927]*y[IDX_OHI]*y[IDX_C2H2I] + k[1965]*y[IDX_C2H2I] -
        k[1970]*y[IDX_C2HI] - k[1971]*y[IDX_C2HI] + k[1975]*y[IDX_C4HI] +
        k[2027]*y[IDX_HC3NI] - k[2100]*y[IDX_C2HI]*y[IDX_CNI] -
        k[2224]*y[IDX_C2HI] + k[2323]*y[IDX_GC2HI] + k[2324]*y[IDX_GC2HI] +
        k[2325]*y[IDX_GC2HI] + k[2326]*y[IDX_GC2HI];
    ydot[IDX_OII] = 0.0 - k[69]*y[IDX_CH2I]*y[IDX_OII] -
        k[90]*y[IDX_CHI]*y[IDX_OII] + k[136]*y[IDX_HII]*y[IDX_OI] -
        k[196]*y[IDX_HI]*y[IDX_OII] - k[297]*y[IDX_NHI]*y[IDX_OII] -
        k[306]*y[IDX_OII]*y[IDX_C2I] - k[307]*y[IDX_OII]*y[IDX_C2H2I] -
        k[308]*y[IDX_OII]*y[IDX_C2HI] - k[309]*y[IDX_OII]*y[IDX_CH4I] -
        k[310]*y[IDX_OII]*y[IDX_COI] - k[311]*y[IDX_OII]*y[IDX_H2COI] -
        k[312]*y[IDX_OII]*y[IDX_H2OI] - k[313]*y[IDX_OII]*y[IDX_H2SI] -
        k[314]*y[IDX_OII]*y[IDX_HCOI] - k[315]*y[IDX_OII]*y[IDX_NH2I] -
        k[316]*y[IDX_OII]*y[IDX_NH3I] - k[317]*y[IDX_OII]*y[IDX_O2I] -
        k[318]*y[IDX_OII]*y[IDX_OCSI] - k[319]*y[IDX_OII]*y[IDX_OHI] -
        k[320]*y[IDX_OII]*y[IDX_SO2I] + k[326]*y[IDX_OI]*y[IDX_CNII] +
        k[327]*y[IDX_OI]*y[IDX_COII] + k[328]*y[IDX_OI]*y[IDX_N2II] +
        k[365]*y[IDX_OI] + k[438]*y[IDX_OI] + k[634]*y[IDX_CII]*y[IDX_O2I] +
        k[734]*y[IDX_CHII]*y[IDX_O2I] - k[857]*y[IDX_CHI]*y[IDX_OII] -
        k[958]*y[IDX_H2I]*y[IDX_OII] + k[1210]*y[IDX_HeII]*y[IDX_CO2I] +
        k[1257]*y[IDX_HeII]*y[IDX_NOI] + k[1261]*y[IDX_HeII]*y[IDX_O2I] +
        k[1263]*y[IDX_HeII]*y[IDX_OCNI] + k[1265]*y[IDX_HeII]*y[IDX_OCSI] +
        k[1268]*y[IDX_HeII]*y[IDX_OHI] + k[1273]*y[IDX_HeII]*y[IDX_SOI] +
        k[1286]*y[IDX_HeII]*y[IDX_SiOI] + k[1311]*y[IDX_NII]*y[IDX_O2I] -
        k[1457]*y[IDX_NHI]*y[IDX_OII] - k[1463]*y[IDX_OII]*y[IDX_C2I] -
        k[1464]*y[IDX_OII]*y[IDX_C2H4I] - k[1465]*y[IDX_OII]*y[IDX_C2HI] -
        k[1466]*y[IDX_OII]*y[IDX_CH3OHI] - k[1467]*y[IDX_OII]*y[IDX_CH3OHI] -
        k[1468]*y[IDX_OII]*y[IDX_CH4I] - k[1469]*y[IDX_OII]*y[IDX_CNI] -
        k[1470]*y[IDX_OII]*y[IDX_CO2I] - k[1471]*y[IDX_OII]*y[IDX_H2COI] -
        k[1472]*y[IDX_OII]*y[IDX_H2SI] - k[1473]*y[IDX_OII]*y[IDX_H2SI] -
        k[1474]*y[IDX_OII]*y[IDX_HCNI] - k[1475]*y[IDX_OII]*y[IDX_HCNI] -
        k[1476]*y[IDX_OII]*y[IDX_HCOI] - k[1477]*y[IDX_OII]*y[IDX_N2I] -
        k[1478]*y[IDX_OII]*y[IDX_NO2I] - k[1479]*y[IDX_OII]*y[IDX_OCSI] -
        k[1480]*y[IDX_OII]*y[IDX_OHI] - k[1481]*y[IDX_OII]*y[IDX_SO2I] +
        k[2060]*y[IDX_O2II] + k[2068]*y[IDX_OHII] - k[2103]*y[IDX_CI]*y[IDX_OII]
        - k[2142]*y[IDX_OII]*y[IDX_EM] - k[2227]*y[IDX_OII];
    ydot[IDX_NOII] = 0.0 + k[22]*y[IDX_CII]*y[IDX_NOI] +
        k[34]*y[IDX_C2II]*y[IDX_NOI] + k[40]*y[IDX_C2HII]*y[IDX_NOI] +
        k[45]*y[IDX_C2H2II]*y[IDX_NOI] + k[58]*y[IDX_CHII]*y[IDX_NOI] +
        k[61]*y[IDX_CH2II]*y[IDX_NOI] + k[74]*y[IDX_CH3II]*y[IDX_NOI] +
        k[97]*y[IDX_CNII]*y[IDX_NOI] + k[104]*y[IDX_COII]*y[IDX_NOI] +
        k[133]*y[IDX_HII]*y[IDX_NOI] + k[169]*y[IDX_H2II]*y[IDX_NOI] +
        k[182]*y[IDX_H2OII]*y[IDX_NOI] + k[197]*y[IDX_HCNII]*y[IDX_NOI] -
        k[227]*y[IDX_MgI]*y[IDX_NOII] + k[248]*y[IDX_NII]*y[IDX_NOI] +
        k[255]*y[IDX_N2II]*y[IDX_NOI] + k[263]*y[IDX_NHII]*y[IDX_NOI] +
        k[269]*y[IDX_NH2II]*y[IDX_NOI] + k[280]*y[IDX_NH3II]*y[IDX_NOI] +
        k[298]*y[IDX_NOI]*y[IDX_H2COII] + k[299]*y[IDX_NOI]*y[IDX_H2SII] +
        k[300]*y[IDX_NOI]*y[IDX_HNOII] + k[301]*y[IDX_NOI]*y[IDX_HSII] +
        k[302]*y[IDX_NOI]*y[IDX_O2II] + k[303]*y[IDX_NOI]*y[IDX_SII] +
        k[304]*y[IDX_NOI]*y[IDX_S2II] + k[305]*y[IDX_NOI]*y[IDX_SiOII] +
        k[336]*y[IDX_OHII]*y[IDX_NOI] - k[352]*y[IDX_SiI]*y[IDX_NOII] +
        k[432]*y[IDX_NOI] - k[575]*y[IDX_NOII]*y[IDX_EM] +
        k[868]*y[IDX_CNII]*y[IDX_O2I] + k[901]*y[IDX_HII]*y[IDX_HNOI] +
        k[903]*y[IDX_HII]*y[IDX_NO2I] + k[1059]*y[IDX_H3II]*y[IDX_NO2I] +
        k[1245]*y[IDX_HeII]*y[IDX_HNOI] + k[1291]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[1297]*y[IDX_NII]*y[IDX_COI] + k[1299]*y[IDX_NII]*y[IDX_H2COI] +
        k[1310]*y[IDX_NII]*y[IDX_O2I] + k[1334]*y[IDX_NI]*y[IDX_H2OII] +
        k[1339]*y[IDX_NI]*y[IDX_O2II] + k[1340]*y[IDX_NI]*y[IDX_OHII] +
        k[1343]*y[IDX_NI]*y[IDX_SiOII] + k[1352]*y[IDX_NHII]*y[IDX_CO2I] +
        k[1368]*y[IDX_NHII]*y[IDX_O2I] + k[1457]*y[IDX_NHI]*y[IDX_OII] +
        k[1469]*y[IDX_OII]*y[IDX_CNI] + k[1475]*y[IDX_OII]*y[IDX_HCNI] +
        k[1477]*y[IDX_OII]*y[IDX_N2I] + k[1478]*y[IDX_OII]*y[IDX_NO2I] +
        k[1505]*y[IDX_OI]*y[IDX_N2II] + k[1509]*y[IDX_OI]*y[IDX_NSII] +
        k[2057]*y[IDX_NOI] - k[2236]*y[IDX_NOII];
    ydot[IDX_H2COII] = 0.0 + k[16]*y[IDX_CII]*y[IDX_H2COI] +
        k[42]*y[IDX_C2H2II]*y[IDX_H2COI] - k[65]*y[IDX_CH2I]*y[IDX_H2COII] +
        k[76]*y[IDX_CH4II]*y[IDX_H2COI] - k[85]*y[IDX_CHI]*y[IDX_H2COII] +
        k[94]*y[IDX_CNII]*y[IDX_H2COI] + k[101]*y[IDX_COII]*y[IDX_H2COI] +
        k[119]*y[IDX_HII]*y[IDX_H2COI] + k[161]*y[IDX_H2II]*y[IDX_H2COI] -
        k[173]*y[IDX_H2COII]*y[IDX_SI] + k[174]*y[IDX_H2COI]*y[IDX_O2II] +
        k[178]*y[IDX_H2OII]*y[IDX_H2COI] - k[202]*y[IDX_HCOI]*y[IDX_H2COII] +
        k[212]*y[IDX_HeII]*y[IDX_H2COI] - k[222]*y[IDX_MgI]*y[IDX_H2COII] +
        k[239]*y[IDX_NII]*y[IDX_H2COI] + k[252]*y[IDX_N2II]*y[IDX_H2COI] +
        k[260]*y[IDX_NHII]*y[IDX_H2COI] - k[284]*y[IDX_NH3I]*y[IDX_H2COII] -
        k[298]*y[IDX_NOI]*y[IDX_H2COII] + k[311]*y[IDX_OII]*y[IDX_H2COI] +
        k[331]*y[IDX_OHII]*y[IDX_H2COI] - k[349]*y[IDX_SiI]*y[IDX_H2COII] -
        k[502]*y[IDX_H2COII]*y[IDX_EM] - k[503]*y[IDX_H2COII]*y[IDX_EM] -
        k[504]*y[IDX_H2COII]*y[IDX_EM] - k[505]*y[IDX_H2COII]*y[IDX_EM] -
        k[651]*y[IDX_C2I]*y[IDX_H2COII] - k[678]*y[IDX_C2HI]*y[IDX_H2COII] +
        k[718]*y[IDX_CHII]*y[IDX_H2OI] + k[741]*y[IDX_CH2II]*y[IDX_CO2I] -
        k[757]*y[IDX_CH2I]*y[IDX_H2COII] + k[769]*y[IDX_CH2I]*y[IDX_O2II] +
        k[783]*y[IDX_CH3II]*y[IDX_OI] + k[786]*y[IDX_CH3II]*y[IDX_OHI] -
        k[809]*y[IDX_CH4I]*y[IDX_H2COII] + k[831]*y[IDX_CH5II]*y[IDX_HCOI] -
        k[842]*y[IDX_CHI]*y[IDX_H2COII] - k[965]*y[IDX_H2COII]*y[IDX_CH3OHI] -
        k[966]*y[IDX_H2COII]*y[IDX_H2COI] - k[967]*y[IDX_H2COII]*y[IDX_O2I] -
        k[968]*y[IDX_H2COII]*y[IDX_SI] + k[985]*y[IDX_H2OII]*y[IDX_HCOI] -
        k[998]*y[IDX_H2OI]*y[IDX_H2COII] + k[1046]*y[IDX_H3II]*y[IDX_HCOI] +
        k[1114]*y[IDX_HCNII]*y[IDX_HCOI] - k[1121]*y[IDX_HCNI]*y[IDX_H2COII] +
        k[1144]*y[IDX_HCOII]*y[IDX_HCOI] - k[1158]*y[IDX_HCOI]*y[IDX_H2COII] +
        k[1159]*y[IDX_HCOI]*y[IDX_HNOII] + k[1160]*y[IDX_HCOI]*y[IDX_N2HII] +
        k[1162]*y[IDX_HCOI]*y[IDX_O2HII] - k[1165]*y[IDX_HNCI]*y[IDX_H2COII] +
        k[1289]*y[IDX_NII]*y[IDX_CH3OHI] + k[1361]*y[IDX_NHII]*y[IDX_HCOI] +
        k[1386]*y[IDX_NH2II]*y[IDX_HCOI] - k[1399]*y[IDX_NH2I]*y[IDX_H2COII] -
        k[1424]*y[IDX_NH3I]*y[IDX_H2COII] - k[1449]*y[IDX_NHI]*y[IDX_H2COII] +
        k[1466]*y[IDX_OII]*y[IDX_CH3OHI] + k[1527]*y[IDX_OHII]*y[IDX_HCOI] +
        k[2013]*y[IDX_H2COI] - k[2136]*y[IDX_H2COII]*y[IDX_EM] -
        k[2244]*y[IDX_H2COII];
    ydot[IDX_CH2II] = 0.0 + k[14]*y[IDX_CII]*y[IDX_CH2I] -
        k[61]*y[IDX_CH2II]*y[IDX_NOI] + k[62]*y[IDX_CH2I]*y[IDX_C2II] +
        k[63]*y[IDX_CH2I]*y[IDX_CNII] + k[64]*y[IDX_CH2I]*y[IDX_COII] +
        k[65]*y[IDX_CH2I]*y[IDX_H2COII] + k[66]*y[IDX_CH2I]*y[IDX_H2OII] +
        k[67]*y[IDX_CH2I]*y[IDX_N2II] + k[68]*y[IDX_CH2I]*y[IDX_NH2II] +
        k[69]*y[IDX_CH2I]*y[IDX_OII] + k[70]*y[IDX_CH2I]*y[IDX_O2II] +
        k[71]*y[IDX_CH2I]*y[IDX_OHII] + k[114]*y[IDX_HII]*y[IDX_CH2I] +
        k[156]*y[IDX_H2II]*y[IDX_CH2I] + k[235]*y[IDX_NII]*y[IDX_CH2I] +
        k[381]*y[IDX_CH2I] - k[478]*y[IDX_CH2II]*y[IDX_EM] -
        k[479]*y[IDX_CH2II]*y[IDX_EM] - k[480]*y[IDX_CH2II]*y[IDX_EM] +
        k[617]*y[IDX_CII]*y[IDX_H2COI] + k[619]*y[IDX_CII]*y[IDX_H2CSI] -
        k[687]*y[IDX_CI]*y[IDX_CH2II] + k[726]*y[IDX_CHII]*y[IDX_HCOI] -
        k[741]*y[IDX_CH2II]*y[IDX_CO2I] - k[742]*y[IDX_CH2II]*y[IDX_H2COI] -
        k[743]*y[IDX_CH2II]*y[IDX_H2OI] - k[744]*y[IDX_CH2II]*y[IDX_H2SI] -
        k[745]*y[IDX_CH2II]*y[IDX_H2SI] - k[746]*y[IDX_CH2II]*y[IDX_H2SI] -
        k[747]*y[IDX_CH2II]*y[IDX_HCOI] - k[748]*y[IDX_CH2II]*y[IDX_NH3I] -
        k[749]*y[IDX_CH2II]*y[IDX_O2I] - k[750]*y[IDX_CH2II]*y[IDX_OI] -
        k[751]*y[IDX_CH2II]*y[IDX_OCSI] - k[752]*y[IDX_CH2II]*y[IDX_OCSI] -
        k[753]*y[IDX_CH2II]*y[IDX_SI] + k[815]*y[IDX_CH4I]*y[IDX_N2II] +
        k[838]*y[IDX_CHI]*y[IDX_C2HII] + k[840]*y[IDX_CHI]*y[IDX_CH5II] +
        k[842]*y[IDX_CHI]*y[IDX_H2COII] + k[843]*y[IDX_CHI]*y[IDX_H2OII] +
        k[844]*y[IDX_CHI]*y[IDX_H3COII] + k[845]*y[IDX_CHI]*y[IDX_H3OII] +
        k[846]*y[IDX_CHI]*y[IDX_HCNII] + k[847]*y[IDX_CHI]*y[IDX_HCNHII] +
        k[848]*y[IDX_CHI]*y[IDX_HCNHII] + k[849]*y[IDX_CHI]*y[IDX_HCOII] +
        k[850]*y[IDX_CHI]*y[IDX_HNOII] + k[851]*y[IDX_CHI]*y[IDX_HSII] +
        k[853]*y[IDX_CHI]*y[IDX_N2HII] + k[854]*y[IDX_CHI]*y[IDX_NHII] +
        k[855]*y[IDX_CHI]*y[IDX_NH2II] + k[859]*y[IDX_CHI]*y[IDX_O2HII] +
        k[860]*y[IDX_CHI]*y[IDX_OHII] + k[863]*y[IDX_CHI]*y[IDX_SiHII] +
        k[916]*y[IDX_H2II]*y[IDX_CHI] + k[937]*y[IDX_H2I]*y[IDX_CHII] -
        k[938]*y[IDX_H2I]*y[IDX_CH2II] + k[1034]*y[IDX_H3II]*y[IDX_CHI] -
        k[1099]*y[IDX_HI]*y[IDX_CH2II] + k[1100]*y[IDX_HI]*y[IDX_CH3II] +
        k[1185]*y[IDX_HeII]*y[IDX_C2H4I] + k[1195]*y[IDX_HeII]*y[IDX_CH2COI] +
        k[1203]*y[IDX_HeII]*y[IDX_CH4I] + k[1218]*y[IDX_HeII]*y[IDX_H2COI] +
        k[1221]*y[IDX_HeII]*y[IDX_H2CSI] - k[1331]*y[IDX_NI]*y[IDX_CH2II] -
        k[1978]*y[IDX_CH2II] - k[1979]*y[IDX_CH2II] - k[1980]*y[IDX_CH2II] +
        k[1981]*y[IDX_CH2I] + k[1985]*y[IDX_CH3II] + k[1993]*y[IDX_CH4II] +
        k[2112]*y[IDX_H2I]*y[IDX_CII] - k[2237]*y[IDX_CH2II];
    ydot[IDX_NH3II] = 0.0 + k[21]*y[IDX_CII]*y[IDX_NH3I] +
        k[57]*y[IDX_CHII]*y[IDX_NH3I] + k[78]*y[IDX_CH4II]*y[IDX_NH3I] +
        k[131]*y[IDX_HII]*y[IDX_NH3I] + k[167]*y[IDX_H2II]*y[IDX_NH3I] +
        k[216]*y[IDX_HeII]*y[IDX_NH3I] + k[246]*y[IDX_NII]*y[IDX_NH3I] +
        k[262]*y[IDX_NHII]*y[IDX_NH3I] + k[268]*y[IDX_NH2II]*y[IDX_NH3I] -
        k[278]*y[IDX_NH3II]*y[IDX_HCOI] - k[279]*y[IDX_NH3II]*y[IDX_MgI] -
        k[280]*y[IDX_NH3II]*y[IDX_NOI] - k[281]*y[IDX_NH3II]*y[IDX_SiI] +
        k[282]*y[IDX_NH3I]*y[IDX_C2H2II] + k[283]*y[IDX_NH3I]*y[IDX_COII] +
        k[284]*y[IDX_NH3I]*y[IDX_H2COII] + k[285]*y[IDX_NH3I]*y[IDX_H2OII] +
        k[286]*y[IDX_NH3I]*y[IDX_H2SII] + k[287]*y[IDX_NH3I]*y[IDX_HCNII] +
        k[288]*y[IDX_NH3I]*y[IDX_HSII] + k[289]*y[IDX_NH3I]*y[IDX_N2II] +
        k[290]*y[IDX_NH3I]*y[IDX_O2II] + k[291]*y[IDX_NH3I]*y[IDX_OCSII] +
        k[292]*y[IDX_NH3I]*y[IDX_SII] + k[293]*y[IDX_NH3I]*y[IDX_SOII] +
        k[316]*y[IDX_OII]*y[IDX_NH3I] + k[335]*y[IDX_OHII]*y[IDX_NH3I] +
        k[427]*y[IDX_NH3I] - k[570]*y[IDX_NH3II]*y[IDX_EM] -
        k[571]*y[IDX_NH3II]*y[IDX_EM] - k[768]*y[IDX_CH2I]*y[IDX_NH3II] -
        k[818]*y[IDX_CH4I]*y[IDX_NH3II] - k[856]*y[IDX_CHI]*y[IDX_NH3II] +
        k[956]*y[IDX_H2I]*y[IDX_NH2II] - k[957]*y[IDX_H2I]*y[IDX_NH3II] +
        k[1056]*y[IDX_H3II]*y[IDX_NH2I] + k[1358]*y[IDX_NHII]*y[IDX_H2OI] +
        k[1364]*y[IDX_NHII]*y[IDX_NH2I] + k[1377]*y[IDX_NH2II]*y[IDX_H2COI] +
        k[1379]*y[IDX_NH2II]*y[IDX_H2OI] + k[1383]*y[IDX_NH2II]*y[IDX_H2SI] +
        k[1388]*y[IDX_NH2II]*y[IDX_NH2I] + k[1395]*y[IDX_NH2I]*y[IDX_C2HII] +
        k[1396]*y[IDX_NH2I]*y[IDX_C2H2II] + k[1397]*y[IDX_NH2I]*y[IDX_CH5II] +
        k[1399]*y[IDX_NH2I]*y[IDX_H2COII] + k[1400]*y[IDX_NH2I]*y[IDX_H2OII] +
        k[1401]*y[IDX_NH2I]*y[IDX_H3COII] + k[1402]*y[IDX_NH2I]*y[IDX_H3OII] +
        k[1403]*y[IDX_NH2I]*y[IDX_HCNII] + k[1404]*y[IDX_NH2I]*y[IDX_HCNHII] +
        k[1405]*y[IDX_NH2I]*y[IDX_HCNHII] + k[1406]*y[IDX_NH2I]*y[IDX_HCOII] +
        k[1407]*y[IDX_NH2I]*y[IDX_HNOII] + k[1408]*y[IDX_NH2I]*y[IDX_N2HII] -
        k[1409]*y[IDX_NH2I]*y[IDX_NH3II] + k[1410]*y[IDX_NH2I]*y[IDX_O2HII] +
        k[1411]*y[IDX_NH2I]*y[IDX_OHII] - k[1412]*y[IDX_NH3II]*y[IDX_C2I] -
        k[1413]*y[IDX_NH3II]*y[IDX_H2COI] - k[1414]*y[IDX_NH3II]*y[IDX_H2OI] -
        k[1415]*y[IDX_NH3II]*y[IDX_H2SI] - k[1416]*y[IDX_NH3II]*y[IDX_HCOI] -
        k[1417]*y[IDX_NH3II]*y[IDX_NH3I] + k[1455]*y[IDX_NHI]*y[IDX_NH2II] -
        k[1456]*y[IDX_NHI]*y[IDX_NH3II] - k[1508]*y[IDX_OI]*y[IDX_NH3II] -
        k[1546]*y[IDX_OHI]*y[IDX_NH3II] + k[2052]*y[IDX_NH3I] -
        k[2243]*y[IDX_NH3II];
    ydot[IDX_H2SI] = 0.0 - k[17]*y[IDX_CII]*y[IDX_H2SI] -
        k[43]*y[IDX_C2H2II]*y[IDX_H2SI] - k[77]*y[IDX_CH4II]*y[IDX_H2SI] -
        k[102]*y[IDX_COII]*y[IDX_H2SI] - k[123]*y[IDX_HII]*y[IDX_H2SI] -
        k[163]*y[IDX_H2II]*y[IDX_H2SI] - k[179]*y[IDX_H2OII]*y[IDX_H2SI] -
        k[190]*y[IDX_H2SI]*y[IDX_OCSII] + k[203]*y[IDX_HCOI]*y[IDX_H2SII] -
        k[214]*y[IDX_HeII]*y[IDX_H2SI] + k[223]*y[IDX_MgI]*y[IDX_H2SII] -
        k[241]*y[IDX_NII]*y[IDX_H2SI] - k[253]*y[IDX_N2II]*y[IDX_H2SI] -
        k[266]*y[IDX_NH2II]*y[IDX_H2SI] + k[286]*y[IDX_NH3I]*y[IDX_H2SII] +
        k[299]*y[IDX_NOI]*y[IDX_H2SII] - k[313]*y[IDX_OII]*y[IDX_H2SI] -
        k[322]*y[IDX_O2II]*y[IDX_H2SI] - k[333]*y[IDX_OHII]*y[IDX_H2SI] +
        k[346]*y[IDX_SI]*y[IDX_H2SII] + k[350]*y[IDX_SiI]*y[IDX_H2SII] -
        k[403]*y[IDX_H2SI] - k[404]*y[IDX_H2SI] + k[532]*y[IDX_H3SII]*y[IDX_EM]
        - k[622]*y[IDX_CII]*y[IDX_H2SI] - k[667]*y[IDX_C2H2II]*y[IDX_H2SI] -
        k[721]*y[IDX_CHII]*y[IDX_H2SI] - k[722]*y[IDX_CHII]*y[IDX_H2SI] -
        k[744]*y[IDX_CH2II]*y[IDX_H2SI] - k[745]*y[IDX_CH2II]*y[IDX_H2SI] -
        k[746]*y[IDX_CH2II]*y[IDX_H2SI] - k[778]*y[IDX_CH3II]*y[IDX_H2SI] -
        k[800]*y[IDX_CH4II]*y[IDX_H2SI] - k[829]*y[IDX_CH5II]*y[IDX_H2SI] -
        k[872]*y[IDX_COII]*y[IDX_H2SI] - k[894]*y[IDX_HII]*y[IDX_H2SI] -
        k[895]*y[IDX_HII]*y[IDX_H2SI] - k[923]*y[IDX_H2II]*y[IDX_H2SI] -
        k[924]*y[IDX_H2II]*y[IDX_H2SI] + k[970]*y[IDX_H2COI]*y[IDX_H3SII] -
        k[981]*y[IDX_H2OII]*y[IDX_H2SI] - k[982]*y[IDX_H2OII]*y[IDX_H2SI] +
        k[1009]*y[IDX_H2OI]*y[IDX_HSiSII] - k[1017]*y[IDX_H2SII]*y[IDX_H2SI] -
        k[1018]*y[IDX_H2SI]*y[IDX_C2NII] - k[1019]*y[IDX_H2SI]*y[IDX_HS2II] -
        k[1020]*y[IDX_H2SI]*y[IDX_HSiSII] - k[1021]*y[IDX_H2SI]*y[IDX_SOII] -
        k[1044]*y[IDX_H3II]*y[IDX_H2SI] - k[1079]*y[IDX_H3COII]*y[IDX_H2SI] -
        k[1087]*y[IDX_H3OII]*y[IDX_H2SI] + k[1123]*y[IDX_HCNI]*y[IDX_H3SII] -
        k[1134]*y[IDX_HCNHII]*y[IDX_H2SI] - k[1135]*y[IDX_HCNHII]*y[IDX_H2SI] -
        k[1143]*y[IDX_HCOII]*y[IDX_H2SI] + k[1167]*y[IDX_HNCI]*y[IDX_H3SII] -
        k[1175]*y[IDX_HSII]*y[IDX_H2SI] - k[1176]*y[IDX_HSII]*y[IDX_H2SI] -
        k[1226]*y[IDX_HeII]*y[IDX_H2SI] - k[1227]*y[IDX_HeII]*y[IDX_H2SI] -
        k[1300]*y[IDX_NII]*y[IDX_H2SI] - k[1301]*y[IDX_NII]*y[IDX_H2SI] -
        k[1302]*y[IDX_NII]*y[IDX_H2SI] - k[1315]*y[IDX_N2II]*y[IDX_H2SI] -
        k[1316]*y[IDX_N2II]*y[IDX_H2SI] - k[1381]*y[IDX_NH2II]*y[IDX_H2SI] -
        k[1382]*y[IDX_NH2II]*y[IDX_H2SI] - k[1383]*y[IDX_NH2II]*y[IDX_H2SI] -
        k[1384]*y[IDX_NH2II]*y[IDX_H2SI] - k[1415]*y[IDX_NH3II]*y[IDX_H2SI] +
        k[1429]*y[IDX_NH3I]*y[IDX_H3SII] - k[1472]*y[IDX_OII]*y[IDX_H2SI] -
        k[1473]*y[IDX_OII]*y[IDX_H2SI] - k[1524]*y[IDX_OHII]*y[IDX_H2SI] -
        k[1550]*y[IDX_SII]*y[IDX_H2SI] - k[1551]*y[IDX_SII]*y[IDX_H2SI] -
        k[1653]*y[IDX_CH3I]*y[IDX_H2SI] + k[1727]*y[IDX_H2I]*y[IDX_HSI] -
        k[1749]*y[IDX_HI]*y[IDX_H2SI] + k[1789]*y[IDX_HSI]*y[IDX_HSI] -
        k[1888]*y[IDX_OI]*y[IDX_H2SI] - k[1938]*y[IDX_OHI]*y[IDX_H2SI] -
        k[2020]*y[IDX_H2SI] - k[2021]*y[IDX_H2SI] - k[2022]*y[IDX_H2SI] +
        k[2138]*y[IDX_H2SII]*y[IDX_EM] - k[2257]*y[IDX_H2SI] +
        k[2391]*y[IDX_GH2SI] + k[2392]*y[IDX_GH2SI] + k[2393]*y[IDX_GH2SI] +
        k[2394]*y[IDX_GH2SI];
    ydot[IDX_H2OII] = 0.0 - k[66]*y[IDX_CH2I]*y[IDX_H2OII] -
        k[86]*y[IDX_CHI]*y[IDX_H2OII] + k[121]*y[IDX_HII]*y[IDX_H2OI] +
        k[162]*y[IDX_H2II]*y[IDX_H2OI] - k[175]*y[IDX_H2OII]*y[IDX_C2I] -
        k[176]*y[IDX_H2OII]*y[IDX_C2H2I] - k[177]*y[IDX_H2OII]*y[IDX_C2HI] -
        k[178]*y[IDX_H2OII]*y[IDX_H2COI] - k[179]*y[IDX_H2OII]*y[IDX_H2SI] -
        k[180]*y[IDX_H2OII]*y[IDX_HCOI] - k[181]*y[IDX_H2OII]*y[IDX_MgI] -
        k[182]*y[IDX_H2OII]*y[IDX_NOI] - k[183]*y[IDX_H2OII]*y[IDX_O2I] -
        k[184]*y[IDX_H2OII]*y[IDX_OCSI] - k[185]*y[IDX_H2OII]*y[IDX_SI] -
        k[186]*y[IDX_H2OII]*y[IDX_SiI] + k[187]*y[IDX_H2OI]*y[IDX_COII] +
        k[188]*y[IDX_H2OI]*y[IDX_HCNII] + k[189]*y[IDX_H2OI]*y[IDX_N2II] +
        k[213]*y[IDX_HeII]*y[IDX_H2OI] + k[240]*y[IDX_NII]*y[IDX_H2OI] +
        k[261]*y[IDX_NHII]*y[IDX_H2OI] - k[274]*y[IDX_NH2I]*y[IDX_H2OII] -
        k[285]*y[IDX_NH3I]*y[IDX_H2OII] + k[312]*y[IDX_OII]*y[IDX_H2OI] +
        k[332]*y[IDX_OHII]*y[IDX_H2OI] - k[512]*y[IDX_H2OII]*y[IDX_EM] -
        k[513]*y[IDX_H2OII]*y[IDX_EM] - k[514]*y[IDX_H2OII]*y[IDX_EM] -
        k[690]*y[IDX_CI]*y[IDX_H2OII] - k[758]*y[IDX_CH2I]*y[IDX_H2OII] -
        k[810]*y[IDX_CH4I]*y[IDX_H2OII] - k[843]*y[IDX_CHI]*y[IDX_H2OII] +
        k[933]*y[IDX_H2II]*y[IDX_OHI] - k[945]*y[IDX_H2I]*y[IDX_H2OII] +
        k[960]*y[IDX_H2I]*y[IDX_OHII] - k[976]*y[IDX_H2OII]*y[IDX_C2I] -
        k[977]*y[IDX_H2OII]*y[IDX_C2HI] - k[978]*y[IDX_H2OII]*y[IDX_COI] -
        k[979]*y[IDX_H2OII]*y[IDX_H2COI] - k[980]*y[IDX_H2OII]*y[IDX_H2OI] -
        k[981]*y[IDX_H2OII]*y[IDX_H2SI] - k[982]*y[IDX_H2OII]*y[IDX_H2SI] -
        k[983]*y[IDX_H2OII]*y[IDX_HCNI] - k[984]*y[IDX_H2OII]*y[IDX_HCOI] -
        k[985]*y[IDX_H2OII]*y[IDX_HCOI] - k[986]*y[IDX_H2OII]*y[IDX_HNCI] -
        k[987]*y[IDX_H2OII]*y[IDX_SI] - k[988]*y[IDX_H2OII]*y[IDX_SI] -
        k[989]*y[IDX_H2OII]*y[IDX_SO2I] + k[1063]*y[IDX_H3II]*y[IDX_OI] +
        k[1066]*y[IDX_H3II]*y[IDX_OHI] - k[1333]*y[IDX_NI]*y[IDX_H2OII] -
        k[1334]*y[IDX_NI]*y[IDX_H2OII] + k[1371]*y[IDX_NHII]*y[IDX_OHI] -
        k[1400]*y[IDX_NH2I]*y[IDX_H2OII] - k[1425]*y[IDX_NH3I]*y[IDX_H2OII] -
        k[1450]*y[IDX_NHI]*y[IDX_H2OII] - k[1497]*y[IDX_OI]*y[IDX_H2OII] +
        k[1526]*y[IDX_OHII]*y[IDX_HCOI] + k[1532]*y[IDX_OHII]*y[IDX_OHI] +
        k[1538]*y[IDX_OHI]*y[IDX_CH5II] - k[1540]*y[IDX_OHI]*y[IDX_H2OII] +
        k[1541]*y[IDX_OHI]*y[IDX_HCNII] + k[1542]*y[IDX_OHI]*y[IDX_HCOII] +
        k[1544]*y[IDX_OHI]*y[IDX_HNOII] + k[1545]*y[IDX_OHI]*y[IDX_N2HII] +
        k[1547]*y[IDX_OHI]*y[IDX_O2HII] - k[2016]*y[IDX_H2OII] +
        k[2017]*y[IDX_H2OI] - k[2239]*y[IDX_H2OII];
    ydot[IDX_OHII] = 0.0 - k[71]*y[IDX_CH2I]*y[IDX_OHII] -
        k[92]*y[IDX_CHI]*y[IDX_OHII] + k[138]*y[IDX_HII]*y[IDX_OHI] +
        k[171]*y[IDX_H2II]*y[IDX_OHI] + k[251]*y[IDX_NII]*y[IDX_OHI] -
        k[277]*y[IDX_NH2I]*y[IDX_OHII] + k[319]*y[IDX_OII]*y[IDX_OHI] -
        k[329]*y[IDX_OHII]*y[IDX_C2I] - k[330]*y[IDX_OHII]*y[IDX_C2HI] -
        k[331]*y[IDX_OHII]*y[IDX_H2COI] - k[332]*y[IDX_OHII]*y[IDX_H2OI] -
        k[333]*y[IDX_OHII]*y[IDX_H2SI] - k[334]*y[IDX_OHII]*y[IDX_HCOI] -
        k[335]*y[IDX_OHII]*y[IDX_NH3I] - k[336]*y[IDX_OHII]*y[IDX_NOI] -
        k[337]*y[IDX_OHII]*y[IDX_O2I] - k[338]*y[IDX_OHII]*y[IDX_SI] +
        k[339]*y[IDX_OHI]*y[IDX_C2II] + k[340]*y[IDX_OHI]*y[IDX_CNII] +
        k[341]*y[IDX_OHI]*y[IDX_COII] + k[342]*y[IDX_OHI]*y[IDX_N2II] -
        k[582]*y[IDX_OHII]*y[IDX_EM] - k[702]*y[IDX_CI]*y[IDX_OHII] -
        k[771]*y[IDX_CH2I]*y[IDX_OHII] - k[819]*y[IDX_CH4I]*y[IDX_OHII] -
        k[820]*y[IDX_CH4I]*y[IDX_OHII] - k[860]*y[IDX_CHI]*y[IDX_OHII] +
        k[932]*y[IDX_H2II]*y[IDX_OI] + k[958]*y[IDX_H2I]*y[IDX_OII] -
        k[960]*y[IDX_H2I]*y[IDX_OHII] + k[1064]*y[IDX_H3II]*y[IDX_OI] +
        k[1200]*y[IDX_HeII]*y[IDX_CH3OHI] + k[1222]*y[IDX_HeII]*y[IDX_H2OI] -
        k[1340]*y[IDX_NI]*y[IDX_OHII] + k[1370]*y[IDX_NHII]*y[IDX_OI] -
        k[1411]*y[IDX_NH2I]*y[IDX_OHII] - k[1460]*y[IDX_NHI]*y[IDX_OHII] +
        k[1476]*y[IDX_OII]*y[IDX_HCOI] + k[1506]*y[IDX_OI]*y[IDX_N2HII] +
        k[1510]*y[IDX_OI]*y[IDX_O2HII] - k[1511]*y[IDX_OI]*y[IDX_OHII] -
        k[1517]*y[IDX_OHII]*y[IDX_C2I] - k[1518]*y[IDX_OHII]*y[IDX_C2HI] -
        k[1519]*y[IDX_OHII]*y[IDX_CNI] - k[1520]*y[IDX_OHII]*y[IDX_CO2I] -
        k[1521]*y[IDX_OHII]*y[IDX_COI] - k[1522]*y[IDX_OHII]*y[IDX_H2COI] -
        k[1523]*y[IDX_OHII]*y[IDX_H2OI] - k[1524]*y[IDX_OHII]*y[IDX_H2SI] -
        k[1525]*y[IDX_OHII]*y[IDX_HCNI] - k[1526]*y[IDX_OHII]*y[IDX_HCOI] -
        k[1527]*y[IDX_OHII]*y[IDX_HCOI] - k[1528]*y[IDX_OHII]*y[IDX_HNCI] -
        k[1529]*y[IDX_OHII]*y[IDX_N2I] - k[1530]*y[IDX_OHII]*y[IDX_NH3I] -
        k[1531]*y[IDX_OHII]*y[IDX_NOI] - k[1532]*y[IDX_OHII]*y[IDX_OHI] -
        k[1533]*y[IDX_OHII]*y[IDX_SI] - k[1534]*y[IDX_OHII]*y[IDX_SI] -
        k[1535]*y[IDX_OHII]*y[IDX_SiI] - k[1536]*y[IDX_OHII]*y[IDX_SiHI] -
        k[1537]*y[IDX_OHII]*y[IDX_SiOI] + k[2016]*y[IDX_H2OII] -
        k[2068]*y[IDX_OHII] + k[2070]*y[IDX_OHI] - k[2233]*y[IDX_OHII];
    ydot[IDX_CH3II] = 0.0 - k[72]*y[IDX_CH3II]*y[IDX_HCOI] -
        k[73]*y[IDX_CH3II]*y[IDX_MgI] - k[74]*y[IDX_CH3II]*y[IDX_NOI] +
        k[115]*y[IDX_HII]*y[IDX_CH3I] + k[385]*y[IDX_CH3I] -
        k[481]*y[IDX_CH3II]*y[IDX_EM] - k[482]*y[IDX_CH3II]*y[IDX_EM] -
        k[483]*y[IDX_CH3II]*y[IDX_EM] + k[613]*y[IDX_CII]*y[IDX_CH3OHI] -
        k[688]*y[IDX_CI]*y[IDX_CH3II] + k[709]*y[IDX_CHII]*y[IDX_CH3OHI] +
        k[715]*y[IDX_CHII]*y[IDX_H2COI] + k[747]*y[IDX_CH2II]*y[IDX_HCOI] +
        k[754]*y[IDX_CH2I]*y[IDX_C2HII] + k[755]*y[IDX_CH2I]*y[IDX_CH5II] +
        k[757]*y[IDX_CH2I]*y[IDX_H2COII] + k[758]*y[IDX_CH2I]*y[IDX_H2OII] +
        k[759]*y[IDX_CH2I]*y[IDX_H3OII] + k[760]*y[IDX_CH2I]*y[IDX_HCNII] +
        k[761]*y[IDX_CH2I]*y[IDX_HCNHII] + k[762]*y[IDX_CH2I]*y[IDX_HCNHII] +
        k[763]*y[IDX_CH2I]*y[IDX_HCOII] + k[764]*y[IDX_CH2I]*y[IDX_HNOII] +
        k[765]*y[IDX_CH2I]*y[IDX_N2HII] + k[766]*y[IDX_CH2I]*y[IDX_NHII] +
        k[767]*y[IDX_CH2I]*y[IDX_NH2II] + k[768]*y[IDX_CH2I]*y[IDX_NH3II] +
        k[770]*y[IDX_CH2I]*y[IDX_O2HII] + k[771]*y[IDX_CH2I]*y[IDX_OHII] -
        k[774]*y[IDX_CH3II]*y[IDX_C2H4I] - k[775]*y[IDX_CH3II]*y[IDX_CH3CNI] -
        k[776]*y[IDX_CH3II]*y[IDX_CH3OHI] - k[777]*y[IDX_CH3II]*y[IDX_H2COI] -
        k[778]*y[IDX_CH3II]*y[IDX_H2SI] - k[779]*y[IDX_CH3II]*y[IDX_HCOI] -
        k[780]*y[IDX_CH3II]*y[IDX_HSI] - k[781]*y[IDX_CH3II]*y[IDX_NH3I] -
        k[782]*y[IDX_CH3II]*y[IDX_O2I] - k[783]*y[IDX_CH3II]*y[IDX_OI] -
        k[784]*y[IDX_CH3II]*y[IDX_OI] - k[785]*y[IDX_CH3II]*y[IDX_OCSI] -
        k[786]*y[IDX_CH3II]*y[IDX_OHI] - k[787]*y[IDX_CH3II]*y[IDX_SI] -
        k[788]*y[IDX_CH3II]*y[IDX_SOI] - k[789]*y[IDX_CH3II]*y[IDX_SiH4I] +
        k[816]*y[IDX_CH4I]*y[IDX_N2II] - k[839]*y[IDX_CHI]*y[IDX_CH3II] +
        k[886]*y[IDX_HII]*y[IDX_CH3CNI] + k[887]*y[IDX_HII]*y[IDX_CH3OHI] +
        k[890]*y[IDX_HII]*y[IDX_CH4I] + k[913]*y[IDX_H2II]*y[IDX_CH2I] +
        k[914]*y[IDX_H2II]*y[IDX_CH4I] + k[938]*y[IDX_H2I]*y[IDX_CH2II] +
        k[1023]*y[IDX_H3II]*y[IDX_C2H5OHI] + k[1028]*y[IDX_H3II]*y[IDX_CH2I] +
        k[1031]*y[IDX_H3II]*y[IDX_CH3OHI] - k[1100]*y[IDX_HI]*y[IDX_CH3II] +
        k[1101]*y[IDX_HI]*y[IDX_CH4II] + k[1199]*y[IDX_HeII]*y[IDX_CH3CNI] +
        k[1201]*y[IDX_HeII]*y[IDX_CH3OHI] + k[1204]*y[IDX_HeII]*y[IDX_CH4I] +
        k[1292]*y[IDX_NII]*y[IDX_CH3OHI] + k[1293]*y[IDX_NII]*y[IDX_CH4I] -
        k[1446]*y[IDX_NHI]*y[IDX_CH3II] + k[1468]*y[IDX_OII]*y[IDX_CH4I] +
        k[1493]*y[IDX_OI]*y[IDX_CH4II] - k[1984]*y[IDX_CH3II] -
        k[1985]*y[IDX_CH3II] + k[1987]*y[IDX_CH3I] + k[1994]*y[IDX_CH4II] -
        k[2107]*y[IDX_CH3II]*y[IDX_H2OI] - k[2108]*y[IDX_CH3II]*y[IDX_HCNI] -
        k[2114]*y[IDX_H2I]*y[IDX_CH3II] - k[2133]*y[IDX_CH3II]*y[IDX_EM] -
        k[2245]*y[IDX_CH3II];
    ydot[IDX_N2I] = 0.0 + k[38]*y[IDX_C2I]*y[IDX_N2II] +
        k[49]*y[IDX_C2HI]*y[IDX_N2II] + k[53]*y[IDX_CI]*y[IDX_N2II] +
        k[67]*y[IDX_CH2I]*y[IDX_N2II] + k[88]*y[IDX_CHI]*y[IDX_N2II] +
        k[100]*y[IDX_CNI]*y[IDX_N2II] + k[107]*y[IDX_COI]*y[IDX_N2II] +
        k[189]*y[IDX_H2OI]*y[IDX_N2II] + k[201]*y[IDX_HCNI]*y[IDX_N2II] -
        k[215]*y[IDX_HeII]*y[IDX_N2I] + k[226]*y[IDX_MgI]*y[IDX_N2II] +
        k[252]*y[IDX_N2II]*y[IDX_H2COI] + k[253]*y[IDX_N2II]*y[IDX_H2SI] +
        k[254]*y[IDX_N2II]*y[IDX_HCOI] + k[255]*y[IDX_N2II]*y[IDX_NOI] +
        k[256]*y[IDX_N2II]*y[IDX_O2I] + k[257]*y[IDX_N2II]*y[IDX_OCSI] +
        k[258]*y[IDX_N2II]*y[IDX_SI] + k[259]*y[IDX_NI]*y[IDX_N2II] +
        k[275]*y[IDX_NH2I]*y[IDX_N2II] + k[289]*y[IDX_NH3I]*y[IDX_N2II] +
        k[296]*y[IDX_NHI]*y[IDX_N2II] + k[328]*y[IDX_OI]*y[IDX_N2II] +
        k[342]*y[IDX_OHI]*y[IDX_N2II] - k[421]*y[IDX_N2I] +
        k[565]*y[IDX_N2HII]*y[IDX_EM] + k[655]*y[IDX_C2I]*y[IDX_N2HII] +
        k[682]*y[IDX_C2HI]*y[IDX_N2HII] + k[698]*y[IDX_CI]*y[IDX_N2HII] +
        k[765]*y[IDX_CH2I]*y[IDX_N2HII] + k[815]*y[IDX_CH4I]*y[IDX_N2II] +
        k[816]*y[IDX_CH4I]*y[IDX_N2II] + k[817]*y[IDX_CH4I]*y[IDX_N2HII] +
        k[853]*y[IDX_CHI]*y[IDX_N2HII] + k[877]*y[IDX_COI]*y[IDX_N2HII] -
        k[927]*y[IDX_H2II]*y[IDX_N2I] + k[1011]*y[IDX_H2OI]*y[IDX_N2HII] -
        k[1055]*y[IDX_H3II]*y[IDX_N2I] + k[1128]*y[IDX_HCNI]*y[IDX_N2HII] +
        k[1160]*y[IDX_HCOI]*y[IDX_N2HII] + k[1171]*y[IDX_HNCI]*y[IDX_N2HII] -
        k[1250]*y[IDX_HeII]*y[IDX_N2I] + k[1304]*y[IDX_NII]*y[IDX_NCCNI] +
        k[1314]*y[IDX_N2II]*y[IDX_H2COI] + k[1315]*y[IDX_N2II]*y[IDX_H2SI] +
        k[1316]*y[IDX_N2II]*y[IDX_H2SI] + k[1318]*y[IDX_N2II]*y[IDX_OCSI] -
        k[1319]*y[IDX_N2I]*y[IDX_HNOII] - k[1320]*y[IDX_N2I]*y[IDX_O2HII] +
        k[1321]*y[IDX_N2HII]*y[IDX_CH3CNI] + k[1322]*y[IDX_N2HII]*y[IDX_CO2I] +
        k[1323]*y[IDX_N2HII]*y[IDX_H2COI] + k[1324]*y[IDX_N2HII]*y[IDX_SI] -
        k[1363]*y[IDX_NHII]*y[IDX_N2I] + k[1408]*y[IDX_NH2I]*y[IDX_N2HII] +
        k[1440]*y[IDX_NH3I]*y[IDX_N2HII] + k[1454]*y[IDX_NHI]*y[IDX_N2HII] -
        k[1477]*y[IDX_OII]*y[IDX_N2I] + k[1506]*y[IDX_OI]*y[IDX_N2HII] -
        k[1529]*y[IDX_OHII]*y[IDX_N2I] + k[1545]*y[IDX_OHI]*y[IDX_N2HII] -
        k[1597]*y[IDX_CI]*y[IDX_N2I] - k[1627]*y[IDX_CH2I]*y[IDX_N2I] -
        k[1681]*y[IDX_CHI]*y[IDX_N2I] + k[1703]*y[IDX_CNI]*y[IDX_CNI] +
        k[1710]*y[IDX_CNI]*y[IDX_NOI] + k[1807]*y[IDX_NI]*y[IDX_CNI] +
        k[1818]*y[IDX_NI]*y[IDX_NCCNI] + k[1819]*y[IDX_NI]*y[IDX_NHI] +
        k[1820]*y[IDX_NI]*y[IDX_NO2I] + k[1822]*y[IDX_NI]*y[IDX_NO2I] +
        k[1823]*y[IDX_NI]*y[IDX_NOI] + k[1824]*y[IDX_NI]*y[IDX_NSI] +
        k[1834]*y[IDX_NH2I]*y[IDX_NOI] + k[1835]*y[IDX_NH2I]*y[IDX_NOI] +
        k[1843]*y[IDX_NHI]*y[IDX_NHI] + k[1844]*y[IDX_NHI]*y[IDX_NHI] +
        k[1847]*y[IDX_NHI]*y[IDX_NOI] + k[1848]*y[IDX_NHI]*y[IDX_NOI] +
        k[1858]*y[IDX_NOI]*y[IDX_NOI] + k[1860]*y[IDX_NOI]*y[IDX_OCNI] -
        k[1901]*y[IDX_OI]*y[IDX_N2I] - k[2045]*y[IDX_N2I] - k[2219]*y[IDX_N2I] +
        k[2347]*y[IDX_GN2I] + k[2348]*y[IDX_GN2I] + k[2349]*y[IDX_GN2I] +
        k[2350]*y[IDX_GN2I];
    ydot[IDX_C2H2II] = 0.0 - k[42]*y[IDX_C2H2II]*y[IDX_H2COI] -
        k[43]*y[IDX_C2H2II]*y[IDX_H2SI] - k[44]*y[IDX_C2H2II]*y[IDX_HCOI] -
        k[45]*y[IDX_C2H2II]*y[IDX_NOI] + k[46]*y[IDX_C2H2I]*y[IDX_HCNII] +
        k[75]*y[IDX_CH4II]*y[IDX_C2H2I] + k[111]*y[IDX_HII]*y[IDX_C2H2I] +
        k[154]*y[IDX_H2II]*y[IDX_C2H2I] + k[176]*y[IDX_H2OII]*y[IDX_C2H2I] +
        k[208]*y[IDX_HeII]*y[IDX_C2H2I] - k[220]*y[IDX_MgI]*y[IDX_C2H2II] -
        k[282]*y[IDX_NH3I]*y[IDX_C2H2II] + k[307]*y[IDX_OII]*y[IDX_C2H2I] +
        k[321]*y[IDX_O2II]*y[IDX_C2H2I] + k[367]*y[IDX_C2H2I] -
        k[461]*y[IDX_C2H2II]*y[IDX_EM] - k[462]*y[IDX_C2H2II]*y[IDX_EM] -
        k[463]*y[IDX_C2H2II]*y[IDX_EM] + k[610]*y[IDX_CII]*y[IDX_CH3I] +
        k[611]*y[IDX_CII]*y[IDX_CH3CCHI] + k[614]*y[IDX_CII]*y[IDX_CH4I] +
        k[661]*y[IDX_C2HII]*y[IDX_HCNI] + k[663]*y[IDX_C2HII]*y[IDX_HCOI] -
        k[665]*y[IDX_C2H2II]*y[IDX_CH3CNI] - k[666]*y[IDX_C2H2II]*y[IDX_CH3CNI]
        - k[667]*y[IDX_C2H2II]*y[IDX_H2SI] - k[668]*y[IDX_C2H2II]*y[IDX_HCNI] -
        k[669]*y[IDX_C2H2II]*y[IDX_HNCI] - k[670]*y[IDX_C2H2II]*y[IDX_SiI] -
        k[671]*y[IDX_C2H2II]*y[IDX_SiH4I] - k[672]*y[IDX_C2H2II]*y[IDX_SiH4I] -
        k[673]*y[IDX_C2H2II]*y[IDX_SiH4I] - k[674]*y[IDX_C2H2II]*y[IDX_SiH4I] +
        k[675]*y[IDX_C2H2I]*y[IDX_C2N2II] + k[678]*y[IDX_C2HI]*y[IDX_H2COII] +
        k[679]*y[IDX_C2HI]*y[IDX_HCNII] + k[680]*y[IDX_C2HI]*y[IDX_HCOII] +
        k[681]*y[IDX_C2HI]*y[IDX_HNOII] + k[682]*y[IDX_C2HI]*y[IDX_N2HII] +
        k[683]*y[IDX_C2HI]*y[IDX_O2HII] + k[711]*y[IDX_CHII]*y[IDX_CH4I] +
        k[804]*y[IDX_CH4I]*y[IDX_C2II] + k[805]*y[IDX_CH4I]*y[IDX_C2HII] -
        k[806]*y[IDX_CH4I]*y[IDX_C2H2II] + k[824]*y[IDX_CH5II]*y[IDX_C2HI] +
        k[839]*y[IDX_CHI]*y[IDX_CH3II] + k[882]*y[IDX_HII]*y[IDX_C2H3I] +
        k[883]*y[IDX_HII]*y[IDX_C2H4I] + k[910]*y[IDX_H2II]*y[IDX_C2H4I] +
        k[911]*y[IDX_H2II]*y[IDX_C2HI] + k[936]*y[IDX_H2I]*y[IDX_C2HII] +
        k[977]*y[IDX_H2OII]*y[IDX_C2HI] - k[991]*y[IDX_H2OI]*y[IDX_C2H2II] +
        k[1025]*y[IDX_H3II]*y[IDX_C2HI] + k[1182]*y[IDX_HeII]*y[IDX_C2H3I] +
        k[1184]*y[IDX_HeII]*y[IDX_C2H4I] - k[1328]*y[IDX_NI]*y[IDX_C2H2II] -
        k[1329]*y[IDX_NI]*y[IDX_C2H2II] - k[1330]*y[IDX_NI]*y[IDX_C2H2II] +
        k[1348]*y[IDX_NHII]*y[IDX_C2HI] + k[1375]*y[IDX_NH2II]*y[IDX_C2HI] -
        k[1396]*y[IDX_NH2I]*y[IDX_C2H2II] + k[1412]*y[IDX_NH3II]*y[IDX_C2I] -
        k[1419]*y[IDX_NH3I]*y[IDX_C2H2II] + k[1464]*y[IDX_OII]*y[IDX_C2H4I] -
        k[1492]*y[IDX_OI]*y[IDX_C2H2II] + k[1518]*y[IDX_OHII]*y[IDX_C2HI] +
        k[1964]*y[IDX_C2H2I] - k[2277]*y[IDX_C2H2II];
    ydot[IDX_C2I] = 0.0 + k[33]*y[IDX_C2II]*y[IDX_HCOI] +
        k[34]*y[IDX_C2II]*y[IDX_NOI] + k[35]*y[IDX_C2II]*y[IDX_SI] -
        k[36]*y[IDX_C2I]*y[IDX_CNII] - k[37]*y[IDX_C2I]*y[IDX_COII] -
        k[38]*y[IDX_C2I]*y[IDX_N2II] - k[39]*y[IDX_C2I]*y[IDX_O2II] +
        k[50]*y[IDX_CI]*y[IDX_C2II] + k[62]*y[IDX_CH2I]*y[IDX_C2II] +
        k[82]*y[IDX_CHI]*y[IDX_C2II] - k[110]*y[IDX_HII]*y[IDX_C2I] -
        k[153]*y[IDX_H2II]*y[IDX_C2I] - k[175]*y[IDX_H2OII]*y[IDX_C2I] -
        k[207]*y[IDX_HeII]*y[IDX_C2I] - k[233]*y[IDX_NII]*y[IDX_C2I] +
        k[271]*y[IDX_NH2I]*y[IDX_C2II] - k[306]*y[IDX_OII]*y[IDX_C2I] -
        k[329]*y[IDX_OHII]*y[IDX_C2I] + k[339]*y[IDX_OHI]*y[IDX_C2II] -
        k[366]*y[IDX_C2I] + k[373]*y[IDX_C2HI] + k[375]*y[IDX_C2NI] +
        k[377]*y[IDX_C3NI] + k[378]*y[IDX_C4HI] + k[459]*y[IDX_C2HII]*y[IDX_EM]
        + k[461]*y[IDX_C2H2II]*y[IDX_EM] + k[468]*y[IDX_C2NII]*y[IDX_EM] +
        k[473]*y[IDX_C3II]*y[IDX_EM] + k[475]*y[IDX_C4NII]*y[IDX_EM] +
        k[588]*y[IDX_SiC2II]*y[IDX_EM] + k[591]*y[IDX_SiC3II]*y[IDX_EM] +
        k[642]*y[IDX_CII]*y[IDX_SiCI] - k[647]*y[IDX_C2II]*y[IDX_C2I] -
        k[651]*y[IDX_C2I]*y[IDX_H2COII] - k[652]*y[IDX_C2I]*y[IDX_HCNII] -
        k[653]*y[IDX_C2I]*y[IDX_HCOII] - k[654]*y[IDX_C2I]*y[IDX_HNOII] -
        k[655]*y[IDX_C2I]*y[IDX_N2HII] - k[656]*y[IDX_C2I]*y[IDX_O2II] -
        k[657]*y[IDX_C2I]*y[IDX_O2HII] - k[658]*y[IDX_C2I]*y[IDX_SII] -
        k[659]*y[IDX_C2I]*y[IDX_SiOII] + k[660]*y[IDX_C2HII]*y[IDX_H2COI] +
        k[662]*y[IDX_C2HII]*y[IDX_HCNI] + k[664]*y[IDX_C2HII]*y[IDX_HNCI] +
        k[677]*y[IDX_C2HI]*y[IDX_COII] - k[705]*y[IDX_CHII]*y[IDX_C2I] +
        k[754]*y[IDX_CH2I]*y[IDX_C2HII] - k[823]*y[IDX_CH5II]*y[IDX_C2I] +
        k[838]*y[IDX_CHI]*y[IDX_C2HII] - k[909]*y[IDX_H2II]*y[IDX_C2I] -
        k[976]*y[IDX_H2OII]*y[IDX_C2I] - k[1022]*y[IDX_H3II]*y[IDX_C2I] -
        k[1080]*y[IDX_H3OII]*y[IDX_C2I] - k[1177]*y[IDX_HeII]*y[IDX_C2I] +
        k[1191]*y[IDX_HeII]*y[IDX_C4HI] + k[1274]*y[IDX_HeII]*y[IDX_SiC2I] -
        k[1345]*y[IDX_NHII]*y[IDX_C2I] - k[1346]*y[IDX_NHII]*y[IDX_C2I] -
        k[1347]*y[IDX_NHII]*y[IDX_C2I] - k[1374]*y[IDX_NH2II]*y[IDX_C2I] +
        k[1395]*y[IDX_NH2I]*y[IDX_C2HII] - k[1412]*y[IDX_NH3II]*y[IDX_C2I] +
        k[1418]*y[IDX_NH3I]*y[IDX_C2HII] - k[1463]*y[IDX_OII]*y[IDX_C2I] -
        k[1517]*y[IDX_OHII]*y[IDX_C2I] - k[1570]*y[IDX_C2I]*y[IDX_C2H2I] -
        k[1571]*y[IDX_C2I]*y[IDX_HCNI] - k[1572]*y[IDX_C2I]*y[IDX_O2I] -
        k[1573]*y[IDX_C2I]*y[IDX_SI] + k[1584]*y[IDX_CI]*y[IDX_C2NI] +
        k[1589]*y[IDX_CI]*y[IDX_CHI] + k[1590]*y[IDX_CI]*y[IDX_CNI] +
        k[1591]*y[IDX_CI]*y[IDX_COI] + k[1592]*y[IDX_CI]*y[IDX_CSI] +
        k[1703]*y[IDX_CNI]*y[IDX_CNI] - k[1736]*y[IDX_HI]*y[IDX_C2I] -
        k[1790]*y[IDX_NI]*y[IDX_C2I] - k[1867]*y[IDX_OI]*y[IDX_C2I] -
        k[1961]*y[IDX_C2I] - k[1962]*y[IDX_C2I] + k[1970]*y[IDX_C2HI] +
        k[1972]*y[IDX_C2NI] + k[1974]*y[IDX_C3NI] + k[1975]*y[IDX_C4HI] +
        k[2078]*y[IDX_SiC2I] + k[2079]*y[IDX_SiC3I] +
        k[2101]*y[IDX_CI]*y[IDX_CI] - k[2209]*y[IDX_C2I] + k[2315]*y[IDX_GC2I] +
        k[2316]*y[IDX_GC2I] + k[2317]*y[IDX_GC2I] + k[2318]*y[IDX_GC2I];
    ydot[IDX_CH2I] = 0.0 - k[14]*y[IDX_CII]*y[IDX_CH2I] +
        k[61]*y[IDX_CH2II]*y[IDX_NOI] - k[62]*y[IDX_CH2I]*y[IDX_C2II] -
        k[63]*y[IDX_CH2I]*y[IDX_CNII] - k[64]*y[IDX_CH2I]*y[IDX_COII] -
        k[65]*y[IDX_CH2I]*y[IDX_H2COII] - k[66]*y[IDX_CH2I]*y[IDX_H2OII] -
        k[67]*y[IDX_CH2I]*y[IDX_N2II] - k[68]*y[IDX_CH2I]*y[IDX_NH2II] -
        k[69]*y[IDX_CH2I]*y[IDX_OII] - k[70]*y[IDX_CH2I]*y[IDX_O2II] -
        k[71]*y[IDX_CH2I]*y[IDX_OHII] - k[114]*y[IDX_HII]*y[IDX_CH2I] -
        k[156]*y[IDX_H2II]*y[IDX_CH2I] - k[235]*y[IDX_NII]*y[IDX_CH2I] -
        k[381]*y[IDX_CH2I] - k[382]*y[IDX_CH2I] + k[383]*y[IDX_CH2COI] +
        k[384]*y[IDX_CH3I] + k[390]*y[IDX_CH4I] + k[481]*y[IDX_CH3II]*y[IDX_EM]
        + k[486]*y[IDX_CH3OH2II]*y[IDX_EM] + k[491]*y[IDX_CH4II]*y[IDX_EM] +
        k[493]*y[IDX_CH5II]*y[IDX_EM] + k[502]*y[IDX_H2COII]*y[IDX_EM] +
        k[521]*y[IDX_H3COII]*y[IDX_EM] - k[608]*y[IDX_CII]*y[IDX_CH2I] -
        k[707]*y[IDX_CHII]*y[IDX_CH2I] + k[710]*y[IDX_CHII]*y[IDX_CH3OHI] +
        k[717]*y[IDX_CHII]*y[IDX_H2COI] - k[754]*y[IDX_CH2I]*y[IDX_C2HII] -
        k[755]*y[IDX_CH2I]*y[IDX_CH5II] - k[756]*y[IDX_CH2I]*y[IDX_COII] -
        k[757]*y[IDX_CH2I]*y[IDX_H2COII] - k[758]*y[IDX_CH2I]*y[IDX_H2OII] -
        k[759]*y[IDX_CH2I]*y[IDX_H3OII] - k[760]*y[IDX_CH2I]*y[IDX_HCNII] -
        k[761]*y[IDX_CH2I]*y[IDX_HCNHII] - k[762]*y[IDX_CH2I]*y[IDX_HCNHII] -
        k[763]*y[IDX_CH2I]*y[IDX_HCOII] - k[764]*y[IDX_CH2I]*y[IDX_HNOII] -
        k[765]*y[IDX_CH2I]*y[IDX_N2HII] - k[766]*y[IDX_CH2I]*y[IDX_NHII] -
        k[767]*y[IDX_CH2I]*y[IDX_NH2II] - k[768]*y[IDX_CH2I]*y[IDX_NH3II] -
        k[769]*y[IDX_CH2I]*y[IDX_O2II] - k[770]*y[IDX_CH2I]*y[IDX_O2HII] -
        k[771]*y[IDX_CH2I]*y[IDX_OHII] - k[772]*y[IDX_CH2I]*y[IDX_SII] -
        k[773]*y[IDX_CH2I]*y[IDX_SiOII] + k[781]*y[IDX_CH3II]*y[IDX_NH3I] +
        k[804]*y[IDX_CH4I]*y[IDX_C2II] + k[820]*y[IDX_CH4I]*y[IDX_OHII] -
        k[885]*y[IDX_HII]*y[IDX_CH2I] - k[913]*y[IDX_H2II]*y[IDX_CH2I] -
        k[1028]*y[IDX_H3II]*y[IDX_CH2I] + k[1185]*y[IDX_HeII]*y[IDX_C2H4I] -
        k[1192]*y[IDX_HeII]*y[IDX_CH2I] - k[1193]*y[IDX_HeII]*y[IDX_CH2I] +
        k[1194]*y[IDX_HeII]*y[IDX_CH2COI] + k[1220]*y[IDX_HeII]*y[IDX_H2CSI] +
        k[1299]*y[IDX_NII]*y[IDX_H2COI] + k[1495]*y[IDX_OI]*y[IDX_CH5II] -
        k[1586]*y[IDX_CI]*y[IDX_CH2I] - k[1587]*y[IDX_CI]*y[IDX_CH2I] -
        k[1618]*y[IDX_CH2I]*y[IDX_CH2I] - k[1618]*y[IDX_CH2I]*y[IDX_CH2I] -
        k[1619]*y[IDX_CH2I]*y[IDX_CH2I] - k[1619]*y[IDX_CH2I]*y[IDX_CH2I] -
        k[1620]*y[IDX_CH2I]*y[IDX_CH2I] - k[1620]*y[IDX_CH2I]*y[IDX_CH2I] -
        k[1621]*y[IDX_CH2I]*y[IDX_CH2I] - k[1621]*y[IDX_CH2I]*y[IDX_CH2I] -
        k[1622]*y[IDX_CH2I]*y[IDX_CH4I] - k[1623]*y[IDX_CH2I]*y[IDX_CNI] -
        k[1624]*y[IDX_CH2I]*y[IDX_H2COI] - k[1625]*y[IDX_CH2I]*y[IDX_HCOI] -
        k[1626]*y[IDX_CH2I]*y[IDX_HNOI] - k[1627]*y[IDX_CH2I]*y[IDX_N2I] -
        k[1628]*y[IDX_CH2I]*y[IDX_NO2I] - k[1629]*y[IDX_CH2I]*y[IDX_NOI] -
        k[1630]*y[IDX_CH2I]*y[IDX_NOI] - k[1631]*y[IDX_CH2I]*y[IDX_NOI] -
        k[1632]*y[IDX_CH2I]*y[IDX_O2I] - k[1633]*y[IDX_CH2I]*y[IDX_O2I] -
        k[1634]*y[IDX_CH2I]*y[IDX_O2I] - k[1635]*y[IDX_CH2I]*y[IDX_O2I] -
        k[1636]*y[IDX_CH2I]*y[IDX_O2I] - k[1637]*y[IDX_CH2I]*y[IDX_OI] -
        k[1638]*y[IDX_CH2I]*y[IDX_OI] - k[1639]*y[IDX_CH2I]*y[IDX_OI] -
        k[1640]*y[IDX_CH2I]*y[IDX_OI] - k[1641]*y[IDX_CH2I]*y[IDX_OHI] -
        k[1642]*y[IDX_CH2I]*y[IDX_OHI] - k[1643]*y[IDX_CH2I]*y[IDX_OHI] -
        k[1644]*y[IDX_CH2I]*y[IDX_SI] - k[1645]*y[IDX_CH2I]*y[IDX_SI] +
        k[1649]*y[IDX_CH3I]*y[IDX_CH3I] + k[1650]*y[IDX_CH3I]*y[IDX_CNI] +
        k[1662]*y[IDX_CH3I]*y[IDX_O2I] + k[1668]*y[IDX_CH3I]*y[IDX_OHI] +
        k[1678]*y[IDX_CHI]*y[IDX_H2COI] + k[1679]*y[IDX_CHI]*y[IDX_HCOI] +
        k[1680]*y[IDX_CHI]*y[IDX_HNOI] + k[1692]*y[IDX_CHI]*y[IDX_O2HI] -
        k[1723]*y[IDX_H2I]*y[IDX_CH2I] + k[1725]*y[IDX_H2I]*y[IDX_CHI] -
        k[1739]*y[IDX_HI]*y[IDX_CH2I] + k[1741]*y[IDX_HI]*y[IDX_CH3I] +
        k[1752]*y[IDX_HI]*y[IDX_HCOI] - k[1801]*y[IDX_NI]*y[IDX_CH2I] -
        k[1802]*y[IDX_NI]*y[IDX_CH2I] - k[1803]*y[IDX_NI]*y[IDX_CH2I] +
        k[1868]*y[IDX_OI]*y[IDX_C2H2I] + k[1872]*y[IDX_OI]*y[IDX_C2H4I] -
        k[1981]*y[IDX_CH2I] - k[1982]*y[IDX_CH2I] + k[1983]*y[IDX_CH2COI] +
        k[1986]*y[IDX_CH3I] + k[1995]*y[IDX_CH4I] + k[2113]*y[IDX_H2I]*y[IDX_CI]
        - k[2213]*y[IDX_CH2I];
    ydot[IDX_CH3I] = 0.0 + k[72]*y[IDX_CH3II]*y[IDX_HCOI] +
        k[73]*y[IDX_CH3II]*y[IDX_MgI] + k[74]*y[IDX_CH3II]*y[IDX_NOI] -
        k[115]*y[IDX_HII]*y[IDX_CH3I] - k[384]*y[IDX_CH3I] - k[385]*y[IDX_CH3I]
        - k[386]*y[IDX_CH3I] + k[387]*y[IDX_CH3CNI] + k[389]*y[IDX_CH3OHI] +
        k[485]*y[IDX_CH3CNHII]*y[IDX_EM] + k[487]*y[IDX_CH3OH2II]*y[IDX_EM] +
        k[488]*y[IDX_CH3OH2II]*y[IDX_EM] + k[492]*y[IDX_CH4II]*y[IDX_EM] +
        k[494]*y[IDX_CH5II]*y[IDX_EM] + k[495]*y[IDX_CH5II]*y[IDX_EM] -
        k[609]*y[IDX_CII]*y[IDX_CH3I] - k[610]*y[IDX_CII]*y[IDX_CH3I] +
        k[676]*y[IDX_C2H4I]*y[IDX_SII] + k[742]*y[IDX_CH2II]*y[IDX_H2COI] -
        k[790]*y[IDX_CH3I]*y[IDX_SII] + k[794]*y[IDX_CH4II]*y[IDX_CH3OHI] +
        k[795]*y[IDX_CH4II]*y[IDX_CH4I] + k[796]*y[IDX_CH4II]*y[IDX_CO2I] +
        k[797]*y[IDX_CH4II]*y[IDX_COI] + k[798]*y[IDX_CH4II]*y[IDX_H2COI] +
        k[799]*y[IDX_CH4II]*y[IDX_H2OI] + k[800]*y[IDX_CH4II]*y[IDX_H2SI] +
        k[801]*y[IDX_CH4II]*y[IDX_NH3I] + k[802]*y[IDX_CH4II]*y[IDX_OCSI] +
        k[803]*y[IDX_CH4I]*y[IDX_C2II] + k[805]*y[IDX_CH4I]*y[IDX_C2HII] +
        k[807]*y[IDX_CH4I]*y[IDX_COII] + k[808]*y[IDX_CH4I]*y[IDX_CSII] +
        k[809]*y[IDX_CH4I]*y[IDX_H2COII] + k[810]*y[IDX_CH4I]*y[IDX_H2OII] +
        k[811]*y[IDX_CH4I]*y[IDX_HCNII] + k[818]*y[IDX_CH4I]*y[IDX_NH3II] -
        k[1029]*y[IDX_H3II]*y[IDX_CH3I] - k[1196]*y[IDX_HeII]*y[IDX_CH3I] +
        k[1198]*y[IDX_HeII]*y[IDX_CH3CNI] + k[1200]*y[IDX_HeII]*y[IDX_CH3OHI] +
        k[1205]*y[IDX_HeII]*y[IDX_CH4I] + k[1238]*y[IDX_HeII]*y[IDX_HCOOCH3I] +
        k[1291]*y[IDX_NII]*y[IDX_CH3OHI] + k[1564]*y[IDX_SiII]*y[IDX_CH3OHI] -
        k[1588]*y[IDX_CI]*y[IDX_CH3I] + k[1621]*y[IDX_CH2I]*y[IDX_CH2I] +
        k[1622]*y[IDX_CH2I]*y[IDX_CH4I] + k[1622]*y[IDX_CH2I]*y[IDX_CH4I] +
        k[1624]*y[IDX_CH2I]*y[IDX_H2COI] + k[1625]*y[IDX_CH2I]*y[IDX_HCOI] +
        k[1626]*y[IDX_CH2I]*y[IDX_HNOI] + k[1643]*y[IDX_CH2I]*y[IDX_OHI] -
        k[1646]*y[IDX_CH3I]*y[IDX_C2H3I] - k[1647]*y[IDX_CH3I]*y[IDX_CH3I] -
        k[1647]*y[IDX_CH3I]*y[IDX_CH3I] - k[1648]*y[IDX_CH3I]*y[IDX_CH3I] -
        k[1648]*y[IDX_CH3I]*y[IDX_CH3I] - k[1649]*y[IDX_CH3I]*y[IDX_CH3I] -
        k[1649]*y[IDX_CH3I]*y[IDX_CH3I] - k[1650]*y[IDX_CH3I]*y[IDX_CNI] -
        k[1651]*y[IDX_CH3I]*y[IDX_H2COI] - k[1652]*y[IDX_CH3I]*y[IDX_H2OI] -
        k[1653]*y[IDX_CH3I]*y[IDX_H2SI] - k[1654]*y[IDX_CH3I]*y[IDX_HCOI] -
        k[1655]*y[IDX_CH3I]*y[IDX_HNOI] - k[1656]*y[IDX_CH3I]*y[IDX_NH2I] -
        k[1657]*y[IDX_CH3I]*y[IDX_NH3I] - k[1658]*y[IDX_CH3I]*y[IDX_NO2I] -
        k[1659]*y[IDX_CH3I]*y[IDX_NOI] - k[1660]*y[IDX_CH3I]*y[IDX_O2I] -
        k[1661]*y[IDX_CH3I]*y[IDX_O2I] - k[1662]*y[IDX_CH3I]*y[IDX_O2I] -
        k[1663]*y[IDX_CH3I]*y[IDX_O2HI] - k[1664]*y[IDX_CH3I]*y[IDX_OI] -
        k[1665]*y[IDX_CH3I]*y[IDX_OI] - k[1666]*y[IDX_CH3I]*y[IDX_OHI] -
        k[1667]*y[IDX_CH3I]*y[IDX_OHI] - k[1668]*y[IDX_CH3I]*y[IDX_OHI] -
        k[1669]*y[IDX_CH3I]*y[IDX_SI] + k[1670]*y[IDX_CH4I]*y[IDX_CNI] +
        k[1671]*y[IDX_CH4I]*y[IDX_O2I] + k[1672]*y[IDX_CH4I]*y[IDX_OHI] +
        k[1673]*y[IDX_CH4I]*y[IDX_SI] + k[1723]*y[IDX_H2I]*y[IDX_CH2I] -
        k[1724]*y[IDX_H2I]*y[IDX_CH3I] + k[1740]*y[IDX_HI]*y[IDX_CH2COI] -
        k[1741]*y[IDX_HI]*y[IDX_CH3I] + k[1742]*y[IDX_HI]*y[IDX_CH4I] +
        k[1792]*y[IDX_NI]*y[IDX_C2H4I] + k[1794]*y[IDX_NI]*y[IDX_C2H5I] -
        k[1804]*y[IDX_NI]*y[IDX_CH3I] - k[1805]*y[IDX_NI]*y[IDX_CH3I] -
        k[1806]*y[IDX_NI]*y[IDX_CH3I] + k[1833]*y[IDX_NH2I]*y[IDX_CH4I] +
        k[1839]*y[IDX_NHI]*y[IDX_CH4I] + k[1873]*y[IDX_OI]*y[IDX_C2H4I] +
        k[1874]*y[IDX_OI]*y[IDX_C2H5I] + k[1879]*y[IDX_OI]*y[IDX_CH4I] +
        k[1929]*y[IDX_OHI]*y[IDX_C2H2I] - k[1986]*y[IDX_CH3I] -
        k[1987]*y[IDX_CH3I] - k[1988]*y[IDX_CH3I] + k[1989]*y[IDX_CH3CNI] +
        k[1992]*y[IDX_CH3OHI] + k[1996]*y[IDX_CH4I] -
        k[2109]*y[IDX_CH3I]*y[IDX_CNI] + k[2115]*y[IDX_H2I]*y[IDX_CHI] +
        k[2133]*y[IDX_CH3II]*y[IDX_EM] - k[2216]*y[IDX_CH3I];
    ydot[IDX_CHII] = 0.0 + k[15]*y[IDX_CII]*y[IDX_CHI] -
        k[55]*y[IDX_CHII]*y[IDX_HCOI] - k[56]*y[IDX_CHII]*y[IDX_MgI] -
        k[57]*y[IDX_CHII]*y[IDX_NH3I] - k[58]*y[IDX_CHII]*y[IDX_NOI] -
        k[59]*y[IDX_CHII]*y[IDX_SI] - k[60]*y[IDX_CHII]*y[IDX_SiI] +
        k[82]*y[IDX_CHI]*y[IDX_C2II] + k[83]*y[IDX_CHI]*y[IDX_CNII] +
        k[84]*y[IDX_CHI]*y[IDX_COII] + k[85]*y[IDX_CHI]*y[IDX_H2COII] +
        k[86]*y[IDX_CHI]*y[IDX_H2OII] + k[87]*y[IDX_CHI]*y[IDX_NII] +
        k[88]*y[IDX_CHI]*y[IDX_N2II] + k[89]*y[IDX_CHI]*y[IDX_NH2II] +
        k[90]*y[IDX_CHI]*y[IDX_OII] + k[91]*y[IDX_CHI]*y[IDX_O2II] +
        k[92]*y[IDX_CHI]*y[IDX_OHII] + k[117]*y[IDX_HII]*y[IDX_CHI] +
        k[158]*y[IDX_H2II]*y[IDX_CHI] + k[211]*y[IDX_HeII]*y[IDX_CHI] -
        k[380]*y[IDX_CHII] - k[477]*y[IDX_CHII]*y[IDX_EM] +
        k[626]*y[IDX_CII]*y[IDX_HCOI] - k[686]*y[IDX_CI]*y[IDX_CHII] +
        k[689]*y[IDX_CI]*y[IDX_CH5II] + k[690]*y[IDX_CI]*y[IDX_H2OII] +
        k[693]*y[IDX_CI]*y[IDX_HCNII] + k[694]*y[IDX_CI]*y[IDX_HCOII] +
        k[695]*y[IDX_CI]*y[IDX_HCO2II] + k[696]*y[IDX_CI]*y[IDX_HNOII] +
        k[698]*y[IDX_CI]*y[IDX_N2HII] + k[699]*y[IDX_CI]*y[IDX_NHII] +
        k[701]*y[IDX_CI]*y[IDX_O2HII] + k[702]*y[IDX_CI]*y[IDX_OHII] -
        k[705]*y[IDX_CHII]*y[IDX_C2I] - k[706]*y[IDX_CHII]*y[IDX_C2HI] -
        k[707]*y[IDX_CHII]*y[IDX_CH2I] - k[708]*y[IDX_CHII]*y[IDX_CH3OHI] -
        k[709]*y[IDX_CHII]*y[IDX_CH3OHI] - k[710]*y[IDX_CHII]*y[IDX_CH3OHI] -
        k[711]*y[IDX_CHII]*y[IDX_CH4I] - k[712]*y[IDX_CHII]*y[IDX_CHI] -
        k[713]*y[IDX_CHII]*y[IDX_CNI] - k[714]*y[IDX_CHII]*y[IDX_CO2I] -
        k[715]*y[IDX_CHII]*y[IDX_H2COI] - k[716]*y[IDX_CHII]*y[IDX_H2COI] -
        k[717]*y[IDX_CHII]*y[IDX_H2COI] - k[718]*y[IDX_CHII]*y[IDX_H2OI] -
        k[719]*y[IDX_CHII]*y[IDX_H2OI] - k[720]*y[IDX_CHII]*y[IDX_H2OI] -
        k[721]*y[IDX_CHII]*y[IDX_H2SI] - k[722]*y[IDX_CHII]*y[IDX_H2SI] -
        k[723]*y[IDX_CHII]*y[IDX_HCNI] - k[724]*y[IDX_CHII]*y[IDX_HCNI] -
        k[725]*y[IDX_CHII]*y[IDX_HCNI] - k[726]*y[IDX_CHII]*y[IDX_HCOI] -
        k[727]*y[IDX_CHII]*y[IDX_HNCI] - k[728]*y[IDX_CHII]*y[IDX_NI] -
        k[729]*y[IDX_CHII]*y[IDX_NH2I] - k[730]*y[IDX_CHII]*y[IDX_NH3I] -
        k[731]*y[IDX_CHII]*y[IDX_NHI] - k[732]*y[IDX_CHII]*y[IDX_O2I] -
        k[733]*y[IDX_CHII]*y[IDX_O2I] - k[734]*y[IDX_CHII]*y[IDX_O2I] -
        k[735]*y[IDX_CHII]*y[IDX_OI] - k[736]*y[IDX_CHII]*y[IDX_OCSI] -
        k[737]*y[IDX_CHII]*y[IDX_OCSI] - k[738]*y[IDX_CHII]*y[IDX_OHI] -
        k[739]*y[IDX_CHII]*y[IDX_SI] - k[740]*y[IDX_CHII]*y[IDX_SI] +
        k[885]*y[IDX_HII]*y[IDX_CH2I] + k[912]*y[IDX_H2II]*y[IDX_CI] +
        k[934]*y[IDX_H2I]*y[IDX_CII] - k[937]*y[IDX_H2I]*y[IDX_CHII] +
        k[1027]*y[IDX_H3II]*y[IDX_CI] - k[1098]*y[IDX_HI]*y[IDX_CHII] +
        k[1099]*y[IDX_HI]*y[IDX_CH2II] + k[1180]*y[IDX_HeII]*y[IDX_C2H2I] +
        k[1187]*y[IDX_HeII]*y[IDX_C2HI] + k[1193]*y[IDX_HeII]*y[IDX_CH2I] +
        k[1196]*y[IDX_HeII]*y[IDX_CH3I] + k[1202]*y[IDX_HeII]*y[IDX_CH4I] +
        k[1234]*y[IDX_HeII]*y[IDX_HCNI] + k[1237]*y[IDX_HeII]*y[IDX_HCOI] +
        k[1327]*y[IDX_NI]*y[IDX_C2HII] + k[1330]*y[IDX_NI]*y[IDX_C2H2II] -
        k[1977]*y[IDX_CHII] + k[1979]*y[IDX_CH2II] + k[1984]*y[IDX_CH3II] +
        k[2000]*y[IDX_CHI] + k[2122]*y[IDX_HI]*y[IDX_CII] - k[2231]*y[IDX_CHII];
    ydot[IDX_NHI] = 0.0 - k[132]*y[IDX_HII]*y[IDX_NHI] -
        k[168]*y[IDX_H2II]*y[IDX_NHI] - k[247]*y[IDX_NII]*y[IDX_NHI] +
        k[260]*y[IDX_NHII]*y[IDX_H2COI] + k[261]*y[IDX_NHII]*y[IDX_H2OI] +
        k[262]*y[IDX_NHII]*y[IDX_NH3I] + k[263]*y[IDX_NHII]*y[IDX_NOI] +
        k[264]*y[IDX_NHII]*y[IDX_O2I] + k[265]*y[IDX_NHII]*y[IDX_SI] -
        k[294]*y[IDX_NHI]*y[IDX_CNII] - k[295]*y[IDX_NHI]*y[IDX_COII] -
        k[296]*y[IDX_NHI]*y[IDX_N2II] - k[297]*y[IDX_NHI]*y[IDX_OII] +
        k[415]*y[IDX_HNCOI] + k[425]*y[IDX_NH2I] + k[428]*y[IDX_NH3I] -
        k[429]*y[IDX_NHI] - k[430]*y[IDX_NHI] + k[566]*y[IDX_N2HII]*y[IDX_EM] +
        k[569]*y[IDX_NH2II]*y[IDX_EM] + k[571]*y[IDX_NH3II]*y[IDX_EM] -
        k[631]*y[IDX_CII]*y[IDX_NHI] - k[731]*y[IDX_CHII]*y[IDX_NHI] +
        k[767]*y[IDX_CH2I]*y[IDX_NH2II] + k[855]*y[IDX_CHI]*y[IDX_NH2II] -
        k[929]*y[IDX_H2II]*y[IDX_NHI] + k[996]*y[IDX_H2OI]*y[IDX_CNII] -
        k[1058]*y[IDX_H3II]*y[IDX_NHI] - k[1256]*y[IDX_HeII]*y[IDX_NHI] +
        k[1289]*y[IDX_NII]*y[IDX_CH3OHI] + k[1290]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[1298]*y[IDX_NII]*y[IDX_H2COI] + k[1300]*y[IDX_NII]*y[IDX_H2SI] +
        k[1302]*y[IDX_NII]*y[IDX_H2SI] + k[1307]*y[IDX_NII]*y[IDX_NH3I] -
        k[1308]*y[IDX_NII]*y[IDX_NHI] - k[1366]*y[IDX_NHII]*y[IDX_NHI] +
        k[1374]*y[IDX_NH2II]*y[IDX_C2I] + k[1375]*y[IDX_NH2II]*y[IDX_C2HI] +
        k[1376]*y[IDX_NH2II]*y[IDX_H2COI] + k[1378]*y[IDX_NH2II]*y[IDX_H2OI] +
        k[1381]*y[IDX_NH2II]*y[IDX_H2SI] + k[1385]*y[IDX_NH2II]*y[IDX_HCNI] +
        k[1386]*y[IDX_NH2II]*y[IDX_HCOI] + k[1387]*y[IDX_NH2II]*y[IDX_HNCI] +
        k[1388]*y[IDX_NH2II]*y[IDX_NH2I] + k[1389]*y[IDX_NH2II]*y[IDX_NH3I] +
        k[1393]*y[IDX_NH2II]*y[IDX_SI] + k[1398]*y[IDX_NH2I]*y[IDX_COII] +
        k[1409]*y[IDX_NH2I]*y[IDX_NH3II] + k[1412]*y[IDX_NH3II]*y[IDX_C2I] -
        k[1444]*y[IDX_NHI]*y[IDX_C2II] - k[1445]*y[IDX_NHI]*y[IDX_C2II] -
        k[1446]*y[IDX_NHI]*y[IDX_CH3II] - k[1447]*y[IDX_NHI]*y[IDX_CH5II] -
        k[1448]*y[IDX_NHI]*y[IDX_COII] - k[1449]*y[IDX_NHI]*y[IDX_H2COII] -
        k[1450]*y[IDX_NHI]*y[IDX_H2OII] - k[1451]*y[IDX_NHI]*y[IDX_HCNII] -
        k[1452]*y[IDX_NHI]*y[IDX_HCOII] - k[1453]*y[IDX_NHI]*y[IDX_HNOII] -
        k[1454]*y[IDX_NHI]*y[IDX_N2HII] - k[1455]*y[IDX_NHI]*y[IDX_NH2II] -
        k[1456]*y[IDX_NHI]*y[IDX_NH3II] - k[1457]*y[IDX_NHI]*y[IDX_OII] -
        k[1458]*y[IDX_NHI]*y[IDX_O2II] - k[1459]*y[IDX_NHI]*y[IDX_O2HII] -
        k[1460]*y[IDX_NHI]*y[IDX_OHII] - k[1461]*y[IDX_NHI]*y[IDX_SII] +
        k[1601]*y[IDX_CI]*y[IDX_NH2I] - k[1602]*y[IDX_CI]*y[IDX_NHI] -
        k[1603]*y[IDX_CI]*y[IDX_NHI] + k[1627]*y[IDX_CH2I]*y[IDX_N2I] +
        k[1656]*y[IDX_CH3I]*y[IDX_NH2I] + k[1683]*y[IDX_CHI]*y[IDX_NI] +
        k[1716]*y[IDX_COI]*y[IDX_HNOI] + k[1728]*y[IDX_H2I]*y[IDX_NI] -
        k[1730]*y[IDX_H2I]*y[IDX_NHI] + k[1757]*y[IDX_HI]*y[IDX_HNOI] +
        k[1760]*y[IDX_HI]*y[IDX_NH2I] - k[1762]*y[IDX_HI]*y[IDX_NHI] +
        k[1764]*y[IDX_HI]*y[IDX_NOI] + k[1767]*y[IDX_HI]*y[IDX_NSI] +
        k[1773]*y[IDX_HI]*y[IDX_OCNI] + k[1791]*y[IDX_NI]*y[IDX_C2H3I] +
        k[1793]*y[IDX_NI]*y[IDX_C2H5I] + k[1803]*y[IDX_NI]*y[IDX_CH2I] +
        k[1810]*y[IDX_NI]*y[IDX_H2CNI] + k[1811]*y[IDX_NI]*y[IDX_HCOI] +
        k[1815]*y[IDX_NI]*y[IDX_HNOI] + k[1817]*y[IDX_NI]*y[IDX_HSI] -
        k[1819]*y[IDX_NI]*y[IDX_NHI] + k[1826]*y[IDX_NI]*y[IDX_O2HI] +
        k[1828]*y[IDX_NI]*y[IDX_OHI] + k[1836]*y[IDX_NH2I]*y[IDX_OHI] -
        k[1839]*y[IDX_NHI]*y[IDX_CH4I] - k[1840]*y[IDX_NHI]*y[IDX_CNI] -
        k[1841]*y[IDX_NHI]*y[IDX_H2OI] - k[1842]*y[IDX_NHI]*y[IDX_NH3I] -
        k[1843]*y[IDX_NHI]*y[IDX_NHI] - k[1843]*y[IDX_NHI]*y[IDX_NHI] -
        k[1844]*y[IDX_NHI]*y[IDX_NHI] - k[1844]*y[IDX_NHI]*y[IDX_NHI] -
        k[1845]*y[IDX_NHI]*y[IDX_NHI] - k[1845]*y[IDX_NHI]*y[IDX_NHI] -
        k[1846]*y[IDX_NHI]*y[IDX_NO2I] - k[1847]*y[IDX_NHI]*y[IDX_NOI] -
        k[1848]*y[IDX_NHI]*y[IDX_NOI] - k[1849]*y[IDX_NHI]*y[IDX_O2I] -
        k[1850]*y[IDX_NHI]*y[IDX_O2I] - k[1851]*y[IDX_NHI]*y[IDX_OI] -
        k[1852]*y[IDX_NHI]*y[IDX_OI] - k[1853]*y[IDX_NHI]*y[IDX_OHI] -
        k[1854]*y[IDX_NHI]*y[IDX_OHI] - k[1855]*y[IDX_NHI]*y[IDX_OHI] -
        k[1856]*y[IDX_NHI]*y[IDX_SI] - k[1857]*y[IDX_NHI]*y[IDX_SI] +
        k[1890]*y[IDX_OI]*y[IDX_HCNI] + k[1898]*y[IDX_OI]*y[IDX_HNOI] +
        k[1903]*y[IDX_OI]*y[IDX_NH2I] + k[2037]*y[IDX_HNCOI] +
        k[2050]*y[IDX_NH2I] + k[2053]*y[IDX_NH3I] - k[2054]*y[IDX_NHI] -
        k[2055]*y[IDX_NHI] - k[2222]*y[IDX_NHI];
    ydot[IDX_SII] = 0.0 + k[35]*y[IDX_C2II]*y[IDX_SI] +
        k[41]*y[IDX_C2HII]*y[IDX_SI] + k[59]*y[IDX_CHII]*y[IDX_SI] +
        k[99]*y[IDX_CNII]*y[IDX_SI] + k[106]*y[IDX_COII]*y[IDX_SI] +
        k[140]*y[IDX_HII]*y[IDX_SI] + k[173]*y[IDX_H2COII]*y[IDX_SI] +
        k[185]*y[IDX_H2OII]*y[IDX_SI] + k[199]*y[IDX_HCNII]*y[IDX_SI] -
        k[205]*y[IDX_HCOI]*y[IDX_SII] - k[229]*y[IDX_MgI]*y[IDX_SII] +
        k[258]*y[IDX_N2II]*y[IDX_SI] + k[265]*y[IDX_NHII]*y[IDX_SI] +
        k[270]*y[IDX_NH2II]*y[IDX_SI] - k[292]*y[IDX_NH3I]*y[IDX_SII] -
        k[303]*y[IDX_NOI]*y[IDX_SII] + k[323]*y[IDX_O2II]*y[IDX_SI] +
        k[338]*y[IDX_OHII]*y[IDX_SI] - k[343]*y[IDX_SII]*y[IDX_SiCI] -
        k[344]*y[IDX_SII]*y[IDX_SiSI] + k[345]*y[IDX_SI]*y[IDX_CII] +
        k[346]*y[IDX_SI]*y[IDX_H2SII] + k[347]*y[IDX_SI]*y[IDX_HSII] -
        k[354]*y[IDX_SiI]*y[IDX_SII] - k[355]*y[IDX_SiHI]*y[IDX_SII] +
        k[444]*y[IDX_SI] + k[640]*y[IDX_CII]*y[IDX_SOI] -
        k[658]*y[IDX_C2I]*y[IDX_SII] - k[676]*y[IDX_C2H4I]*y[IDX_SII] -
        k[772]*y[IDX_CH2I]*y[IDX_SII] - k[790]*y[IDX_CH3I]*y[IDX_SII] -
        k[821]*y[IDX_CH4I]*y[IDX_SII] - k[822]*y[IDX_CH4I]*y[IDX_SII] -
        k[861]*y[IDX_CHI]*y[IDX_SII] + k[895]*y[IDX_HII]*y[IDX_H2SI] +
        k[902]*y[IDX_HII]*y[IDX_HSI] + k[924]*y[IDX_H2II]*y[IDX_H2SI] -
        k[961]*y[IDX_H2I]*y[IDX_SII] - k[974]*y[IDX_H2COI]*y[IDX_SII] -
        k[975]*y[IDX_H2COI]*y[IDX_SII] + k[1105]*y[IDX_HI]*y[IDX_HSII] -
        k[1163]*y[IDX_HCOI]*y[IDX_SII] + k[1214]*y[IDX_HeII]*y[IDX_CSI] +
        k[1220]*y[IDX_HeII]*y[IDX_H2CSI] + k[1227]*y[IDX_HeII]*y[IDX_H2SI] +
        k[1247]*y[IDX_HeII]*y[IDX_HS2I] + k[1249]*y[IDX_HeII]*y[IDX_HSI] +
        k[1259]*y[IDX_HeII]*y[IDX_NSI] + k[1266]*y[IDX_HeII]*y[IDX_OCSI] +
        k[1269]*y[IDX_HeII]*y[IDX_S2I] + k[1270]*y[IDX_HeII]*y[IDX_SO2I] +
        k[1272]*y[IDX_HeII]*y[IDX_SOI] + k[1287]*y[IDX_HeII]*y[IDX_SiSI] +
        k[1302]*y[IDX_NII]*y[IDX_H2SI] + k[1313]*y[IDX_NII]*y[IDX_OCSI] +
        k[1316]*y[IDX_N2II]*y[IDX_H2SI] + k[1318]*y[IDX_N2II]*y[IDX_OCSI] -
        k[1461]*y[IDX_NHI]*y[IDX_SII] + k[1473]*y[IDX_OII]*y[IDX_H2SI] +
        k[1479]*y[IDX_OII]*y[IDX_OCSI] - k[1486]*y[IDX_O2I]*y[IDX_SII] +
        k[1503]*y[IDX_OI]*y[IDX_HSII] - k[1548]*y[IDX_OHI]*y[IDX_SII] -
        k[1550]*y[IDX_SII]*y[IDX_H2SI] - k[1551]*y[IDX_SII]*y[IDX_H2SI] -
        k[1552]*y[IDX_SII]*y[IDX_OCSI] - k[1569]*y[IDX_SiHI]*y[IDX_SII] +
        k[2005]*y[IDX_CSII] + k[2039]*y[IDX_HSII] + k[2073]*y[IDX_SI] -
        k[2105]*y[IDX_CI]*y[IDX_SII] - k[2117]*y[IDX_H2I]*y[IDX_SII] -
        k[2143]*y[IDX_SII]*y[IDX_EM] - k[2264]*y[IDX_SII];
    ydot[IDX_CNI] = 0.0 + k[20]*y[IDX_CII]*y[IDX_NCCNI] +
        k[36]*y[IDX_C2I]*y[IDX_CNII] + k[47]*y[IDX_C2HI]*y[IDX_CNII] +
        k[51]*y[IDX_CI]*y[IDX_CNII] + k[63]*y[IDX_CH2I]*y[IDX_CNII] +
        k[83]*y[IDX_CHI]*y[IDX_CNII] + k[93]*y[IDX_CNII]*y[IDX_COI] +
        k[94]*y[IDX_CNII]*y[IDX_H2COI] + k[95]*y[IDX_CNII]*y[IDX_HCNI] +
        k[96]*y[IDX_CNII]*y[IDX_HCOI] + k[97]*y[IDX_CNII]*y[IDX_NOI] +
        k[98]*y[IDX_CNII]*y[IDX_O2I] + k[99]*y[IDX_CNII]*y[IDX_SI] -
        k[100]*y[IDX_CNI]*y[IDX_N2II] - k[159]*y[IDX_H2II]*y[IDX_CNI] +
        k[191]*y[IDX_HI]*y[IDX_CNII] - k[237]*y[IDX_NII]*y[IDX_CNI] +
        k[272]*y[IDX_NH2I]*y[IDX_CNII] + k[294]*y[IDX_NHI]*y[IDX_CNII] +
        k[326]*y[IDX_OI]*y[IDX_CNII] + k[340]*y[IDX_OHI]*y[IDX_CNII] +
        k[376]*y[IDX_C2NI] + k[377]*y[IDX_C3NI] + k[387]*y[IDX_CH3CNI] -
        k[392]*y[IDX_CNI] + k[407]*y[IDX_HC3NI] + k[408]*y[IDX_HCNI] +
        k[414]*y[IDX_HNCI] + k[423]*y[IDX_NCCNI] + k[423]*y[IDX_NCCNI] +
        k[439]*y[IDX_OCNI] + k[469]*y[IDX_C2NII]*y[IDX_EM] +
        k[471]*y[IDX_C2N2II]*y[IDX_EM] + k[471]*y[IDX_C2N2II]*y[IDX_EM] +
        k[538]*y[IDX_HCNII]*y[IDX_EM] + k[539]*y[IDX_HCNHII]*y[IDX_EM] +
        k[635]*y[IDX_CII]*y[IDX_OCNI] + k[652]*y[IDX_C2I]*y[IDX_HCNII] +
        k[661]*y[IDX_C2HII]*y[IDX_HCNI] + k[665]*y[IDX_C2H2II]*y[IDX_CH3CNI] +
        k[679]*y[IDX_C2HI]*y[IDX_HCNII] + k[693]*y[IDX_CI]*y[IDX_HCNII] -
        k[713]*y[IDX_CHII]*y[IDX_CNI] + k[760]*y[IDX_CH2I]*y[IDX_HCNII] +
        k[846]*y[IDX_CHI]*y[IDX_HCNII] - k[869]*y[IDX_CNI]*y[IDX_HNOII] -
        k[870]*y[IDX_CNI]*y[IDX_O2HII] - k[917]*y[IDX_H2II]*y[IDX_CNI] +
        k[1002]*y[IDX_H2OI]*y[IDX_HCNII] - k[1035]*y[IDX_H3II]*y[IDX_CNI] +
        k[1097]*y[IDX_HI]*y[IDX_C2N2II] + k[1110]*y[IDX_HCNII]*y[IDX_CO2I] +
        k[1111]*y[IDX_HCNII]*y[IDX_COI] + k[1112]*y[IDX_HCNII]*y[IDX_H2COI] +
        k[1113]*y[IDX_HCNII]*y[IDX_HCNI] + k[1114]*y[IDX_HCNII]*y[IDX_HCOI] +
        k[1116]*y[IDX_HCNII]*y[IDX_HNCI] + k[1117]*y[IDX_HCNII]*y[IDX_SI] +
        k[1189]*y[IDX_HeII]*y[IDX_C2NI] + k[1190]*y[IDX_HeII]*y[IDX_C3NI] +
        k[1199]*y[IDX_HeII]*y[IDX_CH3CNI] - k[1207]*y[IDX_HeII]*y[IDX_CNI] -
        k[1208]*y[IDX_HeII]*y[IDX_CNI] + k[1230]*y[IDX_HeII]*y[IDX_HC3NI] +
        k[1251]*y[IDX_HeII]*y[IDX_NCCNI] + k[1263]*y[IDX_HeII]*y[IDX_OCNI] +
        k[1325]*y[IDX_NI]*y[IDX_C2II] + k[1327]*y[IDX_NI]*y[IDX_C2HII] +
        k[1342]*y[IDX_NI]*y[IDX_SiCII] - k[1349]*y[IDX_NHII]*y[IDX_CNI] +
        k[1403]*y[IDX_NH2I]*y[IDX_HCNII] + k[1451]*y[IDX_NHI]*y[IDX_HCNII] -
        k[1469]*y[IDX_OII]*y[IDX_CNI] - k[1519]*y[IDX_OHII]*y[IDX_CNI] +
        k[1541]*y[IDX_OHI]*y[IDX_HCNII] + k[1580]*y[IDX_C2HI]*y[IDX_NCCNI] +
        k[1584]*y[IDX_CI]*y[IDX_C2NI] - k[1590]*y[IDX_CI]*y[IDX_CNI] +
        k[1597]*y[IDX_CI]*y[IDX_N2I] + k[1598]*y[IDX_CI]*y[IDX_NCCNI] +
        k[1602]*y[IDX_CI]*y[IDX_NHI] + k[1604]*y[IDX_CI]*y[IDX_NOI] +
        k[1607]*y[IDX_CI]*y[IDX_NSI] + k[1609]*y[IDX_CI]*y[IDX_OCNI] -
        k[1623]*y[IDX_CH2I]*y[IDX_CNI] - k[1650]*y[IDX_CH3I]*y[IDX_CNI] -
        k[1670]*y[IDX_CH4I]*y[IDX_CNI] + k[1682]*y[IDX_CHI]*y[IDX_NI] -
        k[1701]*y[IDX_CNI]*y[IDX_C2H2I] - k[1702]*y[IDX_CNI]*y[IDX_C2H4I] -
        k[1703]*y[IDX_CNI]*y[IDX_CNI] - k[1703]*y[IDX_CNI]*y[IDX_CNI] -
        k[1704]*y[IDX_CNI]*y[IDX_H2COI] - k[1705]*y[IDX_CNI]*y[IDX_HCNI] -
        k[1706]*y[IDX_CNI]*y[IDX_HCOI] - k[1707]*y[IDX_CNI]*y[IDX_HNCI] -
        k[1708]*y[IDX_CNI]*y[IDX_HNOI] - k[1709]*y[IDX_CNI]*y[IDX_NO2I] -
        k[1710]*y[IDX_CNI]*y[IDX_NOI] - k[1711]*y[IDX_CNI]*y[IDX_NOI] -
        k[1712]*y[IDX_CNI]*y[IDX_O2I] - k[1713]*y[IDX_CNI]*y[IDX_O2I] -
        k[1714]*y[IDX_CNI]*y[IDX_SI] - k[1715]*y[IDX_CNI]*y[IDX_SiH4I] -
        k[1726]*y[IDX_H2I]*y[IDX_CNI] + k[1750]*y[IDX_HI]*y[IDX_HCNI] +
        k[1759]*y[IDX_HI]*y[IDX_NCCNI] + k[1774]*y[IDX_HI]*y[IDX_OCNI] +
        k[1790]*y[IDX_NI]*y[IDX_C2I] + k[1796]*y[IDX_NI]*y[IDX_C2NI] +
        k[1796]*y[IDX_NI]*y[IDX_C2NI] + k[1798]*y[IDX_NI]*y[IDX_C3NI] +
        k[1800]*y[IDX_NI]*y[IDX_C4NI] - k[1807]*y[IDX_NI]*y[IDX_CNI] +
        k[1809]*y[IDX_NI]*y[IDX_CSI] + k[1832]*y[IDX_NI]*y[IDX_SiCI] -
        k[1838]*y[IDX_NH3I]*y[IDX_CNI] - k[1840]*y[IDX_NHI]*y[IDX_CNI] +
        k[1876]*y[IDX_OI]*y[IDX_C2NI] - k[1880]*y[IDX_OI]*y[IDX_CNI] -
        k[1881]*y[IDX_OI]*y[IDX_CNI] + k[1889]*y[IDX_OI]*y[IDX_HCNI] +
        k[1911]*y[IDX_OI]*y[IDX_OCNI] - k[1932]*y[IDX_OHI]*y[IDX_CNI] -
        k[1933]*y[IDX_OHI]*y[IDX_CNI] + k[1939]*y[IDX_OHI]*y[IDX_HCNI] +
        k[1973]*y[IDX_C2NI] + k[1974]*y[IDX_C3NI] + k[1989]*y[IDX_CH3CNI] -
        k[2001]*y[IDX_CNI] + k[2027]*y[IDX_HC3NI] + k[2028]*y[IDX_HCNI] +
        k[2036]*y[IDX_HNCI] + k[2047]*y[IDX_NCCNI] + k[2047]*y[IDX_NCCNI] +
        k[2065]*y[IDX_OCNI] - k[2100]*y[IDX_C2HI]*y[IDX_CNI] +
        k[2102]*y[IDX_CI]*y[IDX_NI] - k[2109]*y[IDX_CH3I]*y[IDX_CNI] -
        k[2220]*y[IDX_CNI];
    ydot[IDX_H2COI] = 0.0 - k[16]*y[IDX_CII]*y[IDX_H2COI] -
        k[42]*y[IDX_C2H2II]*y[IDX_H2COI] + k[65]*y[IDX_CH2I]*y[IDX_H2COII] -
        k[76]*y[IDX_CH4II]*y[IDX_H2COI] + k[85]*y[IDX_CHI]*y[IDX_H2COII] -
        k[94]*y[IDX_CNII]*y[IDX_H2COI] - k[101]*y[IDX_COII]*y[IDX_H2COI] -
        k[119]*y[IDX_HII]*y[IDX_H2COI] - k[161]*y[IDX_H2II]*y[IDX_H2COI] +
        k[173]*y[IDX_H2COII]*y[IDX_SI] - k[174]*y[IDX_H2COI]*y[IDX_O2II] -
        k[178]*y[IDX_H2OII]*y[IDX_H2COI] + k[202]*y[IDX_HCOI]*y[IDX_H2COII] -
        k[212]*y[IDX_HeII]*y[IDX_H2COI] + k[222]*y[IDX_MgI]*y[IDX_H2COII] -
        k[239]*y[IDX_NII]*y[IDX_H2COI] - k[252]*y[IDX_N2II]*y[IDX_H2COI] -
        k[260]*y[IDX_NHII]*y[IDX_H2COI] + k[284]*y[IDX_NH3I]*y[IDX_H2COII] +
        k[298]*y[IDX_NOI]*y[IDX_H2COII] - k[311]*y[IDX_OII]*y[IDX_H2COI] -
        k[331]*y[IDX_OHII]*y[IDX_H2COI] + k[349]*y[IDX_SiI]*y[IDX_H2COII] +
        k[388]*y[IDX_CH3OHI] - k[399]*y[IDX_H2COI] +
        k[490]*y[IDX_CH3OH2II]*y[IDX_EM] + k[524]*y[IDX_H3COII]*y[IDX_EM] -
        k[617]*y[IDX_CII]*y[IDX_H2COI] - k[618]*y[IDX_CII]*y[IDX_H2COI] -
        k[660]*y[IDX_C2HII]*y[IDX_H2COI] + k[709]*y[IDX_CHII]*y[IDX_CH3OHI] -
        k[715]*y[IDX_CHII]*y[IDX_H2COI] - k[716]*y[IDX_CHII]*y[IDX_H2COI] -
        k[717]*y[IDX_CHII]*y[IDX_H2COI] - k[742]*y[IDX_CH2II]*y[IDX_H2COI] +
        k[773]*y[IDX_CH2I]*y[IDX_SiOII] - k[777]*y[IDX_CH3II]*y[IDX_H2COI] +
        k[793]*y[IDX_CH3OHI]*y[IDX_S2II] - k[798]*y[IDX_CH4II]*y[IDX_H2COI] -
        k[827]*y[IDX_CH5II]*y[IDX_H2COI] + k[844]*y[IDX_CHI]*y[IDX_H3COII] -
        k[865]*y[IDX_CNII]*y[IDX_H2COI] - k[871]*y[IDX_COII]*y[IDX_H2COI] -
        k[892]*y[IDX_HII]*y[IDX_H2COI] - k[893]*y[IDX_HII]*y[IDX_H2COI] -
        k[921]*y[IDX_H2II]*y[IDX_H2COI] - k[966]*y[IDX_H2COII]*y[IDX_H2COI] -
        k[969]*y[IDX_H2COI]*y[IDX_CH3OH2II] - k[970]*y[IDX_H2COI]*y[IDX_H3SII] -
        k[971]*y[IDX_H2COI]*y[IDX_HNOII] - k[972]*y[IDX_H2COI]*y[IDX_O2II] -
        k[973]*y[IDX_H2COI]*y[IDX_O2HII] - k[974]*y[IDX_H2COI]*y[IDX_SII] -
        k[975]*y[IDX_H2COI]*y[IDX_SII] - k[979]*y[IDX_H2OII]*y[IDX_H2COI] +
        k[1001]*y[IDX_H2OI]*y[IDX_H3COII] - k[1041]*y[IDX_H3II]*y[IDX_H2COI] +
        k[1078]*y[IDX_H3COII]*y[IDX_CH3OHI] + k[1079]*y[IDX_H3COII]*y[IDX_H2SI]
        - k[1086]*y[IDX_H3OII]*y[IDX_H2COI] - k[1112]*y[IDX_HCNII]*y[IDX_H2COI]
        + k[1122]*y[IDX_HCNI]*y[IDX_H3COII] - k[1132]*y[IDX_HCNHII]*y[IDX_H2COI]
        - k[1133]*y[IDX_HCNHII]*y[IDX_H2COI] - k[1141]*y[IDX_HCOII]*y[IDX_H2COI]
        + k[1166]*y[IDX_HNCI]*y[IDX_H3COII] - k[1216]*y[IDX_HeII]*y[IDX_H2COI] -
        k[1217]*y[IDX_HeII]*y[IDX_H2COI] - k[1218]*y[IDX_HeII]*y[IDX_H2COI] -
        k[1298]*y[IDX_NII]*y[IDX_H2COI] - k[1299]*y[IDX_NII]*y[IDX_H2COI] -
        k[1314]*y[IDX_N2II]*y[IDX_H2COI] - k[1323]*y[IDX_N2HII]*y[IDX_H2COI] -
        k[1354]*y[IDX_NHII]*y[IDX_H2COI] - k[1355]*y[IDX_NHII]*y[IDX_H2COI] -
        k[1376]*y[IDX_NH2II]*y[IDX_H2COI] - k[1377]*y[IDX_NH2II]*y[IDX_H2COI] +
        k[1401]*y[IDX_NH2I]*y[IDX_H3COII] - k[1413]*y[IDX_NH3II]*y[IDX_H2COI] +
        k[1427]*y[IDX_NH3I]*y[IDX_H3COII] - k[1471]*y[IDX_OII]*y[IDX_H2COI] -
        k[1522]*y[IDX_OHII]*y[IDX_H2COI] + k[1559]*y[IDX_SOII]*y[IDX_C2H4I] +
        k[1576]*y[IDX_C2H3I]*y[IDX_O2I] - k[1624]*y[IDX_CH2I]*y[IDX_H2COI] +
        k[1628]*y[IDX_CH2I]*y[IDX_NO2I] + k[1629]*y[IDX_CH2I]*y[IDX_NOI] +
        k[1635]*y[IDX_CH2I]*y[IDX_O2I] + k[1641]*y[IDX_CH2I]*y[IDX_OHI] -
        k[1651]*y[IDX_CH3I]*y[IDX_H2COI] + k[1658]*y[IDX_CH3I]*y[IDX_NO2I] +
        k[1660]*y[IDX_CH3I]*y[IDX_O2I] + k[1665]*y[IDX_CH3I]*y[IDX_OI] +
        k[1667]*y[IDX_CH3I]*y[IDX_OHI] - k[1678]*y[IDX_CHI]*y[IDX_H2COI] -
        k[1704]*y[IDX_CNI]*y[IDX_H2COI] - k[1747]*y[IDX_HI]*y[IDX_H2COI] +
        k[1781]*y[IDX_HCOI]*y[IDX_HCOI] + k[1782]*y[IDX_HCOI]*y[IDX_HNOI] +
        k[1786]*y[IDX_HCOI]*y[IDX_O2HI] + k[1872]*y[IDX_OI]*y[IDX_C2H4I] +
        k[1874]*y[IDX_OI]*y[IDX_C2H5I] - k[1886]*y[IDX_OI]*y[IDX_H2COI] -
        k[1937]*y[IDX_OHI]*y[IDX_H2COI] + k[1990]*y[IDX_CH3OHI] -
        k[2011]*y[IDX_H2COI] - k[2012]*y[IDX_H2COI] - k[2013]*y[IDX_H2COI] -
        k[2014]*y[IDX_H2COI] + k[2032]*y[IDX_HCOOCH3I] + k[2032]*y[IDX_HCOOCH3I]
        + k[2136]*y[IDX_H2COII]*y[IDX_EM] - k[2208]*y[IDX_H2COI] +
        k[2367]*y[IDX_GH2COI] + k[2368]*y[IDX_GH2COI] + k[2369]*y[IDX_GH2COI] +
        k[2370]*y[IDX_GH2COI];
    ydot[IDX_CH4I] = 0.0 + k[75]*y[IDX_CH4II]*y[IDX_C2H2I] +
        k[76]*y[IDX_CH4II]*y[IDX_H2COI] + k[77]*y[IDX_CH4II]*y[IDX_H2SI] +
        k[78]*y[IDX_CH4II]*y[IDX_NH3I] + k[79]*y[IDX_CH4II]*y[IDX_O2I] +
        k[80]*y[IDX_CH4II]*y[IDX_OCSI] - k[81]*y[IDX_CH4I]*y[IDX_COII] -
        k[116]*y[IDX_HII]*y[IDX_CH4I] - k[157]*y[IDX_H2II]*y[IDX_CH4I] -
        k[210]*y[IDX_HeII]*y[IDX_CH4I] - k[236]*y[IDX_NII]*y[IDX_CH4I] -
        k[309]*y[IDX_OII]*y[IDX_CH4I] - k[390]*y[IDX_CH4I] +
        k[411]*y[IDX_HCOOCH3I] + k[496]*y[IDX_CH5II]*y[IDX_EM] -
        k[614]*y[IDX_CII]*y[IDX_CH4I] + k[689]*y[IDX_CI]*y[IDX_CH5II] -
        k[711]*y[IDX_CHII]*y[IDX_CH4I] + k[755]*y[IDX_CH2I]*y[IDX_CH5II] +
        k[776]*y[IDX_CH3II]*y[IDX_CH3OHI] + k[777]*y[IDX_CH3II]*y[IDX_H2COI] +
        k[789]*y[IDX_CH3II]*y[IDX_SiH4I] - k[795]*y[IDX_CH4II]*y[IDX_CH4I] -
        k[803]*y[IDX_CH4I]*y[IDX_C2II] - k[804]*y[IDX_CH4I]*y[IDX_C2II] -
        k[805]*y[IDX_CH4I]*y[IDX_C2HII] - k[806]*y[IDX_CH4I]*y[IDX_C2H2II] -
        k[807]*y[IDX_CH4I]*y[IDX_COII] - k[808]*y[IDX_CH4I]*y[IDX_CSII] -
        k[809]*y[IDX_CH4I]*y[IDX_H2COII] - k[810]*y[IDX_CH4I]*y[IDX_H2OII] -
        k[811]*y[IDX_CH4I]*y[IDX_HCNII] - k[812]*y[IDX_CH4I]*y[IDX_HCO2II] -
        k[813]*y[IDX_CH4I]*y[IDX_HNOII] - k[814]*y[IDX_CH4I]*y[IDX_HSII] -
        k[815]*y[IDX_CH4I]*y[IDX_N2II] - k[816]*y[IDX_CH4I]*y[IDX_N2II] -
        k[817]*y[IDX_CH4I]*y[IDX_N2HII] - k[818]*y[IDX_CH4I]*y[IDX_NH3II] -
        k[819]*y[IDX_CH4I]*y[IDX_OHII] - k[820]*y[IDX_CH4I]*y[IDX_OHII] -
        k[821]*y[IDX_CH4I]*y[IDX_SII] - k[822]*y[IDX_CH4I]*y[IDX_SII] +
        k[823]*y[IDX_CH5II]*y[IDX_C2I] + k[824]*y[IDX_CH5II]*y[IDX_C2HI] +
        k[825]*y[IDX_CH5II]*y[IDX_CO2I] + k[826]*y[IDX_CH5II]*y[IDX_COI] +
        k[827]*y[IDX_CH5II]*y[IDX_H2COI] + k[828]*y[IDX_CH5II]*y[IDX_H2OI] +
        k[829]*y[IDX_CH5II]*y[IDX_H2SI] + k[830]*y[IDX_CH5II]*y[IDX_HCNI] +
        k[831]*y[IDX_CH5II]*y[IDX_HCOI] + k[832]*y[IDX_CH5II]*y[IDX_HClI] +
        k[833]*y[IDX_CH5II]*y[IDX_HNCI] + k[834]*y[IDX_CH5II]*y[IDX_MgI] +
        k[835]*y[IDX_CH5II]*y[IDX_SI] + k[836]*y[IDX_CH5II]*y[IDX_SiH4I] +
        k[840]*y[IDX_CHI]*y[IDX_CH5II] - k[890]*y[IDX_HII]*y[IDX_CH4I] -
        k[914]*y[IDX_H2II]*y[IDX_CH4I] - k[915]*y[IDX_H2II]*y[IDX_CH4I] +
        k[1023]*y[IDX_H3II]*y[IDX_C2H5OHI] + k[1024]*y[IDX_H3II]*y[IDX_C2H5OHI]
        - k[1033]*y[IDX_H3II]*y[IDX_CH4I] - k[1202]*y[IDX_HeII]*y[IDX_CH4I] -
        k[1203]*y[IDX_HeII]*y[IDX_CH4I] - k[1204]*y[IDX_HeII]*y[IDX_CH4I] -
        k[1205]*y[IDX_HeII]*y[IDX_CH4I] - k[1293]*y[IDX_NII]*y[IDX_CH4I] -
        k[1294]*y[IDX_NII]*y[IDX_CH4I] - k[1295]*y[IDX_NII]*y[IDX_CH4I] +
        k[1397]*y[IDX_NH2I]*y[IDX_CH5II] + k[1422]*y[IDX_NH3I]*y[IDX_CH5II] +
        k[1447]*y[IDX_NHI]*y[IDX_CH5II] - k[1468]*y[IDX_OII]*y[IDX_CH4I] +
        k[1538]*y[IDX_OHI]*y[IDX_CH5II] - k[1622]*y[IDX_CH2I]*y[IDX_CH4I] +
        k[1646]*y[IDX_CH3I]*y[IDX_C2H3I] + k[1649]*y[IDX_CH3I]*y[IDX_CH3I] +
        k[1651]*y[IDX_CH3I]*y[IDX_H2COI] + k[1652]*y[IDX_CH3I]*y[IDX_H2OI] +
        k[1653]*y[IDX_CH3I]*y[IDX_H2SI] + k[1654]*y[IDX_CH3I]*y[IDX_HCOI] +
        k[1655]*y[IDX_CH3I]*y[IDX_HNOI] + k[1656]*y[IDX_CH3I]*y[IDX_NH2I] +
        k[1657]*y[IDX_CH3I]*y[IDX_NH3I] + k[1663]*y[IDX_CH3I]*y[IDX_O2HI] +
        k[1666]*y[IDX_CH3I]*y[IDX_OHI] - k[1670]*y[IDX_CH4I]*y[IDX_CNI] -
        k[1671]*y[IDX_CH4I]*y[IDX_O2I] - k[1672]*y[IDX_CH4I]*y[IDX_OHI] -
        k[1673]*y[IDX_CH4I]*y[IDX_SI] - k[1676]*y[IDX_CHI]*y[IDX_CH4I] +
        k[1724]*y[IDX_H2I]*y[IDX_CH3I] - k[1742]*y[IDX_HI]*y[IDX_CH4I] -
        k[1833]*y[IDX_NH2I]*y[IDX_CH4I] - k[1839]*y[IDX_NHI]*y[IDX_CH4I] -
        k[1879]*y[IDX_OI]*y[IDX_CH4I] - k[1995]*y[IDX_CH4I] -
        k[1996]*y[IDX_CH4I] - k[1997]*y[IDX_CH4I] - k[1998]*y[IDX_CH4I] -
        k[2217]*y[IDX_CH4I] + k[2303]*y[IDX_GCH4I] + k[2304]*y[IDX_GCH4I] +
        k[2305]*y[IDX_GCH4I] + k[2306]*y[IDX_GCH4I];
    ydot[IDX_HCOI] = 0.0 - k[18]*y[IDX_CII]*y[IDX_HCOI] -
        k[33]*y[IDX_C2II]*y[IDX_HCOI] - k[44]*y[IDX_C2H2II]*y[IDX_HCOI] -
        k[55]*y[IDX_CHII]*y[IDX_HCOI] - k[72]*y[IDX_CH3II]*y[IDX_HCOI] -
        k[96]*y[IDX_CNII]*y[IDX_HCOI] - k[103]*y[IDX_COII]*y[IDX_HCOI] -
        k[125]*y[IDX_HII]*y[IDX_HCOI] - k[165]*y[IDX_H2II]*y[IDX_HCOI] -
        k[180]*y[IDX_H2OII]*y[IDX_HCOI] - k[202]*y[IDX_HCOI]*y[IDX_H2COII] -
        k[203]*y[IDX_HCOI]*y[IDX_H2SII] - k[204]*y[IDX_HCOI]*y[IDX_O2II] -
        k[205]*y[IDX_HCOI]*y[IDX_SII] - k[206]*y[IDX_HCOI]*y[IDX_SiOII] +
        k[224]*y[IDX_MgI]*y[IDX_HCOII] - k[243]*y[IDX_NII]*y[IDX_HCOI] -
        k[254]*y[IDX_N2II]*y[IDX_HCOI] - k[267]*y[IDX_NH2II]*y[IDX_HCOI] -
        k[278]*y[IDX_NH3II]*y[IDX_HCOI] - k[314]*y[IDX_OII]*y[IDX_HCOI] -
        k[334]*y[IDX_OHII]*y[IDX_HCOI] - k[409]*y[IDX_HCOI] - k[410]*y[IDX_HCOI]
        + k[505]*y[IDX_H2COII]*y[IDX_EM] + k[525]*y[IDX_H3COII]*y[IDX_EM] +
        k[536]*y[IDX_H5C2O2II]*y[IDX_EM] + k[613]*y[IDX_CII]*y[IDX_CH3OHI] -
        k[626]*y[IDX_CII]*y[IDX_HCOI] - k[648]*y[IDX_C2II]*y[IDX_HCOI] +
        k[651]*y[IDX_C2I]*y[IDX_H2COII] - k[663]*y[IDX_C2HII]*y[IDX_HCOI] +
        k[678]*y[IDX_C2HI]*y[IDX_H2COII] - k[726]*y[IDX_CHII]*y[IDX_HCOI] +
        k[734]*y[IDX_CHII]*y[IDX_O2I] - k[747]*y[IDX_CH2II]*y[IDX_HCOI] +
        k[752]*y[IDX_CH2II]*y[IDX_OCSI] + k[757]*y[IDX_CH2I]*y[IDX_H2COII] -
        k[779]*y[IDX_CH3II]*y[IDX_HCOI] - k[831]*y[IDX_CH5II]*y[IDX_HCOI] +
        k[842]*y[IDX_CHI]*y[IDX_H2COII] - k[867]*y[IDX_CNII]*y[IDX_HCOI] +
        k[871]*y[IDX_COII]*y[IDX_H2COI] - k[897]*y[IDX_HII]*y[IDX_HCOI] -
        k[898]*y[IDX_HII]*y[IDX_HCOI] - k[925]*y[IDX_H2II]*y[IDX_HCOI] +
        k[965]*y[IDX_H2COII]*y[IDX_CH3OHI] + k[966]*y[IDX_H2COII]*y[IDX_H2COI] +
        k[968]*y[IDX_H2COII]*y[IDX_SI] - k[984]*y[IDX_H2OII]*y[IDX_HCOI] -
        k[985]*y[IDX_H2OII]*y[IDX_HCOI] + k[998]*y[IDX_H2OI]*y[IDX_H2COII] -
        k[1046]*y[IDX_H3II]*y[IDX_HCOI] - k[1114]*y[IDX_HCNII]*y[IDX_HCOI] -
        k[1115]*y[IDX_HCNII]*y[IDX_HCOI] + k[1121]*y[IDX_HCNI]*y[IDX_H2COII] -
        k[1144]*y[IDX_HCOII]*y[IDX_HCOI] - k[1158]*y[IDX_HCOI]*y[IDX_H2COII] -
        k[1159]*y[IDX_HCOI]*y[IDX_HNOII] - k[1160]*y[IDX_HCOI]*y[IDX_N2HII] -
        k[1161]*y[IDX_HCOI]*y[IDX_O2II] - k[1162]*y[IDX_HCOI]*y[IDX_O2HII] -
        k[1163]*y[IDX_HCOI]*y[IDX_SII] + k[1165]*y[IDX_HNCI]*y[IDX_H2COII] -
        k[1235]*y[IDX_HeII]*y[IDX_HCOI] - k[1236]*y[IDX_HeII]*y[IDX_HCOI] -
        k[1237]*y[IDX_HeII]*y[IDX_HCOI] - k[1303]*y[IDX_NII]*y[IDX_HCOI] -
        k[1317]*y[IDX_N2II]*y[IDX_HCOI] + k[1352]*y[IDX_NHII]*y[IDX_CO2I] -
        k[1361]*y[IDX_NHII]*y[IDX_HCOI] + k[1377]*y[IDX_NH2II]*y[IDX_H2COI] -
        k[1386]*y[IDX_NH2II]*y[IDX_HCOI] + k[1399]*y[IDX_NH2I]*y[IDX_H2COII] +
        k[1413]*y[IDX_NH3II]*y[IDX_H2COI] - k[1416]*y[IDX_NH3II]*y[IDX_HCOI] +
        k[1424]*y[IDX_NH3I]*y[IDX_H2COII] - k[1476]*y[IDX_OII]*y[IDX_HCOI] -
        k[1526]*y[IDX_OHII]*y[IDX_HCOI] - k[1527]*y[IDX_OHII]*y[IDX_HCOI] +
        k[1558]*y[IDX_SOII]*y[IDX_C2H2I] + k[1561]*y[IDX_SOII]*y[IDX_C2H4I] +
        k[1576]*y[IDX_C2H3I]*y[IDX_O2I] + k[1581]*y[IDX_C2HI]*y[IDX_O2I] -
        k[1594]*y[IDX_CI]*y[IDX_HCOI] + k[1624]*y[IDX_CH2I]*y[IDX_H2COI] -
        k[1625]*y[IDX_CH2I]*y[IDX_HCOI] + k[1636]*y[IDX_CH2I]*y[IDX_O2I] +
        k[1639]*y[IDX_CH2I]*y[IDX_OI] + k[1651]*y[IDX_CH3I]*y[IDX_H2COI] -
        k[1654]*y[IDX_CH3I]*y[IDX_HCOI] + k[1661]*y[IDX_CH3I]*y[IDX_O2I] +
        k[1677]*y[IDX_CHI]*y[IDX_CO2I] + k[1678]*y[IDX_CHI]*y[IDX_H2COI] -
        k[1679]*y[IDX_CHI]*y[IDX_HCOI] + k[1685]*y[IDX_CHI]*y[IDX_NOI] +
        k[1690]*y[IDX_CHI]*y[IDX_O2I] + k[1691]*y[IDX_CHI]*y[IDX_O2HI] +
        k[1696]*y[IDX_CHI]*y[IDX_OHI] + k[1704]*y[IDX_CNI]*y[IDX_H2COI] -
        k[1706]*y[IDX_CNI]*y[IDX_HCOI] + k[1747]*y[IDX_HI]*y[IDX_H2COI] -
        k[1751]*y[IDX_HI]*y[IDX_HCOI] - k[1752]*y[IDX_HI]*y[IDX_HCOI] -
        k[1780]*y[IDX_HCOI]*y[IDX_HCOI] - k[1780]*y[IDX_HCOI]*y[IDX_HCOI] -
        k[1781]*y[IDX_HCOI]*y[IDX_HCOI] - k[1781]*y[IDX_HCOI]*y[IDX_HCOI] -
        k[1782]*y[IDX_HCOI]*y[IDX_HNOI] - k[1783]*y[IDX_HCOI]*y[IDX_NOI] -
        k[1784]*y[IDX_HCOI]*y[IDX_O2I] - k[1785]*y[IDX_HCOI]*y[IDX_O2I] -
        k[1786]*y[IDX_HCOI]*y[IDX_O2HI] - k[1811]*y[IDX_NI]*y[IDX_HCOI] -
        k[1812]*y[IDX_NI]*y[IDX_HCOI] - k[1813]*y[IDX_NI]*y[IDX_HCOI] +
        k[1873]*y[IDX_OI]*y[IDX_C2H4I] + k[1886]*y[IDX_OI]*y[IDX_H2COI] -
        k[1892]*y[IDX_OI]*y[IDX_HCOI] - k[1893]*y[IDX_OI]*y[IDX_HCOI] +
        k[1937]*y[IDX_OHI]*y[IDX_H2COI] - k[1941]*y[IDX_OHI]*y[IDX_HCOI] -
        k[1951]*y[IDX_SI]*y[IDX_HCOI] - k[1952]*y[IDX_SI]*y[IDX_HCOI] -
        k[2030]*y[IDX_HCOI] - k[2031]*y[IDX_HCOI] - k[2218]*y[IDX_HCOI];
    ydot[IDX_HCNI] = 0.0 + k[1]*y[IDX_HII]*y[IDX_HNCI] +
        k[46]*y[IDX_C2H2I]*y[IDX_HCNII] - k[95]*y[IDX_CNII]*y[IDX_HCNI] -
        k[124]*y[IDX_HII]*y[IDX_HCNI] - k[164]*y[IDX_H2II]*y[IDX_HCNI] +
        k[188]*y[IDX_H2OI]*y[IDX_HCNII] + k[194]*y[IDX_HI]*y[IDX_HCNII] +
        k[197]*y[IDX_HCNII]*y[IDX_NOI] + k[198]*y[IDX_HCNII]*y[IDX_O2I] +
        k[199]*y[IDX_HCNII]*y[IDX_SI] - k[200]*y[IDX_HCNI]*y[IDX_COII] -
        k[201]*y[IDX_HCNI]*y[IDX_N2II] - k[242]*y[IDX_NII]*y[IDX_HCNI] +
        k[287]*y[IDX_NH3I]*y[IDX_HCNII] + k[398]*y[IDX_H2CNI] -
        k[408]*y[IDX_HCNI] + k[540]*y[IDX_HCNHII]*y[IDX_EM] +
        k[624]*y[IDX_CII]*y[IDX_HC3NI] - k[661]*y[IDX_C2HII]*y[IDX_HCNI] -
        k[662]*y[IDX_C2HII]*y[IDX_HCNI] - k[668]*y[IDX_C2H2II]*y[IDX_HCNI] -
        k[723]*y[IDX_CHII]*y[IDX_HCNI] - k[724]*y[IDX_CHII]*y[IDX_HCNI] -
        k[725]*y[IDX_CHII]*y[IDX_HCNI] + k[761]*y[IDX_CH2I]*y[IDX_HCNHII] -
        k[830]*y[IDX_CH5II]*y[IDX_HCNI] + k[847]*y[IDX_CHI]*y[IDX_HCNHII] +
        k[865]*y[IDX_CNII]*y[IDX_H2COI] - k[866]*y[IDX_CNII]*y[IDX_HCNI] +
        k[886]*y[IDX_HII]*y[IDX_CH3CNI] - k[983]*y[IDX_H2OII]*y[IDX_HCNI] +
        k[993]*y[IDX_H2OI]*y[IDX_C2NII] + k[1018]*y[IDX_H2SI]*y[IDX_C2NII] -
        k[1045]*y[IDX_H3II]*y[IDX_HCNI] - k[1088]*y[IDX_H3OII]*y[IDX_HCNI] -
        k[1113]*y[IDX_HCNII]*y[IDX_HCNI] - k[1118]*y[IDX_HCNI]*y[IDX_C2N2II] -
        k[1119]*y[IDX_HCNI]*y[IDX_C3II] - k[1120]*y[IDX_HCNI]*y[IDX_CH3OH2II] -
        k[1121]*y[IDX_HCNI]*y[IDX_H2COII] - k[1122]*y[IDX_HCNI]*y[IDX_H3COII] -
        k[1123]*y[IDX_HCNI]*y[IDX_H3SII] - k[1124]*y[IDX_HCNI]*y[IDX_HCOII] -
        k[1125]*y[IDX_HCNI]*y[IDX_HNOII] - k[1126]*y[IDX_HCNI]*y[IDX_HSII] -
        k[1127]*y[IDX_HCNI]*y[IDX_HSiSII] - k[1128]*y[IDX_HCNI]*y[IDX_N2HII] -
        k[1129]*y[IDX_HCNI]*y[IDX_O2HII] + k[1130]*y[IDX_HCNHII]*y[IDX_CH3CNI] +
        k[1132]*y[IDX_HCNHII]*y[IDX_H2COI] + k[1134]*y[IDX_HCNHII]*y[IDX_H2SI] -
        k[1231]*y[IDX_HeII]*y[IDX_HCNI] - k[1232]*y[IDX_HeII]*y[IDX_HCNI] -
        k[1233]*y[IDX_HeII]*y[IDX_HCNI] - k[1234]*y[IDX_HeII]*y[IDX_HCNI] +
        k[1330]*y[IDX_NI]*y[IDX_C2H2II] - k[1360]*y[IDX_NHII]*y[IDX_HCNI] -
        k[1385]*y[IDX_NH2II]*y[IDX_HCNI] + k[1404]*y[IDX_NH2I]*y[IDX_HCNHII] +
        k[1421]*y[IDX_NH3I]*y[IDX_C2NII] + k[1431]*y[IDX_NH3I]*y[IDX_HCNHII] -
        k[1474]*y[IDX_OII]*y[IDX_HCNI] - k[1475]*y[IDX_OII]*y[IDX_HCNI] -
        k[1525]*y[IDX_OHII]*y[IDX_HCNI] - k[1571]*y[IDX_C2I]*y[IDX_HCNI] +
        k[1574]*y[IDX_C2H2I]*y[IDX_NOI] - k[1578]*y[IDX_C2HI]*y[IDX_HCNI] +
        k[1599]*y[IDX_CI]*y[IDX_NH2I] + k[1623]*y[IDX_CH2I]*y[IDX_CNI] +
        k[1627]*y[IDX_CH2I]*y[IDX_N2I] + k[1630]*y[IDX_CH2I]*y[IDX_NOI] +
        k[1650]*y[IDX_CH3I]*y[IDX_CNI] + k[1659]*y[IDX_CH3I]*y[IDX_NOI] +
        k[1670]*y[IDX_CH4I]*y[IDX_CNI] + k[1681]*y[IDX_CHI]*y[IDX_N2I] +
        k[1684]*y[IDX_CHI]*y[IDX_NOI] + k[1702]*y[IDX_CNI]*y[IDX_C2H4I] +
        k[1704]*y[IDX_CNI]*y[IDX_H2COI] - k[1705]*y[IDX_CNI]*y[IDX_HCNI] +
        k[1706]*y[IDX_CNI]*y[IDX_HCOI] + k[1708]*y[IDX_CNI]*y[IDX_HNOI] +
        k[1715]*y[IDX_CNI]*y[IDX_SiH4I] + k[1726]*y[IDX_H2I]*y[IDX_CNI] +
        k[1746]*y[IDX_HI]*y[IDX_H2CNI] - k[1750]*y[IDX_HI]*y[IDX_HCNI] +
        k[1754]*y[IDX_HI]*y[IDX_HNCI] + k[1759]*y[IDX_HI]*y[IDX_NCCNI] +
        k[1772]*y[IDX_HI]*y[IDX_OCNI] + k[1792]*y[IDX_NI]*y[IDX_C2H4I] +
        k[1801]*y[IDX_NI]*y[IDX_CH2I] + k[1805]*y[IDX_NI]*y[IDX_CH3I] +
        k[1806]*y[IDX_NI]*y[IDX_CH3I] + k[1810]*y[IDX_NI]*y[IDX_H2CNI] +
        k[1812]*y[IDX_NI]*y[IDX_HCOI] + k[1814]*y[IDX_NI]*y[IDX_HCSI] +
        k[1838]*y[IDX_NH3I]*y[IDX_CNI] + k[1840]*y[IDX_NHI]*y[IDX_CNI] -
        k[1889]*y[IDX_OI]*y[IDX_HCNI] - k[1890]*y[IDX_OI]*y[IDX_HCNI] -
        k[1891]*y[IDX_OI]*y[IDX_HCNI] + k[1932]*y[IDX_OHI]*y[IDX_CNI] -
        k[1939]*y[IDX_OHI]*y[IDX_HCNI] - k[1940]*y[IDX_OHI]*y[IDX_HCNI] +
        k[1943]*y[IDX_OHI]*y[IDX_NCCNI] + k[2010]*y[IDX_H2CNI] -
        k[2028]*y[IDX_HCNI] - k[2108]*y[IDX_CH3II]*y[IDX_HCNI] -
        k[2223]*y[IDX_HCNI] + k[2331]*y[IDX_GHCNI] + k[2332]*y[IDX_GHCNI] +
        k[2333]*y[IDX_GHCNI] + k[2334]*y[IDX_GHCNI];
    ydot[IDX_NOI] = 0.0 - k[22]*y[IDX_CII]*y[IDX_NOI] -
        k[34]*y[IDX_C2II]*y[IDX_NOI] - k[40]*y[IDX_C2HII]*y[IDX_NOI] -
        k[45]*y[IDX_C2H2II]*y[IDX_NOI] - k[58]*y[IDX_CHII]*y[IDX_NOI] -
        k[61]*y[IDX_CH2II]*y[IDX_NOI] - k[74]*y[IDX_CH3II]*y[IDX_NOI] -
        k[97]*y[IDX_CNII]*y[IDX_NOI] - k[104]*y[IDX_COII]*y[IDX_NOI] -
        k[133]*y[IDX_HII]*y[IDX_NOI] - k[169]*y[IDX_H2II]*y[IDX_NOI] -
        k[182]*y[IDX_H2OII]*y[IDX_NOI] - k[197]*y[IDX_HCNII]*y[IDX_NOI] +
        k[227]*y[IDX_MgI]*y[IDX_NOII] - k[248]*y[IDX_NII]*y[IDX_NOI] -
        k[255]*y[IDX_N2II]*y[IDX_NOI] - k[263]*y[IDX_NHII]*y[IDX_NOI] -
        k[269]*y[IDX_NH2II]*y[IDX_NOI] - k[280]*y[IDX_NH3II]*y[IDX_NOI] -
        k[298]*y[IDX_NOI]*y[IDX_H2COII] - k[299]*y[IDX_NOI]*y[IDX_H2SII] -
        k[300]*y[IDX_NOI]*y[IDX_HNOII] - k[301]*y[IDX_NOI]*y[IDX_HSII] -
        k[302]*y[IDX_NOI]*y[IDX_O2II] - k[303]*y[IDX_NOI]*y[IDX_SII] -
        k[304]*y[IDX_NOI]*y[IDX_S2II] - k[305]*y[IDX_NOI]*y[IDX_SiOII] -
        k[336]*y[IDX_OHII]*y[IDX_NOI] + k[352]*y[IDX_SiI]*y[IDX_NOII] +
        k[416]*y[IDX_HNOI] + k[431]*y[IDX_NO2I] - k[432]*y[IDX_NOI] -
        k[433]*y[IDX_NOI] + k[511]*y[IDX_H2NOII]*y[IDX_EM] +
        k[549]*y[IDX_HNOII]*y[IDX_EM] + k[654]*y[IDX_C2I]*y[IDX_HNOII] +
        k[681]*y[IDX_C2HI]*y[IDX_HNOII] + k[696]*y[IDX_CI]*y[IDX_HNOII] +
        k[764]*y[IDX_CH2I]*y[IDX_HNOII] + k[813]*y[IDX_CH4I]*y[IDX_HNOII] +
        k[850]*y[IDX_CHI]*y[IDX_HNOII] + k[869]*y[IDX_CNI]*y[IDX_HNOII] +
        k[876]*y[IDX_COI]*y[IDX_HNOII] - k[930]*y[IDX_H2II]*y[IDX_NOI] +
        k[971]*y[IDX_H2COI]*y[IDX_HNOII] + k[1005]*y[IDX_H2OI]*y[IDX_HNOII] -
        k[1060]*y[IDX_H3II]*y[IDX_NOI] + k[1125]*y[IDX_HCNI]*y[IDX_HNOII] +
        k[1159]*y[IDX_HCOI]*y[IDX_HNOII] + k[1169]*y[IDX_HNCI]*y[IDX_HNOII] +
        k[1173]*y[IDX_HNOII]*y[IDX_CO2I] + k[1174]*y[IDX_HNOII]*y[IDX_SI] +
        k[1246]*y[IDX_HeII]*y[IDX_HNOI] - k[1257]*y[IDX_HeII]*y[IDX_NOI] -
        k[1258]*y[IDX_HeII]*y[IDX_NOI] + k[1292]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[1296]*y[IDX_NII]*y[IDX_CO2I] - k[1309]*y[IDX_NII]*y[IDX_NOI] +
        k[1311]*y[IDX_NII]*y[IDX_O2I] + k[1312]*y[IDX_NII]*y[IDX_OCSI] +
        k[1319]*y[IDX_N2I]*y[IDX_HNOII] + k[1344]*y[IDX_NI]*y[IDX_SiOII] -
        k[1367]*y[IDX_NHII]*y[IDX_NOI] + k[1407]*y[IDX_NH2I]*y[IDX_HNOII] +
        k[1436]*y[IDX_NH3I]*y[IDX_HNOII] + k[1453]*y[IDX_NHI]*y[IDX_HNOII] -
        k[1462]*y[IDX_NOI]*y[IDX_O2HII] - k[1531]*y[IDX_OHII]*y[IDX_NOI] +
        k[1544]*y[IDX_OHI]*y[IDX_HNOII] - k[1574]*y[IDX_C2H2I]*y[IDX_NOI] -
        k[1604]*y[IDX_CI]*y[IDX_NOI] - k[1605]*y[IDX_CI]*y[IDX_NOI] +
        k[1626]*y[IDX_CH2I]*y[IDX_HNOI] + k[1628]*y[IDX_CH2I]*y[IDX_NO2I] -
        k[1629]*y[IDX_CH2I]*y[IDX_NOI] - k[1630]*y[IDX_CH2I]*y[IDX_NOI] -
        k[1631]*y[IDX_CH2I]*y[IDX_NOI] + k[1655]*y[IDX_CH3I]*y[IDX_HNOI] -
        k[1659]*y[IDX_CH3I]*y[IDX_NOI] + k[1680]*y[IDX_CHI]*y[IDX_HNOI] -
        k[1684]*y[IDX_CHI]*y[IDX_NOI] - k[1685]*y[IDX_CHI]*y[IDX_NOI] -
        k[1686]*y[IDX_CHI]*y[IDX_NOI] + k[1708]*y[IDX_CNI]*y[IDX_HNOI] +
        k[1709]*y[IDX_CNI]*y[IDX_NO2I] - k[1710]*y[IDX_CNI]*y[IDX_NOI] -
        k[1711]*y[IDX_CNI]*y[IDX_NOI] + k[1712]*y[IDX_CNI]*y[IDX_O2I] +
        k[1717]*y[IDX_COI]*y[IDX_NO2I] + k[1756]*y[IDX_HI]*y[IDX_HNOI] +
        k[1763]*y[IDX_HI]*y[IDX_NO2I] - k[1764]*y[IDX_HI]*y[IDX_NOI] -
        k[1765]*y[IDX_HI]*y[IDX_NOI] + k[1782]*y[IDX_HCOI]*y[IDX_HNOI] -
        k[1783]*y[IDX_HCOI]*y[IDX_NOI] + k[1808]*y[IDX_NI]*y[IDX_CO2I] +
        k[1815]*y[IDX_NI]*y[IDX_HNOI] + k[1821]*y[IDX_NI]*y[IDX_NO2I] +
        k[1821]*y[IDX_NI]*y[IDX_NO2I] - k[1823]*y[IDX_NI]*y[IDX_NOI] +
        k[1825]*y[IDX_NI]*y[IDX_O2I] + k[1827]*y[IDX_NI]*y[IDX_OHI] +
        k[1831]*y[IDX_NI]*y[IDX_SOI] - k[1834]*y[IDX_NH2I]*y[IDX_NOI] -
        k[1835]*y[IDX_NH2I]*y[IDX_NOI] + k[1846]*y[IDX_NHI]*y[IDX_NO2I] -
        k[1847]*y[IDX_NHI]*y[IDX_NOI] - k[1848]*y[IDX_NHI]*y[IDX_NOI] +
        k[1850]*y[IDX_NHI]*y[IDX_O2I] + k[1851]*y[IDX_NHI]*y[IDX_OI] -
        k[1858]*y[IDX_NOI]*y[IDX_NOI] - k[1858]*y[IDX_NOI]*y[IDX_NOI] -
        k[1859]*y[IDX_NOI]*y[IDX_O2I] - k[1860]*y[IDX_NOI]*y[IDX_OCNI] -
        k[1861]*y[IDX_NOI]*y[IDX_SI] - k[1862]*y[IDX_NOI]*y[IDX_SI] +
        k[1863]*y[IDX_O2I]*y[IDX_OCNI] + k[1881]*y[IDX_OI]*y[IDX_CNI] +
        k[1897]*y[IDX_OI]*y[IDX_HNOI] + k[1901]*y[IDX_OI]*y[IDX_N2I] +
        k[1905]*y[IDX_OI]*y[IDX_NO2I] - k[1906]*y[IDX_OI]*y[IDX_NOI] +
        k[1907]*y[IDX_OI]*y[IDX_NSI] + k[1910]*y[IDX_OI]*y[IDX_OCNI] +
        k[1942]*y[IDX_OHI]*y[IDX_HNOI] - k[1945]*y[IDX_OHI]*y[IDX_NOI] -
        k[1958]*y[IDX_SiI]*y[IDX_NOI] + k[2038]*y[IDX_HNOI] +
        k[2056]*y[IDX_NO2I] - k[2057]*y[IDX_NOI] - k[2058]*y[IDX_NOI] -
        k[2212]*y[IDX_NOI] + k[2363]*y[IDX_GNOI] + k[2364]*y[IDX_GNOI] +
        k[2365]*y[IDX_GNOI] + k[2366]*y[IDX_GNOI];
    ydot[IDX_CHI] = 0.0 - k[0]*y[IDX_CHI]*y[IDX_OI] -
        k[2]*y[IDX_H2I]*y[IDX_CHI] - k[9]*y[IDX_HI]*y[IDX_CHI] -
        k[15]*y[IDX_CII]*y[IDX_CHI] + k[55]*y[IDX_CHII]*y[IDX_HCOI] +
        k[56]*y[IDX_CHII]*y[IDX_MgI] + k[57]*y[IDX_CHII]*y[IDX_NH3I] +
        k[58]*y[IDX_CHII]*y[IDX_NOI] + k[59]*y[IDX_CHII]*y[IDX_SI] +
        k[60]*y[IDX_CHII]*y[IDX_SiI] - k[82]*y[IDX_CHI]*y[IDX_C2II] -
        k[83]*y[IDX_CHI]*y[IDX_CNII] - k[84]*y[IDX_CHI]*y[IDX_COII] -
        k[85]*y[IDX_CHI]*y[IDX_H2COII] - k[86]*y[IDX_CHI]*y[IDX_H2OII] -
        k[87]*y[IDX_CHI]*y[IDX_NII] - k[88]*y[IDX_CHI]*y[IDX_N2II] -
        k[89]*y[IDX_CHI]*y[IDX_NH2II] - k[90]*y[IDX_CHI]*y[IDX_OII] -
        k[91]*y[IDX_CHI]*y[IDX_O2II] - k[92]*y[IDX_CHI]*y[IDX_OHII] -
        k[117]*y[IDX_HII]*y[IDX_CHI] - k[158]*y[IDX_H2II]*y[IDX_CHI] -
        k[211]*y[IDX_HeII]*y[IDX_CHI] + k[382]*y[IDX_CH2I] + k[386]*y[IDX_CH3I]
        - k[391]*y[IDX_CHI] + k[460]*y[IDX_C2HII]*y[IDX_EM] +
        k[463]*y[IDX_C2H2II]*y[IDX_EM] + k[463]*y[IDX_C2H2II]*y[IDX_EM] +
        k[480]*y[IDX_CH2II]*y[IDX_EM] + k[482]*y[IDX_CH3II]*y[IDX_EM] +
        k[483]*y[IDX_CH3II]*y[IDX_EM] + k[497]*y[IDX_CH5II]*y[IDX_EM] +
        k[522]*y[IDX_H3COII]*y[IDX_EM] + k[546]*y[IDX_HCSII]*y[IDX_EM] +
        k[612]*y[IDX_CII]*y[IDX_CH3OHI] - k[615]*y[IDX_CII]*y[IDX_CHI] +
        k[618]*y[IDX_CII]*y[IDX_H2COI] - k[712]*y[IDX_CHII]*y[IDX_CHI] +
        k[745]*y[IDX_CH2II]*y[IDX_H2SI] + k[748]*y[IDX_CH2II]*y[IDX_NH3I] +
        k[756]*y[IDX_CH2I]*y[IDX_COII] - k[837]*y[IDX_CHI]*y[IDX_C2II] -
        k[838]*y[IDX_CHI]*y[IDX_C2HII] - k[839]*y[IDX_CHI]*y[IDX_CH3II] -
        k[840]*y[IDX_CHI]*y[IDX_CH5II] - k[841]*y[IDX_CHI]*y[IDX_COII] -
        k[842]*y[IDX_CHI]*y[IDX_H2COII] - k[843]*y[IDX_CHI]*y[IDX_H2OII] -
        k[844]*y[IDX_CHI]*y[IDX_H3COII] - k[845]*y[IDX_CHI]*y[IDX_H3OII] -
        k[846]*y[IDX_CHI]*y[IDX_HCNII] - k[847]*y[IDX_CHI]*y[IDX_HCNHII] -
        k[848]*y[IDX_CHI]*y[IDX_HCNHII] - k[849]*y[IDX_CHI]*y[IDX_HCOII] -
        k[850]*y[IDX_CHI]*y[IDX_HNOII] - k[851]*y[IDX_CHI]*y[IDX_HSII] -
        k[852]*y[IDX_CHI]*y[IDX_NII] - k[853]*y[IDX_CHI]*y[IDX_N2HII] -
        k[854]*y[IDX_CHI]*y[IDX_NHII] - k[855]*y[IDX_CHI]*y[IDX_NH2II] -
        k[856]*y[IDX_CHI]*y[IDX_NH3II] - k[857]*y[IDX_CHI]*y[IDX_OII] -
        k[858]*y[IDX_CHI]*y[IDX_O2II] - k[859]*y[IDX_CHI]*y[IDX_O2HII] -
        k[860]*y[IDX_CHI]*y[IDX_OHII] - k[861]*y[IDX_CHI]*y[IDX_SII] -
        k[862]*y[IDX_CHI]*y[IDX_SiII] - k[863]*y[IDX_CHI]*y[IDX_SiHII] -
        k[864]*y[IDX_CHI]*y[IDX_SiOII] - k[916]*y[IDX_H2II]*y[IDX_CHI] -
        k[1034]*y[IDX_H3II]*y[IDX_CHI] + k[1180]*y[IDX_HeII]*y[IDX_C2H2I] +
        k[1188]*y[IDX_HeII]*y[IDX_C2HI] - k[1206]*y[IDX_HeII]*y[IDX_CHI] +
        k[1229]*y[IDX_HeII]*y[IDX_HC3NI] + k[1232]*y[IDX_HeII]*y[IDX_HCNI] +
        k[1465]*y[IDX_OII]*y[IDX_C2HI] + k[1475]*y[IDX_OII]*y[IDX_HCNI] +
        k[1492]*y[IDX_OI]*y[IDX_C2H2II] + k[1587]*y[IDX_CI]*y[IDX_CH2I] +
        k[1587]*y[IDX_CI]*y[IDX_CH2I] - k[1589]*y[IDX_CI]*y[IDX_CHI] +
        k[1594]*y[IDX_CI]*y[IDX_HCOI] + k[1596]*y[IDX_CI]*y[IDX_HSI] +
        k[1601]*y[IDX_CI]*y[IDX_NH2I] + k[1603]*y[IDX_CI]*y[IDX_NHI] +
        k[1612]*y[IDX_CI]*y[IDX_OHI] + k[1621]*y[IDX_CH2I]*y[IDX_CH2I] +
        k[1623]*y[IDX_CH2I]*y[IDX_CNI] + k[1640]*y[IDX_CH2I]*y[IDX_OI] +
        k[1642]*y[IDX_CH2I]*y[IDX_OHI] - k[1674]*y[IDX_CHI]*y[IDX_C2H2I] -
        k[1675]*y[IDX_CHI]*y[IDX_C2H4I] - k[1676]*y[IDX_CHI]*y[IDX_CH4I] -
        k[1677]*y[IDX_CHI]*y[IDX_CO2I] - k[1678]*y[IDX_CHI]*y[IDX_H2COI] -
        k[1679]*y[IDX_CHI]*y[IDX_HCOI] - k[1680]*y[IDX_CHI]*y[IDX_HNOI] -
        k[1681]*y[IDX_CHI]*y[IDX_N2I] - k[1682]*y[IDX_CHI]*y[IDX_NI] -
        k[1683]*y[IDX_CHI]*y[IDX_NI] - k[1684]*y[IDX_CHI]*y[IDX_NOI] -
        k[1685]*y[IDX_CHI]*y[IDX_NOI] - k[1686]*y[IDX_CHI]*y[IDX_NOI] -
        k[1687]*y[IDX_CHI]*y[IDX_O2I] - k[1688]*y[IDX_CHI]*y[IDX_O2I] -
        k[1689]*y[IDX_CHI]*y[IDX_O2I] - k[1690]*y[IDX_CHI]*y[IDX_O2I] -
        k[1691]*y[IDX_CHI]*y[IDX_O2HI] - k[1692]*y[IDX_CHI]*y[IDX_O2HI] -
        k[1693]*y[IDX_CHI]*y[IDX_OI] - k[1694]*y[IDX_CHI]*y[IDX_OI] -
        k[1695]*y[IDX_CHI]*y[IDX_OCSI] - k[1696]*y[IDX_CHI]*y[IDX_OHI] -
        k[1697]*y[IDX_CHI]*y[IDX_SI] - k[1698]*y[IDX_CHI]*y[IDX_SI] -
        k[1699]*y[IDX_CHI]*y[IDX_SOI] - k[1700]*y[IDX_CHI]*y[IDX_SOI] +
        k[1722]*y[IDX_H2I]*y[IDX_CI] - k[1725]*y[IDX_H2I]*y[IDX_CHI] +
        k[1736]*y[IDX_HI]*y[IDX_C2I] + k[1739]*y[IDX_HI]*y[IDX_CH2I] -
        k[1743]*y[IDX_HI]*y[IDX_CHI] + k[1803]*y[IDX_NI]*y[IDX_CH2I] +
        k[1875]*y[IDX_OI]*y[IDX_C2HI] + k[1980]*y[IDX_CH2II] +
        k[1982]*y[IDX_CH2I] + k[1988]*y[IDX_CH3I] + k[1998]*y[IDX_CH4I] -
        k[1999]*y[IDX_CHI] - k[2000]*y[IDX_CHI] - k[2115]*y[IDX_H2I]*y[IDX_CHI]
        + k[2123]*y[IDX_HI]*y[IDX_CI] - k[2210]*y[IDX_CHI];
    ydot[IDX_NH3I] = 0.0 - k[21]*y[IDX_CII]*y[IDX_NH3I] -
        k[57]*y[IDX_CHII]*y[IDX_NH3I] - k[78]*y[IDX_CH4II]*y[IDX_NH3I] -
        k[131]*y[IDX_HII]*y[IDX_NH3I] - k[167]*y[IDX_H2II]*y[IDX_NH3I] -
        k[216]*y[IDX_HeII]*y[IDX_NH3I] - k[246]*y[IDX_NII]*y[IDX_NH3I] -
        k[262]*y[IDX_NHII]*y[IDX_NH3I] - k[268]*y[IDX_NH2II]*y[IDX_NH3I] +
        k[278]*y[IDX_NH3II]*y[IDX_HCOI] + k[279]*y[IDX_NH3II]*y[IDX_MgI] +
        k[280]*y[IDX_NH3II]*y[IDX_NOI] + k[281]*y[IDX_NH3II]*y[IDX_SiI] -
        k[282]*y[IDX_NH3I]*y[IDX_C2H2II] - k[283]*y[IDX_NH3I]*y[IDX_COII] -
        k[284]*y[IDX_NH3I]*y[IDX_H2COII] - k[285]*y[IDX_NH3I]*y[IDX_H2OII] -
        k[286]*y[IDX_NH3I]*y[IDX_H2SII] - k[287]*y[IDX_NH3I]*y[IDX_HCNII] -
        k[288]*y[IDX_NH3I]*y[IDX_HSII] - k[289]*y[IDX_NH3I]*y[IDX_N2II] -
        k[290]*y[IDX_NH3I]*y[IDX_O2II] - k[291]*y[IDX_NH3I]*y[IDX_OCSII] -
        k[292]*y[IDX_NH3I]*y[IDX_SII] - k[293]*y[IDX_NH3I]*y[IDX_SOII] -
        k[316]*y[IDX_OII]*y[IDX_NH3I] - k[335]*y[IDX_OHII]*y[IDX_NH3I] -
        k[426]*y[IDX_NH3I] - k[427]*y[IDX_NH3I] - k[428]*y[IDX_NH3I] +
        k[574]*y[IDX_NH4II]*y[IDX_EM] - k[630]*y[IDX_CII]*y[IDX_NH3I] -
        k[730]*y[IDX_CHII]*y[IDX_NH3I] - k[748]*y[IDX_CH2II]*y[IDX_NH3I] -
        k[781]*y[IDX_CH3II]*y[IDX_NH3I] - k[792]*y[IDX_CH3OH2II]*y[IDX_NH3I] -
        k[801]*y[IDX_CH4II]*y[IDX_NH3I] - k[1057]*y[IDX_H3II]*y[IDX_NH3I] -
        k[1254]*y[IDX_HeII]*y[IDX_NH3I] - k[1255]*y[IDX_HeII]*y[IDX_NH3I] -
        k[1306]*y[IDX_NII]*y[IDX_NH3I] - k[1307]*y[IDX_NII]*y[IDX_NH3I] -
        k[1365]*y[IDX_NHII]*y[IDX_NH3I] + k[1382]*y[IDX_NH2II]*y[IDX_H2SI] -
        k[1389]*y[IDX_NH2II]*y[IDX_NH3I] - k[1417]*y[IDX_NH3II]*y[IDX_NH3I] -
        k[1418]*y[IDX_NH3I]*y[IDX_C2HII] - k[1419]*y[IDX_NH3I]*y[IDX_C2H2II] -
        k[1420]*y[IDX_NH3I]*y[IDX_C2H5OH2II] - k[1421]*y[IDX_NH3I]*y[IDX_C2NII]
        - k[1422]*y[IDX_NH3I]*y[IDX_CH5II] - k[1423]*y[IDX_NH3I]*y[IDX_COII] -
        k[1424]*y[IDX_NH3I]*y[IDX_H2COII] - k[1425]*y[IDX_NH3I]*y[IDX_H2OII] -
        k[1426]*y[IDX_NH3I]*y[IDX_H2SII] - k[1427]*y[IDX_NH3I]*y[IDX_H3COII] -
        k[1428]*y[IDX_NH3I]*y[IDX_H3OII] - k[1429]*y[IDX_NH3I]*y[IDX_H3SII] -
        k[1430]*y[IDX_NH3I]*y[IDX_HCNII] - k[1431]*y[IDX_NH3I]*y[IDX_HCNHII] -
        k[1432]*y[IDX_NH3I]*y[IDX_HCNHII] - k[1433]*y[IDX_NH3I]*y[IDX_HCOII] -
        k[1434]*y[IDX_NH3I]*y[IDX_HCO2II] - k[1435]*y[IDX_NH3I]*y[IDX_HCSII] -
        k[1436]*y[IDX_NH3I]*y[IDX_HNOII] - k[1437]*y[IDX_NH3I]*y[IDX_HSII] -
        k[1438]*y[IDX_NH3I]*y[IDX_HSO2II] - k[1439]*y[IDX_NH3I]*y[IDX_HSiSII] -
        k[1440]*y[IDX_NH3I]*y[IDX_N2HII] - k[1441]*y[IDX_NH3I]*y[IDX_O2HII] -
        k[1442]*y[IDX_NH3I]*y[IDX_SiHII] - k[1443]*y[IDX_NH3I]*y[IDX_SiOHII] -
        k[1530]*y[IDX_OHII]*y[IDX_NH3I] - k[1657]*y[IDX_CH3I]*y[IDX_NH3I] +
        k[1729]*y[IDX_H2I]*y[IDX_NH2I] - k[1761]*y[IDX_HI]*y[IDX_NH3I] +
        k[1833]*y[IDX_NH2I]*y[IDX_CH4I] + k[1837]*y[IDX_NH2I]*y[IDX_OHI] -
        k[1838]*y[IDX_NH3I]*y[IDX_CNI] - k[1842]*y[IDX_NHI]*y[IDX_NH3I] -
        k[1904]*y[IDX_OI]*y[IDX_NH3I] - k[1944]*y[IDX_OHI]*y[IDX_NH3I] -
        k[2051]*y[IDX_NH3I] - k[2052]*y[IDX_NH3I] - k[2053]*y[IDX_NH3I] -
        k[2225]*y[IDX_NH3I] + k[2307]*y[IDX_GNH3I] + k[2308]*y[IDX_GNH3I] +
        k[2309]*y[IDX_GNH3I] + k[2310]*y[IDX_GNH3I];
    ydot[IDX_H3OII] = 0.0 - k[528]*y[IDX_H3OII]*y[IDX_EM] -
        k[529]*y[IDX_H3OII]*y[IDX_EM] - k[530]*y[IDX_H3OII]*y[IDX_EM] -
        k[531]*y[IDX_H3OII]*y[IDX_EM] - k[692]*y[IDX_CI]*y[IDX_H3OII] +
        k[719]*y[IDX_CHII]*y[IDX_H2OI] - k[759]*y[IDX_CH2I]*y[IDX_H3OII] +
        k[799]*y[IDX_CH4II]*y[IDX_H2OI] + k[810]*y[IDX_CH4I]*y[IDX_H2OII] +
        k[820]*y[IDX_CH4I]*y[IDX_OHII] + k[828]*y[IDX_CH5II]*y[IDX_H2OI] -
        k[845]*y[IDX_CHI]*y[IDX_H3OII] + k[922]*y[IDX_H2II]*y[IDX_H2OI] +
        k[945]*y[IDX_H2I]*y[IDX_H2OII] + k[980]*y[IDX_H2OII]*y[IDX_H2OI] +
        k[982]*y[IDX_H2OII]*y[IDX_H2SI] + k[984]*y[IDX_H2OII]*y[IDX_HCOI] +
        k[991]*y[IDX_H2OI]*y[IDX_C2H2II] + k[998]*y[IDX_H2OI]*y[IDX_H2COII] +
        k[999]*y[IDX_H2OI]*y[IDX_H2ClII] + k[1000]*y[IDX_H2OI]*y[IDX_H2SII] +
        k[1001]*y[IDX_H2OI]*y[IDX_H3COII] + k[1002]*y[IDX_H2OI]*y[IDX_HCNII] +
        k[1003]*y[IDX_H2OI]*y[IDX_HCOII] + k[1004]*y[IDX_H2OI]*y[IDX_HCO2II] +
        k[1005]*y[IDX_H2OI]*y[IDX_HNOII] + k[1006]*y[IDX_H2OI]*y[IDX_HOCSII] +
        k[1007]*y[IDX_H2OI]*y[IDX_HSII] + k[1008]*y[IDX_H2OI]*y[IDX_HSO2II] +
        k[1011]*y[IDX_H2OI]*y[IDX_N2HII] + k[1012]*y[IDX_H2OI]*y[IDX_O2HII] +
        k[1014]*y[IDX_H2OI]*y[IDX_SiHII] + k[1015]*y[IDX_H2OI]*y[IDX_SiH4II] +
        k[1016]*y[IDX_H2OI]*y[IDX_SiH5II] + k[1043]*y[IDX_H3II]*y[IDX_H2OI] -
        k[1080]*y[IDX_H3OII]*y[IDX_C2I] - k[1081]*y[IDX_H3OII]*y[IDX_C2H5OHI] -
        k[1082]*y[IDX_H3OII]*y[IDX_CH3CCHI] - k[1083]*y[IDX_H3OII]*y[IDX_CH3CNI]
        - k[1084]*y[IDX_H3OII]*y[IDX_CH3OHI] - k[1085]*y[IDX_H3OII]*y[IDX_CSI] -
        k[1086]*y[IDX_H3OII]*y[IDX_H2COI] - k[1087]*y[IDX_H3OII]*y[IDX_H2SI] -
        k[1088]*y[IDX_H3OII]*y[IDX_HCNI] - k[1089]*y[IDX_H3OII]*y[IDX_HCOOCH3I]
        - k[1090]*y[IDX_H3OII]*y[IDX_HNCI] - k[1091]*y[IDX_H3OII]*y[IDX_HS2I] -
        k[1092]*y[IDX_H3OII]*y[IDX_S2I] - k[1093]*y[IDX_H3OII]*y[IDX_SiI] -
        k[1094]*y[IDX_H3OII]*y[IDX_SiH2I] - k[1095]*y[IDX_H3OII]*y[IDX_SiHI] -
        k[1096]*y[IDX_H3OII]*y[IDX_SiOI] + k[1356]*y[IDX_NHII]*y[IDX_H2OI] +
        k[1378]*y[IDX_NH2II]*y[IDX_H2OI] - k[1402]*y[IDX_NH2I]*y[IDX_H3OII] -
        k[1428]*y[IDX_NH3I]*y[IDX_H3OII] + k[1450]*y[IDX_NHI]*y[IDX_H2OII] +
        k[1495]*y[IDX_OI]*y[IDX_CH5II] + k[1523]*y[IDX_OHII]*y[IDX_H2OI] +
        k[1540]*y[IDX_OHI]*y[IDX_H2OII] - k[2121]*y[IDX_H3OII]*y[IDX_C2H4I] -
        k[2246]*y[IDX_H3OII];
    ydot[IDX_O2I] = 0.0 - k[6]*y[IDX_H2I]*y[IDX_O2I] -
        k[12]*y[IDX_HI]*y[IDX_O2I] + k[39]*y[IDX_C2I]*y[IDX_O2II] +
        k[54]*y[IDX_CI]*y[IDX_O2II] + k[70]*y[IDX_CH2I]*y[IDX_O2II] -
        k[79]*y[IDX_CH4II]*y[IDX_O2I] + k[91]*y[IDX_CHI]*y[IDX_O2II] -
        k[98]*y[IDX_CNII]*y[IDX_O2I] - k[105]*y[IDX_COII]*y[IDX_O2I] -
        k[135]*y[IDX_HII]*y[IDX_O2I] - k[170]*y[IDX_H2II]*y[IDX_O2I] +
        k[174]*y[IDX_H2COI]*y[IDX_O2II] - k[183]*y[IDX_H2OII]*y[IDX_O2I] -
        k[198]*y[IDX_HCNII]*y[IDX_O2I] + k[204]*y[IDX_HCOI]*y[IDX_O2II] -
        k[217]*y[IDX_HeII]*y[IDX_O2I] + k[228]*y[IDX_MgI]*y[IDX_O2II] -
        k[249]*y[IDX_NII]*y[IDX_O2I] - k[256]*y[IDX_N2II]*y[IDX_O2I] -
        k[264]*y[IDX_NHII]*y[IDX_O2I] + k[276]*y[IDX_NH2I]*y[IDX_O2II] +
        k[290]*y[IDX_NH3I]*y[IDX_O2II] + k[302]*y[IDX_NOI]*y[IDX_O2II] -
        k[317]*y[IDX_OII]*y[IDX_O2I] + k[321]*y[IDX_O2II]*y[IDX_C2H2I] +
        k[322]*y[IDX_O2II]*y[IDX_H2SI] + k[323]*y[IDX_O2II]*y[IDX_SI] -
        k[324]*y[IDX_O2I]*y[IDX_ClII] - k[325]*y[IDX_O2I]*y[IDX_SO2II] -
        k[337]*y[IDX_OHII]*y[IDX_O2I] + k[353]*y[IDX_SiI]*y[IDX_O2II] -
        k[435]*y[IDX_O2I] - k[436]*y[IDX_O2I] + k[437]*y[IDX_O2HI] +
        k[578]*y[IDX_O2HII]*y[IDX_EM] - k[633]*y[IDX_CII]*y[IDX_O2I] -
        k[634]*y[IDX_CII]*y[IDX_O2I] - k[649]*y[IDX_C2II]*y[IDX_O2I] +
        k[657]*y[IDX_C2I]*y[IDX_O2HII] + k[683]*y[IDX_C2HI]*y[IDX_O2HII] +
        k[701]*y[IDX_CI]*y[IDX_O2HII] - k[732]*y[IDX_CHII]*y[IDX_O2I] -
        k[733]*y[IDX_CHII]*y[IDX_O2I] - k[734]*y[IDX_CHII]*y[IDX_O2I] -
        k[749]*y[IDX_CH2II]*y[IDX_O2I] + k[770]*y[IDX_CH2I]*y[IDX_O2HII] -
        k[782]*y[IDX_CH3II]*y[IDX_O2I] + k[859]*y[IDX_CHI]*y[IDX_O2HII] -
        k[868]*y[IDX_CNII]*y[IDX_O2I] + k[870]*y[IDX_CNI]*y[IDX_O2HII] +
        k[878]*y[IDX_COI]*y[IDX_O2HII] - k[931]*y[IDX_H2II]*y[IDX_O2I] +
        k[959]*y[IDX_H2I]*y[IDX_O2HII] - k[967]*y[IDX_H2COII]*y[IDX_O2I] +
        k[972]*y[IDX_H2COI]*y[IDX_O2II] + k[973]*y[IDX_H2COI]*y[IDX_O2HII] +
        k[1012]*y[IDX_H2OI]*y[IDX_O2HII] - k[1062]*y[IDX_H3II]*y[IDX_O2I] +
        k[1129]*y[IDX_HCNI]*y[IDX_O2HII] + k[1162]*y[IDX_HCOI]*y[IDX_O2HII] +
        k[1172]*y[IDX_HNCI]*y[IDX_O2HII] + k[1212]*y[IDX_HeII]*y[IDX_CO2I] -
        k[1261]*y[IDX_HeII]*y[IDX_O2I] + k[1270]*y[IDX_HeII]*y[IDX_SO2I] -
        k[1310]*y[IDX_NII]*y[IDX_O2I] - k[1311]*y[IDX_NII]*y[IDX_O2I] +
        k[1320]*y[IDX_N2I]*y[IDX_O2HII] - k[1368]*y[IDX_NHII]*y[IDX_O2I] -
        k[1369]*y[IDX_NHII]*y[IDX_O2I] - k[1390]*y[IDX_NH2II]*y[IDX_O2I] -
        k[1391]*y[IDX_NH2II]*y[IDX_O2I] + k[1410]*y[IDX_NH2I]*y[IDX_O2HII] +
        k[1441]*y[IDX_NH3I]*y[IDX_O2HII] + k[1459]*y[IDX_NHI]*y[IDX_O2HII] +
        k[1462]*y[IDX_NOI]*y[IDX_O2HII] + k[1478]*y[IDX_OII]*y[IDX_NO2I] +
        k[1481]*y[IDX_OII]*y[IDX_SO2I] + k[1483]*y[IDX_O2II]*y[IDX_CH3OHI] -
        k[1485]*y[IDX_O2I]*y[IDX_CSII] - k[1486]*y[IDX_O2I]*y[IDX_SII] -
        k[1487]*y[IDX_O2I]*y[IDX_SiSII] - k[1488]*y[IDX_O2I]*y[IDX_SiSII] +
        k[1489]*y[IDX_O2HII]*y[IDX_CO2I] + k[1500]*y[IDX_OI]*y[IDX_HCO2II] +
        k[1510]*y[IDX_OI]*y[IDX_O2HII] + k[1516]*y[IDX_OI]*y[IDX_SiOII] +
        k[1547]*y[IDX_OHI]*y[IDX_O2HII] + k[1554]*y[IDX_SI]*y[IDX_O2HII] -
        k[1567]*y[IDX_SiH2II]*y[IDX_O2I] - k[1572]*y[IDX_C2I]*y[IDX_O2I] -
        k[1576]*y[IDX_C2H3I]*y[IDX_O2I] - k[1577]*y[IDX_C2H3I]*y[IDX_O2I] -
        k[1581]*y[IDX_C2HI]*y[IDX_O2I] - k[1608]*y[IDX_CI]*y[IDX_O2I] -
        k[1632]*y[IDX_CH2I]*y[IDX_O2I] - k[1633]*y[IDX_CH2I]*y[IDX_O2I] -
        k[1634]*y[IDX_CH2I]*y[IDX_O2I] - k[1635]*y[IDX_CH2I]*y[IDX_O2I] -
        k[1636]*y[IDX_CH2I]*y[IDX_O2I] - k[1660]*y[IDX_CH3I]*y[IDX_O2I] -
        k[1661]*y[IDX_CH3I]*y[IDX_O2I] - k[1662]*y[IDX_CH3I]*y[IDX_O2I] +
        k[1663]*y[IDX_CH3I]*y[IDX_O2HI] - k[1671]*y[IDX_CH4I]*y[IDX_O2I] -
        k[1687]*y[IDX_CHI]*y[IDX_O2I] - k[1688]*y[IDX_CHI]*y[IDX_O2I] -
        k[1689]*y[IDX_CHI]*y[IDX_O2I] - k[1690]*y[IDX_CHI]*y[IDX_O2I] +
        k[1692]*y[IDX_CHI]*y[IDX_O2HI] - k[1712]*y[IDX_CNI]*y[IDX_O2I] -
        k[1713]*y[IDX_CNI]*y[IDX_O2I] - k[1718]*y[IDX_COI]*y[IDX_O2I] -
        k[1731]*y[IDX_H2I]*y[IDX_O2I] - k[1732]*y[IDX_H2I]*y[IDX_O2I] -
        k[1768]*y[IDX_HI]*y[IDX_O2I] + k[1770]*y[IDX_HI]*y[IDX_O2HI] -
        k[1784]*y[IDX_HCOI]*y[IDX_O2I] - k[1785]*y[IDX_HCOI]*y[IDX_O2I] +
        k[1786]*y[IDX_HCOI]*y[IDX_O2HI] + k[1822]*y[IDX_NI]*y[IDX_NO2I] -
        k[1825]*y[IDX_NI]*y[IDX_O2I] + k[1826]*y[IDX_NI]*y[IDX_O2HI] -
        k[1849]*y[IDX_NHI]*y[IDX_O2I] - k[1850]*y[IDX_NHI]*y[IDX_O2I] +
        k[1858]*y[IDX_NOI]*y[IDX_NOI] - k[1859]*y[IDX_NOI]*y[IDX_O2I] -
        k[1863]*y[IDX_O2I]*y[IDX_OCNI] - k[1864]*y[IDX_O2I]*y[IDX_OCNI] -
        k[1865]*y[IDX_O2I]*y[IDX_SI] - k[1866]*y[IDX_O2I]*y[IDX_SOI] +
        k[1882]*y[IDX_OI]*y[IDX_CO2I] + k[1898]*y[IDX_OI]*y[IDX_HNOI] +
        k[1905]*y[IDX_OI]*y[IDX_NO2I] + k[1906]*y[IDX_OI]*y[IDX_NOI] +
        k[1909]*y[IDX_OI]*y[IDX_O2HI] + k[1911]*y[IDX_OI]*y[IDX_OCNI] +
        k[1914]*y[IDX_OI]*y[IDX_OHI] + k[1916]*y[IDX_OI]*y[IDX_SO2I] +
        k[1917]*y[IDX_OI]*y[IDX_SOI] + k[1946]*y[IDX_OHI]*y[IDX_O2HI] -
        k[1959]*y[IDX_SiI]*y[IDX_O2I] - k[2061]*y[IDX_O2I] - k[2062]*y[IDX_O2I]
        + k[2063]*y[IDX_O2HI] + k[2128]*y[IDX_OI]*y[IDX_OI] - k[2253]*y[IDX_O2I]
        + k[2375]*y[IDX_GO2I] + k[2376]*y[IDX_GO2I] + k[2377]*y[IDX_GO2I] +
        k[2378]*y[IDX_GO2I];
    ydot[IDX_CII] = 0.0 - k[14]*y[IDX_CII]*y[IDX_CH2I] -
        k[15]*y[IDX_CII]*y[IDX_CHI] - k[16]*y[IDX_CII]*y[IDX_H2COI] -
        k[17]*y[IDX_CII]*y[IDX_H2SI] - k[18]*y[IDX_CII]*y[IDX_HCOI] -
        k[19]*y[IDX_CII]*y[IDX_MgI] - k[20]*y[IDX_CII]*y[IDX_NCCNI] -
        k[21]*y[IDX_CII]*y[IDX_NH3I] - k[22]*y[IDX_CII]*y[IDX_NOI] -
        k[23]*y[IDX_CII]*y[IDX_NSI] - k[24]*y[IDX_CII]*y[IDX_OCSI] -
        k[25]*y[IDX_CII]*y[IDX_SOI] - k[26]*y[IDX_CII]*y[IDX_SiI] -
        k[27]*y[IDX_CII]*y[IDX_SiC2I] - k[28]*y[IDX_CII]*y[IDX_SiC3I] -
        k[29]*y[IDX_CII]*y[IDX_SiCI] - k[30]*y[IDX_CII]*y[IDX_SiH2I] -
        k[31]*y[IDX_CII]*y[IDX_SiH3I] - k[32]*y[IDX_CII]*y[IDX_SiSI] +
        k[50]*y[IDX_CI]*y[IDX_C2II] + k[51]*y[IDX_CI]*y[IDX_CNII] +
        k[52]*y[IDX_CI]*y[IDX_COII] + k[53]*y[IDX_CI]*y[IDX_N2II] +
        k[54]*y[IDX_CI]*y[IDX_O2II] + k[209]*y[IDX_HeII]*y[IDX_CI] -
        k[345]*y[IDX_SI]*y[IDX_CII] + k[356]*y[IDX_CI] + k[379]*y[IDX_CI] +
        k[380]*y[IDX_CHII] - k[606]*y[IDX_CII]*y[IDX_C2H5OHI] -
        k[607]*y[IDX_CII]*y[IDX_C2HI] - k[608]*y[IDX_CII]*y[IDX_CH2I] -
        k[609]*y[IDX_CII]*y[IDX_CH3I] - k[610]*y[IDX_CII]*y[IDX_CH3I] -
        k[611]*y[IDX_CII]*y[IDX_CH3CCHI] - k[612]*y[IDX_CII]*y[IDX_CH3OHI] -
        k[613]*y[IDX_CII]*y[IDX_CH3OHI] - k[614]*y[IDX_CII]*y[IDX_CH4I] -
        k[615]*y[IDX_CII]*y[IDX_CHI] - k[616]*y[IDX_CII]*y[IDX_CO2I] -
        k[617]*y[IDX_CII]*y[IDX_H2COI] - k[618]*y[IDX_CII]*y[IDX_H2COI] -
        k[619]*y[IDX_CII]*y[IDX_H2CSI] - k[620]*y[IDX_CII]*y[IDX_H2OI] -
        k[621]*y[IDX_CII]*y[IDX_H2OI] - k[622]*y[IDX_CII]*y[IDX_H2SI] -
        k[623]*y[IDX_CII]*y[IDX_HC3NI] - k[624]*y[IDX_CII]*y[IDX_HC3NI] -
        k[625]*y[IDX_CII]*y[IDX_HC3NI] - k[626]*y[IDX_CII]*y[IDX_HCOI] -
        k[627]*y[IDX_CII]*y[IDX_HNCI] - k[628]*y[IDX_CII]*y[IDX_HSI] -
        k[629]*y[IDX_CII]*y[IDX_NH2I] - k[630]*y[IDX_CII]*y[IDX_NH3I] -
        k[631]*y[IDX_CII]*y[IDX_NHI] - k[632]*y[IDX_CII]*y[IDX_NSI] -
        k[633]*y[IDX_CII]*y[IDX_O2I] - k[634]*y[IDX_CII]*y[IDX_O2I] -
        k[635]*y[IDX_CII]*y[IDX_OCNI] - k[636]*y[IDX_CII]*y[IDX_OCSI] -
        k[637]*y[IDX_CII]*y[IDX_OHI] - k[638]*y[IDX_CII]*y[IDX_SO2I] -
        k[639]*y[IDX_CII]*y[IDX_SOI] - k[640]*y[IDX_CII]*y[IDX_SOI] -
        k[641]*y[IDX_CII]*y[IDX_SOI] - k[642]*y[IDX_CII]*y[IDX_SiCI] -
        k[643]*y[IDX_CII]*y[IDX_SiH2I] - k[644]*y[IDX_CII]*y[IDX_SiHI] -
        k[645]*y[IDX_CII]*y[IDX_SiOI] - k[646]*y[IDX_CII]*y[IDX_SiSI] -
        k[934]*y[IDX_H2I]*y[IDX_CII] + k[1098]*y[IDX_HI]*y[IDX_CHII] +
        k[1177]*y[IDX_HeII]*y[IDX_C2I] + k[1188]*y[IDX_HeII]*y[IDX_C2HI] +
        k[1189]*y[IDX_HeII]*y[IDX_C2NI] + k[1192]*y[IDX_HeII]*y[IDX_CH2I] +
        k[1206]*y[IDX_HeII]*y[IDX_CHI] + k[1208]*y[IDX_HeII]*y[IDX_CNI] +
        k[1212]*y[IDX_HeII]*y[IDX_CO2I] + k[1213]*y[IDX_HeII]*y[IDX_COI] +
        k[1215]*y[IDX_HeII]*y[IDX_CSI] + k[1233]*y[IDX_HeII]*y[IDX_HCNI] +
        k[1243]*y[IDX_HeII]*y[IDX_HNCI] + k[1277]*y[IDX_HeII]*y[IDX_SiCI] +
        k[1325]*y[IDX_NI]*y[IDX_C2II] + k[1960]*y[IDX_C2II] + k[1976]*y[IDX_CI]
        + k[1978]*y[IDX_CH2II] + k[2002]*y[IDX_COII] -
        k[2096]*y[IDX_CII]*y[IDX_CI] - k[2097]*y[IDX_CII]*y[IDX_NI] -
        k[2098]*y[IDX_CII]*y[IDX_OI] - k[2099]*y[IDX_CII]*y[IDX_SI] -
        k[2112]*y[IDX_H2I]*y[IDX_CII] - k[2122]*y[IDX_HI]*y[IDX_CII] -
        k[2132]*y[IDX_CII]*y[IDX_EM] - k[2221]*y[IDX_CII];
    ydot[IDX_SI] = 0.0 - k[35]*y[IDX_C2II]*y[IDX_SI] -
        k[41]*y[IDX_C2HII]*y[IDX_SI] - k[59]*y[IDX_CHII]*y[IDX_SI] -
        k[99]*y[IDX_CNII]*y[IDX_SI] - k[106]*y[IDX_COII]*y[IDX_SI] -
        k[140]*y[IDX_HII]*y[IDX_SI] - k[173]*y[IDX_H2COII]*y[IDX_SI] -
        k[185]*y[IDX_H2OII]*y[IDX_SI] - k[199]*y[IDX_HCNII]*y[IDX_SI] +
        k[205]*y[IDX_HCOI]*y[IDX_SII] + k[229]*y[IDX_MgI]*y[IDX_SII] -
        k[258]*y[IDX_N2II]*y[IDX_SI] - k[265]*y[IDX_NHII]*y[IDX_SI] -
        k[270]*y[IDX_NH2II]*y[IDX_SI] + k[292]*y[IDX_NH3I]*y[IDX_SII] +
        k[303]*y[IDX_NOI]*y[IDX_SII] - k[323]*y[IDX_O2II]*y[IDX_SI] -
        k[338]*y[IDX_OHII]*y[IDX_SI] + k[343]*y[IDX_SII]*y[IDX_SiCI] +
        k[344]*y[IDX_SII]*y[IDX_SiSI] - k[345]*y[IDX_SI]*y[IDX_CII] -
        k[346]*y[IDX_SI]*y[IDX_H2SII] - k[347]*y[IDX_SI]*y[IDX_HSII] +
        k[354]*y[IDX_SiI]*y[IDX_SII] + k[355]*y[IDX_SiHI]*y[IDX_SII] +
        k[396]*y[IDX_CSI] + k[404]*y[IDX_H2SI] + k[417]*y[IDX_HS2I] +
        k[418]*y[IDX_HSI] + k[434]*y[IDX_NSI] + k[441]*y[IDX_OCSI] +
        k[443]*y[IDX_S2I] + k[443]*y[IDX_S2I] - k[444]*y[IDX_SI] +
        k[446]*y[IDX_SOI] + k[457]*y[IDX_SiSI] + k[500]*y[IDX_CSII]*y[IDX_EM] +
        k[516]*y[IDX_H2SII]*y[IDX_EM] + k[535]*y[IDX_H3SII]*y[IDX_EM] +
        k[546]*y[IDX_HCSII]*y[IDX_EM] + k[554]*y[IDX_HSII]*y[IDX_EM] +
        k[555]*y[IDX_HS2II]*y[IDX_EM] + k[576]*y[IDX_NSII]*y[IDX_EM] +
        k[581]*y[IDX_OCSII]*y[IDX_EM] + k[583]*y[IDX_S2II]*y[IDX_EM] +
        k[583]*y[IDX_S2II]*y[IDX_EM] + k[584]*y[IDX_SOII]*y[IDX_EM] +
        k[585]*y[IDX_SO2II]*y[IDX_EM] + k[605]*y[IDX_SiSII]*y[IDX_EM] +
        k[641]*y[IDX_CII]*y[IDX_SOI] + k[646]*y[IDX_CII]*y[IDX_SiSI] -
        k[650]*y[IDX_C2II]*y[IDX_SI] - k[739]*y[IDX_CHII]*y[IDX_SI] -
        k[740]*y[IDX_CHII]*y[IDX_SI] - k[753]*y[IDX_CH2II]*y[IDX_SI] -
        k[787]*y[IDX_CH3II]*y[IDX_SI] - k[835]*y[IDX_CH5II]*y[IDX_SI] +
        k[851]*y[IDX_CHI]*y[IDX_HSII] - k[968]*y[IDX_H2COII]*y[IDX_SI] -
        k[987]*y[IDX_H2OII]*y[IDX_SI] - k[988]*y[IDX_H2OII]*y[IDX_SI] +
        k[1007]*y[IDX_H2OI]*y[IDX_HSII] - k[1068]*y[IDX_H3II]*y[IDX_SI] -
        k[1117]*y[IDX_HCNII]*y[IDX_SI] + k[1126]*y[IDX_HCNI]*y[IDX_HSII] -
        k[1151]*y[IDX_HCOII]*y[IDX_SI] + k[1170]*y[IDX_HNCI]*y[IDX_HSII] -
        k[1174]*y[IDX_HNOII]*y[IDX_SI] + k[1175]*y[IDX_HSII]*y[IDX_H2SI] +
        k[1215]*y[IDX_HeII]*y[IDX_CSI] + k[1221]*y[IDX_HeII]*y[IDX_H2CSI] +
        k[1260]*y[IDX_HeII]*y[IDX_NSI] + k[1267]*y[IDX_HeII]*y[IDX_OCSI] +
        k[1269]*y[IDX_HeII]*y[IDX_S2I] + k[1273]*y[IDX_HeII]*y[IDX_SOI] +
        k[1288]*y[IDX_HeII]*y[IDX_SiSI] - k[1324]*y[IDX_N2HII]*y[IDX_SI] -
        k[1372]*y[IDX_NHII]*y[IDX_SI] - k[1373]*y[IDX_NHII]*y[IDX_SI] +
        k[1384]*y[IDX_NH2II]*y[IDX_H2SI] - k[1392]*y[IDX_NH2II]*y[IDX_SI] -
        k[1393]*y[IDX_NH2II]*y[IDX_SI] + k[1437]*y[IDX_NH3I]*y[IDX_HSII] -
        k[1484]*y[IDX_O2II]*y[IDX_SI] + k[1496]*y[IDX_OI]*y[IDX_CSII] +
        k[1502]*y[IDX_OI]*y[IDX_HCSII] + k[1509]*y[IDX_OI]*y[IDX_NSII] -
        k[1533]*y[IDX_OHII]*y[IDX_SI] - k[1534]*y[IDX_OHII]*y[IDX_SI] -
        k[1553]*y[IDX_SI]*y[IDX_H3SII] - k[1554]*y[IDX_SI]*y[IDX_O2HII] -
        k[1555]*y[IDX_SI]*y[IDX_SiOII] - k[1568]*y[IDX_SiH2II]*y[IDX_SI] -
        k[1573]*y[IDX_C2I]*y[IDX_SI] + k[1592]*y[IDX_CI]*y[IDX_CSI] +
        k[1596]*y[IDX_CI]*y[IDX_HSI] + k[1607]*y[IDX_CI]*y[IDX_NSI] +
        k[1613]*y[IDX_CI]*y[IDX_S2I] + k[1616]*y[IDX_CI]*y[IDX_SOI] -
        k[1644]*y[IDX_CH2I]*y[IDX_SI] - k[1645]*y[IDX_CH2I]*y[IDX_SI] -
        k[1669]*y[IDX_CH3I]*y[IDX_SI] - k[1673]*y[IDX_CH4I]*y[IDX_SI] -
        k[1697]*y[IDX_CHI]*y[IDX_SI] - k[1698]*y[IDX_CHI]*y[IDX_SI] -
        k[1714]*y[IDX_CNI]*y[IDX_SI] - k[1735]*y[IDX_H2I]*y[IDX_SI] +
        k[1758]*y[IDX_HI]*y[IDX_HSI] + k[1767]*y[IDX_HI]*y[IDX_NSI] +
        k[1777]*y[IDX_HI]*y[IDX_S2I] + k[1779]*y[IDX_HI]*y[IDX_SOI] +
        k[1789]*y[IDX_HSI]*y[IDX_HSI] + k[1809]*y[IDX_NI]*y[IDX_CSI] +
        k[1814]*y[IDX_NI]*y[IDX_HCSI] + k[1817]*y[IDX_NI]*y[IDX_HSI] +
        k[1824]*y[IDX_NI]*y[IDX_NSI] + k[1829]*y[IDX_NI]*y[IDX_S2I] +
        k[1831]*y[IDX_NI]*y[IDX_SOI] - k[1856]*y[IDX_NHI]*y[IDX_SI] -
        k[1857]*y[IDX_NHI]*y[IDX_SI] - k[1861]*y[IDX_NOI]*y[IDX_SI] -
        k[1862]*y[IDX_NOI]*y[IDX_SI] - k[1865]*y[IDX_O2I]*y[IDX_SI] +
        k[1883]*y[IDX_OI]*y[IDX_CSI] + k[1899]*y[IDX_OI]*y[IDX_HSI] +
        k[1907]*y[IDX_OI]*y[IDX_NSI] + k[1912]*y[IDX_OI]*y[IDX_OCSI] +
        k[1915]*y[IDX_OI]*y[IDX_S2I] + k[1917]*y[IDX_OI]*y[IDX_SOI] -
        k[1948]*y[IDX_OHI]*y[IDX_SI] - k[1951]*y[IDX_SI]*y[IDX_HCOI] -
        k[1952]*y[IDX_SI]*y[IDX_HCOI] - k[1953]*y[IDX_SI]*y[IDX_HSI] -
        k[1954]*y[IDX_SI]*y[IDX_SO2I] - k[1955]*y[IDX_SI]*y[IDX_SOI] +
        k[2007]*y[IDX_CSI] + k[2022]*y[IDX_H2SI] + k[2040]*y[IDX_HSII] +
        k[2042]*y[IDX_HS2I] + k[2043]*y[IDX_HSI] + k[2059]*y[IDX_NSI] +
        k[2067]*y[IDX_OCSI] + k[2072]*y[IDX_S2I] + k[2072]*y[IDX_S2I] -
        k[2073]*y[IDX_SI] + k[2075]*y[IDX_SOI] + k[2095]*y[IDX_SiSI] -
        k[2099]*y[IDX_CII]*y[IDX_SI] - k[2106]*y[IDX_CI]*y[IDX_SI] +
        k[2143]*y[IDX_SII]*y[IDX_EM] - k[2261]*y[IDX_SI];
    ydot[IDX_OHI] = 0.0 + k[4]*y[IDX_H2I]*y[IDX_H2OI] -
        k[7]*y[IDX_H2I]*y[IDX_OHI] + k[11]*y[IDX_HI]*y[IDX_H2OI] -
        k[13]*y[IDX_HI]*y[IDX_OHI] + k[71]*y[IDX_CH2I]*y[IDX_OHII] +
        k[92]*y[IDX_CHI]*y[IDX_OHII] - k[138]*y[IDX_HII]*y[IDX_OHI] -
        k[171]*y[IDX_H2II]*y[IDX_OHI] - k[251]*y[IDX_NII]*y[IDX_OHI] +
        k[277]*y[IDX_NH2I]*y[IDX_OHII] - k[319]*y[IDX_OII]*y[IDX_OHI] +
        k[329]*y[IDX_OHII]*y[IDX_C2I] + k[330]*y[IDX_OHII]*y[IDX_C2HI] +
        k[331]*y[IDX_OHII]*y[IDX_H2COI] + k[332]*y[IDX_OHII]*y[IDX_H2OI] +
        k[333]*y[IDX_OHII]*y[IDX_H2SI] + k[334]*y[IDX_OHII]*y[IDX_HCOI] +
        k[335]*y[IDX_OHII]*y[IDX_NH3I] + k[336]*y[IDX_OHII]*y[IDX_NOI] +
        k[337]*y[IDX_OHII]*y[IDX_O2I] + k[338]*y[IDX_OHII]*y[IDX_SI] -
        k[339]*y[IDX_OHI]*y[IDX_C2II] - k[340]*y[IDX_OHI]*y[IDX_CNII] -
        k[341]*y[IDX_OHI]*y[IDX_COII] - k[342]*y[IDX_OHI]*y[IDX_N2II] +
        k[372]*y[IDX_C2H5OHI] + k[389]*y[IDX_CH3OHI] + k[401]*y[IDX_H2OI] -
        k[442]*y[IDX_OHI] + k[466]*y[IDX_C2H5OH2II]*y[IDX_EM] +
        k[488]*y[IDX_CH3OH2II]*y[IDX_EM] + k[514]*y[IDX_H2OII]*y[IDX_EM] +
        k[521]*y[IDX_H3COII]*y[IDX_EM] + k[530]*y[IDX_H3OII]*y[IDX_EM] +
        k[531]*y[IDX_H3OII]*y[IDX_EM] + k[545]*y[IDX_HCO2II]*y[IDX_EM] +
        k[552]*y[IDX_HOCSII]*y[IDX_EM] + k[560]*y[IDX_HSO2II]*y[IDX_EM] +
        k[603]*y[IDX_SiOHII]*y[IDX_EM] - k[637]*y[IDX_CII]*y[IDX_OHI] +
        k[690]*y[IDX_CI]*y[IDX_H2OII] + k[732]*y[IDX_CHII]*y[IDX_O2I] -
        k[738]*y[IDX_CHII]*y[IDX_OHI] + k[749]*y[IDX_CH2II]*y[IDX_O2I] +
        k[758]*y[IDX_CH2I]*y[IDX_H2OII] - k[786]*y[IDX_CH3II]*y[IDX_OHI] +
        k[843]*y[IDX_CHI]*y[IDX_H2OII] + k[903]*y[IDX_HII]*y[IDX_NO2I] -
        k[933]*y[IDX_H2II]*y[IDX_OHI] + k[976]*y[IDX_H2OII]*y[IDX_C2I] +
        k[977]*y[IDX_H2OII]*y[IDX_C2HI] + k[978]*y[IDX_H2OII]*y[IDX_COI] +
        k[979]*y[IDX_H2OII]*y[IDX_H2COI] + k[980]*y[IDX_H2OII]*y[IDX_H2OI] +
        k[981]*y[IDX_H2OII]*y[IDX_H2SI] + k[983]*y[IDX_H2OII]*y[IDX_HCNI] +
        k[985]*y[IDX_H2OII]*y[IDX_HCOI] + k[986]*y[IDX_H2OII]*y[IDX_HNCI] +
        k[987]*y[IDX_H2OII]*y[IDX_SI] + k[989]*y[IDX_H2OII]*y[IDX_SO2I] +
        k[990]*y[IDX_H2OI]*y[IDX_C2II] + k[992]*y[IDX_H2OI]*y[IDX_C2NII] +
        k[995]*y[IDX_H2OI]*y[IDX_CNII] + k[997]*y[IDX_H2OI]*y[IDX_COII] +
        k[1010]*y[IDX_H2OI]*y[IDX_N2II] + k[1059]*y[IDX_H3II]*y[IDX_NO2I] -
        k[1066]*y[IDX_H3II]*y[IDX_OHI] + k[1107]*y[IDX_HI]*y[IDX_SO2II] +
        k[1201]*y[IDX_HeII]*y[IDX_CH3OHI] + k[1223]*y[IDX_HeII]*y[IDX_H2OI] -
        k[1268]*y[IDX_HeII]*y[IDX_OHI] + k[1359]*y[IDX_NHII]*y[IDX_H2OI] +
        k[1368]*y[IDX_NHII]*y[IDX_O2I] - k[1371]*y[IDX_NHII]*y[IDX_OHI] +
        k[1379]*y[IDX_NH2II]*y[IDX_H2OI] + k[1391]*y[IDX_NH2II]*y[IDX_O2I] +
        k[1400]*y[IDX_NH2I]*y[IDX_H2OII] + k[1414]*y[IDX_NH3II]*y[IDX_H2OI] +
        k[1425]*y[IDX_NH3I]*y[IDX_H2OII] + k[1467]*y[IDX_OII]*y[IDX_CH3OHI] +
        k[1468]*y[IDX_OII]*y[IDX_CH4I] + k[1471]*y[IDX_OII]*y[IDX_H2COI] +
        k[1472]*y[IDX_OII]*y[IDX_H2SI] - k[1480]*y[IDX_OII]*y[IDX_OHI] +
        k[1493]*y[IDX_OI]*y[IDX_CH4II] + k[1498]*y[IDX_OI]*y[IDX_H2SII] +
        k[1503]*y[IDX_OI]*y[IDX_HSII] - k[1532]*y[IDX_OHII]*y[IDX_OHI] -
        k[1538]*y[IDX_OHI]*y[IDX_CH5II] - k[1539]*y[IDX_OHI]*y[IDX_COII] -
        k[1540]*y[IDX_OHI]*y[IDX_H2OII] - k[1541]*y[IDX_OHI]*y[IDX_HCNII] -
        k[1542]*y[IDX_OHI]*y[IDX_HCOII] - k[1543]*y[IDX_OHI]*y[IDX_HCOII] -
        k[1544]*y[IDX_OHI]*y[IDX_HNOII] - k[1545]*y[IDX_OHI]*y[IDX_N2HII] -
        k[1546]*y[IDX_OHI]*y[IDX_NH3II] - k[1547]*y[IDX_OHI]*y[IDX_O2HII] -
        k[1548]*y[IDX_OHI]*y[IDX_SII] - k[1549]*y[IDX_OHI]*y[IDX_SiII] +
        k[1567]*y[IDX_SiH2II]*y[IDX_O2I] - k[1611]*y[IDX_CI]*y[IDX_OHI] -
        k[1612]*y[IDX_CI]*y[IDX_OHI] + k[1630]*y[IDX_CH2I]*y[IDX_NOI] +
        k[1636]*y[IDX_CH2I]*y[IDX_O2I] + k[1640]*y[IDX_CH2I]*y[IDX_OI] -
        k[1641]*y[IDX_CH2I]*y[IDX_OHI] - k[1642]*y[IDX_CH2I]*y[IDX_OHI] -
        k[1643]*y[IDX_CH2I]*y[IDX_OHI] + k[1652]*y[IDX_CH3I]*y[IDX_H2OI] +
        k[1660]*y[IDX_CH3I]*y[IDX_O2I] - k[1666]*y[IDX_CH3I]*y[IDX_OHI] -
        k[1667]*y[IDX_CH3I]*y[IDX_OHI] - k[1668]*y[IDX_CH3I]*y[IDX_OHI] -
        k[1672]*y[IDX_CH4I]*y[IDX_OHI] + k[1689]*y[IDX_CHI]*y[IDX_O2I] +
        k[1691]*y[IDX_CHI]*y[IDX_O2HI] + k[1694]*y[IDX_CHI]*y[IDX_OI] -
        k[1696]*y[IDX_CHI]*y[IDX_OHI] + k[1719]*y[IDX_COI]*y[IDX_O2HI] +
        k[1732]*y[IDX_H2I]*y[IDX_O2I] + k[1732]*y[IDX_H2I]*y[IDX_O2I] +
        k[1733]*y[IDX_H2I]*y[IDX_OI] - k[1734]*y[IDX_H2I]*y[IDX_OHI] +
        k[1744]*y[IDX_HI]*y[IDX_CO2I] + k[1745]*y[IDX_HI]*y[IDX_COI] +
        k[1748]*y[IDX_HI]*y[IDX_H2OI] + k[1757]*y[IDX_HI]*y[IDX_HNOI] +
        k[1763]*y[IDX_HI]*y[IDX_NO2I] + k[1765]*y[IDX_HI]*y[IDX_NOI] +
        k[1768]*y[IDX_HI]*y[IDX_O2I] + k[1771]*y[IDX_HI]*y[IDX_O2HI] +
        k[1771]*y[IDX_HI]*y[IDX_O2HI] + k[1774]*y[IDX_HI]*y[IDX_OCNI] -
        k[1776]*y[IDX_HI]*y[IDX_OHI] + k[1779]*y[IDX_HI]*y[IDX_SOI] +
        k[1784]*y[IDX_HCOI]*y[IDX_O2I] - k[1827]*y[IDX_NI]*y[IDX_OHI] -
        k[1828]*y[IDX_NI]*y[IDX_OHI] + k[1835]*y[IDX_NH2I]*y[IDX_NOI] -
        k[1836]*y[IDX_NH2I]*y[IDX_OHI] - k[1837]*y[IDX_NH2I]*y[IDX_OHI] +
        k[1841]*y[IDX_NHI]*y[IDX_H2OI] + k[1848]*y[IDX_NHI]*y[IDX_NOI] +
        k[1850]*y[IDX_NHI]*y[IDX_O2I] + k[1852]*y[IDX_NHI]*y[IDX_OI] -
        k[1853]*y[IDX_NHI]*y[IDX_OHI] - k[1854]*y[IDX_NHI]*y[IDX_OHI] -
        k[1855]*y[IDX_NHI]*y[IDX_OHI] + k[1870]*y[IDX_OI]*y[IDX_C2H4I] +
        k[1879]*y[IDX_OI]*y[IDX_CH4I] + k[1886]*y[IDX_OI]*y[IDX_H2COI] +
        k[1887]*y[IDX_OI]*y[IDX_H2OI] + k[1887]*y[IDX_OI]*y[IDX_H2OI] +
        k[1888]*y[IDX_OI]*y[IDX_H2SI] + k[1889]*y[IDX_OI]*y[IDX_HCNI] +
        k[1893]*y[IDX_OI]*y[IDX_HCOI] + k[1897]*y[IDX_OI]*y[IDX_HNOI] +
        k[1899]*y[IDX_OI]*y[IDX_HSI] + k[1903]*y[IDX_OI]*y[IDX_NH2I] +
        k[1904]*y[IDX_OI]*y[IDX_NH3I] + k[1909]*y[IDX_OI]*y[IDX_O2HI] -
        k[1914]*y[IDX_OI]*y[IDX_OHI] + k[1925]*y[IDX_OI]*y[IDX_SiH4I] -
        k[1927]*y[IDX_OHI]*y[IDX_C2H2I] - k[1928]*y[IDX_OHI]*y[IDX_C2H2I] -
        k[1929]*y[IDX_OHI]*y[IDX_C2H2I] - k[1930]*y[IDX_OHI]*y[IDX_C2H3I] -
        k[1931]*y[IDX_OHI]*y[IDX_C2H5I] - k[1932]*y[IDX_OHI]*y[IDX_CNI] -
        k[1933]*y[IDX_OHI]*y[IDX_CNI] - k[1934]*y[IDX_OHI]*y[IDX_COI] -
        k[1935]*y[IDX_OHI]*y[IDX_CSI] - k[1936]*y[IDX_OHI]*y[IDX_CSI] -
        k[1937]*y[IDX_OHI]*y[IDX_H2COI] - k[1938]*y[IDX_OHI]*y[IDX_H2SI] -
        k[1939]*y[IDX_OHI]*y[IDX_HCNI] - k[1940]*y[IDX_OHI]*y[IDX_HCNI] -
        k[1941]*y[IDX_OHI]*y[IDX_HCOI] - k[1942]*y[IDX_OHI]*y[IDX_HNOI] -
        k[1943]*y[IDX_OHI]*y[IDX_NCCNI] - k[1944]*y[IDX_OHI]*y[IDX_NH3I] -
        k[1945]*y[IDX_OHI]*y[IDX_NOI] - k[1946]*y[IDX_OHI]*y[IDX_O2HI] -
        k[1947]*y[IDX_OHI]*y[IDX_OHI] - k[1947]*y[IDX_OHI]*y[IDX_OHI] -
        k[1948]*y[IDX_OHI]*y[IDX_SI] - k[1949]*y[IDX_OHI]*y[IDX_SOI] -
        k[1950]*y[IDX_OHI]*y[IDX_SiI] + k[1969]*y[IDX_C2H5OHI] +
        k[1992]*y[IDX_CH3OHI] + k[2018]*y[IDX_H2OI] + k[2064]*y[IDX_O2HI] -
        k[2069]*y[IDX_OHI] - k[2070]*y[IDX_OHI] + k[2124]*y[IDX_HI]*y[IDX_OI] -
        k[2125]*y[IDX_HI]*y[IDX_OHI] - k[2211]*y[IDX_OHI];
    ydot[IDX_NI] = 0.0 + k[87]*y[IDX_CHI]*y[IDX_NII] +
        k[233]*y[IDX_NII]*y[IDX_C2I] + k[234]*y[IDX_NII]*y[IDX_C2HI] +
        k[235]*y[IDX_NII]*y[IDX_CH2I] + k[236]*y[IDX_NII]*y[IDX_CH4I] +
        k[237]*y[IDX_NII]*y[IDX_CNI] + k[238]*y[IDX_NII]*y[IDX_COI] +
        k[239]*y[IDX_NII]*y[IDX_H2COI] + k[240]*y[IDX_NII]*y[IDX_H2OI] +
        k[241]*y[IDX_NII]*y[IDX_H2SI] + k[242]*y[IDX_NII]*y[IDX_HCNI] +
        k[243]*y[IDX_NII]*y[IDX_HCOI] + k[244]*y[IDX_NII]*y[IDX_MgI] +
        k[245]*y[IDX_NII]*y[IDX_NH2I] + k[246]*y[IDX_NII]*y[IDX_NH3I] +
        k[247]*y[IDX_NII]*y[IDX_NHI] + k[248]*y[IDX_NII]*y[IDX_NOI] +
        k[249]*y[IDX_NII]*y[IDX_O2I] + k[250]*y[IDX_NII]*y[IDX_OCSI] +
        k[251]*y[IDX_NII]*y[IDX_OHI] - k[259]*y[IDX_NI]*y[IDX_N2II] -
        k[364]*y[IDX_NI] + k[375]*y[IDX_C2NI] + k[392]*y[IDX_CNI] +
        k[421]*y[IDX_N2I] + k[421]*y[IDX_N2I] - k[422]*y[IDX_NI] +
        k[429]*y[IDX_NHI] + k[433]*y[IDX_NOI] + k[434]*y[IDX_NSI] +
        k[468]*y[IDX_C2NII]*y[IDX_EM] + k[470]*y[IDX_C2N2II]*y[IDX_EM] +
        k[498]*y[IDX_CNII]*y[IDX_EM] + k[564]*y[IDX_N2II]*y[IDX_EM] +
        k[564]*y[IDX_N2II]*y[IDX_EM] + k[566]*y[IDX_N2HII]*y[IDX_EM] +
        k[567]*y[IDX_NHII]*y[IDX_EM] + k[568]*y[IDX_NH2II]*y[IDX_EM] +
        k[575]*y[IDX_NOII]*y[IDX_EM] + k[576]*y[IDX_NSII]*y[IDX_EM] +
        k[632]*y[IDX_CII]*y[IDX_NSI] + k[699]*y[IDX_CI]*y[IDX_NHII] -
        k[728]*y[IDX_CHII]*y[IDX_NI] + k[766]*y[IDX_CH2I]*y[IDX_NHII] +
        k[854]*y[IDX_CHI]*y[IDX_NHII] - k[928]*y[IDX_H2II]*y[IDX_NI] +
        k[954]*y[IDX_H2I]*y[IDX_NHII] + k[1208]*y[IDX_HeII]*y[IDX_CNI] +
        k[1233]*y[IDX_HeII]*y[IDX_HCNI] + k[1234]*y[IDX_HeII]*y[IDX_HCNI] +
        k[1243]*y[IDX_HeII]*y[IDX_HNCI] + k[1250]*y[IDX_HeII]*y[IDX_N2I] +
        k[1257]*y[IDX_HeII]*y[IDX_NOI] + k[1259]*y[IDX_HeII]*y[IDX_NSI] +
        k[1293]*y[IDX_NII]*y[IDX_CH4I] + k[1305]*y[IDX_NII]*y[IDX_NCCNI] +
        k[1313]*y[IDX_NII]*y[IDX_OCSI] - k[1325]*y[IDX_NI]*y[IDX_C2II] -
        k[1326]*y[IDX_NI]*y[IDX_C2HII] - k[1327]*y[IDX_NI]*y[IDX_C2HII] -
        k[1328]*y[IDX_NI]*y[IDX_C2H2II] - k[1329]*y[IDX_NI]*y[IDX_C2H2II] -
        k[1330]*y[IDX_NI]*y[IDX_C2H2II] - k[1331]*y[IDX_NI]*y[IDX_CH2II] -
        k[1332]*y[IDX_NI]*y[IDX_CNII] - k[1333]*y[IDX_NI]*y[IDX_H2OII] -
        k[1334]*y[IDX_NI]*y[IDX_H2OII] - k[1335]*y[IDX_NI]*y[IDX_H2SII] -
        k[1336]*y[IDX_NI]*y[IDX_HSII] - k[1337]*y[IDX_NI]*y[IDX_NHII] -
        k[1338]*y[IDX_NI]*y[IDX_NH2II] - k[1339]*y[IDX_NI]*y[IDX_O2II] -
        k[1340]*y[IDX_NI]*y[IDX_OHII] - k[1341]*y[IDX_NI]*y[IDX_SOII] -
        k[1342]*y[IDX_NI]*y[IDX_SiCII] - k[1343]*y[IDX_NI]*y[IDX_SiOII] -
        k[1344]*y[IDX_NI]*y[IDX_SiOII] + k[1345]*y[IDX_NHII]*y[IDX_C2I] +
        k[1348]*y[IDX_NHII]*y[IDX_C2HI] + k[1349]*y[IDX_NHII]*y[IDX_CNI] +
        k[1350]*y[IDX_NHII]*y[IDX_CO2I] + k[1353]*y[IDX_NHII]*y[IDX_COI] +
        k[1354]*y[IDX_NHII]*y[IDX_H2COI] + k[1356]*y[IDX_NHII]*y[IDX_H2OI] +
        k[1360]*y[IDX_NHII]*y[IDX_HCNI] + k[1361]*y[IDX_NHII]*y[IDX_HCOI] +
        k[1362]*y[IDX_NHII]*y[IDX_HNCI] + k[1363]*y[IDX_NHII]*y[IDX_N2I] +
        k[1364]*y[IDX_NHII]*y[IDX_NH2I] + k[1365]*y[IDX_NHII]*y[IDX_NH3I] +
        k[1366]*y[IDX_NHII]*y[IDX_NHI] + k[1369]*y[IDX_NHII]*y[IDX_O2I] +
        k[1370]*y[IDX_NHII]*y[IDX_OI] + k[1371]*y[IDX_NHII]*y[IDX_OHI] +
        k[1372]*y[IDX_NHII]*y[IDX_SI] + k[1444]*y[IDX_NHI]*y[IDX_C2II] +
        k[1448]*y[IDX_NHI]*y[IDX_COII] + k[1449]*y[IDX_NHI]*y[IDX_H2COII] +
        k[1450]*y[IDX_NHI]*y[IDX_H2OII] + k[1455]*y[IDX_NHI]*y[IDX_NH2II] +
        k[1456]*y[IDX_NHI]*y[IDX_NH3II] + k[1474]*y[IDX_OII]*y[IDX_HCNI] +
        k[1477]*y[IDX_OII]*y[IDX_N2I] + k[1505]*y[IDX_OI]*y[IDX_N2II] +
        k[1590]*y[IDX_CI]*y[IDX_CNI] + k[1597]*y[IDX_CI]*y[IDX_N2I] +
        k[1603]*y[IDX_CI]*y[IDX_NHI] + k[1605]*y[IDX_CI]*y[IDX_NOI] +
        k[1606]*y[IDX_CI]*y[IDX_NSI] + k[1629]*y[IDX_CH2I]*y[IDX_NOI] +
        k[1681]*y[IDX_CHI]*y[IDX_N2I] - k[1682]*y[IDX_CHI]*y[IDX_NI] -
        k[1683]*y[IDX_CHI]*y[IDX_NI] + k[1685]*y[IDX_CHI]*y[IDX_NOI] +
        k[1711]*y[IDX_CNI]*y[IDX_NOI] - k[1728]*y[IDX_H2I]*y[IDX_NI] +
        k[1762]*y[IDX_HI]*y[IDX_NHI] + k[1765]*y[IDX_HI]*y[IDX_NOI] +
        k[1766]*y[IDX_HI]*y[IDX_NSI] - k[1790]*y[IDX_NI]*y[IDX_C2I] -
        k[1791]*y[IDX_NI]*y[IDX_C2H3I] - k[1792]*y[IDX_NI]*y[IDX_C2H4I] -
        k[1793]*y[IDX_NI]*y[IDX_C2H5I] - k[1794]*y[IDX_NI]*y[IDX_C2H5I] -
        k[1795]*y[IDX_NI]*y[IDX_C2HI] - k[1796]*y[IDX_NI]*y[IDX_C2NI] -
        k[1797]*y[IDX_NI]*y[IDX_C3H2I] - k[1798]*y[IDX_NI]*y[IDX_C3NI] -
        k[1799]*y[IDX_NI]*y[IDX_C4HI] - k[1800]*y[IDX_NI]*y[IDX_C4NI] -
        k[1801]*y[IDX_NI]*y[IDX_CH2I] - k[1802]*y[IDX_NI]*y[IDX_CH2I] -
        k[1803]*y[IDX_NI]*y[IDX_CH2I] - k[1804]*y[IDX_NI]*y[IDX_CH3I] -
        k[1805]*y[IDX_NI]*y[IDX_CH3I] - k[1806]*y[IDX_NI]*y[IDX_CH3I] -
        k[1807]*y[IDX_NI]*y[IDX_CNI] - k[1808]*y[IDX_NI]*y[IDX_CO2I] -
        k[1809]*y[IDX_NI]*y[IDX_CSI] - k[1810]*y[IDX_NI]*y[IDX_H2CNI] -
        k[1811]*y[IDX_NI]*y[IDX_HCOI] - k[1812]*y[IDX_NI]*y[IDX_HCOI] -
        k[1813]*y[IDX_NI]*y[IDX_HCOI] - k[1814]*y[IDX_NI]*y[IDX_HCSI] -
        k[1815]*y[IDX_NI]*y[IDX_HNOI] - k[1816]*y[IDX_NI]*y[IDX_HSI] -
        k[1817]*y[IDX_NI]*y[IDX_HSI] - k[1818]*y[IDX_NI]*y[IDX_NCCNI] -
        k[1819]*y[IDX_NI]*y[IDX_NHI] - k[1820]*y[IDX_NI]*y[IDX_NO2I] -
        k[1821]*y[IDX_NI]*y[IDX_NO2I] - k[1822]*y[IDX_NI]*y[IDX_NO2I] -
        k[1823]*y[IDX_NI]*y[IDX_NOI] - k[1824]*y[IDX_NI]*y[IDX_NSI] -
        k[1825]*y[IDX_NI]*y[IDX_O2I] - k[1826]*y[IDX_NI]*y[IDX_O2HI] -
        k[1827]*y[IDX_NI]*y[IDX_OHI] - k[1828]*y[IDX_NI]*y[IDX_OHI] -
        k[1829]*y[IDX_NI]*y[IDX_S2I] - k[1830]*y[IDX_NI]*y[IDX_SOI] -
        k[1831]*y[IDX_NI]*y[IDX_SOI] - k[1832]*y[IDX_NI]*y[IDX_SiCI] +
        k[1840]*y[IDX_NHI]*y[IDX_CNI] + k[1845]*y[IDX_NHI]*y[IDX_NHI] +
        k[1852]*y[IDX_NHI]*y[IDX_OI] + k[1853]*y[IDX_NHI]*y[IDX_OHI] +
        k[1856]*y[IDX_NHI]*y[IDX_SI] + k[1862]*y[IDX_NOI]*y[IDX_SI] +
        k[1880]*y[IDX_OI]*y[IDX_CNI] + k[1901]*y[IDX_OI]*y[IDX_N2I] +
        k[1906]*y[IDX_OI]*y[IDX_NOI] + k[1908]*y[IDX_OI]*y[IDX_NSI] +
        k[1958]*y[IDX_SiI]*y[IDX_NOI] + k[1972]*y[IDX_C2NI] + k[2001]*y[IDX_CNI]
        + k[2045]*y[IDX_N2I] + k[2045]*y[IDX_N2I] + k[2048]*y[IDX_NHII] +
        k[2054]*y[IDX_NHI] + k[2058]*y[IDX_NOI] + k[2059]*y[IDX_NSI] -
        k[2097]*y[IDX_CII]*y[IDX_NI] - k[2102]*y[IDX_CI]*y[IDX_NI] -
        k[2127]*y[IDX_NII]*y[IDX_NI] + k[2141]*y[IDX_NII]*y[IDX_EM] -
        k[2251]*y[IDX_NI];
    ydot[IDX_HeII] = 0.0 - k[172]*y[IDX_H2I]*y[IDX_HeII] -
        k[195]*y[IDX_HI]*y[IDX_HeII] - k[207]*y[IDX_HeII]*y[IDX_C2I] -
        k[208]*y[IDX_HeII]*y[IDX_C2H2I] - k[209]*y[IDX_HeII]*y[IDX_CI] -
        k[210]*y[IDX_HeII]*y[IDX_CH4I] - k[211]*y[IDX_HeII]*y[IDX_CHI] -
        k[212]*y[IDX_HeII]*y[IDX_H2COI] - k[213]*y[IDX_HeII]*y[IDX_H2OI] -
        k[214]*y[IDX_HeII]*y[IDX_H2SI] - k[215]*y[IDX_HeII]*y[IDX_N2I] -
        k[216]*y[IDX_HeII]*y[IDX_NH3I] - k[217]*y[IDX_HeII]*y[IDX_O2I] -
        k[218]*y[IDX_HeII]*y[IDX_SO2I] - k[219]*y[IDX_HeII]*y[IDX_SiI] +
        k[363]*y[IDX_HeI] + k[419]*y[IDX_HeI] - k[950]*y[IDX_H2I]*y[IDX_HeII] -
        k[1177]*y[IDX_HeII]*y[IDX_C2I] - k[1178]*y[IDX_HeII]*y[IDX_C2H2I] -
        k[1179]*y[IDX_HeII]*y[IDX_C2H2I] - k[1180]*y[IDX_HeII]*y[IDX_C2H2I] -
        k[1181]*y[IDX_HeII]*y[IDX_C2H3I] - k[1182]*y[IDX_HeII]*y[IDX_C2H3I] -
        k[1183]*y[IDX_HeII]*y[IDX_C2H4I] - k[1184]*y[IDX_HeII]*y[IDX_C2H4I] -
        k[1185]*y[IDX_HeII]*y[IDX_C2H4I] - k[1186]*y[IDX_HeII]*y[IDX_C2HI] -
        k[1187]*y[IDX_HeII]*y[IDX_C2HI] - k[1188]*y[IDX_HeII]*y[IDX_C2HI] -
        k[1189]*y[IDX_HeII]*y[IDX_C2NI] - k[1190]*y[IDX_HeII]*y[IDX_C3NI] -
        k[1191]*y[IDX_HeII]*y[IDX_C4HI] - k[1192]*y[IDX_HeII]*y[IDX_CH2I] -
        k[1193]*y[IDX_HeII]*y[IDX_CH2I] - k[1194]*y[IDX_HeII]*y[IDX_CH2COI] -
        k[1195]*y[IDX_HeII]*y[IDX_CH2COI] - k[1196]*y[IDX_HeII]*y[IDX_CH3I] -
        k[1197]*y[IDX_HeII]*y[IDX_CH3CCHI] - k[1198]*y[IDX_HeII]*y[IDX_CH3CNI] -
        k[1199]*y[IDX_HeII]*y[IDX_CH3CNI] - k[1200]*y[IDX_HeII]*y[IDX_CH3OHI] -
        k[1201]*y[IDX_HeII]*y[IDX_CH3OHI] - k[1202]*y[IDX_HeII]*y[IDX_CH4I] -
        k[1203]*y[IDX_HeII]*y[IDX_CH4I] - k[1204]*y[IDX_HeII]*y[IDX_CH4I] -
        k[1205]*y[IDX_HeII]*y[IDX_CH4I] - k[1206]*y[IDX_HeII]*y[IDX_CHI] -
        k[1207]*y[IDX_HeII]*y[IDX_CNI] - k[1208]*y[IDX_HeII]*y[IDX_CNI] -
        k[1209]*y[IDX_HeII]*y[IDX_CO2I] - k[1210]*y[IDX_HeII]*y[IDX_CO2I] -
        k[1211]*y[IDX_HeII]*y[IDX_CO2I] - k[1212]*y[IDX_HeII]*y[IDX_CO2I] -
        k[1213]*y[IDX_HeII]*y[IDX_COI] - k[1214]*y[IDX_HeII]*y[IDX_CSI] -
        k[1215]*y[IDX_HeII]*y[IDX_CSI] - k[1216]*y[IDX_HeII]*y[IDX_H2COI] -
        k[1217]*y[IDX_HeII]*y[IDX_H2COI] - k[1218]*y[IDX_HeII]*y[IDX_H2COI] -
        k[1219]*y[IDX_HeII]*y[IDX_H2CSI] - k[1220]*y[IDX_HeII]*y[IDX_H2CSI] -
        k[1221]*y[IDX_HeII]*y[IDX_H2CSI] - k[1222]*y[IDX_HeII]*y[IDX_H2OI] -
        k[1223]*y[IDX_HeII]*y[IDX_H2OI] - k[1224]*y[IDX_HeII]*y[IDX_H2S2I] -
        k[1225]*y[IDX_HeII]*y[IDX_H2S2I] - k[1226]*y[IDX_HeII]*y[IDX_H2SI] -
        k[1227]*y[IDX_HeII]*y[IDX_H2SI] - k[1228]*y[IDX_HeII]*y[IDX_H2SiOI] -
        k[1229]*y[IDX_HeII]*y[IDX_HC3NI] - k[1230]*y[IDX_HeII]*y[IDX_HC3NI] -
        k[1231]*y[IDX_HeII]*y[IDX_HCNI] - k[1232]*y[IDX_HeII]*y[IDX_HCNI] -
        k[1233]*y[IDX_HeII]*y[IDX_HCNI] - k[1234]*y[IDX_HeII]*y[IDX_HCNI] -
        k[1235]*y[IDX_HeII]*y[IDX_HCOI] - k[1236]*y[IDX_HeII]*y[IDX_HCOI] -
        k[1237]*y[IDX_HeII]*y[IDX_HCOI] - k[1238]*y[IDX_HeII]*y[IDX_HCOOCH3I] -
        k[1239]*y[IDX_HeII]*y[IDX_HCSI] - k[1240]*y[IDX_HeII]*y[IDX_HCSI] -
        k[1241]*y[IDX_HeII]*y[IDX_HClI] - k[1242]*y[IDX_HeII]*y[IDX_HNCI] -
        k[1243]*y[IDX_HeII]*y[IDX_HNCI] - k[1244]*y[IDX_HeII]*y[IDX_HNCI] -
        k[1245]*y[IDX_HeII]*y[IDX_HNOI] - k[1246]*y[IDX_HeII]*y[IDX_HNOI] -
        k[1247]*y[IDX_HeII]*y[IDX_HS2I] - k[1248]*y[IDX_HeII]*y[IDX_HS2I] -
        k[1249]*y[IDX_HeII]*y[IDX_HSI] - k[1250]*y[IDX_HeII]*y[IDX_N2I] -
        k[1251]*y[IDX_HeII]*y[IDX_NCCNI] - k[1252]*y[IDX_HeII]*y[IDX_NH2I] -
        k[1253]*y[IDX_HeII]*y[IDX_NH2I] - k[1254]*y[IDX_HeII]*y[IDX_NH3I] -
        k[1255]*y[IDX_HeII]*y[IDX_NH3I] - k[1256]*y[IDX_HeII]*y[IDX_NHI] -
        k[1257]*y[IDX_HeII]*y[IDX_NOI] - k[1258]*y[IDX_HeII]*y[IDX_NOI] -
        k[1259]*y[IDX_HeII]*y[IDX_NSI] - k[1260]*y[IDX_HeII]*y[IDX_NSI] -
        k[1261]*y[IDX_HeII]*y[IDX_O2I] - k[1262]*y[IDX_HeII]*y[IDX_OCNI] -
        k[1263]*y[IDX_HeII]*y[IDX_OCNI] - k[1264]*y[IDX_HeII]*y[IDX_OCSI] -
        k[1265]*y[IDX_HeII]*y[IDX_OCSI] - k[1266]*y[IDX_HeII]*y[IDX_OCSI] -
        k[1267]*y[IDX_HeII]*y[IDX_OCSI] - k[1268]*y[IDX_HeII]*y[IDX_OHI] -
        k[1269]*y[IDX_HeII]*y[IDX_S2I] - k[1270]*y[IDX_HeII]*y[IDX_SO2I] -
        k[1271]*y[IDX_HeII]*y[IDX_SO2I] - k[1272]*y[IDX_HeII]*y[IDX_SOI] -
        k[1273]*y[IDX_HeII]*y[IDX_SOI] - k[1274]*y[IDX_HeII]*y[IDX_SiC2I] -
        k[1275]*y[IDX_HeII]*y[IDX_SiC3I] - k[1276]*y[IDX_HeII]*y[IDX_SiCI] -
        k[1277]*y[IDX_HeII]*y[IDX_SiCI] - k[1278]*y[IDX_HeII]*y[IDX_SiH2I] -
        k[1279]*y[IDX_HeII]*y[IDX_SiH2I] - k[1280]*y[IDX_HeII]*y[IDX_SiH3I] -
        k[1281]*y[IDX_HeII]*y[IDX_SiH3I] - k[1282]*y[IDX_HeII]*y[IDX_SiH4I] -
        k[1283]*y[IDX_HeII]*y[IDX_SiH4I] - k[1284]*y[IDX_HeII]*y[IDX_SiHI] -
        k[1285]*y[IDX_HeII]*y[IDX_SiOI] - k[1286]*y[IDX_HeII]*y[IDX_SiOI] -
        k[1287]*y[IDX_HeII]*y[IDX_SiSI] - k[1288]*y[IDX_HeII]*y[IDX_SiSI] -
        k[2139]*y[IDX_HeII]*y[IDX_EM];
    ydot[IDX_HeI] = 0.0 + k[172]*y[IDX_H2I]*y[IDX_HeII] +
        k[195]*y[IDX_HI]*y[IDX_HeII] + k[207]*y[IDX_HeII]*y[IDX_C2I] +
        k[208]*y[IDX_HeII]*y[IDX_C2H2I] + k[209]*y[IDX_HeII]*y[IDX_CI] +
        k[210]*y[IDX_HeII]*y[IDX_CH4I] + k[211]*y[IDX_HeII]*y[IDX_CHI] +
        k[212]*y[IDX_HeII]*y[IDX_H2COI] + k[213]*y[IDX_HeII]*y[IDX_H2OI] +
        k[214]*y[IDX_HeII]*y[IDX_H2SI] + k[215]*y[IDX_HeII]*y[IDX_N2I] +
        k[216]*y[IDX_HeII]*y[IDX_NH3I] + k[217]*y[IDX_HeII]*y[IDX_O2I] +
        k[218]*y[IDX_HeII]*y[IDX_SO2I] + k[219]*y[IDX_HeII]*y[IDX_SiI] -
        k[363]*y[IDX_HeI] - k[419]*y[IDX_HeI] + k[563]*y[IDX_HeHII]*y[IDX_EM] -
        k[926]*y[IDX_H2II]*y[IDX_HeI] + k[950]*y[IDX_H2I]*y[IDX_HeII] +
        k[951]*y[IDX_H2I]*y[IDX_HeHII] + k[1106]*y[IDX_HI]*y[IDX_HeHII] +
        k[1177]*y[IDX_HeII]*y[IDX_C2I] + k[1178]*y[IDX_HeII]*y[IDX_C2H2I] +
        k[1179]*y[IDX_HeII]*y[IDX_C2H2I] + k[1180]*y[IDX_HeII]*y[IDX_C2H2I] +
        k[1181]*y[IDX_HeII]*y[IDX_C2H3I] + k[1182]*y[IDX_HeII]*y[IDX_C2H3I] +
        k[1183]*y[IDX_HeII]*y[IDX_C2H4I] + k[1184]*y[IDX_HeII]*y[IDX_C2H4I] +
        k[1185]*y[IDX_HeII]*y[IDX_C2H4I] + k[1186]*y[IDX_HeII]*y[IDX_C2HI] +
        k[1187]*y[IDX_HeII]*y[IDX_C2HI] + k[1188]*y[IDX_HeII]*y[IDX_C2HI] +
        k[1189]*y[IDX_HeII]*y[IDX_C2NI] + k[1190]*y[IDX_HeII]*y[IDX_C3NI] +
        k[1191]*y[IDX_HeII]*y[IDX_C4HI] + k[1192]*y[IDX_HeII]*y[IDX_CH2I] +
        k[1193]*y[IDX_HeII]*y[IDX_CH2I] + k[1194]*y[IDX_HeII]*y[IDX_CH2COI] +
        k[1195]*y[IDX_HeII]*y[IDX_CH2COI] + k[1196]*y[IDX_HeII]*y[IDX_CH3I] +
        k[1197]*y[IDX_HeII]*y[IDX_CH3CCHI] + k[1198]*y[IDX_HeII]*y[IDX_CH3CNI] +
        k[1199]*y[IDX_HeII]*y[IDX_CH3CNI] + k[1200]*y[IDX_HeII]*y[IDX_CH3OHI] +
        k[1201]*y[IDX_HeII]*y[IDX_CH3OHI] + k[1202]*y[IDX_HeII]*y[IDX_CH4I] +
        k[1203]*y[IDX_HeII]*y[IDX_CH4I] + k[1204]*y[IDX_HeII]*y[IDX_CH4I] +
        k[1205]*y[IDX_HeII]*y[IDX_CH4I] + k[1206]*y[IDX_HeII]*y[IDX_CHI] +
        k[1207]*y[IDX_HeII]*y[IDX_CNI] + k[1208]*y[IDX_HeII]*y[IDX_CNI] +
        k[1209]*y[IDX_HeII]*y[IDX_CO2I] + k[1210]*y[IDX_HeII]*y[IDX_CO2I] +
        k[1211]*y[IDX_HeII]*y[IDX_CO2I] + k[1212]*y[IDX_HeII]*y[IDX_CO2I] +
        k[1213]*y[IDX_HeII]*y[IDX_COI] + k[1214]*y[IDX_HeII]*y[IDX_CSI] +
        k[1215]*y[IDX_HeII]*y[IDX_CSI] + k[1216]*y[IDX_HeII]*y[IDX_H2COI] +
        k[1217]*y[IDX_HeII]*y[IDX_H2COI] + k[1218]*y[IDX_HeII]*y[IDX_H2COI] +
        k[1219]*y[IDX_HeII]*y[IDX_H2CSI] + k[1220]*y[IDX_HeII]*y[IDX_H2CSI] +
        k[1221]*y[IDX_HeII]*y[IDX_H2CSI] + k[1222]*y[IDX_HeII]*y[IDX_H2OI] +
        k[1223]*y[IDX_HeII]*y[IDX_H2OI] + k[1224]*y[IDX_HeII]*y[IDX_H2S2I] +
        k[1225]*y[IDX_HeII]*y[IDX_H2S2I] + k[1226]*y[IDX_HeII]*y[IDX_H2SI] +
        k[1227]*y[IDX_HeII]*y[IDX_H2SI] + k[1228]*y[IDX_HeII]*y[IDX_H2SiOI] +
        k[1229]*y[IDX_HeII]*y[IDX_HC3NI] + k[1230]*y[IDX_HeII]*y[IDX_HC3NI] +
        k[1231]*y[IDX_HeII]*y[IDX_HCNI] + k[1232]*y[IDX_HeII]*y[IDX_HCNI] +
        k[1233]*y[IDX_HeII]*y[IDX_HCNI] + k[1234]*y[IDX_HeII]*y[IDX_HCNI] +
        k[1235]*y[IDX_HeII]*y[IDX_HCOI] + k[1237]*y[IDX_HeII]*y[IDX_HCOI] +
        k[1238]*y[IDX_HeII]*y[IDX_HCOOCH3I] + k[1239]*y[IDX_HeII]*y[IDX_HCSI] +
        k[1240]*y[IDX_HeII]*y[IDX_HCSI] + k[1241]*y[IDX_HeII]*y[IDX_HClI] +
        k[1242]*y[IDX_HeII]*y[IDX_HNCI] + k[1243]*y[IDX_HeII]*y[IDX_HNCI] +
        k[1244]*y[IDX_HeII]*y[IDX_HNCI] + k[1245]*y[IDX_HeII]*y[IDX_HNOI] +
        k[1246]*y[IDX_HeII]*y[IDX_HNOI] + k[1247]*y[IDX_HeII]*y[IDX_HS2I] +
        k[1248]*y[IDX_HeII]*y[IDX_HS2I] + k[1249]*y[IDX_HeII]*y[IDX_HSI] +
        k[1250]*y[IDX_HeII]*y[IDX_N2I] + k[1251]*y[IDX_HeII]*y[IDX_NCCNI] +
        k[1252]*y[IDX_HeII]*y[IDX_NH2I] + k[1253]*y[IDX_HeII]*y[IDX_NH2I] +
        k[1254]*y[IDX_HeII]*y[IDX_NH3I] + k[1255]*y[IDX_HeII]*y[IDX_NH3I] +
        k[1256]*y[IDX_HeII]*y[IDX_NHI] + k[1257]*y[IDX_HeII]*y[IDX_NOI] +
        k[1258]*y[IDX_HeII]*y[IDX_NOI] + k[1259]*y[IDX_HeII]*y[IDX_NSI] +
        k[1260]*y[IDX_HeII]*y[IDX_NSI] + k[1261]*y[IDX_HeII]*y[IDX_O2I] +
        k[1262]*y[IDX_HeII]*y[IDX_OCNI] + k[1263]*y[IDX_HeII]*y[IDX_OCNI] +
        k[1264]*y[IDX_HeII]*y[IDX_OCSI] + k[1265]*y[IDX_HeII]*y[IDX_OCSI] +
        k[1266]*y[IDX_HeII]*y[IDX_OCSI] + k[1267]*y[IDX_HeII]*y[IDX_OCSI] +
        k[1268]*y[IDX_HeII]*y[IDX_OHI] + k[1269]*y[IDX_HeII]*y[IDX_S2I] +
        k[1270]*y[IDX_HeII]*y[IDX_SO2I] + k[1271]*y[IDX_HeII]*y[IDX_SO2I] +
        k[1272]*y[IDX_HeII]*y[IDX_SOI] + k[1273]*y[IDX_HeII]*y[IDX_SOI] +
        k[1274]*y[IDX_HeII]*y[IDX_SiC2I] + k[1275]*y[IDX_HeII]*y[IDX_SiC3I] +
        k[1276]*y[IDX_HeII]*y[IDX_SiCI] + k[1277]*y[IDX_HeII]*y[IDX_SiCI] +
        k[1278]*y[IDX_HeII]*y[IDX_SiH2I] + k[1279]*y[IDX_HeII]*y[IDX_SiH2I] +
        k[1280]*y[IDX_HeII]*y[IDX_SiH3I] + k[1281]*y[IDX_HeII]*y[IDX_SiH3I] +
        k[1282]*y[IDX_HeII]*y[IDX_SiH4I] + k[1283]*y[IDX_HeII]*y[IDX_SiH4I] +
        k[1284]*y[IDX_HeII]*y[IDX_SiHI] + k[1285]*y[IDX_HeII]*y[IDX_SiOI] +
        k[1286]*y[IDX_HeII]*y[IDX_SiOI] + k[1287]*y[IDX_HeII]*y[IDX_SiSI] +
        k[1288]*y[IDX_HeII]*y[IDX_SiSI] - k[2111]*y[IDX_HII]*y[IDX_HeI] +
        k[2139]*y[IDX_HeII]*y[IDX_EM];
    ydot[IDX_H3II] = 0.0 - k[519]*y[IDX_H3II]*y[IDX_EM] -
        k[520]*y[IDX_H3II]*y[IDX_EM] + k[920]*y[IDX_H2II]*y[IDX_H2I] +
        k[925]*y[IDX_H2II]*y[IDX_HCOI] + k[951]*y[IDX_H2I]*y[IDX_HeHII] +
        k[954]*y[IDX_H2I]*y[IDX_NHII] + k[959]*y[IDX_H2I]*y[IDX_O2HII] -
        k[1022]*y[IDX_H3II]*y[IDX_C2I] - k[1023]*y[IDX_H3II]*y[IDX_C2H5OHI] -
        k[1024]*y[IDX_H3II]*y[IDX_C2H5OHI] - k[1025]*y[IDX_H3II]*y[IDX_C2HI] -
        k[1026]*y[IDX_H3II]*y[IDX_C2NI] - k[1027]*y[IDX_H3II]*y[IDX_CI] -
        k[1028]*y[IDX_H3II]*y[IDX_CH2I] - k[1029]*y[IDX_H3II]*y[IDX_CH3I] -
        k[1030]*y[IDX_H3II]*y[IDX_CH3CNI] - k[1031]*y[IDX_H3II]*y[IDX_CH3OHI] -
        k[1032]*y[IDX_H3II]*y[IDX_CH3OHI] - k[1033]*y[IDX_H3II]*y[IDX_CH4I] -
        k[1034]*y[IDX_H3II]*y[IDX_CHI] - k[1035]*y[IDX_H3II]*y[IDX_CNI] -
        k[1036]*y[IDX_H3II]*y[IDX_CO2I] - k[1037]*y[IDX_H3II]*y[IDX_COI] -
        k[1038]*y[IDX_H3II]*y[IDX_COI] - k[1039]*y[IDX_H3II]*y[IDX_CSI] -
        k[1040]*y[IDX_H3II]*y[IDX_ClI] - k[1041]*y[IDX_H3II]*y[IDX_H2COI] -
        k[1042]*y[IDX_H3II]*y[IDX_H2CSI] - k[1043]*y[IDX_H3II]*y[IDX_H2OI] -
        k[1044]*y[IDX_H3II]*y[IDX_H2SI] - k[1045]*y[IDX_H3II]*y[IDX_HCNI] -
        k[1046]*y[IDX_H3II]*y[IDX_HCOI] - k[1047]*y[IDX_H3II]*y[IDX_HCOOCH3I] -
        k[1048]*y[IDX_H3II]*y[IDX_HCSI] - k[1049]*y[IDX_H3II]*y[IDX_HClI] -
        k[1050]*y[IDX_H3II]*y[IDX_HNCI] - k[1051]*y[IDX_H3II]*y[IDX_HNOI] -
        k[1052]*y[IDX_H3II]*y[IDX_HS2I] - k[1053]*y[IDX_H3II]*y[IDX_HSI] -
        k[1054]*y[IDX_H3II]*y[IDX_MgI] - k[1055]*y[IDX_H3II]*y[IDX_N2I] -
        k[1056]*y[IDX_H3II]*y[IDX_NH2I] - k[1057]*y[IDX_H3II]*y[IDX_NH3I] -
        k[1058]*y[IDX_H3II]*y[IDX_NHI] - k[1059]*y[IDX_H3II]*y[IDX_NO2I] -
        k[1060]*y[IDX_H3II]*y[IDX_NOI] - k[1061]*y[IDX_H3II]*y[IDX_NSI] -
        k[1062]*y[IDX_H3II]*y[IDX_O2I] - k[1063]*y[IDX_H3II]*y[IDX_OI] -
        k[1064]*y[IDX_H3II]*y[IDX_OI] - k[1065]*y[IDX_H3II]*y[IDX_OCSI] -
        k[1066]*y[IDX_H3II]*y[IDX_OHI] - k[1067]*y[IDX_H3II]*y[IDX_S2I] -
        k[1068]*y[IDX_H3II]*y[IDX_SI] - k[1069]*y[IDX_H3II]*y[IDX_SO2I] -
        k[1070]*y[IDX_H3II]*y[IDX_SOI] - k[1071]*y[IDX_H3II]*y[IDX_SiI] -
        k[1072]*y[IDX_H3II]*y[IDX_SiH2I] - k[1073]*y[IDX_H3II]*y[IDX_SiH3I] -
        k[1074]*y[IDX_H3II]*y[IDX_SiH4I] - k[1075]*y[IDX_H3II]*y[IDX_SiHI] -
        k[1076]*y[IDX_H3II]*y[IDX_SiOI] - k[1077]*y[IDX_H3II]*y[IDX_SiSI] -
        k[2025]*y[IDX_H3II] - k[2026]*y[IDX_H3II];
    ydot[IDX_HII] = 0.0 - k[1]*y[IDX_HII]*y[IDX_HNCI] +
        k[1]*y[IDX_HII]*y[IDX_HNCI] + k[108]*y[IDX_ClII]*y[IDX_HI] -
        k[109]*y[IDX_ClI]*y[IDX_HII] - k[110]*y[IDX_HII]*y[IDX_C2I] -
        k[111]*y[IDX_HII]*y[IDX_C2H2I] - k[112]*y[IDX_HII]*y[IDX_C2HI] -
        k[113]*y[IDX_HII]*y[IDX_C2NI] - k[114]*y[IDX_HII]*y[IDX_CH2I] -
        k[115]*y[IDX_HII]*y[IDX_CH3I] - k[116]*y[IDX_HII]*y[IDX_CH4I] -
        k[117]*y[IDX_HII]*y[IDX_CHI] - k[118]*y[IDX_HII]*y[IDX_CSI] -
        k[119]*y[IDX_HII]*y[IDX_H2COI] - k[120]*y[IDX_HII]*y[IDX_H2CSI] -
        k[121]*y[IDX_HII]*y[IDX_H2OI] - k[122]*y[IDX_HII]*y[IDX_H2S2I] -
        k[123]*y[IDX_HII]*y[IDX_H2SI] - k[124]*y[IDX_HII]*y[IDX_HCNI] -
        k[125]*y[IDX_HII]*y[IDX_HCOI] - k[126]*y[IDX_HII]*y[IDX_HClI] -
        k[127]*y[IDX_HII]*y[IDX_HS2I] - k[128]*y[IDX_HII]*y[IDX_HSI] -
        k[129]*y[IDX_HII]*y[IDX_MgI] - k[130]*y[IDX_HII]*y[IDX_NH2I] -
        k[131]*y[IDX_HII]*y[IDX_NH3I] - k[132]*y[IDX_HII]*y[IDX_NHI] -
        k[133]*y[IDX_HII]*y[IDX_NOI] - k[134]*y[IDX_HII]*y[IDX_NSI] -
        k[135]*y[IDX_HII]*y[IDX_O2I] - k[136]*y[IDX_HII]*y[IDX_OI] -
        k[137]*y[IDX_HII]*y[IDX_OCSI] - k[138]*y[IDX_HII]*y[IDX_OHI] -
        k[139]*y[IDX_HII]*y[IDX_S2I] - k[140]*y[IDX_HII]*y[IDX_SI] -
        k[141]*y[IDX_HII]*y[IDX_SO2I] - k[142]*y[IDX_HII]*y[IDX_SOI] -
        k[143]*y[IDX_HII]*y[IDX_SiI] - k[144]*y[IDX_HII]*y[IDX_SiC2I] -
        k[145]*y[IDX_HII]*y[IDX_SiC3I] - k[146]*y[IDX_HII]*y[IDX_SiCI] -
        k[147]*y[IDX_HII]*y[IDX_SiH2I] - k[148]*y[IDX_HII]*y[IDX_SiH3I] -
        k[149]*y[IDX_HII]*y[IDX_SiH4I] - k[150]*y[IDX_HII]*y[IDX_SiHI] -
        k[151]*y[IDX_HII]*y[IDX_SiOI] - k[152]*y[IDX_HII]*y[IDX_SiSI] +
        k[191]*y[IDX_HI]*y[IDX_CNII] + k[192]*y[IDX_HI]*y[IDX_COII] +
        k[193]*y[IDX_HI]*y[IDX_H2II] + k[194]*y[IDX_HI]*y[IDX_HCNII] +
        k[195]*y[IDX_HI]*y[IDX_HeII] + k[196]*y[IDX_HI]*y[IDX_OII] +
        k[359]*y[IDX_H2I] + k[362]*y[IDX_HI] + k[406]*y[IDX_HI] -
        k[882]*y[IDX_HII]*y[IDX_C2H3I] - k[883]*y[IDX_HII]*y[IDX_C2H4I] -
        k[884]*y[IDX_HII]*y[IDX_C2HI] - k[885]*y[IDX_HII]*y[IDX_CH2I] -
        k[886]*y[IDX_HII]*y[IDX_CH3CNI] - k[887]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[888]*y[IDX_HII]*y[IDX_CH3OHI] - k[889]*y[IDX_HII]*y[IDX_CH3OHI] -
        k[890]*y[IDX_HII]*y[IDX_CH4I] - k[891]*y[IDX_HII]*y[IDX_CO2I] -
        k[892]*y[IDX_HII]*y[IDX_H2COI] - k[893]*y[IDX_HII]*y[IDX_H2COI] -
        k[894]*y[IDX_HII]*y[IDX_H2SI] - k[895]*y[IDX_HII]*y[IDX_H2SI] -
        k[896]*y[IDX_HII]*y[IDX_H2SiOI] - k[897]*y[IDX_HII]*y[IDX_HCOI] -
        k[898]*y[IDX_HII]*y[IDX_HCOI] - k[899]*y[IDX_HII]*y[IDX_HCSI] -
        k[900]*y[IDX_HII]*y[IDX_HNCOI] - k[901]*y[IDX_HII]*y[IDX_HNOI] -
        k[902]*y[IDX_HII]*y[IDX_HSI] - k[903]*y[IDX_HII]*y[IDX_NO2I] -
        k[904]*y[IDX_HII]*y[IDX_OCSI] - k[905]*y[IDX_HII]*y[IDX_SiH2I] -
        k[906]*y[IDX_HII]*y[IDX_SiH3I] - k[907]*y[IDX_HII]*y[IDX_SiH4I] -
        k[908]*y[IDX_HII]*y[IDX_SiHI] + k[950]*y[IDX_H2I]*y[IDX_HeII] +
        k[1205]*y[IDX_HeII]*y[IDX_CH4I] + k[1223]*y[IDX_HeII]*y[IDX_H2OI] +
        k[1240]*y[IDX_HeII]*y[IDX_HCSI] + k[1246]*y[IDX_HeII]*y[IDX_HNOI] +
        k[1977]*y[IDX_CHII] + k[1980]*y[IDX_CH2II] + k[2009]*y[IDX_H2II] +
        k[2026]*y[IDX_H3II] + k[2040]*y[IDX_HSII] + k[2048]*y[IDX_NHII] -
        k[2110]*y[IDX_HII]*y[IDX_HI] - k[2111]*y[IDX_HII]*y[IDX_HeI] -
        k[2135]*y[IDX_HII]*y[IDX_EM];
    ydot[IDX_OI] = 0.0 - k[0]*y[IDX_CHI]*y[IDX_OI] +
        k[6]*y[IDX_H2I]*y[IDX_O2I] + k[6]*y[IDX_H2I]*y[IDX_O2I] +
        k[7]*y[IDX_H2I]*y[IDX_OHI] + k[12]*y[IDX_HI]*y[IDX_O2I] +
        k[12]*y[IDX_HI]*y[IDX_O2I] + k[13]*y[IDX_HI]*y[IDX_OHI] +
        k[69]*y[IDX_CH2I]*y[IDX_OII] + k[90]*y[IDX_CHI]*y[IDX_OII] -
        k[136]*y[IDX_HII]*y[IDX_OI] + k[196]*y[IDX_HI]*y[IDX_OII] +
        k[297]*y[IDX_NHI]*y[IDX_OII] + k[306]*y[IDX_OII]*y[IDX_C2I] +
        k[307]*y[IDX_OII]*y[IDX_C2H2I] + k[308]*y[IDX_OII]*y[IDX_C2HI] +
        k[309]*y[IDX_OII]*y[IDX_CH4I] + k[310]*y[IDX_OII]*y[IDX_COI] +
        k[311]*y[IDX_OII]*y[IDX_H2COI] + k[312]*y[IDX_OII]*y[IDX_H2OI] +
        k[313]*y[IDX_OII]*y[IDX_H2SI] + k[314]*y[IDX_OII]*y[IDX_HCOI] +
        k[315]*y[IDX_OII]*y[IDX_NH2I] + k[316]*y[IDX_OII]*y[IDX_NH3I] +
        k[317]*y[IDX_OII]*y[IDX_O2I] + k[318]*y[IDX_OII]*y[IDX_OCSI] +
        k[319]*y[IDX_OII]*y[IDX_OHI] + k[320]*y[IDX_OII]*y[IDX_SO2I] -
        k[326]*y[IDX_OI]*y[IDX_CNII] - k[327]*y[IDX_OI]*y[IDX_COII] -
        k[328]*y[IDX_OI]*y[IDX_N2II] - k[365]*y[IDX_OI] + k[393]*y[IDX_CO2I] +
        k[394]*y[IDX_COI] + k[431]*y[IDX_NO2I] + k[433]*y[IDX_NOI] +
        k[436]*y[IDX_O2I] + k[436]*y[IDX_O2I] - k[438]*y[IDX_OI] +
        k[439]*y[IDX_OCNI] + k[442]*y[IDX_OHI] + k[445]*y[IDX_SO2I] +
        k[446]*y[IDX_SOI] + k[456]*y[IDX_SiOI] + k[499]*y[IDX_COII]*y[IDX_EM] +
        k[502]*y[IDX_H2COII]*y[IDX_EM] + k[512]*y[IDX_H2OII]*y[IDX_EM] +
        k[513]*y[IDX_H2OII]*y[IDX_EM] + k[529]*y[IDX_H3OII]*y[IDX_EM] +
        k[544]*y[IDX_HCO2II]*y[IDX_EM] + k[559]*y[IDX_HSO2II]*y[IDX_EM] +
        k[575]*y[IDX_NOII]*y[IDX_EM] + k[577]*y[IDX_O2II]*y[IDX_EM] +
        k[577]*y[IDX_O2II]*y[IDX_EM] + k[580]*y[IDX_OCSII]*y[IDX_EM] +
        k[582]*y[IDX_OHII]*y[IDX_EM] + k[584]*y[IDX_SOII]*y[IDX_EM] +
        k[585]*y[IDX_SO2II]*y[IDX_EM] + k[585]*y[IDX_SO2II]*y[IDX_EM] +
        k[586]*y[IDX_SO2II]*y[IDX_EM] + k[602]*y[IDX_SiOII]*y[IDX_EM] +
        k[633]*y[IDX_CII]*y[IDX_O2I] + k[639]*y[IDX_CII]*y[IDX_SOI] +
        k[700]*y[IDX_CI]*y[IDX_O2II] + k[702]*y[IDX_CI]*y[IDX_OHII] +
        k[733]*y[IDX_CHII]*y[IDX_O2I] - k[735]*y[IDX_CHII]*y[IDX_OI] -
        k[750]*y[IDX_CH2II]*y[IDX_OI] + k[769]*y[IDX_CH2I]*y[IDX_O2II] +
        k[771]*y[IDX_CH2I]*y[IDX_OHII] + k[782]*y[IDX_CH3II]*y[IDX_O2I] -
        k[783]*y[IDX_CH3II]*y[IDX_OI] - k[784]*y[IDX_CH3II]*y[IDX_OI] +
        k[819]*y[IDX_CH4I]*y[IDX_OHII] + k[858]*y[IDX_CHI]*y[IDX_O2II] +
        k[860]*y[IDX_CHI]*y[IDX_OHII] + k[891]*y[IDX_HII]*y[IDX_CO2I] -
        k[932]*y[IDX_H2II]*y[IDX_OI] - k[1063]*y[IDX_H3II]*y[IDX_OI] -
        k[1064]*y[IDX_H3II]*y[IDX_OI] + k[1209]*y[IDX_HeII]*y[IDX_CO2I] +
        k[1213]*y[IDX_HeII]*y[IDX_COI] + k[1218]*y[IDX_HeII]*y[IDX_H2COI] +
        k[1237]*y[IDX_HeII]*y[IDX_HCOI] + k[1258]*y[IDX_HeII]*y[IDX_NOI] +
        k[1261]*y[IDX_HeII]*y[IDX_O2I] + k[1262]*y[IDX_HeII]*y[IDX_OCNI] +
        k[1264]*y[IDX_HeII]*y[IDX_OCSI] + k[1271]*y[IDX_HeII]*y[IDX_SO2I] +
        k[1272]*y[IDX_HeII]*y[IDX_SOI] + k[1285]*y[IDX_HeII]*y[IDX_SiOI] +
        k[1309]*y[IDX_NII]*y[IDX_NOI] + k[1310]*y[IDX_NII]*y[IDX_O2I] +
        k[1339]*y[IDX_NI]*y[IDX_O2II] + k[1341]*y[IDX_NI]*y[IDX_SOII] +
        k[1358]*y[IDX_NHII]*y[IDX_H2OI] + k[1367]*y[IDX_NHII]*y[IDX_NOI] -
        k[1370]*y[IDX_NHII]*y[IDX_OI] + k[1380]*y[IDX_NH2II]*y[IDX_H2OI] +
        k[1390]*y[IDX_NH2II]*y[IDX_O2I] + k[1411]*y[IDX_NH2I]*y[IDX_OHII] +
        k[1458]*y[IDX_NHI]*y[IDX_O2II] + k[1460]*y[IDX_NHI]*y[IDX_OHII] +
        k[1484]*y[IDX_O2II]*y[IDX_SI] + k[1485]*y[IDX_O2I]*y[IDX_CSII] +
        k[1486]*y[IDX_O2I]*y[IDX_SII] - k[1490]*y[IDX_OI]*y[IDX_C2II] -
        k[1491]*y[IDX_OI]*y[IDX_C2HII] - k[1492]*y[IDX_OI]*y[IDX_C2H2II] -
        k[1493]*y[IDX_OI]*y[IDX_CH4II] - k[1494]*y[IDX_OI]*y[IDX_CH5II] -
        k[1495]*y[IDX_OI]*y[IDX_CH5II] - k[1496]*y[IDX_OI]*y[IDX_CSII] -
        k[1497]*y[IDX_OI]*y[IDX_H2OII] - k[1498]*y[IDX_OI]*y[IDX_H2SII] -
        k[1499]*y[IDX_OI]*y[IDX_H2SII] - k[1500]*y[IDX_OI]*y[IDX_HCO2II] -
        k[1501]*y[IDX_OI]*y[IDX_HCSII] - k[1502]*y[IDX_OI]*y[IDX_HCSII] -
        k[1503]*y[IDX_OI]*y[IDX_HSII] - k[1504]*y[IDX_OI]*y[IDX_HSII] -
        k[1505]*y[IDX_OI]*y[IDX_N2II] - k[1506]*y[IDX_OI]*y[IDX_N2HII] -
        k[1507]*y[IDX_OI]*y[IDX_NH2II] - k[1508]*y[IDX_OI]*y[IDX_NH3II] -
        k[1509]*y[IDX_OI]*y[IDX_NSII] - k[1510]*y[IDX_OI]*y[IDX_O2HII] -
        k[1511]*y[IDX_OI]*y[IDX_OHII] - k[1512]*y[IDX_OI]*y[IDX_SiCII] -
        k[1513]*y[IDX_OI]*y[IDX_SiHII] - k[1514]*y[IDX_OI]*y[IDX_SiH2II] -
        k[1515]*y[IDX_OI]*y[IDX_SiH3II] - k[1516]*y[IDX_OI]*y[IDX_SiOII] +
        k[1517]*y[IDX_OHII]*y[IDX_C2I] + k[1518]*y[IDX_OHII]*y[IDX_C2HI] +
        k[1519]*y[IDX_OHII]*y[IDX_CNI] + k[1520]*y[IDX_OHII]*y[IDX_CO2I] +
        k[1521]*y[IDX_OHII]*y[IDX_COI] + k[1522]*y[IDX_OHII]*y[IDX_H2COI] +
        k[1523]*y[IDX_OHII]*y[IDX_H2OI] + k[1524]*y[IDX_OHII]*y[IDX_H2SI] +
        k[1525]*y[IDX_OHII]*y[IDX_HCNI] + k[1527]*y[IDX_OHII]*y[IDX_HCOI] +
        k[1528]*y[IDX_OHII]*y[IDX_HNCI] + k[1529]*y[IDX_OHII]*y[IDX_N2I] +
        k[1530]*y[IDX_OHII]*y[IDX_NH3I] + k[1531]*y[IDX_OHII]*y[IDX_NOI] +
        k[1532]*y[IDX_OHII]*y[IDX_OHI] + k[1533]*y[IDX_OHII]*y[IDX_SI] +
        k[1535]*y[IDX_OHII]*y[IDX_SiI] + k[1536]*y[IDX_OHII]*y[IDX_SiHI] +
        k[1537]*y[IDX_OHII]*y[IDX_SiOI] + k[1539]*y[IDX_OHI]*y[IDX_COII] +
        k[1540]*y[IDX_OHI]*y[IDX_H2OII] + k[1546]*y[IDX_OHI]*y[IDX_NH3II] +
        k[1591]*y[IDX_CI]*y[IDX_COI] + k[1604]*y[IDX_CI]*y[IDX_NOI] +
        k[1608]*y[IDX_CI]*y[IDX_O2I] + k[1612]*y[IDX_CI]*y[IDX_OHI] +
        k[1615]*y[IDX_CI]*y[IDX_SOI] + k[1635]*y[IDX_CH2I]*y[IDX_O2I] -
        k[1637]*y[IDX_CH2I]*y[IDX_OI] - k[1638]*y[IDX_CH2I]*y[IDX_OI] -
        k[1639]*y[IDX_CH2I]*y[IDX_OI] - k[1640]*y[IDX_CH2I]*y[IDX_OI] +
        k[1643]*y[IDX_CH2I]*y[IDX_OHI] - k[1664]*y[IDX_CH3I]*y[IDX_OI] -
        k[1665]*y[IDX_CH3I]*y[IDX_OI] + k[1666]*y[IDX_CH3I]*y[IDX_OHI] +
        k[1684]*y[IDX_CHI]*y[IDX_NOI] + k[1688]*y[IDX_CHI]*y[IDX_O2I] +
        k[1690]*y[IDX_CHI]*y[IDX_O2I] - k[1693]*y[IDX_CHI]*y[IDX_OI] -
        k[1694]*y[IDX_CHI]*y[IDX_OI] + k[1713]*y[IDX_CNI]*y[IDX_O2I] +
        k[1718]*y[IDX_COI]*y[IDX_O2I] - k[1733]*y[IDX_H2I]*y[IDX_OI] +
        k[1752]*y[IDX_HI]*y[IDX_HCOI] + k[1755]*y[IDX_HI]*y[IDX_HNOI] +
        k[1764]*y[IDX_HI]*y[IDX_NOI] + k[1768]*y[IDX_HI]*y[IDX_O2I] +
        k[1769]*y[IDX_HI]*y[IDX_O2HI] + k[1772]*y[IDX_HI]*y[IDX_OCNI] +
        k[1776]*y[IDX_HI]*y[IDX_OHI] + k[1778]*y[IDX_HI]*y[IDX_SOI] +
        k[1812]*y[IDX_NI]*y[IDX_HCOI] + k[1820]*y[IDX_NI]*y[IDX_NO2I] +
        k[1820]*y[IDX_NI]*y[IDX_NO2I] + k[1823]*y[IDX_NI]*y[IDX_NOI] +
        k[1825]*y[IDX_NI]*y[IDX_O2I] + k[1828]*y[IDX_NI]*y[IDX_OHI] +
        k[1830]*y[IDX_NI]*y[IDX_SOI] + k[1837]*y[IDX_NH2I]*y[IDX_OHI] +
        k[1847]*y[IDX_NHI]*y[IDX_NOI] + k[1849]*y[IDX_NHI]*y[IDX_O2I] -
        k[1851]*y[IDX_NHI]*y[IDX_OI] - k[1852]*y[IDX_NHI]*y[IDX_OI] +
        k[1855]*y[IDX_NHI]*y[IDX_OHI] + k[1859]*y[IDX_NOI]*y[IDX_O2I] +
        k[1861]*y[IDX_NOI]*y[IDX_SI] + k[1865]*y[IDX_O2I]*y[IDX_SI] +
        k[1866]*y[IDX_O2I]*y[IDX_SOI] - k[1867]*y[IDX_OI]*y[IDX_C2I] -
        k[1868]*y[IDX_OI]*y[IDX_C2H2I] - k[1869]*y[IDX_OI]*y[IDX_C2H3I] -
        k[1870]*y[IDX_OI]*y[IDX_C2H4I] - k[1871]*y[IDX_OI]*y[IDX_C2H4I] -
        k[1872]*y[IDX_OI]*y[IDX_C2H4I] - k[1873]*y[IDX_OI]*y[IDX_C2H4I] -
        k[1874]*y[IDX_OI]*y[IDX_C2H5I] - k[1875]*y[IDX_OI]*y[IDX_C2HI] -
        k[1876]*y[IDX_OI]*y[IDX_C2NI] - k[1877]*y[IDX_OI]*y[IDX_C3NI] -
        k[1878]*y[IDX_OI]*y[IDX_C4NI] - k[1879]*y[IDX_OI]*y[IDX_CH4I] -
        k[1880]*y[IDX_OI]*y[IDX_CNI] - k[1881]*y[IDX_OI]*y[IDX_CNI] -
        k[1882]*y[IDX_OI]*y[IDX_CO2I] - k[1883]*y[IDX_OI]*y[IDX_CSI] -
        k[1884]*y[IDX_OI]*y[IDX_CSI] - k[1885]*y[IDX_OI]*y[IDX_H2CNI] -
        k[1886]*y[IDX_OI]*y[IDX_H2COI] - k[1887]*y[IDX_OI]*y[IDX_H2OI] -
        k[1888]*y[IDX_OI]*y[IDX_H2SI] - k[1889]*y[IDX_OI]*y[IDX_HCNI] -
        k[1890]*y[IDX_OI]*y[IDX_HCNI] - k[1891]*y[IDX_OI]*y[IDX_HCNI] -
        k[1892]*y[IDX_OI]*y[IDX_HCOI] - k[1893]*y[IDX_OI]*y[IDX_HCOI] -
        k[1894]*y[IDX_OI]*y[IDX_HCSI] - k[1895]*y[IDX_OI]*y[IDX_HCSI] -
        k[1896]*y[IDX_OI]*y[IDX_HNOI] - k[1897]*y[IDX_OI]*y[IDX_HNOI] -
        k[1898]*y[IDX_OI]*y[IDX_HNOI] - k[1899]*y[IDX_OI]*y[IDX_HSI] -
        k[1900]*y[IDX_OI]*y[IDX_HSI] - k[1901]*y[IDX_OI]*y[IDX_N2I] -
        k[1902]*y[IDX_OI]*y[IDX_NH2I] - k[1903]*y[IDX_OI]*y[IDX_NH2I] -
        k[1904]*y[IDX_OI]*y[IDX_NH3I] - k[1905]*y[IDX_OI]*y[IDX_NO2I] -
        k[1906]*y[IDX_OI]*y[IDX_NOI] - k[1907]*y[IDX_OI]*y[IDX_NSI] -
        k[1908]*y[IDX_OI]*y[IDX_NSI] - k[1909]*y[IDX_OI]*y[IDX_O2HI] -
        k[1910]*y[IDX_OI]*y[IDX_OCNI] - k[1911]*y[IDX_OI]*y[IDX_OCNI] -
        k[1912]*y[IDX_OI]*y[IDX_OCSI] - k[1913]*y[IDX_OI]*y[IDX_OCSI] -
        k[1914]*y[IDX_OI]*y[IDX_OHI] - k[1915]*y[IDX_OI]*y[IDX_S2I] -
        k[1916]*y[IDX_OI]*y[IDX_SO2I] - k[1917]*y[IDX_OI]*y[IDX_SOI] -
        k[1918]*y[IDX_OI]*y[IDX_SiC2I] - k[1919]*y[IDX_OI]*y[IDX_SiC3I] -
        k[1920]*y[IDX_OI]*y[IDX_SiCI] - k[1921]*y[IDX_OI]*y[IDX_SiCI] -
        k[1922]*y[IDX_OI]*y[IDX_SiH2I] - k[1923]*y[IDX_OI]*y[IDX_SiH2I] -
        k[1924]*y[IDX_OI]*y[IDX_SiH3I] - k[1925]*y[IDX_OI]*y[IDX_SiH4I] -
        k[1926]*y[IDX_OI]*y[IDX_SiHI] + k[1932]*y[IDX_OHI]*y[IDX_CNI] +
        k[1947]*y[IDX_OHI]*y[IDX_OHI] + k[1955]*y[IDX_SI]*y[IDX_SOI] +
        k[1959]*y[IDX_SiI]*y[IDX_O2I] + k[2002]*y[IDX_COII] +
        k[2003]*y[IDX_CO2I] + k[2004]*y[IDX_COI] + k[2056]*y[IDX_NO2I] +
        k[2058]*y[IDX_NOI] + k[2060]*y[IDX_O2II] + k[2062]*y[IDX_O2I] +
        k[2062]*y[IDX_O2I] + k[2064]*y[IDX_O2HI] + k[2065]*y[IDX_OCNI] +
        k[2069]*y[IDX_OHI] + k[2074]*y[IDX_SO2I] + k[2075]*y[IDX_SOI] +
        k[2092]*y[IDX_SiOII] + k[2093]*y[IDX_SiOI] -
        k[2098]*y[IDX_CII]*y[IDX_OI] - k[2104]*y[IDX_CI]*y[IDX_OI] -
        k[2124]*y[IDX_HI]*y[IDX_OI] - k[2128]*y[IDX_OI]*y[IDX_OI] -
        k[2128]*y[IDX_OI]*y[IDX_OI] - k[2129]*y[IDX_OI]*y[IDX_SOI] -
        k[2130]*y[IDX_OI]*y[IDX_SiII] - k[2131]*y[IDX_OI]*y[IDX_SiI] +
        k[2142]*y[IDX_OII]*y[IDX_EM] - k[2252]*y[IDX_OI];
    ydot[IDX_CI] = 0.0 + k[2]*y[IDX_H2I]*y[IDX_CHI] +
        k[9]*y[IDX_HI]*y[IDX_CHI] + k[14]*y[IDX_CII]*y[IDX_CH2I] +
        k[15]*y[IDX_CII]*y[IDX_CHI] + k[16]*y[IDX_CII]*y[IDX_H2COI] +
        k[17]*y[IDX_CII]*y[IDX_H2SI] + k[18]*y[IDX_CII]*y[IDX_HCOI] +
        k[19]*y[IDX_CII]*y[IDX_MgI] + k[21]*y[IDX_CII]*y[IDX_NH3I] +
        k[22]*y[IDX_CII]*y[IDX_NOI] + k[23]*y[IDX_CII]*y[IDX_NSI] +
        k[24]*y[IDX_CII]*y[IDX_OCSI] + k[25]*y[IDX_CII]*y[IDX_SOI] +
        k[26]*y[IDX_CII]*y[IDX_SiI] + k[27]*y[IDX_CII]*y[IDX_SiC2I] +
        k[28]*y[IDX_CII]*y[IDX_SiC3I] + k[29]*y[IDX_CII]*y[IDX_SiCI] +
        k[30]*y[IDX_CII]*y[IDX_SiH2I] + k[31]*y[IDX_CII]*y[IDX_SiH3I] +
        k[32]*y[IDX_CII]*y[IDX_SiSI] - k[50]*y[IDX_CI]*y[IDX_C2II] -
        k[51]*y[IDX_CI]*y[IDX_CNII] - k[52]*y[IDX_CI]*y[IDX_COII] -
        k[53]*y[IDX_CI]*y[IDX_N2II] - k[54]*y[IDX_CI]*y[IDX_O2II] -
        k[209]*y[IDX_HeII]*y[IDX_CI] + k[345]*y[IDX_SI]*y[IDX_CII] -
        k[356]*y[IDX_CI] + k[366]*y[IDX_C2I] + k[366]*y[IDX_C2I] +
        k[376]*y[IDX_C2NI] - k[379]*y[IDX_CI] + k[391]*y[IDX_CHI] +
        k[392]*y[IDX_CNI] + k[394]*y[IDX_COI] + k[396]*y[IDX_CSI] +
        k[449]*y[IDX_SiC2I] + k[450]*y[IDX_SiC3I] + k[451]*y[IDX_SiCI] +
        k[458]*y[IDX_C2II]*y[IDX_EM] + k[458]*y[IDX_C2II]*y[IDX_EM] +
        k[460]*y[IDX_C2HII]*y[IDX_EM] + k[469]*y[IDX_C2NII]*y[IDX_EM] +
        k[473]*y[IDX_C3II]*y[IDX_EM] + k[476]*y[IDX_C4NII]*y[IDX_EM] +
        k[477]*y[IDX_CHII]*y[IDX_EM] + k[478]*y[IDX_CH2II]*y[IDX_EM] +
        k[479]*y[IDX_CH2II]*y[IDX_EM] + k[498]*y[IDX_CNII]*y[IDX_EM] +
        k[499]*y[IDX_COII]*y[IDX_EM] + k[500]*y[IDX_CSII]*y[IDX_EM] +
        k[579]*y[IDX_OCSII]*y[IDX_EM] + k[587]*y[IDX_SiCII]*y[IDX_EM] +
        k[589]*y[IDX_SiC2II]*y[IDX_EM] + k[590]*y[IDX_SiC3II]*y[IDX_EM] +
        k[647]*y[IDX_C2II]*y[IDX_C2I] + k[650]*y[IDX_C2II]*y[IDX_SI] +
        k[658]*y[IDX_C2I]*y[IDX_SII] - k[685]*y[IDX_CI]*y[IDX_C2HII] -
        k[686]*y[IDX_CI]*y[IDX_CHII] - k[687]*y[IDX_CI]*y[IDX_CH2II] -
        k[688]*y[IDX_CI]*y[IDX_CH3II] - k[689]*y[IDX_CI]*y[IDX_CH5II] -
        k[690]*y[IDX_CI]*y[IDX_H2OII] - k[691]*y[IDX_CI]*y[IDX_H2SII] -
        k[692]*y[IDX_CI]*y[IDX_H3OII] - k[693]*y[IDX_CI]*y[IDX_HCNII] -
        k[694]*y[IDX_CI]*y[IDX_HCOII] - k[695]*y[IDX_CI]*y[IDX_HCO2II] -
        k[696]*y[IDX_CI]*y[IDX_HNOII] - k[697]*y[IDX_CI]*y[IDX_HSII] -
        k[698]*y[IDX_CI]*y[IDX_N2HII] - k[699]*y[IDX_CI]*y[IDX_NHII] -
        k[700]*y[IDX_CI]*y[IDX_O2II] - k[701]*y[IDX_CI]*y[IDX_O2HII] -
        k[702]*y[IDX_CI]*y[IDX_OHII] - k[703]*y[IDX_CI]*y[IDX_SiHII] -
        k[704]*y[IDX_CI]*y[IDX_SiOII] + k[708]*y[IDX_CHII]*y[IDX_CH3OHI] +
        k[716]*y[IDX_CHII]*y[IDX_H2COI] + k[719]*y[IDX_CHII]*y[IDX_H2OI] +
        k[721]*y[IDX_CHII]*y[IDX_H2SI] + k[725]*y[IDX_CHII]*y[IDX_HCNI] +
        k[727]*y[IDX_CHII]*y[IDX_HNCI] + k[730]*y[IDX_CHII]*y[IDX_NH3I] +
        k[737]*y[IDX_CHII]*y[IDX_OCSI] + k[740]*y[IDX_CHII]*y[IDX_SI] +
        k[841]*y[IDX_CHI]*y[IDX_COII] + k[856]*y[IDX_CHI]*y[IDX_NH3II] -
        k[912]*y[IDX_H2II]*y[IDX_CI] - k[1027]*y[IDX_H3II]*y[IDX_CI] +
        k[1177]*y[IDX_HeII]*y[IDX_C2I] + k[1187]*y[IDX_HeII]*y[IDX_C2HI] +
        k[1207]*y[IDX_HeII]*y[IDX_CNI] + k[1211]*y[IDX_HeII]*y[IDX_CO2I] +
        k[1214]*y[IDX_HeII]*y[IDX_CSI] + k[1244]*y[IDX_HeII]*y[IDX_HNCI] +
        k[1275]*y[IDX_HeII]*y[IDX_SiC3I] + k[1276]*y[IDX_HeII]*y[IDX_SiCI] +
        k[1297]*y[IDX_NII]*y[IDX_COI] + k[1332]*y[IDX_NI]*y[IDX_CNII] +
        k[1347]*y[IDX_NHII]*y[IDX_C2I] + k[1463]*y[IDX_OII]*y[IDX_C2I] +
        k[1469]*y[IDX_OII]*y[IDX_CNI] + k[1490]*y[IDX_OI]*y[IDX_C2II] +
        k[1491]*y[IDX_OI]*y[IDX_C2HII] + k[1512]*y[IDX_OI]*y[IDX_SiCII] +
        k[1573]*y[IDX_C2I]*y[IDX_SI] - k[1582]*y[IDX_CI]*y[IDX_C2H3I] -
        k[1583]*y[IDX_CI]*y[IDX_C2H5I] - k[1584]*y[IDX_CI]*y[IDX_C2NI] -
        k[1585]*y[IDX_CI]*y[IDX_C3H2I] - k[1586]*y[IDX_CI]*y[IDX_CH2I] -
        k[1587]*y[IDX_CI]*y[IDX_CH2I] - k[1588]*y[IDX_CI]*y[IDX_CH3I] -
        k[1589]*y[IDX_CI]*y[IDX_CHI] - k[1590]*y[IDX_CI]*y[IDX_CNI] -
        k[1591]*y[IDX_CI]*y[IDX_COI] - k[1592]*y[IDX_CI]*y[IDX_CSI] -
        k[1593]*y[IDX_CI]*y[IDX_H2CNI] - k[1594]*y[IDX_CI]*y[IDX_HCOI] -
        k[1595]*y[IDX_CI]*y[IDX_HSI] - k[1596]*y[IDX_CI]*y[IDX_HSI] -
        k[1597]*y[IDX_CI]*y[IDX_N2I] - k[1598]*y[IDX_CI]*y[IDX_NCCNI] -
        k[1599]*y[IDX_CI]*y[IDX_NH2I] - k[1600]*y[IDX_CI]*y[IDX_NH2I] -
        k[1601]*y[IDX_CI]*y[IDX_NH2I] - k[1602]*y[IDX_CI]*y[IDX_NHI] -
        k[1603]*y[IDX_CI]*y[IDX_NHI] - k[1604]*y[IDX_CI]*y[IDX_NOI] -
        k[1605]*y[IDX_CI]*y[IDX_NOI] - k[1606]*y[IDX_CI]*y[IDX_NSI] -
        k[1607]*y[IDX_CI]*y[IDX_NSI] - k[1608]*y[IDX_CI]*y[IDX_O2I] -
        k[1609]*y[IDX_CI]*y[IDX_OCNI] - k[1610]*y[IDX_CI]*y[IDX_OCSI] -
        k[1611]*y[IDX_CI]*y[IDX_OHI] - k[1612]*y[IDX_CI]*y[IDX_OHI] -
        k[1613]*y[IDX_CI]*y[IDX_S2I] - k[1614]*y[IDX_CI]*y[IDX_SO2I] -
        k[1615]*y[IDX_CI]*y[IDX_SOI] - k[1616]*y[IDX_CI]*y[IDX_SOI] -
        k[1617]*y[IDX_CI]*y[IDX_SiHI] + k[1683]*y[IDX_CHI]*y[IDX_NI] +
        k[1694]*y[IDX_CHI]*y[IDX_OI] + k[1698]*y[IDX_CHI]*y[IDX_SI] +
        k[1714]*y[IDX_CNI]*y[IDX_SI] - k[1722]*y[IDX_H2I]*y[IDX_CI] +
        k[1736]*y[IDX_HI]*y[IDX_C2I] + k[1743]*y[IDX_HI]*y[IDX_CHI] +
        k[1745]*y[IDX_HI]*y[IDX_COI] - k[1788]*y[IDX_HNCOI]*y[IDX_CI] +
        k[1790]*y[IDX_NI]*y[IDX_C2I] + k[1807]*y[IDX_NI]*y[IDX_CNI] +
        k[1867]*y[IDX_OI]*y[IDX_C2I] + k[1881]*y[IDX_OI]*y[IDX_CNI] +
        k[1884]*y[IDX_OI]*y[IDX_CSI] + k[1921]*y[IDX_OI]*y[IDX_SiCI] +
        k[1957]*y[IDX_SiI]*y[IDX_COI] + k[1960]*y[IDX_C2II] + k[1962]*y[IDX_C2I]
        + k[1962]*y[IDX_C2I] + k[1973]*y[IDX_C2NI] - k[1976]*y[IDX_CI] +
        k[1977]*y[IDX_CHII] + k[1999]*y[IDX_CHI] + k[2001]*y[IDX_CNI] +
        k[2004]*y[IDX_COI] + k[2005]*y[IDX_CSII] + k[2007]*y[IDX_CSI] +
        k[2080]*y[IDX_SiC3I] + k[2081]*y[IDX_SiCI] -
        k[2096]*y[IDX_CII]*y[IDX_CI] - k[2101]*y[IDX_CI]*y[IDX_CI] -
        k[2101]*y[IDX_CI]*y[IDX_CI] - k[2102]*y[IDX_CI]*y[IDX_NI] -
        k[2103]*y[IDX_CI]*y[IDX_OII] - k[2104]*y[IDX_CI]*y[IDX_OI] -
        k[2105]*y[IDX_CI]*y[IDX_SII] - k[2106]*y[IDX_CI]*y[IDX_SI] -
        k[2113]*y[IDX_H2I]*y[IDX_CI] - k[2123]*y[IDX_HI]*y[IDX_CI] +
        k[2132]*y[IDX_CII]*y[IDX_EM] - k[2206]*y[IDX_CI] + k[2292]*y[IDX_C3II];
    ydot[IDX_HCOII] = 0.0 + k[0]*y[IDX_CHI]*y[IDX_OI] +
        k[5]*y[IDX_H2I]*y[IDX_HOCII] + k[18]*y[IDX_CII]*y[IDX_HCOI] +
        k[33]*y[IDX_C2II]*y[IDX_HCOI] + k[44]*y[IDX_C2H2II]*y[IDX_HCOI] +
        k[55]*y[IDX_CHII]*y[IDX_HCOI] + k[72]*y[IDX_CH3II]*y[IDX_HCOI] +
        k[96]*y[IDX_CNII]*y[IDX_HCOI] + k[103]*y[IDX_COII]*y[IDX_HCOI] +
        k[125]*y[IDX_HII]*y[IDX_HCOI] + k[165]*y[IDX_H2II]*y[IDX_HCOI] +
        k[180]*y[IDX_H2OII]*y[IDX_HCOI] + k[202]*y[IDX_HCOI]*y[IDX_H2COII] +
        k[203]*y[IDX_HCOI]*y[IDX_H2SII] + k[204]*y[IDX_HCOI]*y[IDX_O2II] +
        k[205]*y[IDX_HCOI]*y[IDX_SII] + k[206]*y[IDX_HCOI]*y[IDX_SiOII] -
        k[224]*y[IDX_MgI]*y[IDX_HCOII] + k[243]*y[IDX_NII]*y[IDX_HCOI] +
        k[254]*y[IDX_N2II]*y[IDX_HCOI] + k[267]*y[IDX_NH2II]*y[IDX_HCOI] +
        k[278]*y[IDX_NH3II]*y[IDX_HCOI] + k[314]*y[IDX_OII]*y[IDX_HCOI] +
        k[334]*y[IDX_OHII]*y[IDX_HCOI] + k[410]*y[IDX_HCOI] -
        k[542]*y[IDX_HCOII]*y[IDX_EM] + k[618]*y[IDX_CII]*y[IDX_H2COI] +
        k[620]*y[IDX_CII]*y[IDX_H2OI] - k[653]*y[IDX_C2I]*y[IDX_HCOII] +
        k[677]*y[IDX_C2HI]*y[IDX_COII] - k[680]*y[IDX_C2HI]*y[IDX_HCOII] +
        k[692]*y[IDX_CI]*y[IDX_H3OII] - k[694]*y[IDX_CI]*y[IDX_HCOII] +
        k[714]*y[IDX_CHII]*y[IDX_CO2I] + k[717]*y[IDX_CHII]*y[IDX_H2COI] +
        k[720]*y[IDX_CHII]*y[IDX_H2OI] + k[733]*y[IDX_CHII]*y[IDX_O2I] +
        k[742]*y[IDX_CH2II]*y[IDX_H2COI] + k[749]*y[IDX_CH2II]*y[IDX_O2I] +
        k[750]*y[IDX_CH2II]*y[IDX_OI] + k[756]*y[IDX_CH2I]*y[IDX_COII] -
        k[763]*y[IDX_CH2I]*y[IDX_HCOII] + k[777]*y[IDX_CH3II]*y[IDX_H2COI] +
        k[784]*y[IDX_CH3II]*y[IDX_OI] + k[797]*y[IDX_CH4II]*y[IDX_COI] +
        k[807]*y[IDX_CH4I]*y[IDX_COII] + k[826]*y[IDX_CH5II]*y[IDX_COI] +
        k[841]*y[IDX_CHI]*y[IDX_COII] - k[849]*y[IDX_CHI]*y[IDX_HCOII] +
        k[858]*y[IDX_CHI]*y[IDX_O2II] + k[864]*y[IDX_CHI]*y[IDX_SiOII] +
        k[865]*y[IDX_CNII]*y[IDX_H2COI] + k[871]*y[IDX_COII]*y[IDX_H2COI] +
        k[872]*y[IDX_COII]*y[IDX_H2SI] + k[874]*y[IDX_COI]*y[IDX_H2ClII] +
        k[875]*y[IDX_COI]*y[IDX_HCO2II] + k[876]*y[IDX_COI]*y[IDX_HNOII] +
        k[877]*y[IDX_COI]*y[IDX_N2HII] + k[878]*y[IDX_COI]*y[IDX_O2HII] +
        k[880]*y[IDX_COI]*y[IDX_SiH4II] + k[889]*y[IDX_HII]*y[IDX_CH3OHI] +
        k[891]*y[IDX_HII]*y[IDX_CO2I] + k[893]*y[IDX_HII]*y[IDX_H2COI] +
        k[919]*y[IDX_H2II]*y[IDX_COI] + k[921]*y[IDX_H2II]*y[IDX_H2COI] +
        k[941]*y[IDX_H2I]*y[IDX_COII] + k[967]*y[IDX_H2COII]*y[IDX_O2I] +
        k[972]*y[IDX_H2COI]*y[IDX_O2II] + k[975]*y[IDX_H2COI]*y[IDX_SII] +
        k[978]*y[IDX_H2OII]*y[IDX_COI] + k[993]*y[IDX_H2OI]*y[IDX_C2NII] +
        k[994]*y[IDX_H2OI]*y[IDX_C4NII] + k[996]*y[IDX_H2OI]*y[IDX_CNII] +
        k[997]*y[IDX_H2OI]*y[IDX_COII] - k[1003]*y[IDX_H2OI]*y[IDX_HCOII] +
        k[1037]*y[IDX_H3II]*y[IDX_COI] + k[1111]*y[IDX_HCNII]*y[IDX_COI] -
        k[1124]*y[IDX_HCNI]*y[IDX_HCOII] - k[1136]*y[IDX_HCOII]*y[IDX_C2H5OHI] -
        k[1137]*y[IDX_HCOII]*y[IDX_CH3CCHI] - k[1138]*y[IDX_HCOII]*y[IDX_CH3CNI]
        - k[1139]*y[IDX_HCOII]*y[IDX_CH3OHI] - k[1140]*y[IDX_HCOII]*y[IDX_CSI] -
        k[1141]*y[IDX_HCOII]*y[IDX_H2COI] - k[1142]*y[IDX_HCOII]*y[IDX_H2CSI] -
        k[1143]*y[IDX_HCOII]*y[IDX_H2SI] - k[1144]*y[IDX_HCOII]*y[IDX_HCOI] -
        k[1145]*y[IDX_HCOII]*y[IDX_HCOOCH3I] - k[1146]*y[IDX_HCOII]*y[IDX_HS2I]
        - k[1147]*y[IDX_HCOII]*y[IDX_HSI] - k[1148]*y[IDX_HCOII]*y[IDX_NSI] -
        k[1149]*y[IDX_HCOII]*y[IDX_OCSI] - k[1150]*y[IDX_HCOII]*y[IDX_S2I] -
        k[1151]*y[IDX_HCOII]*y[IDX_SI] - k[1152]*y[IDX_HCOII]*y[IDX_SOI] -
        k[1153]*y[IDX_HCOII]*y[IDX_SiH2I] - k[1154]*y[IDX_HCOII]*y[IDX_SiH4I] -
        k[1155]*y[IDX_HCOII]*y[IDX_SiHI] - k[1156]*y[IDX_HCOII]*y[IDX_SiOI] -
        k[1157]*y[IDX_HCOII]*y[IDX_SiSI] - k[1168]*y[IDX_HNCI]*y[IDX_HCOII] +
        k[1217]*y[IDX_HeII]*y[IDX_H2COI] + k[1298]*y[IDX_NII]*y[IDX_H2COI] +
        k[1314]*y[IDX_N2II]*y[IDX_H2COI] + k[1353]*y[IDX_NHII]*y[IDX_COI] +
        k[1355]*y[IDX_NHII]*y[IDX_H2COI] + k[1398]*y[IDX_NH2I]*y[IDX_COII] -
        k[1406]*y[IDX_NH2I]*y[IDX_HCOII] + k[1423]*y[IDX_NH3I]*y[IDX_COII] -
        k[1433]*y[IDX_NH3I]*y[IDX_HCOII] + k[1448]*y[IDX_NHI]*y[IDX_COII] -
        k[1452]*y[IDX_NHI]*y[IDX_HCOII] + k[1471]*y[IDX_OII]*y[IDX_H2COI] +
        k[1474]*y[IDX_OII]*y[IDX_HCNI] + k[1482]*y[IDX_O2II]*y[IDX_C2H2I] +
        k[1491]*y[IDX_OI]*y[IDX_C2HII] + k[1492]*y[IDX_OI]*y[IDX_C2H2II] +
        k[1500]*y[IDX_OI]*y[IDX_HCO2II] + k[1502]*y[IDX_OI]*y[IDX_HCSII] +
        k[1521]*y[IDX_OHII]*y[IDX_COI] + k[1539]*y[IDX_OHI]*y[IDX_COII] -
        k[1542]*y[IDX_OHI]*y[IDX_HCOII] - k[1543]*y[IDX_OHI]*y[IDX_HCOII] +
        k[1557]*y[IDX_SOII]*y[IDX_C2H2I] - k[1566]*y[IDX_SiI]*y[IDX_HCOII] +
        k[2014]*y[IDX_H2COI] - k[2029]*y[IDX_HCOII] + k[2031]*y[IDX_HCOI] -
        k[2240]*y[IDX_HCOII];
    ydot[IDX_H2OI] = 0.0 - k[4]*y[IDX_H2I]*y[IDX_H2OI] -
        k[11]*y[IDX_HI]*y[IDX_H2OI] + k[66]*y[IDX_CH2I]*y[IDX_H2OII] +
        k[86]*y[IDX_CHI]*y[IDX_H2OII] - k[121]*y[IDX_HII]*y[IDX_H2OI] -
        k[162]*y[IDX_H2II]*y[IDX_H2OI] + k[175]*y[IDX_H2OII]*y[IDX_C2I] +
        k[176]*y[IDX_H2OII]*y[IDX_C2H2I] + k[177]*y[IDX_H2OII]*y[IDX_C2HI] +
        k[178]*y[IDX_H2OII]*y[IDX_H2COI] + k[179]*y[IDX_H2OII]*y[IDX_H2SI] +
        k[180]*y[IDX_H2OII]*y[IDX_HCOI] + k[181]*y[IDX_H2OII]*y[IDX_MgI] +
        k[182]*y[IDX_H2OII]*y[IDX_NOI] + k[183]*y[IDX_H2OII]*y[IDX_O2I] +
        k[184]*y[IDX_H2OII]*y[IDX_OCSI] + k[185]*y[IDX_H2OII]*y[IDX_SI] +
        k[186]*y[IDX_H2OII]*y[IDX_SiI] - k[187]*y[IDX_H2OI]*y[IDX_COII] -
        k[188]*y[IDX_H2OI]*y[IDX_HCNII] - k[189]*y[IDX_H2OI]*y[IDX_N2II] -
        k[213]*y[IDX_HeII]*y[IDX_H2OI] - k[240]*y[IDX_NII]*y[IDX_H2OI] -
        k[261]*y[IDX_NHII]*y[IDX_H2OI] + k[274]*y[IDX_NH2I]*y[IDX_H2OII] +
        k[285]*y[IDX_NH3I]*y[IDX_H2OII] - k[312]*y[IDX_OII]*y[IDX_H2OI] -
        k[332]*y[IDX_OHII]*y[IDX_H2OI] - k[401]*y[IDX_H2OI] +
        k[464]*y[IDX_C2H5OH2II]*y[IDX_EM] + k[465]*y[IDX_C2H5OH2II]*y[IDX_EM] +
        k[486]*y[IDX_CH3OH2II]*y[IDX_EM] + k[487]*y[IDX_CH3OH2II]*y[IDX_EM] +
        k[522]*y[IDX_H3COII]*y[IDX_EM] + k[528]*y[IDX_H3OII]*y[IDX_EM] -
        k[620]*y[IDX_CII]*y[IDX_H2OI] - k[621]*y[IDX_CII]*y[IDX_H2OI] -
        k[718]*y[IDX_CHII]*y[IDX_H2OI] - k[719]*y[IDX_CHII]*y[IDX_H2OI] -
        k[720]*y[IDX_CHII]*y[IDX_H2OI] - k[743]*y[IDX_CH2II]*y[IDX_H2OI] +
        k[759]*y[IDX_CH2I]*y[IDX_H3OII] - k[799]*y[IDX_CH4II]*y[IDX_H2OI] -
        k[828]*y[IDX_CH5II]*y[IDX_H2OI] + k[845]*y[IDX_CHI]*y[IDX_H3OII] +
        k[887]*y[IDX_HII]*y[IDX_CH3OHI] - k[922]*y[IDX_H2II]*y[IDX_H2OI] -
        k[980]*y[IDX_H2OII]*y[IDX_H2OI] - k[990]*y[IDX_H2OI]*y[IDX_C2II] -
        k[991]*y[IDX_H2OI]*y[IDX_C2H2II] - k[992]*y[IDX_H2OI]*y[IDX_C2NII] -
        k[993]*y[IDX_H2OI]*y[IDX_C2NII] - k[994]*y[IDX_H2OI]*y[IDX_C4NII] -
        k[995]*y[IDX_H2OI]*y[IDX_CNII] - k[996]*y[IDX_H2OI]*y[IDX_CNII] -
        k[997]*y[IDX_H2OI]*y[IDX_COII] - k[998]*y[IDX_H2OI]*y[IDX_H2COII] -
        k[999]*y[IDX_H2OI]*y[IDX_H2ClII] - k[1000]*y[IDX_H2OI]*y[IDX_H2SII] -
        k[1001]*y[IDX_H2OI]*y[IDX_H3COII] - k[1002]*y[IDX_H2OI]*y[IDX_HCNII] -
        k[1003]*y[IDX_H2OI]*y[IDX_HCOII] - k[1004]*y[IDX_H2OI]*y[IDX_HCO2II] -
        k[1005]*y[IDX_H2OI]*y[IDX_HNOII] - k[1006]*y[IDX_H2OI]*y[IDX_HOCSII] -
        k[1007]*y[IDX_H2OI]*y[IDX_HSII] - k[1008]*y[IDX_H2OI]*y[IDX_HSO2II] -
        k[1009]*y[IDX_H2OI]*y[IDX_HSiSII] - k[1010]*y[IDX_H2OI]*y[IDX_N2II] -
        k[1011]*y[IDX_H2OI]*y[IDX_N2HII] - k[1012]*y[IDX_H2OI]*y[IDX_O2HII] -
        k[1013]*y[IDX_H2OI]*y[IDX_SiII] - k[1014]*y[IDX_H2OI]*y[IDX_SiHII] -
        k[1015]*y[IDX_H2OI]*y[IDX_SiH4II] - k[1016]*y[IDX_H2OI]*y[IDX_SiH5II] +
        k[1021]*y[IDX_H2SI]*y[IDX_SOII] + k[1023]*y[IDX_H3II]*y[IDX_C2H5OHI] +
        k[1031]*y[IDX_H3II]*y[IDX_CH3OHI] - k[1043]*y[IDX_H3II]*y[IDX_H2OI] +
        k[1080]*y[IDX_H3OII]*y[IDX_C2I] + k[1081]*y[IDX_H3OII]*y[IDX_C2H5OHI] +
        k[1082]*y[IDX_H3OII]*y[IDX_CH3CCHI] + k[1083]*y[IDX_H3OII]*y[IDX_CH3CNI]
        + k[1084]*y[IDX_H3OII]*y[IDX_CH3OHI] + k[1085]*y[IDX_H3OII]*y[IDX_CSI] +
        k[1086]*y[IDX_H3OII]*y[IDX_H2COI] + k[1087]*y[IDX_H3OII]*y[IDX_H2SI] +
        k[1088]*y[IDX_H3OII]*y[IDX_HCNI] + k[1089]*y[IDX_H3OII]*y[IDX_HCOOCH3I]
        + k[1090]*y[IDX_H3OII]*y[IDX_HNCI] + k[1091]*y[IDX_H3OII]*y[IDX_HS2I] +
        k[1092]*y[IDX_H3OII]*y[IDX_S2I] + k[1093]*y[IDX_H3OII]*y[IDX_SiI] +
        k[1094]*y[IDX_H3OII]*y[IDX_SiH2I] + k[1095]*y[IDX_H3OII]*y[IDX_SiHI] +
        k[1096]*y[IDX_H3OII]*y[IDX_SiOI] + k[1120]*y[IDX_HCNI]*y[IDX_CH3OH2II] -
        k[1222]*y[IDX_HeII]*y[IDX_H2OI] - k[1223]*y[IDX_HeII]*y[IDX_H2OI] -
        k[1356]*y[IDX_NHII]*y[IDX_H2OI] - k[1357]*y[IDX_NHII]*y[IDX_H2OI] -
        k[1358]*y[IDX_NHII]*y[IDX_H2OI] - k[1359]*y[IDX_NHII]*y[IDX_H2OI] -
        k[1378]*y[IDX_NH2II]*y[IDX_H2OI] - k[1379]*y[IDX_NH2II]*y[IDX_H2OI] -
        k[1380]*y[IDX_NH2II]*y[IDX_H2OI] + k[1402]*y[IDX_NH2I]*y[IDX_H3OII] -
        k[1414]*y[IDX_NH3II]*y[IDX_H2OI] + k[1428]*y[IDX_NH3I]*y[IDX_H3OII] +
        k[1464]*y[IDX_OII]*y[IDX_C2H4I] + k[1466]*y[IDX_OII]*y[IDX_CH3OHI] +
        k[1473]*y[IDX_OII]*y[IDX_H2SI] - k[1523]*y[IDX_OHII]*y[IDX_H2OI] +
        k[1634]*y[IDX_CH2I]*y[IDX_O2I] + k[1642]*y[IDX_CH2I]*y[IDX_OHI] -
        k[1652]*y[IDX_CH3I]*y[IDX_H2OI] + k[1659]*y[IDX_CH3I]*y[IDX_NOI] +
        k[1661]*y[IDX_CH3I]*y[IDX_O2I] + k[1668]*y[IDX_CH3I]*y[IDX_OHI] +
        k[1672]*y[IDX_CH4I]*y[IDX_OHI] + k[1734]*y[IDX_H2I]*y[IDX_OHI] -
        k[1748]*y[IDX_HI]*y[IDX_H2OI] + k[1769]*y[IDX_HI]*y[IDX_O2HI] +
        k[1834]*y[IDX_NH2I]*y[IDX_NOI] + k[1836]*y[IDX_NH2I]*y[IDX_OHI] -
        k[1841]*y[IDX_NHI]*y[IDX_H2OI] + k[1853]*y[IDX_NHI]*y[IDX_OHI] -
        k[1887]*y[IDX_OI]*y[IDX_H2OI] + k[1927]*y[IDX_OHI]*y[IDX_C2H2I] +
        k[1930]*y[IDX_OHI]*y[IDX_C2H3I] + k[1931]*y[IDX_OHI]*y[IDX_C2H5I] +
        k[1937]*y[IDX_OHI]*y[IDX_H2COI] + k[1938]*y[IDX_OHI]*y[IDX_H2SI] +
        k[1939]*y[IDX_OHI]*y[IDX_HCNI] + k[1941]*y[IDX_OHI]*y[IDX_HCOI] +
        k[1942]*y[IDX_OHI]*y[IDX_HNOI] + k[1944]*y[IDX_OHI]*y[IDX_NH3I] +
        k[1946]*y[IDX_OHI]*y[IDX_O2HI] + k[1947]*y[IDX_OHI]*y[IDX_OHI] -
        k[2017]*y[IDX_H2OI] - k[2018]*y[IDX_H2OI] -
        k[2107]*y[IDX_CH3II]*y[IDX_H2OI] + k[2125]*y[IDX_HI]*y[IDX_OHI] -
        k[2214]*y[IDX_H2OI] + k[2311]*y[IDX_GH2OI] + k[2312]*y[IDX_GH2OI] +
        k[2313]*y[IDX_GH2OI] + k[2314]*y[IDX_GH2OI];
    ydot[IDX_COI] = 0.0 + k[37]*y[IDX_C2I]*y[IDX_COII] +
        k[48]*y[IDX_C2HI]*y[IDX_COII] + k[52]*y[IDX_CI]*y[IDX_COII] +
        k[64]*y[IDX_CH2I]*y[IDX_COII] + k[81]*y[IDX_CH4I]*y[IDX_COII] +
        k[84]*y[IDX_CHI]*y[IDX_COII] - k[93]*y[IDX_CNII]*y[IDX_COI] +
        k[101]*y[IDX_COII]*y[IDX_H2COI] + k[102]*y[IDX_COII]*y[IDX_H2SI] +
        k[103]*y[IDX_COII]*y[IDX_HCOI] + k[104]*y[IDX_COII]*y[IDX_NOI] +
        k[105]*y[IDX_COII]*y[IDX_O2I] + k[106]*y[IDX_COII]*y[IDX_SI] -
        k[107]*y[IDX_COI]*y[IDX_N2II] - k[160]*y[IDX_H2II]*y[IDX_COI] +
        k[187]*y[IDX_H2OI]*y[IDX_COII] + k[192]*y[IDX_HI]*y[IDX_COII] +
        k[200]*y[IDX_HCNI]*y[IDX_COII] - k[238]*y[IDX_NII]*y[IDX_COI] +
        k[273]*y[IDX_NH2I]*y[IDX_COII] + k[283]*y[IDX_NH3I]*y[IDX_COII] +
        k[295]*y[IDX_NHI]*y[IDX_COII] - k[310]*y[IDX_OII]*y[IDX_COI] +
        k[327]*y[IDX_OI]*y[IDX_COII] + k[341]*y[IDX_OHI]*y[IDX_COII] -
        k[357]*y[IDX_COI] + k[383]*y[IDX_CH2COI] + k[393]*y[IDX_CO2I] -
        k[394]*y[IDX_COI] + k[399]*y[IDX_H2COI] + k[409]*y[IDX_HCOI] +
        k[415]*y[IDX_HNCOI] + k[441]*y[IDX_OCSI] +
        k[503]*y[IDX_H2COII]*y[IDX_EM] + k[504]*y[IDX_H2COII]*y[IDX_EM] +
        k[523]*y[IDX_H3COII]*y[IDX_EM] + k[542]*y[IDX_HCOII]*y[IDX_EM] +
        k[544]*y[IDX_HCO2II]*y[IDX_EM] + k[545]*y[IDX_HCO2II]*y[IDX_EM] +
        k[551]*y[IDX_HOCII]*y[IDX_EM] + k[581]*y[IDX_OCSII]*y[IDX_EM] +
        k[616]*y[IDX_CII]*y[IDX_CO2I] + k[617]*y[IDX_CII]*y[IDX_H2COI] +
        k[626]*y[IDX_CII]*y[IDX_HCOI] + k[634]*y[IDX_CII]*y[IDX_O2I] +
        k[636]*y[IDX_CII]*y[IDX_OCSI] + k[638]*y[IDX_CII]*y[IDX_SO2I] +
        k[640]*y[IDX_CII]*y[IDX_SOI] + k[645]*y[IDX_CII]*y[IDX_SiOI] +
        k[648]*y[IDX_C2II]*y[IDX_HCOI] + k[649]*y[IDX_C2II]*y[IDX_O2I] +
        k[653]*y[IDX_C2I]*y[IDX_HCOII] + k[656]*y[IDX_C2I]*y[IDX_O2II] +
        k[659]*y[IDX_C2I]*y[IDX_SiOII] + k[663]*y[IDX_C2HII]*y[IDX_HCOI] +
        k[680]*y[IDX_C2HI]*y[IDX_HCOII] + k[694]*y[IDX_CI]*y[IDX_HCOII] +
        k[704]*y[IDX_CI]*y[IDX_SiOII] + k[714]*y[IDX_CHII]*y[IDX_CO2I] +
        k[715]*y[IDX_CHII]*y[IDX_H2COI] + k[726]*y[IDX_CHII]*y[IDX_HCOI] +
        k[736]*y[IDX_CHII]*y[IDX_OCSI] + k[741]*y[IDX_CH2II]*y[IDX_CO2I] +
        k[747]*y[IDX_CH2II]*y[IDX_HCOI] + k[751]*y[IDX_CH2II]*y[IDX_OCSI] +
        k[763]*y[IDX_CH2I]*y[IDX_HCOII] + k[779]*y[IDX_CH3II]*y[IDX_HCOI] +
        k[785]*y[IDX_CH3II]*y[IDX_OCSI] - k[797]*y[IDX_CH4II]*y[IDX_COI] -
        k[826]*y[IDX_CH5II]*y[IDX_COI] + k[849]*y[IDX_CHI]*y[IDX_HCOII] +
        k[867]*y[IDX_CNII]*y[IDX_HCOI] + k[868]*y[IDX_CNII]*y[IDX_O2I] -
        k[874]*y[IDX_COI]*y[IDX_H2ClII] - k[875]*y[IDX_COI]*y[IDX_HCO2II] -
        k[876]*y[IDX_COI]*y[IDX_HNOII] - k[877]*y[IDX_COI]*y[IDX_N2HII] -
        k[878]*y[IDX_COI]*y[IDX_O2HII] - k[879]*y[IDX_COI]*y[IDX_SO2II] -
        k[880]*y[IDX_COI]*y[IDX_SiH4II] - k[881]*y[IDX_COI]*y[IDX_SiOII] +
        k[898]*y[IDX_HII]*y[IDX_HCOI] + k[900]*y[IDX_HII]*y[IDX_HNCOI] +
        k[904]*y[IDX_HII]*y[IDX_OCSI] - k[919]*y[IDX_H2II]*y[IDX_COI] +
        k[925]*y[IDX_H2II]*y[IDX_HCOI] + k[974]*y[IDX_H2COI]*y[IDX_SII] -
        k[978]*y[IDX_H2OII]*y[IDX_COI] + k[984]*y[IDX_H2OII]*y[IDX_HCOI] +
        k[1003]*y[IDX_H2OI]*y[IDX_HCOII] - k[1037]*y[IDX_H3II]*y[IDX_COI] -
        k[1038]*y[IDX_H3II]*y[IDX_COI] - k[1111]*y[IDX_HCNII]*y[IDX_COI] +
        k[1115]*y[IDX_HCNII]*y[IDX_HCOI] + k[1124]*y[IDX_HCNI]*y[IDX_HCOII] +
        k[1136]*y[IDX_HCOII]*y[IDX_C2H5OHI] +
        k[1137]*y[IDX_HCOII]*y[IDX_CH3CCHI] + k[1138]*y[IDX_HCOII]*y[IDX_CH3CNI]
        + k[1139]*y[IDX_HCOII]*y[IDX_CH3OHI] + k[1140]*y[IDX_HCOII]*y[IDX_CSI] +
        k[1141]*y[IDX_HCOII]*y[IDX_H2COI] + k[1142]*y[IDX_HCOII]*y[IDX_H2CSI] +
        k[1143]*y[IDX_HCOII]*y[IDX_H2SI] + k[1144]*y[IDX_HCOII]*y[IDX_HCOI] +
        k[1145]*y[IDX_HCOII]*y[IDX_HCOOCH3I] + k[1146]*y[IDX_HCOII]*y[IDX_HS2I]
        + k[1147]*y[IDX_HCOII]*y[IDX_HSI] + k[1148]*y[IDX_HCOII]*y[IDX_NSI] +
        k[1149]*y[IDX_HCOII]*y[IDX_OCSI] + k[1150]*y[IDX_HCOII]*y[IDX_S2I] +
        k[1151]*y[IDX_HCOII]*y[IDX_SI] + k[1152]*y[IDX_HCOII]*y[IDX_SOI] +
        k[1153]*y[IDX_HCOII]*y[IDX_SiH2I] + k[1154]*y[IDX_HCOII]*y[IDX_SiH4I] +
        k[1155]*y[IDX_HCOII]*y[IDX_SiHI] + k[1156]*y[IDX_HCOII]*y[IDX_SiOI] +
        k[1157]*y[IDX_HCOII]*y[IDX_SiSI] + k[1158]*y[IDX_HCOI]*y[IDX_H2COII] +
        k[1161]*y[IDX_HCOI]*y[IDX_O2II] + k[1163]*y[IDX_HCOI]*y[IDX_SII] +
        k[1168]*y[IDX_HNCI]*y[IDX_HCOII] + k[1195]*y[IDX_HeII]*y[IDX_CH2COI] +
        k[1210]*y[IDX_HeII]*y[IDX_CO2I] - k[1213]*y[IDX_HeII]*y[IDX_COI] +
        k[1236]*y[IDX_HeII]*y[IDX_HCOI] + k[1266]*y[IDX_HeII]*y[IDX_OCSI] -
        k[1297]*y[IDX_NII]*y[IDX_COI] + k[1303]*y[IDX_NII]*y[IDX_HCOI] +
        k[1313]*y[IDX_NII]*y[IDX_OCSI] + k[1317]*y[IDX_N2II]*y[IDX_HCOI] +
        k[1318]*y[IDX_N2II]*y[IDX_OCSI] + k[1351]*y[IDX_NHII]*y[IDX_CO2I] -
        k[1353]*y[IDX_NHII]*y[IDX_COI] + k[1406]*y[IDX_NH2I]*y[IDX_HCOII] +
        k[1416]*y[IDX_NH3II]*y[IDX_HCOI] + k[1433]*y[IDX_NH3I]*y[IDX_HCOII] +
        k[1452]*y[IDX_NHI]*y[IDX_HCOII] + k[1470]*y[IDX_OII]*y[IDX_CO2I] +
        k[1476]*y[IDX_OII]*y[IDX_HCOI] + k[1482]*y[IDX_O2II]*y[IDX_C2H2I] -
        k[1521]*y[IDX_OHII]*y[IDX_COI] + k[1526]*y[IDX_OHII]*y[IDX_HCOI] +
        k[1542]*y[IDX_OHI]*y[IDX_HCOII] + k[1552]*y[IDX_SII]*y[IDX_OCSI] +
        k[1556]*y[IDX_SOII]*y[IDX_C2H2I] + k[1565]*y[IDX_SiII]*y[IDX_OCSI] +
        k[1566]*y[IDX_SiI]*y[IDX_HCOII] + k[1572]*y[IDX_C2I]*y[IDX_O2I] +
        k[1572]*y[IDX_C2I]*y[IDX_O2I] + k[1574]*y[IDX_C2H2I]*y[IDX_NOI] +
        k[1581]*y[IDX_C2HI]*y[IDX_O2I] - k[1591]*y[IDX_CI]*y[IDX_COI] +
        k[1594]*y[IDX_CI]*y[IDX_HCOI] + k[1605]*y[IDX_CI]*y[IDX_NOI] +
        k[1608]*y[IDX_CI]*y[IDX_O2I] + k[1609]*y[IDX_CI]*y[IDX_OCNI] +
        k[1610]*y[IDX_CI]*y[IDX_OCSI] + k[1611]*y[IDX_CI]*y[IDX_OHI] +
        k[1614]*y[IDX_CI]*y[IDX_SO2I] + k[1616]*y[IDX_CI]*y[IDX_SOI] +
        k[1625]*y[IDX_CH2I]*y[IDX_HCOI] + k[1634]*y[IDX_CH2I]*y[IDX_O2I] +
        k[1637]*y[IDX_CH2I]*y[IDX_OI] + k[1638]*y[IDX_CH2I]*y[IDX_OI] +
        k[1654]*y[IDX_CH3I]*y[IDX_HCOI] + k[1664]*y[IDX_CH3I]*y[IDX_OI] +
        k[1677]*y[IDX_CHI]*y[IDX_CO2I] + k[1679]*y[IDX_CHI]*y[IDX_HCOI] +
        k[1688]*y[IDX_CHI]*y[IDX_O2I] + k[1689]*y[IDX_CHI]*y[IDX_O2I] +
        k[1693]*y[IDX_CHI]*y[IDX_OI] + k[1695]*y[IDX_CHI]*y[IDX_OCSI] +
        k[1699]*y[IDX_CHI]*y[IDX_SOI] + k[1706]*y[IDX_CNI]*y[IDX_HCOI] +
        k[1710]*y[IDX_CNI]*y[IDX_NOI] + k[1712]*y[IDX_CNI]*y[IDX_O2I] -
        k[1716]*y[IDX_COI]*y[IDX_HNOI] - k[1717]*y[IDX_COI]*y[IDX_NO2I] -
        k[1718]*y[IDX_COI]*y[IDX_O2I] - k[1719]*y[IDX_COI]*y[IDX_O2HI] +
        k[1740]*y[IDX_HI]*y[IDX_CH2COI] + k[1744]*y[IDX_HI]*y[IDX_CO2I] -
        k[1745]*y[IDX_HI]*y[IDX_COI] + k[1751]*y[IDX_HI]*y[IDX_HCOI] +
        k[1773]*y[IDX_HI]*y[IDX_OCNI] + k[1775]*y[IDX_HI]*y[IDX_OCSI] +
        k[1780]*y[IDX_HCOI]*y[IDX_HCOI] + k[1780]*y[IDX_HCOI]*y[IDX_HCOI] +
        k[1781]*y[IDX_HCOI]*y[IDX_HCOI] + k[1783]*y[IDX_HCOI]*y[IDX_NOI] +
        k[1785]*y[IDX_HCOI]*y[IDX_O2I] + k[1788]*y[IDX_HNCOI]*y[IDX_CI] +
        k[1808]*y[IDX_NI]*y[IDX_CO2I] + k[1811]*y[IDX_NI]*y[IDX_HCOI] +
        k[1864]*y[IDX_O2I]*y[IDX_OCNI] + k[1867]*y[IDX_OI]*y[IDX_C2I] +
        k[1868]*y[IDX_OI]*y[IDX_C2H2I] + k[1875]*y[IDX_OI]*y[IDX_C2HI] +
        k[1876]*y[IDX_OI]*y[IDX_C2NI] + k[1877]*y[IDX_OI]*y[IDX_C3NI] +
        k[1878]*y[IDX_OI]*y[IDX_C4NI] + k[1880]*y[IDX_OI]*y[IDX_CNI] +
        k[1882]*y[IDX_OI]*y[IDX_CO2I] + k[1883]*y[IDX_OI]*y[IDX_CSI] +
        k[1890]*y[IDX_OI]*y[IDX_HCNI] + k[1893]*y[IDX_OI]*y[IDX_HCOI] +
        k[1894]*y[IDX_OI]*y[IDX_HCSI] + k[1910]*y[IDX_OI]*y[IDX_OCNI] +
        k[1913]*y[IDX_OI]*y[IDX_OCSI] + k[1918]*y[IDX_OI]*y[IDX_SiC2I] +
        k[1919]*y[IDX_OI]*y[IDX_SiC3I] + k[1920]*y[IDX_OI]*y[IDX_SiCI] +
        k[1929]*y[IDX_OHI]*y[IDX_C2H2I] - k[1934]*y[IDX_OHI]*y[IDX_COI] +
        k[1935]*y[IDX_OHI]*y[IDX_CSI] + k[1940]*y[IDX_OHI]*y[IDX_HCNI] +
        k[1941]*y[IDX_OHI]*y[IDX_HCOI] + k[1952]*y[IDX_SI]*y[IDX_HCOI] +
        k[1956]*y[IDX_SiI]*y[IDX_CO2I] - k[1957]*y[IDX_SiI]*y[IDX_COI] +
        k[1983]*y[IDX_CH2COI] + k[2003]*y[IDX_CO2I] - k[2004]*y[IDX_COI] +
        k[2011]*y[IDX_H2COI] + k[2012]*y[IDX_H2COI] + k[2030]*y[IDX_HCOI] +
        k[2037]*y[IDX_HNCOI] + k[2067]*y[IDX_OCSI] + k[2104]*y[IDX_CI]*y[IDX_OI]
        - k[2169]*y[IDX_COI] - k[2207]*y[IDX_COI] - k[2296]*y[IDX_COI] +
        k[2343]*y[IDX_GCOI] + k[2344]*y[IDX_GCOI] + k[2345]*y[IDX_GCOI] +
        k[2346]*y[IDX_GCOI];
    ydot[IDX_H2I] = 0.0 - k[2]*y[IDX_H2I]*y[IDX_CHI] +
        k[2]*y[IDX_H2I]*y[IDX_CHI] - k[3]*y[IDX_H2I]*y[IDX_H2I] -
        k[3]*y[IDX_H2I]*y[IDX_H2I] + k[3]*y[IDX_H2I]*y[IDX_H2I] -
        k[4]*y[IDX_H2I]*y[IDX_H2OI] + k[4]*y[IDX_H2I]*y[IDX_H2OI] -
        k[5]*y[IDX_H2I]*y[IDX_HOCII] + k[5]*y[IDX_H2I]*y[IDX_HOCII] -
        k[6]*y[IDX_H2I]*y[IDX_O2I] + k[6]*y[IDX_H2I]*y[IDX_O2I] -
        k[7]*y[IDX_H2I]*y[IDX_OHI] + k[7]*y[IDX_H2I]*y[IDX_OHI] -
        k[8]*y[IDX_H2I]*y[IDX_EM] - k[10]*y[IDX_HI]*y[IDX_H2I] +
        k[153]*y[IDX_H2II]*y[IDX_C2I] + k[154]*y[IDX_H2II]*y[IDX_C2H2I] +
        k[155]*y[IDX_H2II]*y[IDX_C2HI] + k[156]*y[IDX_H2II]*y[IDX_CH2I] +
        k[157]*y[IDX_H2II]*y[IDX_CH4I] + k[158]*y[IDX_H2II]*y[IDX_CHI] +
        k[159]*y[IDX_H2II]*y[IDX_CNI] + k[160]*y[IDX_H2II]*y[IDX_COI] +
        k[161]*y[IDX_H2II]*y[IDX_H2COI] + k[162]*y[IDX_H2II]*y[IDX_H2OI] +
        k[163]*y[IDX_H2II]*y[IDX_H2SI] + k[164]*y[IDX_H2II]*y[IDX_HCNI] +
        k[165]*y[IDX_H2II]*y[IDX_HCOI] + k[166]*y[IDX_H2II]*y[IDX_NH2I] +
        k[167]*y[IDX_H2II]*y[IDX_NH3I] + k[168]*y[IDX_H2II]*y[IDX_NHI] +
        k[169]*y[IDX_H2II]*y[IDX_NOI] + k[170]*y[IDX_H2II]*y[IDX_O2I] +
        k[171]*y[IDX_H2II]*y[IDX_OHI] - k[172]*y[IDX_H2I]*y[IDX_HeII] +
        k[193]*y[IDX_HI]*y[IDX_H2II] - k[359]*y[IDX_H2I] - k[360]*y[IDX_H2I] -
        k[361]*y[IDX_H2I] + k[370]*y[IDX_C2H4I] + k[371]*y[IDX_C2H5I] +
        k[386]*y[IDX_CH3I] + k[388]*y[IDX_CH3OHI] + k[390]*y[IDX_CH4I] +
        k[399]*y[IDX_H2COI] + k[400]*y[IDX_H2CSI] + k[404]*y[IDX_H2SI] +
        k[405]*y[IDX_H2SiOI] + k[428]*y[IDX_NH3I] + k[454]*y[IDX_SiH4I] +
        k[478]*y[IDX_CH2II]*y[IDX_EM] + k[482]*y[IDX_CH3II]*y[IDX_EM] +
        k[490]*y[IDX_CH3OH2II]*y[IDX_EM] + k[493]*y[IDX_CH5II]*y[IDX_EM] +
        k[494]*y[IDX_CH5II]*y[IDX_EM] + k[497]*y[IDX_CH5II]*y[IDX_EM] +
        k[497]*y[IDX_CH5II]*y[IDX_EM] + k[503]*y[IDX_H2COII]*y[IDX_EM] +
        k[511]*y[IDX_H2NOII]*y[IDX_EM] + k[512]*y[IDX_H2OII]*y[IDX_EM] +
        k[519]*y[IDX_H3II]*y[IDX_EM] + k[523]*y[IDX_H3COII]*y[IDX_EM] +
        k[526]*y[IDX_H3CSII]*y[IDX_EM] + k[529]*y[IDX_H3OII]*y[IDX_EM] +
        k[530]*y[IDX_H3OII]*y[IDX_EM] + k[533]*y[IDX_H3SII]*y[IDX_EM] +
        k[535]*y[IDX_H3SII]*y[IDX_EM] + k[572]*y[IDX_NH4II]*y[IDX_EM] +
        k[593]*y[IDX_SiH2II]*y[IDX_EM] + k[597]*y[IDX_SiH3II]*y[IDX_EM] +
        k[598]*y[IDX_SiH4II]*y[IDX_EM] + k[600]*y[IDX_SiH5II]*y[IDX_EM] +
        k[609]*y[IDX_CII]*y[IDX_CH3I] + k[614]*y[IDX_CII]*y[IDX_CH4I] +
        k[630]*y[IDX_CII]*y[IDX_NH3I] + k[643]*y[IDX_CII]*y[IDX_SiH2I] +
        k[670]*y[IDX_C2H2II]*y[IDX_SiI] + k[671]*y[IDX_C2H2II]*y[IDX_SiH4I] +
        k[688]*y[IDX_CI]*y[IDX_CH3II] + k[692]*y[IDX_CI]*y[IDX_H3OII] +
        k[706]*y[IDX_CHII]*y[IDX_C2HI] + k[707]*y[IDX_CHII]*y[IDX_CH2I] +
        k[711]*y[IDX_CHII]*y[IDX_CH4I] + k[712]*y[IDX_CHII]*y[IDX_CHI] +
        k[720]*y[IDX_CHII]*y[IDX_H2OI] + k[722]*y[IDX_CHII]*y[IDX_H2SI] +
        k[723]*y[IDX_CHII]*y[IDX_HCNI] + k[729]*y[IDX_CHII]*y[IDX_NH2I] +
        k[731]*y[IDX_CHII]*y[IDX_NHI] + k[738]*y[IDX_CHII]*y[IDX_OHI] +
        k[746]*y[IDX_CH2II]*y[IDX_H2SI] + k[774]*y[IDX_CH3II]*y[IDX_C2H4I] +
        k[778]*y[IDX_CH3II]*y[IDX_H2SI] + k[780]*y[IDX_CH3II]*y[IDX_HSI] +
        k[784]*y[IDX_CH3II]*y[IDX_OI] + k[786]*y[IDX_CH3II]*y[IDX_OHI] +
        k[787]*y[IDX_CH3II]*y[IDX_SI] + k[788]*y[IDX_CH3II]*y[IDX_SOI] +
        k[814]*y[IDX_CH4I]*y[IDX_HSII] + k[815]*y[IDX_CH4I]*y[IDX_N2II] +
        k[822]*y[IDX_CH4I]*y[IDX_SII] + k[836]*y[IDX_CH5II]*y[IDX_SiH4I] +
        k[839]*y[IDX_CHI]*y[IDX_CH3II] + k[882]*y[IDX_HII]*y[IDX_C2H3I] +
        k[883]*y[IDX_HII]*y[IDX_C2H4I] + k[884]*y[IDX_HII]*y[IDX_C2HI] +
        k[885]*y[IDX_HII]*y[IDX_CH2I] + k[888]*y[IDX_HII]*y[IDX_CH3OHI] +
        k[889]*y[IDX_HII]*y[IDX_CH3OHI] + k[889]*y[IDX_HII]*y[IDX_CH3OHI] +
        k[890]*y[IDX_HII]*y[IDX_CH4I] + k[892]*y[IDX_HII]*y[IDX_H2COI] +
        k[893]*y[IDX_HII]*y[IDX_H2COI] + k[894]*y[IDX_HII]*y[IDX_H2SI] +
        k[895]*y[IDX_HII]*y[IDX_H2SI] + k[896]*y[IDX_HII]*y[IDX_H2SiOI] +
        k[897]*y[IDX_HII]*y[IDX_HCOI] + k[899]*y[IDX_HII]*y[IDX_HCSI] +
        k[901]*y[IDX_HII]*y[IDX_HNOI] + k[902]*y[IDX_HII]*y[IDX_HSI] +
        k[905]*y[IDX_HII]*y[IDX_SiH2I] + k[906]*y[IDX_HII]*y[IDX_SiH3I] +
        k[907]*y[IDX_HII]*y[IDX_SiH4I] + k[908]*y[IDX_HII]*y[IDX_SiHI] +
        k[910]*y[IDX_H2II]*y[IDX_C2H4I] + k[910]*y[IDX_H2II]*y[IDX_C2H4I] +
        k[914]*y[IDX_H2II]*y[IDX_CH4I] - k[920]*y[IDX_H2II]*y[IDX_H2I] +
        k[921]*y[IDX_H2II]*y[IDX_H2COI] + k[923]*y[IDX_H2II]*y[IDX_H2SI] +
        k[924]*y[IDX_H2II]*y[IDX_H2SI] + k[924]*y[IDX_H2II]*y[IDX_H2SI] -
        k[934]*y[IDX_H2I]*y[IDX_CII] - k[935]*y[IDX_H2I]*y[IDX_C2II] -
        k[936]*y[IDX_H2I]*y[IDX_C2HII] - k[937]*y[IDX_H2I]*y[IDX_CHII] -
        k[938]*y[IDX_H2I]*y[IDX_CH2II] - k[939]*y[IDX_H2I]*y[IDX_CH4II] -
        k[940]*y[IDX_H2I]*y[IDX_CNII] - k[941]*y[IDX_H2I]*y[IDX_COII] -
        k[942]*y[IDX_H2I]*y[IDX_COII] - k[943]*y[IDX_H2I]*y[IDX_CSII] -
        k[944]*y[IDX_H2I]*y[IDX_ClII] - k[945]*y[IDX_H2I]*y[IDX_H2OII] -
        k[946]*y[IDX_H2I]*y[IDX_H2SII] - k[947]*y[IDX_H2I]*y[IDX_HCNII] -
        k[948]*y[IDX_H2I]*y[IDX_HClII] - k[949]*y[IDX_H2I]*y[IDX_HSII] -
        k[950]*y[IDX_H2I]*y[IDX_HeII] - k[951]*y[IDX_H2I]*y[IDX_HeHII] -
        k[952]*y[IDX_H2I]*y[IDX_NII] - k[953]*y[IDX_H2I]*y[IDX_N2II] -
        k[954]*y[IDX_H2I]*y[IDX_NHII] - k[955]*y[IDX_H2I]*y[IDX_NHII] -
        k[956]*y[IDX_H2I]*y[IDX_NH2II] - k[957]*y[IDX_H2I]*y[IDX_NH3II] -
        k[958]*y[IDX_H2I]*y[IDX_OII] - k[959]*y[IDX_H2I]*y[IDX_O2HII] -
        k[960]*y[IDX_H2I]*y[IDX_OHII] - k[961]*y[IDX_H2I]*y[IDX_SII] -
        k[962]*y[IDX_H2I]*y[IDX_SO2II] - k[963]*y[IDX_H2I]*y[IDX_SiH4II] -
        k[964]*y[IDX_H2I]*y[IDX_SiOII] + k[969]*y[IDX_H2COI]*y[IDX_CH3OH2II] +
        k[1022]*y[IDX_H3II]*y[IDX_C2I] + k[1024]*y[IDX_H3II]*y[IDX_C2H5OHI] +
        k[1025]*y[IDX_H3II]*y[IDX_C2HI] + k[1026]*y[IDX_H3II]*y[IDX_C2NI] +
        k[1027]*y[IDX_H3II]*y[IDX_CI] + k[1028]*y[IDX_H3II]*y[IDX_CH2I] +
        k[1029]*y[IDX_H3II]*y[IDX_CH3I] + k[1030]*y[IDX_H3II]*y[IDX_CH3CNI] +
        k[1031]*y[IDX_H3II]*y[IDX_CH3OHI] + k[1032]*y[IDX_H3II]*y[IDX_CH3OHI] +
        k[1033]*y[IDX_H3II]*y[IDX_CH4I] + k[1034]*y[IDX_H3II]*y[IDX_CHI] +
        k[1035]*y[IDX_H3II]*y[IDX_CNI] + k[1036]*y[IDX_H3II]*y[IDX_CO2I] +
        k[1037]*y[IDX_H3II]*y[IDX_COI] + k[1038]*y[IDX_H3II]*y[IDX_COI] +
        k[1039]*y[IDX_H3II]*y[IDX_CSI] + k[1040]*y[IDX_H3II]*y[IDX_ClI] +
        k[1041]*y[IDX_H3II]*y[IDX_H2COI] + k[1042]*y[IDX_H3II]*y[IDX_H2CSI] +
        k[1043]*y[IDX_H3II]*y[IDX_H2OI] + k[1044]*y[IDX_H3II]*y[IDX_H2SI] +
        k[1045]*y[IDX_H3II]*y[IDX_HCNI] + k[1046]*y[IDX_H3II]*y[IDX_HCOI] +
        k[1047]*y[IDX_H3II]*y[IDX_HCOOCH3I] + k[1048]*y[IDX_H3II]*y[IDX_HCSI] +
        k[1049]*y[IDX_H3II]*y[IDX_HClI] + k[1050]*y[IDX_H3II]*y[IDX_HNCI] +
        k[1051]*y[IDX_H3II]*y[IDX_HNOI] + k[1052]*y[IDX_H3II]*y[IDX_HS2I] +
        k[1053]*y[IDX_H3II]*y[IDX_HSI] + k[1054]*y[IDX_H3II]*y[IDX_MgI] +
        k[1055]*y[IDX_H3II]*y[IDX_N2I] + k[1056]*y[IDX_H3II]*y[IDX_NH2I] +
        k[1057]*y[IDX_H3II]*y[IDX_NH3I] + k[1058]*y[IDX_H3II]*y[IDX_NHI] +
        k[1059]*y[IDX_H3II]*y[IDX_NO2I] + k[1060]*y[IDX_H3II]*y[IDX_NOI] +
        k[1061]*y[IDX_H3II]*y[IDX_NSI] + k[1062]*y[IDX_H3II]*y[IDX_O2I] +
        k[1064]*y[IDX_H3II]*y[IDX_OI] + k[1065]*y[IDX_H3II]*y[IDX_OCSI] +
        k[1066]*y[IDX_H3II]*y[IDX_OHI] + k[1067]*y[IDX_H3II]*y[IDX_S2I] +
        k[1068]*y[IDX_H3II]*y[IDX_SI] + k[1069]*y[IDX_H3II]*y[IDX_SO2I] +
        k[1070]*y[IDX_H3II]*y[IDX_SOI] + k[1071]*y[IDX_H3II]*y[IDX_SiI] +
        k[1072]*y[IDX_H3II]*y[IDX_SiH2I] + k[1073]*y[IDX_H3II]*y[IDX_SiH3I] +
        k[1074]*y[IDX_H3II]*y[IDX_SiH4I] + k[1075]*y[IDX_H3II]*y[IDX_SiHI] +
        k[1076]*y[IDX_H3II]*y[IDX_SiOI] + k[1077]*y[IDX_H3II]*y[IDX_SiSI] +
        k[1098]*y[IDX_HI]*y[IDX_CHII] + k[1099]*y[IDX_HI]*y[IDX_CH2II] +
        k[1100]*y[IDX_HI]*y[IDX_CH3II] + k[1101]*y[IDX_HI]*y[IDX_CH4II] +
        k[1102]*y[IDX_HI]*y[IDX_CH5II] + k[1103]*y[IDX_HI]*y[IDX_H2SII] +
        k[1104]*y[IDX_HI]*y[IDX_H3SII] + k[1105]*y[IDX_HI]*y[IDX_HSII] +
        k[1108]*y[IDX_HI]*y[IDX_SiHII] + k[1176]*y[IDX_HSII]*y[IDX_H2SI] +
        k[1178]*y[IDX_HeII]*y[IDX_C2H2I] + k[1181]*y[IDX_HeII]*y[IDX_C2H3I] +
        k[1183]*y[IDX_HeII]*y[IDX_C2H4I] + k[1184]*y[IDX_HeII]*y[IDX_C2H4I] +
        k[1192]*y[IDX_HeII]*y[IDX_CH2I] + k[1196]*y[IDX_HeII]*y[IDX_CH3I] +
        k[1197]*y[IDX_HeII]*y[IDX_CH3CCHI] + k[1197]*y[IDX_HeII]*y[IDX_CH3CCHI]
        + k[1202]*y[IDX_HeII]*y[IDX_CH4I] + k[1203]*y[IDX_HeII]*y[IDX_CH4I] +
        k[1216]*y[IDX_HeII]*y[IDX_H2COI] + k[1219]*y[IDX_HeII]*y[IDX_H2CSI] +
        k[1227]*y[IDX_HeII]*y[IDX_H2SI] + k[1252]*y[IDX_HeII]*y[IDX_NH2I] +
        k[1254]*y[IDX_HeII]*y[IDX_NH3I] + k[1278]*y[IDX_HeII]*y[IDX_SiH2I] +
        k[1280]*y[IDX_HeII]*y[IDX_SiH3I] + k[1282]*y[IDX_HeII]*y[IDX_SiH4I] +
        k[1282]*y[IDX_HeII]*y[IDX_SiH4I] + k[1283]*y[IDX_HeII]*y[IDX_SiH4I] +
        k[1294]*y[IDX_NII]*y[IDX_CH4I] + k[1306]*y[IDX_NII]*y[IDX_NH3I] +
        k[1316]*y[IDX_N2II]*y[IDX_H2SI] + k[1328]*y[IDX_NI]*y[IDX_C2H2II] +
        k[1334]*y[IDX_NI]*y[IDX_H2OII] + k[1335]*y[IDX_NI]*y[IDX_H2SII] +
        k[1357]*y[IDX_NHII]*y[IDX_H2OI] + k[1446]*y[IDX_NHI]*y[IDX_CH3II] +
        k[1494]*y[IDX_OI]*y[IDX_CH5II] + k[1497]*y[IDX_OI]*y[IDX_H2OII] +
        k[1499]*y[IDX_OI]*y[IDX_H2SII] + k[1508]*y[IDX_OI]*y[IDX_NH3II] +
        k[1515]*y[IDX_OI]*y[IDX_SiH3II] + k[1551]*y[IDX_SII]*y[IDX_H2SI] +
        k[1575]*y[IDX_C2H2I]*y[IDX_SiI] + k[1593]*y[IDX_CI]*y[IDX_H2CNI] +
        k[1618]*y[IDX_CH2I]*y[IDX_CH2I] + k[1632]*y[IDX_CH2I]*y[IDX_O2I] +
        k[1637]*y[IDX_CH2I]*y[IDX_OI] + k[1644]*y[IDX_CH2I]*y[IDX_SI] +
        k[1647]*y[IDX_CH3I]*y[IDX_CH3I] + k[1664]*y[IDX_CH3I]*y[IDX_OI] +
        k[1667]*y[IDX_CH3I]*y[IDX_OHI] - k[1720]*y[IDX_ClI]*y[IDX_H2I] -
        k[1721]*y[IDX_H2I]*y[IDX_C2HI] - k[1722]*y[IDX_H2I]*y[IDX_CI] -
        k[1723]*y[IDX_H2I]*y[IDX_CH2I] - k[1724]*y[IDX_H2I]*y[IDX_CH3I] -
        k[1725]*y[IDX_H2I]*y[IDX_CHI] - k[1726]*y[IDX_H2I]*y[IDX_CNI] -
        k[1727]*y[IDX_H2I]*y[IDX_HSI] - k[1728]*y[IDX_H2I]*y[IDX_NI] -
        k[1729]*y[IDX_H2I]*y[IDX_NH2I] - k[1730]*y[IDX_H2I]*y[IDX_NHI] -
        k[1731]*y[IDX_H2I]*y[IDX_O2I] - k[1732]*y[IDX_H2I]*y[IDX_O2I] -
        k[1733]*y[IDX_H2I]*y[IDX_OI] - k[1734]*y[IDX_H2I]*y[IDX_OHI] -
        k[1735]*y[IDX_H2I]*y[IDX_SI] + k[1737]*y[IDX_HI]*y[IDX_C2H2I] +
        k[1738]*y[IDX_HI]*y[IDX_C2H3I] + k[1739]*y[IDX_HI]*y[IDX_CH2I] +
        k[1741]*y[IDX_HI]*y[IDX_CH3I] + k[1742]*y[IDX_HI]*y[IDX_CH4I] +
        k[1743]*y[IDX_HI]*y[IDX_CHI] + k[1746]*y[IDX_HI]*y[IDX_H2CNI] +
        k[1747]*y[IDX_HI]*y[IDX_H2COI] + k[1748]*y[IDX_HI]*y[IDX_H2OI] +
        k[1749]*y[IDX_HI]*y[IDX_H2SI] + k[1750]*y[IDX_HI]*y[IDX_HCNI] +
        k[1751]*y[IDX_HI]*y[IDX_HCOI] + k[1753]*y[IDX_HI]*y[IDX_HCSI] +
        k[1756]*y[IDX_HI]*y[IDX_HNOI] + k[1758]*y[IDX_HI]*y[IDX_HSI] +
        k[1760]*y[IDX_HI]*y[IDX_NH2I] + k[1761]*y[IDX_HI]*y[IDX_NH3I] +
        k[1762]*y[IDX_HI]*y[IDX_NHI] + k[1770]*y[IDX_HI]*y[IDX_O2HI] +
        k[1776]*y[IDX_HI]*y[IDX_OHI] + k[1780]*y[IDX_HCOI]*y[IDX_HCOI] +
        k[1787]*y[IDX_HClI]*y[IDX_HI] + k[1805]*y[IDX_NI]*y[IDX_CH3I] +
        k[1843]*y[IDX_NHI]*y[IDX_NHI] + k[1871]*y[IDX_OI]*y[IDX_C2H4I] +
        k[1885]*y[IDX_OI]*y[IDX_H2CNI] + k[1922]*y[IDX_OI]*y[IDX_SiH2I] +
        k[1967]*y[IDX_C2H4I] + k[1968]*y[IDX_C2H5I] + k[1978]*y[IDX_CH2II] +
        k[1984]*y[IDX_CH3II] + k[1988]*y[IDX_CH3I] + k[1990]*y[IDX_CH3OHI] +
        k[1993]*y[IDX_CH4II] + k[1995]*y[IDX_CH4I] + k[1998]*y[IDX_CH4I] +
        k[2011]*y[IDX_H2COI] + k[2015]*y[IDX_H2CSI] + k[2022]*y[IDX_H2SI] +
        k[2023]*y[IDX_H2SiOI] + k[2026]*y[IDX_H3II] + k[2053]*y[IDX_NH3I] +
        k[2087]*y[IDX_SiH3I] + k[2088]*y[IDX_SiH4I] + k[2090]*y[IDX_SiH4I] -
        k[2112]*y[IDX_H2I]*y[IDX_CII] - k[2113]*y[IDX_H2I]*y[IDX_CI] -
        k[2114]*y[IDX_H2I]*y[IDX_CH3II] - k[2115]*y[IDX_H2I]*y[IDX_CHI] -
        k[2116]*y[IDX_H2I]*y[IDX_HSII] - k[2117]*y[IDX_H2I]*y[IDX_SII] -
        k[2118]*y[IDX_H2I]*y[IDX_SiII] - k[2119]*y[IDX_H2I]*y[IDX_SiHII] -
        k[2120]*y[IDX_H2I]*y[IDX_SiH3II] + (H2formation) * y[IDX_HI] +
        (-H2dissociation) * y[IDX_H2I];
    ydot[IDX_EM] = 0.0 + k[0]*y[IDX_CHI]*y[IDX_OI] -
        k[8]*y[IDX_H2I]*y[IDX_EM] + k[8]*y[IDX_H2I]*y[IDX_EM] + k[356]*y[IDX_CI]
        + k[357]*y[IDX_COI] + k[358]*y[IDX_ClI] + k[359]*y[IDX_H2I] +
        k[360]*y[IDX_H2I] + k[362]*y[IDX_HI] + k[363]*y[IDX_HeI] +
        k[364]*y[IDX_NI] + k[365]*y[IDX_OI] + k[367]*y[IDX_C2H2I] +
        k[374]*y[IDX_C2HI] + k[379]*y[IDX_CI] + k[381]*y[IDX_CH2I] +
        k[385]*y[IDX_CH3I] + k[395]*y[IDX_CSI] + k[397]*y[IDX_ClI] +
        k[403]*y[IDX_H2SI] + k[406]*y[IDX_HI] + k[410]*y[IDX_HCOI] +
        k[412]*y[IDX_HCSI] + k[419]*y[IDX_HeI] + k[420]*y[IDX_MgI] +
        k[422]*y[IDX_NI] + k[424]*y[IDX_NH2I] + k[427]*y[IDX_NH3I] +
        k[430]*y[IDX_NHI] + k[432]*y[IDX_NOI] + k[435]*y[IDX_O2I] +
        k[438]*y[IDX_OI] + k[440]*y[IDX_OCSI] + k[444]*y[IDX_SI] +
        k[447]*y[IDX_SOI] + k[448]*y[IDX_SiI] - k[458]*y[IDX_C2II]*y[IDX_EM] -
        k[459]*y[IDX_C2HII]*y[IDX_EM] - k[460]*y[IDX_C2HII]*y[IDX_EM] -
        k[461]*y[IDX_C2H2II]*y[IDX_EM] - k[462]*y[IDX_C2H2II]*y[IDX_EM] -
        k[463]*y[IDX_C2H2II]*y[IDX_EM] - k[464]*y[IDX_C2H5OH2II]*y[IDX_EM] -
        k[465]*y[IDX_C2H5OH2II]*y[IDX_EM] - k[466]*y[IDX_C2H5OH2II]*y[IDX_EM] -
        k[467]*y[IDX_C2H5OH2II]*y[IDX_EM] - k[468]*y[IDX_C2NII]*y[IDX_EM] -
        k[469]*y[IDX_C2NII]*y[IDX_EM] - k[470]*y[IDX_C2N2II]*y[IDX_EM] -
        k[471]*y[IDX_C2N2II]*y[IDX_EM] - k[472]*y[IDX_C2NHII]*y[IDX_EM] -
        k[473]*y[IDX_C3II]*y[IDX_EM] - k[474]*y[IDX_C3H5II]*y[IDX_EM] -
        k[475]*y[IDX_C4NII]*y[IDX_EM] - k[476]*y[IDX_C4NII]*y[IDX_EM] -
        k[477]*y[IDX_CHII]*y[IDX_EM] - k[478]*y[IDX_CH2II]*y[IDX_EM] -
        k[479]*y[IDX_CH2II]*y[IDX_EM] - k[480]*y[IDX_CH2II]*y[IDX_EM] -
        k[481]*y[IDX_CH3II]*y[IDX_EM] - k[482]*y[IDX_CH3II]*y[IDX_EM] -
        k[483]*y[IDX_CH3II]*y[IDX_EM] - k[484]*y[IDX_CH3CNHII]*y[IDX_EM] -
        k[485]*y[IDX_CH3CNHII]*y[IDX_EM] - k[486]*y[IDX_CH3OH2II]*y[IDX_EM] -
        k[487]*y[IDX_CH3OH2II]*y[IDX_EM] - k[488]*y[IDX_CH3OH2II]*y[IDX_EM] -
        k[489]*y[IDX_CH3OH2II]*y[IDX_EM] - k[490]*y[IDX_CH3OH2II]*y[IDX_EM] -
        k[491]*y[IDX_CH4II]*y[IDX_EM] - k[492]*y[IDX_CH4II]*y[IDX_EM] -
        k[493]*y[IDX_CH5II]*y[IDX_EM] - k[494]*y[IDX_CH5II]*y[IDX_EM] -
        k[495]*y[IDX_CH5II]*y[IDX_EM] - k[496]*y[IDX_CH5II]*y[IDX_EM] -
        k[497]*y[IDX_CH5II]*y[IDX_EM] - k[498]*y[IDX_CNII]*y[IDX_EM] -
        k[499]*y[IDX_COII]*y[IDX_EM] - k[500]*y[IDX_CSII]*y[IDX_EM] -
        k[501]*y[IDX_H2II]*y[IDX_EM] - k[502]*y[IDX_H2COII]*y[IDX_EM] -
        k[503]*y[IDX_H2COII]*y[IDX_EM] - k[504]*y[IDX_H2COII]*y[IDX_EM] -
        k[505]*y[IDX_H2COII]*y[IDX_EM] - k[506]*y[IDX_H2CSII]*y[IDX_EM] -
        k[507]*y[IDX_H2CSII]*y[IDX_EM] - k[508]*y[IDX_H2ClII]*y[IDX_EM] -
        k[509]*y[IDX_H2ClII]*y[IDX_EM] - k[510]*y[IDX_H2NOII]*y[IDX_EM] -
        k[511]*y[IDX_H2NOII]*y[IDX_EM] - k[512]*y[IDX_H2OII]*y[IDX_EM] -
        k[513]*y[IDX_H2OII]*y[IDX_EM] - k[514]*y[IDX_H2OII]*y[IDX_EM] -
        k[515]*y[IDX_H2SII]*y[IDX_EM] - k[516]*y[IDX_H2SII]*y[IDX_EM] -
        k[517]*y[IDX_H2S2II]*y[IDX_EM] - k[518]*y[IDX_H2S2II]*y[IDX_EM] -
        k[519]*y[IDX_H3II]*y[IDX_EM] - k[520]*y[IDX_H3II]*y[IDX_EM] -
        k[521]*y[IDX_H3COII]*y[IDX_EM] - k[522]*y[IDX_H3COII]*y[IDX_EM] -
        k[523]*y[IDX_H3COII]*y[IDX_EM] - k[524]*y[IDX_H3COII]*y[IDX_EM] -
        k[525]*y[IDX_H3COII]*y[IDX_EM] - k[526]*y[IDX_H3CSII]*y[IDX_EM] -
        k[527]*y[IDX_H3CSII]*y[IDX_EM] - k[528]*y[IDX_H3OII]*y[IDX_EM] -
        k[529]*y[IDX_H3OII]*y[IDX_EM] - k[530]*y[IDX_H3OII]*y[IDX_EM] -
        k[531]*y[IDX_H3OII]*y[IDX_EM] - k[532]*y[IDX_H3SII]*y[IDX_EM] -
        k[533]*y[IDX_H3SII]*y[IDX_EM] - k[534]*y[IDX_H3SII]*y[IDX_EM] -
        k[535]*y[IDX_H3SII]*y[IDX_EM] - k[536]*y[IDX_H5C2O2II]*y[IDX_EM] -
        k[537]*y[IDX_H5C2O2II]*y[IDX_EM] - k[538]*y[IDX_HCNII]*y[IDX_EM] -
        k[539]*y[IDX_HCNHII]*y[IDX_EM] - k[540]*y[IDX_HCNHII]*y[IDX_EM] -
        k[541]*y[IDX_HCNHII]*y[IDX_EM] - k[542]*y[IDX_HCOII]*y[IDX_EM] -
        k[543]*y[IDX_HCO2II]*y[IDX_EM] - k[544]*y[IDX_HCO2II]*y[IDX_EM] -
        k[545]*y[IDX_HCO2II]*y[IDX_EM] - k[546]*y[IDX_HCSII]*y[IDX_EM] -
        k[547]*y[IDX_HCSII]*y[IDX_EM] - k[548]*y[IDX_HClII]*y[IDX_EM] -
        k[549]*y[IDX_HNOII]*y[IDX_EM] - k[550]*y[IDX_HNSII]*y[IDX_EM] -
        k[551]*y[IDX_HOCII]*y[IDX_EM] - k[552]*y[IDX_HOCSII]*y[IDX_EM] -
        k[553]*y[IDX_HOCSII]*y[IDX_EM] - k[554]*y[IDX_HSII]*y[IDX_EM] -
        k[555]*y[IDX_HS2II]*y[IDX_EM] - k[556]*y[IDX_HS2II]*y[IDX_EM] -
        k[557]*y[IDX_HSOII]*y[IDX_EM] - k[558]*y[IDX_HSO2II]*y[IDX_EM] -
        k[559]*y[IDX_HSO2II]*y[IDX_EM] - k[560]*y[IDX_HSO2II]*y[IDX_EM] -
        k[561]*y[IDX_HSiSII]*y[IDX_EM] - k[562]*y[IDX_HSiSII]*y[IDX_EM] -
        k[563]*y[IDX_HeHII]*y[IDX_EM] - k[564]*y[IDX_N2II]*y[IDX_EM] -
        k[565]*y[IDX_N2HII]*y[IDX_EM] - k[566]*y[IDX_N2HII]*y[IDX_EM] -
        k[567]*y[IDX_NHII]*y[IDX_EM] - k[568]*y[IDX_NH2II]*y[IDX_EM] -
        k[569]*y[IDX_NH2II]*y[IDX_EM] - k[570]*y[IDX_NH3II]*y[IDX_EM] -
        k[571]*y[IDX_NH3II]*y[IDX_EM] - k[572]*y[IDX_NH4II]*y[IDX_EM] -
        k[573]*y[IDX_NH4II]*y[IDX_EM] - k[574]*y[IDX_NH4II]*y[IDX_EM] -
        k[575]*y[IDX_NOII]*y[IDX_EM] - k[576]*y[IDX_NSII]*y[IDX_EM] -
        k[577]*y[IDX_O2II]*y[IDX_EM] - k[578]*y[IDX_O2HII]*y[IDX_EM] -
        k[579]*y[IDX_OCSII]*y[IDX_EM] - k[580]*y[IDX_OCSII]*y[IDX_EM] -
        k[581]*y[IDX_OCSII]*y[IDX_EM] - k[582]*y[IDX_OHII]*y[IDX_EM] -
        k[583]*y[IDX_S2II]*y[IDX_EM] - k[584]*y[IDX_SOII]*y[IDX_EM] -
        k[585]*y[IDX_SO2II]*y[IDX_EM] - k[586]*y[IDX_SO2II]*y[IDX_EM] -
        k[587]*y[IDX_SiCII]*y[IDX_EM] - k[588]*y[IDX_SiC2II]*y[IDX_EM] -
        k[589]*y[IDX_SiC2II]*y[IDX_EM] - k[590]*y[IDX_SiC3II]*y[IDX_EM] -
        k[591]*y[IDX_SiC3II]*y[IDX_EM] - k[592]*y[IDX_SiHII]*y[IDX_EM] -
        k[593]*y[IDX_SiH2II]*y[IDX_EM] - k[594]*y[IDX_SiH2II]*y[IDX_EM] -
        k[595]*y[IDX_SiH2II]*y[IDX_EM] - k[596]*y[IDX_SiH3II]*y[IDX_EM] -
        k[597]*y[IDX_SiH3II]*y[IDX_EM] - k[598]*y[IDX_SiH4II]*y[IDX_EM] -
        k[599]*y[IDX_SiH4II]*y[IDX_EM] - k[600]*y[IDX_SiH5II]*y[IDX_EM] -
        k[601]*y[IDX_SiH5II]*y[IDX_EM] - k[602]*y[IDX_SiOII]*y[IDX_EM] -
        k[603]*y[IDX_SiOHII]*y[IDX_EM] - k[604]*y[IDX_SiOHII]*y[IDX_EM] -
        k[605]*y[IDX_SiSII]*y[IDX_EM] + k[1961]*y[IDX_C2I] +
        k[1964]*y[IDX_C2H2I] + k[1971]*y[IDX_C2HI] + k[1976]*y[IDX_CI] +
        k[1981]*y[IDX_CH2I] + k[1987]*y[IDX_CH3I] + k[1991]*y[IDX_CH3OHI] +
        k[1997]*y[IDX_CH4I] + k[2000]*y[IDX_CHI] + k[2006]*y[IDX_CSI] +
        k[2008]*y[IDX_ClI] + k[2013]*y[IDX_H2COI] + k[2014]*y[IDX_H2COI] +
        k[2017]*y[IDX_H2OI] + k[2020]*y[IDX_H2SI] + k[2031]*y[IDX_HCOI] +
        k[2033]*y[IDX_HCSI] + k[2035]*y[IDX_HClI] + k[2041]*y[IDX_HS2I] +
        k[2044]*y[IDX_MgI] + k[2046]*y[IDX_NCCNI] + k[2049]*y[IDX_NH2I] +
        k[2052]*y[IDX_NH3I] + k[2055]*y[IDX_NHI] + k[2057]*y[IDX_NOI] +
        k[2061]*y[IDX_O2I] + k[2066]*y[IDX_OCSI] + k[2070]*y[IDX_OHI] +
        k[2071]*y[IDX_S2I] + k[2073]*y[IDX_SI] + k[2076]*y[IDX_SOI] +
        k[2077]*y[IDX_SiI] + k[2083]*y[IDX_SiH2I] + k[2086]*y[IDX_SiH3I] +
        k[2094]*y[IDX_SiOI] - k[2132]*y[IDX_CII]*y[IDX_EM] -
        k[2133]*y[IDX_CH3II]*y[IDX_EM] - k[2134]*y[IDX_ClII]*y[IDX_EM] -
        k[2135]*y[IDX_HII]*y[IDX_EM] - k[2136]*y[IDX_H2COII]*y[IDX_EM] -
        k[2137]*y[IDX_H2CSII]*y[IDX_EM] - k[2138]*y[IDX_H2SII]*y[IDX_EM] -
        k[2139]*y[IDX_HeII]*y[IDX_EM] - k[2140]*y[IDX_MgII]*y[IDX_EM] -
        k[2141]*y[IDX_NII]*y[IDX_EM] - k[2142]*y[IDX_OII]*y[IDX_EM] -
        k[2143]*y[IDX_SII]*y[IDX_EM] - k[2144]*y[IDX_SiII]*y[IDX_EM] -
        k[2302]*y[IDX_EM];
    ydot[IDX_HI] = 0.0 + k[2]*y[IDX_H2I]*y[IDX_CHI] +
        k[3]*y[IDX_H2I]*y[IDX_H2I] + k[3]*y[IDX_H2I]*y[IDX_H2I] +
        k[4]*y[IDX_H2I]*y[IDX_H2OI] + k[7]*y[IDX_H2I]*y[IDX_OHI] +
        k[8]*y[IDX_H2I]*y[IDX_EM] + k[8]*y[IDX_H2I]*y[IDX_EM] -
        k[9]*y[IDX_HI]*y[IDX_CHI] + k[9]*y[IDX_HI]*y[IDX_CHI] +
        k[9]*y[IDX_HI]*y[IDX_CHI] - k[10]*y[IDX_HI]*y[IDX_H2I] +
        k[10]*y[IDX_HI]*y[IDX_H2I] + k[10]*y[IDX_HI]*y[IDX_H2I] +
        k[10]*y[IDX_HI]*y[IDX_H2I] - k[11]*y[IDX_HI]*y[IDX_H2OI] +
        k[11]*y[IDX_HI]*y[IDX_H2OI] + k[11]*y[IDX_HI]*y[IDX_H2OI] -
        k[12]*y[IDX_HI]*y[IDX_O2I] + k[12]*y[IDX_HI]*y[IDX_O2I] -
        k[13]*y[IDX_HI]*y[IDX_OHI] + k[13]*y[IDX_HI]*y[IDX_OHI] +
        k[13]*y[IDX_HI]*y[IDX_OHI] - k[108]*y[IDX_ClII]*y[IDX_HI] +
        k[109]*y[IDX_ClI]*y[IDX_HII] + k[110]*y[IDX_HII]*y[IDX_C2I] +
        k[111]*y[IDX_HII]*y[IDX_C2H2I] + k[112]*y[IDX_HII]*y[IDX_C2HI] +
        k[113]*y[IDX_HII]*y[IDX_C2NI] + k[114]*y[IDX_HII]*y[IDX_CH2I] +
        k[115]*y[IDX_HII]*y[IDX_CH3I] + k[116]*y[IDX_HII]*y[IDX_CH4I] +
        k[117]*y[IDX_HII]*y[IDX_CHI] + k[118]*y[IDX_HII]*y[IDX_CSI] +
        k[119]*y[IDX_HII]*y[IDX_H2COI] + k[120]*y[IDX_HII]*y[IDX_H2CSI] +
        k[121]*y[IDX_HII]*y[IDX_H2OI] + k[122]*y[IDX_HII]*y[IDX_H2S2I] +
        k[123]*y[IDX_HII]*y[IDX_H2SI] + k[124]*y[IDX_HII]*y[IDX_HCNI] +
        k[125]*y[IDX_HII]*y[IDX_HCOI] + k[126]*y[IDX_HII]*y[IDX_HClI] +
        k[127]*y[IDX_HII]*y[IDX_HS2I] + k[128]*y[IDX_HII]*y[IDX_HSI] +
        k[129]*y[IDX_HII]*y[IDX_MgI] + k[130]*y[IDX_HII]*y[IDX_NH2I] +
        k[131]*y[IDX_HII]*y[IDX_NH3I] + k[132]*y[IDX_HII]*y[IDX_NHI] +
        k[133]*y[IDX_HII]*y[IDX_NOI] + k[134]*y[IDX_HII]*y[IDX_NSI] +
        k[135]*y[IDX_HII]*y[IDX_O2I] + k[136]*y[IDX_HII]*y[IDX_OI] +
        k[137]*y[IDX_HII]*y[IDX_OCSI] + k[138]*y[IDX_HII]*y[IDX_OHI] +
        k[139]*y[IDX_HII]*y[IDX_S2I] + k[140]*y[IDX_HII]*y[IDX_SI] +
        k[141]*y[IDX_HII]*y[IDX_SO2I] + k[142]*y[IDX_HII]*y[IDX_SOI] +
        k[143]*y[IDX_HII]*y[IDX_SiI] + k[144]*y[IDX_HII]*y[IDX_SiC2I] +
        k[145]*y[IDX_HII]*y[IDX_SiC3I] + k[146]*y[IDX_HII]*y[IDX_SiCI] +
        k[147]*y[IDX_HII]*y[IDX_SiH2I] + k[148]*y[IDX_HII]*y[IDX_SiH3I] +
        k[149]*y[IDX_HII]*y[IDX_SiH4I] + k[150]*y[IDX_HII]*y[IDX_SiHI] +
        k[151]*y[IDX_HII]*y[IDX_SiOI] + k[152]*y[IDX_HII]*y[IDX_SiSI] -
        k[191]*y[IDX_HI]*y[IDX_CNII] - k[192]*y[IDX_HI]*y[IDX_COII] -
        k[193]*y[IDX_HI]*y[IDX_H2II] - k[194]*y[IDX_HI]*y[IDX_HCNII] -
        k[195]*y[IDX_HI]*y[IDX_HeII] - k[196]*y[IDX_HI]*y[IDX_OII] +
        k[359]*y[IDX_H2I] + k[361]*y[IDX_H2I] + k[361]*y[IDX_H2I] -
        k[362]*y[IDX_HI] + k[368]*y[IDX_C2H2I] + k[369]*y[IDX_C2H3I] +
        k[373]*y[IDX_C2HI] + k[380]*y[IDX_CHII] + k[382]*y[IDX_CH2I] +
        k[384]*y[IDX_CH3I] + k[391]*y[IDX_CHI] + k[398]*y[IDX_H2CNI] +
        k[401]*y[IDX_H2OI] - k[406]*y[IDX_HI] + k[408]*y[IDX_HCNI] +
        k[409]*y[IDX_HCOI] + k[413]*y[IDX_HClI] + k[414]*y[IDX_HNCI] +
        k[416]*y[IDX_HNOI] + k[418]*y[IDX_HSI] + k[425]*y[IDX_NH2I] +
        k[426]*y[IDX_NH3I] + k[429]*y[IDX_NHI] + k[437]*y[IDX_O2HI] +
        k[442]*y[IDX_OHI] + k[452]*y[IDX_SiH2I] + k[453]*y[IDX_SiH3I] +
        k[455]*y[IDX_SiHI] + k[459]*y[IDX_C2HII]*y[IDX_EM] +
        k[461]*y[IDX_C2H2II]*y[IDX_EM] + k[461]*y[IDX_C2H2II]*y[IDX_EM] +
        k[462]*y[IDX_C2H2II]*y[IDX_EM] + k[464]*y[IDX_C2H5OH2II]*y[IDX_EM] +
        k[466]*y[IDX_C2H5OH2II]*y[IDX_EM] + k[467]*y[IDX_C2H5OH2II]*y[IDX_EM] +
        k[472]*y[IDX_C2NHII]*y[IDX_EM] + k[474]*y[IDX_C3H5II]*y[IDX_EM] +
        k[477]*y[IDX_CHII]*y[IDX_EM] + k[479]*y[IDX_CH2II]*y[IDX_EM] +
        k[479]*y[IDX_CH2II]*y[IDX_EM] + k[480]*y[IDX_CH2II]*y[IDX_EM] +
        k[481]*y[IDX_CH3II]*y[IDX_EM] + k[483]*y[IDX_CH3II]*y[IDX_EM] +
        k[483]*y[IDX_CH3II]*y[IDX_EM] + k[484]*y[IDX_CH3CNHII]*y[IDX_EM] +
        k[486]*y[IDX_CH3OH2II]*y[IDX_EM] + k[488]*y[IDX_CH3OH2II]*y[IDX_EM] +
        k[489]*y[IDX_CH3OH2II]*y[IDX_EM] + k[490]*y[IDX_CH3OH2II]*y[IDX_EM] +
        k[491]*y[IDX_CH4II]*y[IDX_EM] + k[491]*y[IDX_CH4II]*y[IDX_EM] +
        k[492]*y[IDX_CH4II]*y[IDX_EM] + k[493]*y[IDX_CH5II]*y[IDX_EM] +
        k[495]*y[IDX_CH5II]*y[IDX_EM] + k[495]*y[IDX_CH5II]*y[IDX_EM] +
        k[496]*y[IDX_CH5II]*y[IDX_EM] + k[501]*y[IDX_H2II]*y[IDX_EM] +
        k[501]*y[IDX_H2II]*y[IDX_EM] + k[504]*y[IDX_H2COII]*y[IDX_EM] +
        k[504]*y[IDX_H2COII]*y[IDX_EM] + k[505]*y[IDX_H2COII]*y[IDX_EM] +
        k[506]*y[IDX_H2CSII]*y[IDX_EM] + k[506]*y[IDX_H2CSII]*y[IDX_EM] +
        k[507]*y[IDX_H2CSII]*y[IDX_EM] + k[508]*y[IDX_H2ClII]*y[IDX_EM] +
        k[508]*y[IDX_H2ClII]*y[IDX_EM] + k[509]*y[IDX_H2ClII]*y[IDX_EM] +
        k[510]*y[IDX_H2NOII]*y[IDX_EM] + k[513]*y[IDX_H2OII]*y[IDX_EM] +
        k[513]*y[IDX_H2OII]*y[IDX_EM] + k[514]*y[IDX_H2OII]*y[IDX_EM] +
        k[515]*y[IDX_H2SII]*y[IDX_EM] + k[516]*y[IDX_H2SII]*y[IDX_EM] +
        k[516]*y[IDX_H2SII]*y[IDX_EM] + k[517]*y[IDX_H2S2II]*y[IDX_EM] +
        k[519]*y[IDX_H3II]*y[IDX_EM] + k[520]*y[IDX_H3II]*y[IDX_EM] +
        k[520]*y[IDX_H3II]*y[IDX_EM] + k[520]*y[IDX_H3II]*y[IDX_EM] +
        k[523]*y[IDX_H3COII]*y[IDX_EM] + k[524]*y[IDX_H3COII]*y[IDX_EM] +
        k[525]*y[IDX_H3COII]*y[IDX_EM] + k[525]*y[IDX_H3COII]*y[IDX_EM] +
        k[526]*y[IDX_H3CSII]*y[IDX_EM] + k[527]*y[IDX_H3CSII]*y[IDX_EM] +
        k[528]*y[IDX_H3OII]*y[IDX_EM] + k[529]*y[IDX_H3OII]*y[IDX_EM] +
        k[531]*y[IDX_H3OII]*y[IDX_EM] + k[531]*y[IDX_H3OII]*y[IDX_EM] +
        k[532]*y[IDX_H3SII]*y[IDX_EM] + k[534]*y[IDX_H3SII]*y[IDX_EM] +
        k[534]*y[IDX_H3SII]*y[IDX_EM] + k[535]*y[IDX_H3SII]*y[IDX_EM] +
        k[537]*y[IDX_H5C2O2II]*y[IDX_EM] + k[538]*y[IDX_HCNII]*y[IDX_EM] +
        k[539]*y[IDX_HCNHII]*y[IDX_EM] + k[539]*y[IDX_HCNHII]*y[IDX_EM] +
        k[540]*y[IDX_HCNHII]*y[IDX_EM] + k[541]*y[IDX_HCNHII]*y[IDX_EM] +
        k[542]*y[IDX_HCOII]*y[IDX_EM] + k[543]*y[IDX_HCO2II]*y[IDX_EM] +
        k[544]*y[IDX_HCO2II]*y[IDX_EM] + k[547]*y[IDX_HCSII]*y[IDX_EM] +
        k[548]*y[IDX_HClII]*y[IDX_EM] + k[549]*y[IDX_HNOII]*y[IDX_EM] +
        k[550]*y[IDX_HNSII]*y[IDX_EM] + k[551]*y[IDX_HOCII]*y[IDX_EM] +
        k[553]*y[IDX_HOCSII]*y[IDX_EM] + k[554]*y[IDX_HSII]*y[IDX_EM] +
        k[556]*y[IDX_HS2II]*y[IDX_EM] + k[557]*y[IDX_HSOII]*y[IDX_EM] +
        k[558]*y[IDX_HSO2II]*y[IDX_EM] + k[559]*y[IDX_HSO2II]*y[IDX_EM] +
        k[562]*y[IDX_HSiSII]*y[IDX_EM] + k[563]*y[IDX_HeHII]*y[IDX_EM] +
        k[565]*y[IDX_N2HII]*y[IDX_EM] + k[567]*y[IDX_NHII]*y[IDX_EM] +
        k[568]*y[IDX_NH2II]*y[IDX_EM] + k[568]*y[IDX_NH2II]*y[IDX_EM] +
        k[569]*y[IDX_NH2II]*y[IDX_EM] + k[570]*y[IDX_NH3II]*y[IDX_EM] +
        k[571]*y[IDX_NH3II]*y[IDX_EM] + k[571]*y[IDX_NH3II]*y[IDX_EM] +
        k[573]*y[IDX_NH4II]*y[IDX_EM] + k[573]*y[IDX_NH4II]*y[IDX_EM] +
        k[574]*y[IDX_NH4II]*y[IDX_EM] + k[578]*y[IDX_O2HII]*y[IDX_EM] +
        k[582]*y[IDX_OHII]*y[IDX_EM] + k[592]*y[IDX_SiHII]*y[IDX_EM] +
        k[594]*y[IDX_SiH2II]*y[IDX_EM] + k[594]*y[IDX_SiH2II]*y[IDX_EM] +
        k[595]*y[IDX_SiH2II]*y[IDX_EM] + k[596]*y[IDX_SiH3II]*y[IDX_EM] +
        k[599]*y[IDX_SiH4II]*y[IDX_EM] + k[601]*y[IDX_SiH5II]*y[IDX_EM] +
        k[604]*y[IDX_SiOHII]*y[IDX_EM] + k[607]*y[IDX_CII]*y[IDX_C2HI] +
        k[608]*y[IDX_CII]*y[IDX_CH2I] + k[610]*y[IDX_CII]*y[IDX_CH3I] +
        k[615]*y[IDX_CII]*y[IDX_CHI] + k[620]*y[IDX_CII]*y[IDX_H2OI] +
        k[621]*y[IDX_CII]*y[IDX_H2OI] + k[622]*y[IDX_CII]*y[IDX_H2SI] +
        k[625]*y[IDX_CII]*y[IDX_HC3NI] + k[627]*y[IDX_CII]*y[IDX_HNCI] +
        k[628]*y[IDX_CII]*y[IDX_HSI] + k[629]*y[IDX_CII]*y[IDX_NH2I] +
        k[631]*y[IDX_CII]*y[IDX_NHI] + k[637]*y[IDX_CII]*y[IDX_OHI] +
        k[644]*y[IDX_CII]*y[IDX_SiHI] + k[684]*y[IDX_C2HI]*y[IDX_SiII] +
        k[685]*y[IDX_CI]*y[IDX_C2HII] + k[686]*y[IDX_CI]*y[IDX_CHII] +
        k[687]*y[IDX_CI]*y[IDX_CH2II] + k[691]*y[IDX_CI]*y[IDX_H2SII] +
        k[697]*y[IDX_CI]*y[IDX_HSII] + k[703]*y[IDX_CI]*y[IDX_SiHII] +
        k[705]*y[IDX_CHII]*y[IDX_C2I] + k[711]*y[IDX_CHII]*y[IDX_CH4I] +
        k[713]*y[IDX_CHII]*y[IDX_CNI] + k[718]*y[IDX_CHII]*y[IDX_H2OI] +
        k[724]*y[IDX_CHII]*y[IDX_HCNI] + k[728]*y[IDX_CHII]*y[IDX_NI] +
        k[735]*y[IDX_CHII]*y[IDX_OI] + k[739]*y[IDX_CHII]*y[IDX_SI] +
        k[743]*y[IDX_CH2II]*y[IDX_H2OI] + k[744]*y[IDX_CH2II]*y[IDX_H2SI] +
        k[746]*y[IDX_CH2II]*y[IDX_H2SI] + k[750]*y[IDX_CH2II]*y[IDX_OI] +
        k[753]*y[IDX_CH2II]*y[IDX_SI] + k[772]*y[IDX_CH2I]*y[IDX_SII] +
        k[783]*y[IDX_CH3II]*y[IDX_OI] + k[790]*y[IDX_CH3I]*y[IDX_SII] +
        k[806]*y[IDX_CH4I]*y[IDX_C2H2II] + k[816]*y[IDX_CH4I]*y[IDX_N2II] +
        k[821]*y[IDX_CH4I]*y[IDX_SII] + k[822]*y[IDX_CH4I]*y[IDX_SII] +
        k[834]*y[IDX_CH5II]*y[IDX_MgI] + k[837]*y[IDX_CHI]*y[IDX_C2II] +
        k[852]*y[IDX_CHI]*y[IDX_NII] + k[857]*y[IDX_CHI]*y[IDX_OII] +
        k[861]*y[IDX_CHI]*y[IDX_SII] + k[862]*y[IDX_CHI]*y[IDX_SiII] +
        k[866]*y[IDX_CNII]*y[IDX_HCNI] + k[883]*y[IDX_HII]*y[IDX_C2H4I] +
        k[892]*y[IDX_HII]*y[IDX_H2COI] + k[895]*y[IDX_HII]*y[IDX_H2SI] +
        k[909]*y[IDX_H2II]*y[IDX_C2I] + k[911]*y[IDX_H2II]*y[IDX_C2HI] +
        k[912]*y[IDX_H2II]*y[IDX_CI] + k[913]*y[IDX_H2II]*y[IDX_CH2I] +
        k[914]*y[IDX_H2II]*y[IDX_CH4I] + k[915]*y[IDX_H2II]*y[IDX_CH4I] +
        k[916]*y[IDX_H2II]*y[IDX_CHI] + k[917]*y[IDX_H2II]*y[IDX_CNI] +
        k[918]*y[IDX_H2II]*y[IDX_CO2I] + k[919]*y[IDX_H2II]*y[IDX_COI] +
        k[920]*y[IDX_H2II]*y[IDX_H2I] + k[921]*y[IDX_H2II]*y[IDX_H2COI] +
        k[922]*y[IDX_H2II]*y[IDX_H2OI] + k[923]*y[IDX_H2II]*y[IDX_H2SI] +
        k[926]*y[IDX_H2II]*y[IDX_HeI] + k[927]*y[IDX_H2II]*y[IDX_N2I] +
        k[928]*y[IDX_H2II]*y[IDX_NI] + k[929]*y[IDX_H2II]*y[IDX_NHI] +
        k[930]*y[IDX_H2II]*y[IDX_NOI] + k[931]*y[IDX_H2II]*y[IDX_O2I] +
        k[932]*y[IDX_H2II]*y[IDX_OI] + k[933]*y[IDX_H2II]*y[IDX_OHI] +
        k[934]*y[IDX_H2I]*y[IDX_CII] + k[935]*y[IDX_H2I]*y[IDX_C2II] +
        k[936]*y[IDX_H2I]*y[IDX_C2HII] + k[937]*y[IDX_H2I]*y[IDX_CHII] +
        k[938]*y[IDX_H2I]*y[IDX_CH2II] + k[939]*y[IDX_H2I]*y[IDX_CH4II] +
        k[940]*y[IDX_H2I]*y[IDX_CNII] + k[941]*y[IDX_H2I]*y[IDX_COII] +
        k[942]*y[IDX_H2I]*y[IDX_COII] + k[943]*y[IDX_H2I]*y[IDX_CSII] +
        k[944]*y[IDX_H2I]*y[IDX_ClII] + k[945]*y[IDX_H2I]*y[IDX_H2OII] +
        k[946]*y[IDX_H2I]*y[IDX_H2SII] + k[947]*y[IDX_H2I]*y[IDX_HCNII] +
        k[948]*y[IDX_H2I]*y[IDX_HClII] + k[949]*y[IDX_H2I]*y[IDX_HSII] +
        k[950]*y[IDX_H2I]*y[IDX_HeII] + k[952]*y[IDX_H2I]*y[IDX_NII] +
        k[953]*y[IDX_H2I]*y[IDX_N2II] + k[955]*y[IDX_H2I]*y[IDX_NHII] +
        k[956]*y[IDX_H2I]*y[IDX_NH2II] + k[957]*y[IDX_H2I]*y[IDX_NH3II] +
        k[958]*y[IDX_H2I]*y[IDX_OII] + k[960]*y[IDX_H2I]*y[IDX_OHII] +
        k[961]*y[IDX_H2I]*y[IDX_SII] + k[962]*y[IDX_H2I]*y[IDX_SO2II] +
        k[963]*y[IDX_H2I]*y[IDX_SiH4II] + k[964]*y[IDX_H2I]*y[IDX_SiOII] +
        k[972]*y[IDX_H2COI]*y[IDX_O2II] + k[988]*y[IDX_H2OII]*y[IDX_SI] +
        k[1013]*y[IDX_H2OI]*y[IDX_SiII] + k[1054]*y[IDX_H3II]*y[IDX_MgI] +
        k[1063]*y[IDX_H3II]*y[IDX_OI] - k[1097]*y[IDX_HI]*y[IDX_C2N2II] -
        k[1098]*y[IDX_HI]*y[IDX_CHII] - k[1099]*y[IDX_HI]*y[IDX_CH2II] -
        k[1100]*y[IDX_HI]*y[IDX_CH3II] - k[1101]*y[IDX_HI]*y[IDX_CH4II] -
        k[1102]*y[IDX_HI]*y[IDX_CH5II] - k[1103]*y[IDX_HI]*y[IDX_H2SII] -
        k[1104]*y[IDX_HI]*y[IDX_H3SII] - k[1105]*y[IDX_HI]*y[IDX_HSII] -
        k[1106]*y[IDX_HI]*y[IDX_HeHII] - k[1107]*y[IDX_HI]*y[IDX_SO2II] -
        k[1108]*y[IDX_HI]*y[IDX_SiHII] - k[1109]*y[IDX_HI]*y[IDX_SiSII] +
        k[1119]*y[IDX_HCNI]*y[IDX_C3II] + k[1179]*y[IDX_HeII]*y[IDX_C2H2I] +
        k[1182]*y[IDX_HeII]*y[IDX_C2H3I] + k[1183]*y[IDX_HeII]*y[IDX_C2H4I] +
        k[1186]*y[IDX_HeII]*y[IDX_C2HI] + k[1193]*y[IDX_HeII]*y[IDX_CH2I] +
        k[1202]*y[IDX_HeII]*y[IDX_CH4I] + k[1204]*y[IDX_HeII]*y[IDX_CH4I] +
        k[1206]*y[IDX_HeII]*y[IDX_CHI] + k[1217]*y[IDX_HeII]*y[IDX_H2COI] +
        k[1222]*y[IDX_HeII]*y[IDX_H2OI] + k[1225]*y[IDX_HeII]*y[IDX_H2S2I] +
        k[1226]*y[IDX_HeII]*y[IDX_H2SI] + k[1228]*y[IDX_HeII]*y[IDX_H2SiOI] +
        k[1231]*y[IDX_HeII]*y[IDX_HCNI] + k[1233]*y[IDX_HeII]*y[IDX_HCNI] +
        k[1235]*y[IDX_HeII]*y[IDX_HCOI] + k[1239]*y[IDX_HeII]*y[IDX_HCSI] +
        k[1241]*y[IDX_HeII]*y[IDX_HClI] + k[1242]*y[IDX_HeII]*y[IDX_HNCI] +
        k[1243]*y[IDX_HeII]*y[IDX_HNCI] + k[1245]*y[IDX_HeII]*y[IDX_HNOI] +
        k[1248]*y[IDX_HeII]*y[IDX_HS2I] + k[1249]*y[IDX_HeII]*y[IDX_HSI] +
        k[1253]*y[IDX_HeII]*y[IDX_NH2I] + k[1255]*y[IDX_HeII]*y[IDX_NH3I] +
        k[1256]*y[IDX_HeII]*y[IDX_NHI] + k[1268]*y[IDX_HeII]*y[IDX_OHI] +
        k[1279]*y[IDX_HeII]*y[IDX_SiH2I] + k[1281]*y[IDX_HeII]*y[IDX_SiH3I] +
        k[1283]*y[IDX_HeII]*y[IDX_SiH4I] + k[1284]*y[IDX_HeII]*y[IDX_SiHI] +
        k[1289]*y[IDX_NII]*y[IDX_CH3OHI] + k[1291]*y[IDX_NII]*y[IDX_CH3OHI] +
        k[1292]*y[IDX_NII]*y[IDX_CH3OHI] + k[1293]*y[IDX_NII]*y[IDX_CH4I] +
        k[1294]*y[IDX_NII]*y[IDX_CH4I] + k[1295]*y[IDX_NII]*y[IDX_CH4I] +
        k[1295]*y[IDX_NII]*y[IDX_CH4I] + k[1302]*y[IDX_NII]*y[IDX_H2SI] +
        k[1308]*y[IDX_NII]*y[IDX_NHI] + k[1314]*y[IDX_N2II]*y[IDX_H2COI] +
        k[1315]*y[IDX_N2II]*y[IDX_H2SI] + k[1326]*y[IDX_NI]*y[IDX_C2HII] +
        k[1329]*y[IDX_NI]*y[IDX_C2H2II] + k[1331]*y[IDX_NI]*y[IDX_CH2II] +
        k[1333]*y[IDX_NI]*y[IDX_H2OII] + k[1336]*y[IDX_NI]*y[IDX_HSII] +
        k[1337]*y[IDX_NI]*y[IDX_NHII] + k[1338]*y[IDX_NI]*y[IDX_NH2II] +
        k[1340]*y[IDX_NI]*y[IDX_OHII] + k[1346]*y[IDX_NHII]*y[IDX_C2I] +
        k[1373]*y[IDX_NHII]*y[IDX_SI] + k[1392]*y[IDX_NH2II]*y[IDX_SI] +
        k[1394]*y[IDX_NH2I]*y[IDX_C2II] + k[1445]*y[IDX_NHI]*y[IDX_C2II] +
        k[1457]*y[IDX_NHI]*y[IDX_OII] + k[1461]*y[IDX_NHI]*y[IDX_SII] +
        k[1480]*y[IDX_OII]*y[IDX_OHI] + k[1482]*y[IDX_O2II]*y[IDX_C2H2I] +
        k[1483]*y[IDX_O2II]*y[IDX_CH3OHI] + k[1501]*y[IDX_OI]*y[IDX_HCSII] +
        k[1504]*y[IDX_OI]*y[IDX_HSII] + k[1507]*y[IDX_OI]*y[IDX_NH2II] +
        k[1511]*y[IDX_OI]*y[IDX_OHII] + k[1513]*y[IDX_OI]*y[IDX_SiHII] +
        k[1514]*y[IDX_OI]*y[IDX_SiH2II] + k[1534]*y[IDX_OHII]*y[IDX_SI] +
        k[1543]*y[IDX_OHI]*y[IDX_HCOII] + k[1548]*y[IDX_OHI]*y[IDX_SII] +
        k[1549]*y[IDX_OHI]*y[IDX_SiII] + k[1550]*y[IDX_SII]*y[IDX_H2SI] +
        k[1553]*y[IDX_SI]*y[IDX_H3SII] + k[1568]*y[IDX_SiH2II]*y[IDX_SI] +
        k[1569]*y[IDX_SiHI]*y[IDX_SII] + k[1570]*y[IDX_C2I]*y[IDX_C2H2I] +
        k[1571]*y[IDX_C2I]*y[IDX_HCNI] + k[1574]*y[IDX_C2H2I]*y[IDX_NOI] +
        k[1578]*y[IDX_C2HI]*y[IDX_HCNI] + k[1579]*y[IDX_C2HI]*y[IDX_HNCI] +
        k[1582]*y[IDX_CI]*y[IDX_C2H3I] + k[1583]*y[IDX_CI]*y[IDX_C2H5I] +
        k[1585]*y[IDX_CI]*y[IDX_C3H2I] + k[1586]*y[IDX_CI]*y[IDX_CH2I] +
        k[1588]*y[IDX_CI]*y[IDX_CH3I] + k[1589]*y[IDX_CI]*y[IDX_CHI] +
        k[1595]*y[IDX_CI]*y[IDX_HSI] + k[1599]*y[IDX_CI]*y[IDX_NH2I] +
        k[1600]*y[IDX_CI]*y[IDX_NH2I] + k[1602]*y[IDX_CI]*y[IDX_NHI] +
        k[1611]*y[IDX_CI]*y[IDX_OHI] + k[1617]*y[IDX_CI]*y[IDX_SiHI] +
        k[1619]*y[IDX_CH2I]*y[IDX_CH2I] + k[1619]*y[IDX_CH2I]*y[IDX_CH2I] +
        k[1620]*y[IDX_CH2I]*y[IDX_CH2I] + k[1631]*y[IDX_CH2I]*y[IDX_NOI] +
        k[1633]*y[IDX_CH2I]*y[IDX_O2I] + k[1633]*y[IDX_CH2I]*y[IDX_O2I] +
        k[1638]*y[IDX_CH2I]*y[IDX_OI] + k[1638]*y[IDX_CH2I]*y[IDX_OI] +
        k[1639]*y[IDX_CH2I]*y[IDX_OI] + k[1641]*y[IDX_CH2I]*y[IDX_OHI] +
        k[1645]*y[IDX_CH2I]*y[IDX_SI] + k[1648]*y[IDX_CH3I]*y[IDX_CH3I] +
        k[1664]*y[IDX_CH3I]*y[IDX_OI] + k[1665]*y[IDX_CH3I]*y[IDX_OI] +
        k[1669]*y[IDX_CH3I]*y[IDX_SI] + k[1674]*y[IDX_CHI]*y[IDX_C2H2I] +
        k[1675]*y[IDX_CHI]*y[IDX_C2H4I] + k[1676]*y[IDX_CHI]*y[IDX_CH4I] +
        k[1682]*y[IDX_CHI]*y[IDX_NI] + k[1686]*y[IDX_CHI]*y[IDX_NOI] +
        k[1687]*y[IDX_CHI]*y[IDX_O2I] + k[1688]*y[IDX_CHI]*y[IDX_O2I] +
        k[1693]*y[IDX_CHI]*y[IDX_OI] + k[1695]*y[IDX_CHI]*y[IDX_OCSI] +
        k[1696]*y[IDX_CHI]*y[IDX_OHI] + k[1697]*y[IDX_CHI]*y[IDX_SI] +
        k[1700]*y[IDX_CHI]*y[IDX_SOI] + k[1701]*y[IDX_CNI]*y[IDX_C2H2I] +
        k[1705]*y[IDX_CNI]*y[IDX_HCNI] + k[1707]*y[IDX_CNI]*y[IDX_HNCI] +
        k[1720]*y[IDX_ClI]*y[IDX_H2I] + k[1721]*y[IDX_H2I]*y[IDX_C2HI] +
        k[1722]*y[IDX_H2I]*y[IDX_CI] + k[1723]*y[IDX_H2I]*y[IDX_CH2I] +
        k[1724]*y[IDX_H2I]*y[IDX_CH3I] + k[1725]*y[IDX_H2I]*y[IDX_CHI] +
        k[1726]*y[IDX_H2I]*y[IDX_CNI] + k[1727]*y[IDX_H2I]*y[IDX_HSI] +
        k[1728]*y[IDX_H2I]*y[IDX_NI] + k[1729]*y[IDX_H2I]*y[IDX_NH2I] +
        k[1730]*y[IDX_H2I]*y[IDX_NHI] + k[1731]*y[IDX_H2I]*y[IDX_O2I] +
        k[1733]*y[IDX_H2I]*y[IDX_OI] + k[1734]*y[IDX_H2I]*y[IDX_OHI] +
        k[1735]*y[IDX_H2I]*y[IDX_SI] - k[1736]*y[IDX_HI]*y[IDX_C2I] -
        k[1737]*y[IDX_HI]*y[IDX_C2H2I] - k[1738]*y[IDX_HI]*y[IDX_C2H3I] -
        k[1739]*y[IDX_HI]*y[IDX_CH2I] - k[1740]*y[IDX_HI]*y[IDX_CH2COI] -
        k[1741]*y[IDX_HI]*y[IDX_CH3I] - k[1742]*y[IDX_HI]*y[IDX_CH4I] -
        k[1743]*y[IDX_HI]*y[IDX_CHI] - k[1744]*y[IDX_HI]*y[IDX_CO2I] -
        k[1745]*y[IDX_HI]*y[IDX_COI] - k[1746]*y[IDX_HI]*y[IDX_H2CNI] -
        k[1747]*y[IDX_HI]*y[IDX_H2COI] - k[1748]*y[IDX_HI]*y[IDX_H2OI] -
        k[1749]*y[IDX_HI]*y[IDX_H2SI] - k[1750]*y[IDX_HI]*y[IDX_HCNI] -
        k[1751]*y[IDX_HI]*y[IDX_HCOI] - k[1752]*y[IDX_HI]*y[IDX_HCOI] -
        k[1753]*y[IDX_HI]*y[IDX_HCSI] - k[1754]*y[IDX_HI]*y[IDX_HNCI] +
        k[1754]*y[IDX_HI]*y[IDX_HNCI] - k[1755]*y[IDX_HI]*y[IDX_HNOI] -
        k[1756]*y[IDX_HI]*y[IDX_HNOI] - k[1757]*y[IDX_HI]*y[IDX_HNOI] -
        k[1758]*y[IDX_HI]*y[IDX_HSI] - k[1759]*y[IDX_HI]*y[IDX_NCCNI] -
        k[1760]*y[IDX_HI]*y[IDX_NH2I] - k[1761]*y[IDX_HI]*y[IDX_NH3I] -
        k[1762]*y[IDX_HI]*y[IDX_NHI] - k[1763]*y[IDX_HI]*y[IDX_NO2I] -
        k[1764]*y[IDX_HI]*y[IDX_NOI] - k[1765]*y[IDX_HI]*y[IDX_NOI] -
        k[1766]*y[IDX_HI]*y[IDX_NSI] - k[1767]*y[IDX_HI]*y[IDX_NSI] -
        k[1768]*y[IDX_HI]*y[IDX_O2I] - k[1769]*y[IDX_HI]*y[IDX_O2HI] -
        k[1770]*y[IDX_HI]*y[IDX_O2HI] - k[1771]*y[IDX_HI]*y[IDX_O2HI] -
        k[1772]*y[IDX_HI]*y[IDX_OCNI] - k[1773]*y[IDX_HI]*y[IDX_OCNI] -
        k[1774]*y[IDX_HI]*y[IDX_OCNI] - k[1775]*y[IDX_HI]*y[IDX_OCSI] -
        k[1776]*y[IDX_HI]*y[IDX_OHI] - k[1777]*y[IDX_HI]*y[IDX_S2I] -
        k[1778]*y[IDX_HI]*y[IDX_SOI] - k[1779]*y[IDX_HI]*y[IDX_SOI] -
        k[1787]*y[IDX_HClI]*y[IDX_HI] + k[1795]*y[IDX_NI]*y[IDX_C2HI] +
        k[1797]*y[IDX_NI]*y[IDX_C3H2I] + k[1799]*y[IDX_NI]*y[IDX_C4HI] +
        k[1801]*y[IDX_NI]*y[IDX_CH2I] + k[1802]*y[IDX_NI]*y[IDX_CH2I] +
        k[1804]*y[IDX_NI]*y[IDX_CH3I] + k[1806]*y[IDX_NI]*y[IDX_CH3I] +
        k[1806]*y[IDX_NI]*y[IDX_CH3I] + k[1813]*y[IDX_NI]*y[IDX_HCOI] +
        k[1816]*y[IDX_NI]*y[IDX_HSI] + k[1819]*y[IDX_NI]*y[IDX_NHI] +
        k[1827]*y[IDX_NI]*y[IDX_OHI] + k[1835]*y[IDX_NH2I]*y[IDX_NOI] +
        k[1844]*y[IDX_NHI]*y[IDX_NHI] + k[1844]*y[IDX_NHI]*y[IDX_NHI] +
        k[1847]*y[IDX_NHI]*y[IDX_NOI] + k[1851]*y[IDX_NHI]*y[IDX_OI] +
        k[1854]*y[IDX_NHI]*y[IDX_OHI] + k[1857]*y[IDX_NHI]*y[IDX_SI] +
        k[1869]*y[IDX_OI]*y[IDX_C2H3I] + k[1891]*y[IDX_OI]*y[IDX_HCNI] +
        k[1892]*y[IDX_OI]*y[IDX_HCOI] + k[1895]*y[IDX_OI]*y[IDX_HCSI] +
        k[1896]*y[IDX_OI]*y[IDX_HNOI] + k[1900]*y[IDX_OI]*y[IDX_HSI] +
        k[1902]*y[IDX_OI]*y[IDX_NH2I] + k[1914]*y[IDX_OI]*y[IDX_OHI] +
        k[1923]*y[IDX_OI]*y[IDX_SiH2I] + k[1923]*y[IDX_OI]*y[IDX_SiH2I] +
        k[1924]*y[IDX_OI]*y[IDX_SiH3I] + k[1926]*y[IDX_OI]*y[IDX_SiHI] +
        k[1928]*y[IDX_OHI]*y[IDX_C2H2I] + k[1933]*y[IDX_OHI]*y[IDX_CNI] +
        k[1934]*y[IDX_OHI]*y[IDX_COI] + k[1936]*y[IDX_OHI]*y[IDX_CSI] +
        k[1945]*y[IDX_OHI]*y[IDX_NOI] + k[1948]*y[IDX_OHI]*y[IDX_SI] +
        k[1949]*y[IDX_OHI]*y[IDX_SOI] + k[1950]*y[IDX_OHI]*y[IDX_SiI] +
        k[1951]*y[IDX_SI]*y[IDX_HCOI] + k[1953]*y[IDX_SI]*y[IDX_HSI] +
        k[1963]*y[IDX_C2HII] + k[1965]*y[IDX_C2H2I] + k[1966]*y[IDX_C2H3I] +
        k[1970]*y[IDX_C2HI] + k[1979]*y[IDX_CH2II] + k[1982]*y[IDX_CH2I] +
        k[1985]*y[IDX_CH3II] + k[1986]*y[IDX_CH3I] + k[1991]*y[IDX_CH3OHI] +
        k[1994]*y[IDX_CH4II] + k[1996]*y[IDX_CH4I] + k[1998]*y[IDX_CH4I] +
        k[1999]*y[IDX_CHI] + k[2009]*y[IDX_H2II] + k[2010]*y[IDX_H2CNI] +
        k[2012]*y[IDX_H2COI] + k[2012]*y[IDX_H2COI] + k[2014]*y[IDX_H2COI] +
        k[2016]*y[IDX_H2OII] + k[2018]*y[IDX_H2OI] + k[2021]*y[IDX_H2SI] +
        k[2024]*y[IDX_H2SiOI] + k[2024]*y[IDX_H2SiOI] + k[2025]*y[IDX_H3II] +
        k[2028]*y[IDX_HCNI] + k[2029]*y[IDX_HCOII] + k[2030]*y[IDX_HCOI] +
        k[2034]*y[IDX_HClI] + k[2036]*y[IDX_HNCI] + k[2038]*y[IDX_HNOI] +
        k[2039]*y[IDX_HSII] + k[2043]*y[IDX_HSI] + k[2050]*y[IDX_NH2I] +
        k[2051]*y[IDX_NH3I] + k[2054]*y[IDX_NHI] + k[2063]*y[IDX_O2HI] +
        k[2068]*y[IDX_OHII] + k[2069]*y[IDX_OHI] + k[2082]*y[IDX_SiHII] +
        k[2084]*y[IDX_SiH2I] + k[2085]*y[IDX_SiH3I] + k[2089]*y[IDX_SiH4I] +
        k[2090]*y[IDX_SiH4I] + k[2091]*y[IDX_SiHI] -
        k[2110]*y[IDX_HII]*y[IDX_HI] - k[2122]*y[IDX_HI]*y[IDX_CII] -
        k[2123]*y[IDX_HI]*y[IDX_CI] - k[2124]*y[IDX_HI]*y[IDX_OI] -
        k[2125]*y[IDX_HI]*y[IDX_OHI] - k[2126]*y[IDX_HI]*y[IDX_SiII] +
        k[2135]*y[IDX_HII]*y[IDX_EM] + k[2148]*y[IDX_C3H5II] +
        k[2151]*y[IDX_C2H5OH2II] + k[2152]*y[IDX_CH3OH2II] +
        k[2167]*y[IDX_H5C2O2II] + k[2191]*y[IDX_HSiSII] + k[2193]*y[IDX_SiH5II]
        + k[2198]*y[IDX_H2ClII] + k[2246]*y[IDX_H3OII] + k[2247]*y[IDX_HCO2II] +
        k[2248]*y[IDX_CH5II] + k[2273]*y[IDX_H2NOII] + k[2278]*y[IDX_H3SII] +
        k[2282]*y[IDX_H3CSII] + k[2283]*y[IDX_HSOII] + k[2285]*y[IDX_HOCSII] +
        k[2288]*y[IDX_NH4II] + k[2289]*y[IDX_HCNHII] + k[2290]*y[IDX_N2HII] +
        k[2291]*y[IDX_HNSII] + k[2295]*y[IDX_HSO2II] + (-2.0 * H2formation) *
        y[IDX_HI] + (2.0 * H2dissociation) * y[IDX_H2I];
    
    
#if ((NHEATPROCS || NCOOLPROCS) && NAUNET_DEBUG)
    printf("Total heating/cooling rate: %13.7e\n", ydot[IDX_TGAS]);
#endif

    // clang-format on

    /* */

    return NAUNET_SUCCESS;
}