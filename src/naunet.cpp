#include <cvode/cvode.h>  // prototypes for CVODE fcts., consts.
/* */
#include <nvector/nvector_serial.h>      // access to serial N_Vector
#include <sunlinsol/sunlinsol_dense.h>   // access to dense SUNLinearSolver
#include <sunlinsol/sunlinsol_klu.h>     // access to KLU sparse direct solver
#include <sunmatrix/sunmatrix_sparse.h>  // access to sparse SUNMatrix
/* */
/*  */
#include "naunet.h"
/*  */
#include "naunet_ode.h"
#include "naunet_physics.h"
/* */
#include "naunet_constants.h"
#include "naunet_utilities.h"
/* */

#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

Naunet::Naunet(){};

Naunet::~Naunet(){};

// Adaptedfrom the cvDiurnals_ky.c example from the CVODE package.
// Check function return value...
//   opt == 0 means SUNDIALS function allocates memory so check if
//            returned NULL pointer
//   opt == 1 means SUNDIALS function returns a flag so check if
//            flag >= 0
//   opt == 2 means function allocates memory so check if returned
//            NULL pointer
int Naunet::CheckFlag(void *flagvalue, const char *funcname, int opt,
                      FILE *errf) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(errf,
                "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return NAUNET_FAIL;
    }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *)flagvalue;
        if (*errflag < 0) {
            fprintf(errf, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return NAUNET_FAIL;
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(errf, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return NAUNET_FAIL;
    }

    return NAUNET_SUCCESS;
};

int Naunet::Finalize() {
    /* */

    N_VDestroy(cv_y_);
    // N_VFreeEmpty(cv_y_);
    SUNMatDestroy(cv_a_);
    SUNLinSolFree(cv_ls_);
    // delete m_data;

    /*  */

    fclose(errfp_);

    return NAUNET_SUCCESS;
};

int Naunet::GetCVStates(void *cv_mem, long int &nst, long int &nfe,
                        long int &nsetups, long int &nje, long int &netf,
                        long int &nge, long int &nni, long int &ncfn) {
    int flag;

    flag = CVodeGetNumSteps(cv_mem, &nst);
    if (CheckFlag(&flag, "CVodeGetNumSteps", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumRhsEvals(cv_mem, &nfe);
    if (CheckFlag(&flag, "CVodeGetNumRhsEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumLinSolvSetups(cv_mem, &nsetups);
    if (CheckFlag(&flag, "CVodeGetNumLinSolvSetups", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumErrTestFails(cv_mem, &netf);
    if (CheckFlag(&flag, "CVodeGetNumErrTestFails", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumNonlinSolvIters(cv_mem, &nni);
    if (CheckFlag(&flag, "CVodeGetNumNonlinSolvIters", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumNonlinSolvConvFails(cv_mem, &ncfn);
    if (CheckFlag(&flag, "CVodeGetNumNonlinSolvConvFails", 1, errfp_) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumJacEvals(cv_mem, &nje);
    if (CheckFlag(&flag, "CVodeGetNumJacEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    flag = CVodeGetNumGEvals(cv_mem, &nge);
    if (CheckFlag(&flag, "CVodeGetNumGEvals", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    return NAUNET_SUCCESS;
};

int Naunet::HandleError(int cvflag, realtype *ab, realtype dt, realtype t0) {
    if (cvflag >= 0) {
        return NAUNET_SUCCESS;
    }

    fprintf(errfp_, "CVode failed in Naunet! Flag = %d\n", cvflag);
    fprintf(errfp_, "Calling HandleError to fix the problem\n");

    /* */

    realtype dt_init = dt;

    for (int level = 1; level < 6; level++) {
        int nsubsteps = 10 * level;

        if (cvflag < 0 && cvflag > -5) {
            for (int i = 0; i < NEQUATIONS; i++) {
                ab_tmp_[i] = ab[i];
            }
            dt -= t0;
        } else if (cvflag == -6) {
            // The state may have something wrong
            // Reset to the initial state and try finer steps
            for (int i = 0; i < NEQUATIONS; i++) {
                ab_tmp_[i] = ab_init_[i];
            }
            dt = dt_init;
        } else if (cvflag < 0) {
            fprintf(
                errfp_,
                "The error cannot be recovered by Naunet! Exit from Naunet!\n");
            fprintf(errfp_, "cvFlag = %d, level = %d\n", cvflag, level);
            return NAUNET_FAIL;
        }

        // Reset initial conditions
        t0 = 0.0;
        for (int i = 0; i < NEQUATIONS; i++) {
            ab[i] = ab_tmp_[i];
        }

        // Reinitialize
        cvflag = CVodeReInit(cv_mem_, t0, cv_y_);
        if (CheckFlag(&cvflag, "CVodeReInit", 1, errfp_) == NAUNET_FAIL) {
            return NAUNET_FAIL;
        }

        for (int step = 0; step < nsubsteps; step++) {
            realtype tout = pow(
                10.0, log10(dt) * (realtype)(step + 1) / (realtype)nsubsteps);
            // printf("tout: %13.7e, step: %d, level: %d\n", tout, step, level);
            // realtype tcur = 0.0;
            // cvflag = CVodeGetCurrentTime(cv_mem_, &tcur);
            cvflag = CVode(cv_mem_, tout, cv_y_, &t0, CV_NORMAL);
            if (cvflag < 0) {
                fprintf(errfp_,
                        "CVode failed in Naunet! Flag = %d in the %dth substep "
                        "of %dth level! \n",
                        cvflag, step, level);
                if (level < 5) {
                    fprintf(errfp_,
                            "Tyring to fix the error in the next level\n");
                }
                // fprintf(errfp_, "Failed to fix the error! cvflag = %d in the
                // %dth substep! \n", cvflag, i);
                break;
            }
        }

        // if CVode succeeded, leave the loop
        if (cvflag >= 0) {
            if (level > 0) {
                fprintf(errfp_,
                        "The error was successfully fixed in %dth level\n",
                        level);
            }
            // break;
            return NAUNET_SUCCESS;
        }
    }

    /* */

    return NAUNET_FAIL;
}

int Naunet::Init(int nsystem, double atol, double rtol, int mxsteps) {
    n_system_ = nsystem;
    mxsteps_  = mxsteps;
    atol_     = atol;
    rtol_     = rtol;
    errfp_    = fopen("naunet_error_record.txt", "a");

    /* */
    if (nsystem != 1) {
        printf("This solver doesn't support nsystem > 1!");
        return NAUNET_FAIL;
    }

    cv_y_  = N_VNewEmpty_Serial((sunindextype)NEQUATIONS);
    cv_a_  = SUNSparseMatrix(NEQUATIONS, NEQUATIONS, NNZ, CSR_MAT);
    cv_ls_ = SUNLinSol_KLU(cv_y_, cv_a_);

    /*  */

    // reset the n_vector to empty, maybe not necessary
    /* */

    // N_VDestroy(cv_y_);
    // cv_y_ = N_VNewEmpty_Serial((sunindextype)NEQUATIONS);

    /* */

    /* */

    // TODO: The result is not saved
    // double *x1  = vector(1, 6);
    // double *x2  = vector(1, 7);
    // double **y  = matrix(1, 6, 1, 7);
    // double **y2 = matrix(1, 6, 1, 7);

    // for (int i=1; i<=6; i++) x1[i] = COShieldingTableX[i-1];
    // for (int i=1; i<=7; i++) x2[i] = COShieldingTableY[i-1];

    // for (int i=1; i<=6; i++) {
    //     for (int j=1; j<=7; j++) {
    //         y[i][j] = COShieldingTable[i-1][j-1];
    //     }
    // }

    // splie2(x1, x2, y, 6, 7, y2);

    // free_vector(x1, 1, 6);
    // free_vector(x2, 1, 7);
    // free_matrix(y, 1, 6, 1, 7);
    // free_matrix(y2, 1, 6, 1, 7);

    // splie2(COShieldingTableX, COShieldingTableY, COShieldingTable, 6, 7,
    // COShieldingTableD2);

    /* */

    return NAUNET_SUCCESS;
};

int Naunet::PrintDebugInfo() {
    long int nst, nfe, nsetups, nje, netf, nge, nni, ncfn;
    int flag;

    /* */

    if (GetCVStates(cv_mem_, nst, nfe, nsetups, nje, netf, nge, nni, ncfn) ==
        NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    printf("\nFinal Statistics:\n");
    printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nje = %ld\n", nst, nfe,
           nsetups, nje);
    printf("nni = %-6ld ncfn = %-6ld netf = %-6ld    nge = %ld\n \n", nni, ncfn,
           netf, nge);

    /*  */

    return NAUNET_SUCCESS;
};

#ifdef IDX_ELEM_H
int Naunet::Renorm(realtype *ab) {
    N_Vector b     = N_VMake_Serial(NELEMENTS, ab_ref_);
    N_Vector r     = N_VNew_Serial(NELEMENTS);
    SUNMatrix A    = SUNDenseMatrix(NELEMENTS, NELEMENTS);
    double Hnuclei = GetHNuclei(ab);

    N_VConst(0.0, r);
    // clang-format off
    IJth(A, IDX_ELEM_C, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 12.0 * ab[IDX_GCH4I] / 16.0 / Hnuclei + 12.0 * ab[IDX_GCOI] /
        28.0 / Hnuclei + 12.0 * ab[IDX_GCO2I] / 44.0 / Hnuclei + 12.0 *
        ab[IDX_GH2CNI] / 28.0 / Hnuclei + 12.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei
        + 12.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 12.0 * ab[IDX_GHNCI] / 27.0 /
        Hnuclei + 12.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 12.0 * ab[IDX_GSiCI]
        / 40.0 / Hnuclei + 48.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei + 108.0 *
        ab[IDX_GSiC3I] / 64.0 / Hnuclei + 12.0 * ab[IDX_CI] / 12.0 / Hnuclei +
        12.0 * ab[IDX_CII] / 12.0 / Hnuclei + 12.0 * ab[IDX_CHI] / 13.0 /
        Hnuclei + 12.0 * ab[IDX_CHII] / 13.0 / Hnuclei + 12.0 * ab[IDX_CH2I] /
        14.0 / Hnuclei + 12.0 * ab[IDX_CH2II] / 14.0 / Hnuclei + 12.0 *
        ab[IDX_CH3I] / 15.0 / Hnuclei + 12.0 * ab[IDX_CH3II] / 15.0 / Hnuclei +
        12.0 * ab[IDX_CH3OHI] / 32.0 / Hnuclei + 12.0 * ab[IDX_CH4I] / 16.0 /
        Hnuclei + 12.0 * ab[IDX_CH4II] / 16.0 / Hnuclei + 12.0 * ab[IDX_CNI] /
        26.0 / Hnuclei + 12.0 * ab[IDX_CNII] / 26.0 / Hnuclei + 12.0 *
        ab[IDX_COI] / 28.0 / Hnuclei + 12.0 * ab[IDX_COII] / 28.0 / Hnuclei +
        12.0 * ab[IDX_CO2I] / 44.0 / Hnuclei + 12.0 * ab[IDX_H2CNI] / 28.0 /
        Hnuclei + 12.0 * ab[IDX_H2COI] / 30.0 / Hnuclei + 12.0 * ab[IDX_H2COII]
        / 30.0 / Hnuclei + 12.0 * ab[IDX_H3COII] / 31.0 / Hnuclei + 12.0 *
        ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 * ab[IDX_HCNII] / 27.0 / Hnuclei +
        12.0 * ab[IDX_HCNHII] / 28.0 / Hnuclei + 12.0 * ab[IDX_HCOI] / 29.0 /
        Hnuclei + 12.0 * ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 * ab[IDX_HCO2II]
        / 45.0 / Hnuclei + 12.0 * ab[IDX_HNCI] / 27.0 / Hnuclei + 12.0 *
        ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 * ab[IDX_HOCII] / 29.0 / Hnuclei +
        12.0 * ab[IDX_OCNI] / 42.0 / Hnuclei + 12.0 * ab[IDX_SiCI] / 40.0 /
        Hnuclei + 12.0 * ab[IDX_SiCII] / 40.0 / Hnuclei + 48.0 * ab[IDX_SiC2I] /
        52.0 / Hnuclei + 48.0 * ab[IDX_SiC2II] / 52.0 / Hnuclei + 108.0 *
        ab[IDX_SiC3I] / 64.0 / Hnuclei + 108.0 * ab[IDX_SiC3II] / 64.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_H) = 0.0 + 4.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 4.0 * ab[IDX_GCH4I] / 16.0 / Hnuclei + 2.0 * ab[IDX_GH2CNI] /
        28.0 / Hnuclei + 2.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei + 1.0 *
        ab[IDX_GHCNI] / 27.0 / Hnuclei + 1.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei +
        1.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 * ab[IDX_CHI] / 13.0 /
        Hnuclei + 1.0 * ab[IDX_CHII] / 13.0 / Hnuclei + 2.0 * ab[IDX_CH2I] /
        14.0 / Hnuclei + 2.0 * ab[IDX_CH2II] / 14.0 / Hnuclei + 3.0 *
        ab[IDX_CH3I] / 15.0 / Hnuclei + 3.0 * ab[IDX_CH3II] / 15.0 / Hnuclei +
        4.0 * ab[IDX_CH3OHI] / 32.0 / Hnuclei + 4.0 * ab[IDX_CH4I] / 16.0 /
        Hnuclei + 4.0 * ab[IDX_CH4II] / 16.0 / Hnuclei + 2.0 * ab[IDX_H2CNI] /
        28.0 / Hnuclei + 2.0 * ab[IDX_H2COI] / 30.0 / Hnuclei + 2.0 *
        ab[IDX_H2COII] / 30.0 / Hnuclei + 3.0 * ab[IDX_H3COII] / 31.0 / Hnuclei
        + 1.0 * ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 * ab[IDX_HCNII] / 27.0 /
        Hnuclei + 2.0 * ab[IDX_HCNHII] / 28.0 / Hnuclei + 1.0 * ab[IDX_HCOI] /
        29.0 / Hnuclei + 1.0 * ab[IDX_HCOII] / 29.0 / Hnuclei + 1.0 *
        ab[IDX_HCO2II] / 45.0 / Hnuclei + 1.0 * ab[IDX_HNCI] / 27.0 / Hnuclei +
        1.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei + 1.0 * ab[IDX_HOCII] / 29.0 /
        Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_C, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GH2CNI] / 28.0 /
        Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 14.0 * ab[IDX_GHNCI] /
        27.0 / Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 14.0 *
        ab[IDX_CNI] / 26.0 / Hnuclei + 14.0 * ab[IDX_CNII] / 26.0 / Hnuclei +
        14.0 * ab[IDX_H2CNI] / 28.0 / Hnuclei + 14.0 * ab[IDX_HCNI] / 27.0 /
        Hnuclei + 14.0 * ab[IDX_HCNII] / 27.0 / Hnuclei + 14.0 * ab[IDX_HCNHII]
        / 28.0 / Hnuclei + 14.0 * ab[IDX_HNCI] / 27.0 / Hnuclei + 14.0 *
        ab[IDX_HNCOI] / 43.0 / Hnuclei + 14.0 * ab[IDX_OCNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 16.0 * ab[IDX_GCOI] / 28.0 / Hnuclei + 32.0 * ab[IDX_GCO2I] /
        44.0 / Hnuclei + 16.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei + 16.0 *
        ab[IDX_GHNCOI] / 43.0 / Hnuclei + 16.0 * ab[IDX_CH3OHI] / 32.0 / Hnuclei
        + 16.0 * ab[IDX_COI] / 28.0 / Hnuclei + 16.0 * ab[IDX_COII] / 28.0 /
        Hnuclei + 32.0 * ab[IDX_CO2I] / 44.0 / Hnuclei + 16.0 * ab[IDX_H2COI] /
        30.0 / Hnuclei + 16.0 * ab[IDX_H2COII] / 30.0 / Hnuclei + 16.0 *
        ab[IDX_H3COII] / 31.0 / Hnuclei + 16.0 * ab[IDX_HCOI] / 29.0 / Hnuclei +
        16.0 * ab[IDX_HCOII] / 29.0 / Hnuclei + 32.0 * ab[IDX_HCO2II] / 45.0 /
        Hnuclei + 16.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei + 16.0 * ab[IDX_HOCII] /
        29.0 / Hnuclei + 16.0 * ab[IDX_OCNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_C, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GSiCI] / 40.0 /
        Hnuclei + 56.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei + 84.0 * ab[IDX_GSiC3I]
        / 64.0 / Hnuclei + 28.0 * ab[IDX_SiCI] / 40.0 / Hnuclei + 28.0 *
        ab[IDX_SiCII] / 40.0 / Hnuclei + 56.0 * ab[IDX_SiC2I] / 52.0 / Hnuclei +
        56.0 * ab[IDX_SiC2II] / 52.0 / Hnuclei + 84.0 * ab[IDX_SiC3I] / 64.0 /
        Hnuclei + 84.0 * ab[IDX_SiC3II] / 64.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_C) = 0.0 + 48.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 48.0 * ab[IDX_GCH4I] / 16.0 / Hnuclei + 24.0 * ab[IDX_GH2CNI]
        / 28.0 / Hnuclei + 24.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei + 12.0 *
        ab[IDX_GHCNI] / 27.0 / Hnuclei + 12.0 * ab[IDX_GHNCI] / 27.0 / Hnuclei +
        12.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 12.0 * ab[IDX_CHI] / 13.0 /
        Hnuclei + 12.0 * ab[IDX_CHII] / 13.0 / Hnuclei + 24.0 * ab[IDX_CH2I] /
        14.0 / Hnuclei + 24.0 * ab[IDX_CH2II] / 14.0 / Hnuclei + 36.0 *
        ab[IDX_CH3I] / 15.0 / Hnuclei + 36.0 * ab[IDX_CH3II] / 15.0 / Hnuclei +
        48.0 * ab[IDX_CH3OHI] / 32.0 / Hnuclei + 48.0 * ab[IDX_CH4I] / 16.0 /
        Hnuclei + 48.0 * ab[IDX_CH4II] / 16.0 / Hnuclei + 24.0 * ab[IDX_H2CNI] /
        28.0 / Hnuclei + 24.0 * ab[IDX_H2COI] / 30.0 / Hnuclei + 24.0 *
        ab[IDX_H2COII] / 30.0 / Hnuclei + 36.0 * ab[IDX_H3COII] / 31.0 / Hnuclei
        + 12.0 * ab[IDX_HCNI] / 27.0 / Hnuclei + 12.0 * ab[IDX_HCNII] / 27.0 /
        Hnuclei + 24.0 * ab[IDX_HCNHII] / 28.0 / Hnuclei + 12.0 * ab[IDX_HCOI] /
        29.0 / Hnuclei + 12.0 * ab[IDX_HCOII] / 29.0 / Hnuclei + 12.0 *
        ab[IDX_HCO2II] / 45.0 / Hnuclei + 12.0 * ab[IDX_HNCI] / 27.0 / Hnuclei +
        12.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 * ab[IDX_HOCII] / 29.0 /
        Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_H) = 0.0 + 16.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 16.0 * ab[IDX_GCH4I] / 16.0 / Hnuclei + 4.0 * ab[IDX_GH2CNI] /
        28.0 / Hnuclei + 4.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei + 4.0 *
        ab[IDX_GH2OI] / 18.0 / Hnuclei + 4.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei
        + 1.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 1.0 * ab[IDX_GHNCI] / 27.0 /
        Hnuclei + 1.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 * ab[IDX_GHNOI] /
        31.0 / Hnuclei + 9.0 * ab[IDX_GNH3I] / 17.0 / Hnuclei + 1.0 *
        ab[IDX_GO2HI] / 33.0 / Hnuclei + 16.0 * ab[IDX_GSiH4I] / 32.0 / Hnuclei
        + 1.0 * ab[IDX_CHI] / 13.0 / Hnuclei + 1.0 * ab[IDX_CHII] / 13.0 /
        Hnuclei + 4.0 * ab[IDX_CH2I] / 14.0 / Hnuclei + 4.0 * ab[IDX_CH2II] /
        14.0 / Hnuclei + 9.0 * ab[IDX_CH3I] / 15.0 / Hnuclei + 9.0 *
        ab[IDX_CH3II] / 15.0 / Hnuclei + 16.0 * ab[IDX_CH3OHI] / 32.0 / Hnuclei
        + 16.0 * ab[IDX_CH4I] / 16.0 / Hnuclei + 16.0 * ab[IDX_CH4II] / 16.0 /
        Hnuclei + 1.0 * ab[IDX_HI] / 1.0 / Hnuclei + 1.0 * ab[IDX_HII] / 1.0 /
        Hnuclei + 4.0 * ab[IDX_H2I] / 2.0 / Hnuclei + 4.0 * ab[IDX_H2II] / 2.0 /
        Hnuclei + 4.0 * ab[IDX_H2CNI] / 28.0 / Hnuclei + 4.0 * ab[IDX_H2COI] /
        30.0 / Hnuclei + 4.0 * ab[IDX_H2COII] / 30.0 / Hnuclei + 4.0 *
        ab[IDX_H2NOII] / 32.0 / Hnuclei + 4.0 * ab[IDX_H2OI] / 18.0 / Hnuclei +
        4.0 * ab[IDX_H2OII] / 18.0 / Hnuclei + 4.0 * ab[IDX_H2SiOI] / 46.0 /
        Hnuclei + 9.0 * ab[IDX_H3II] / 3.0 / Hnuclei + 9.0 * ab[IDX_H3COII] /
        31.0 / Hnuclei + 9.0 * ab[IDX_H3OII] / 19.0 / Hnuclei + 1.0 *
        ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 * ab[IDX_HCNII] / 27.0 / Hnuclei +
        4.0 * ab[IDX_HCNHII] / 28.0 / Hnuclei + 1.0 * ab[IDX_HCOI] / 29.0 /
        Hnuclei + 1.0 * ab[IDX_HCOII] / 29.0 / Hnuclei + 1.0 * ab[IDX_HCO2II] /
        45.0 / Hnuclei + 1.0 * ab[IDX_HeHII] / 5.0 / Hnuclei + 1.0 *
        ab[IDX_HNCI] / 27.0 / Hnuclei + 1.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei +
        1.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 1.0 * ab[IDX_HNOII] / 31.0 /
        Hnuclei + 1.0 * ab[IDX_HOCII] / 29.0 / Hnuclei + 1.0 * ab[IDX_N2HII] /
        29.0 / Hnuclei + 1.0 * ab[IDX_NHI] / 15.0 / Hnuclei + 1.0 * ab[IDX_NHII]
        / 15.0 / Hnuclei + 4.0 * ab[IDX_NH2I] / 16.0 / Hnuclei + 4.0 *
        ab[IDX_NH2II] / 16.0 / Hnuclei + 9.0 * ab[IDX_NH3I] / 17.0 / Hnuclei +
        9.0 * ab[IDX_NH3II] / 17.0 / Hnuclei + 1.0 * ab[IDX_O2HI] / 33.0 /
        Hnuclei + 1.0 * ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 * ab[IDX_OHI] /
        17.0 / Hnuclei + 1.0 * ab[IDX_OHII] / 17.0 / Hnuclei + 1.0 *
        ab[IDX_SiHI] / 29.0 / Hnuclei + 1.0 * ab[IDX_SiHII] / 29.0 / Hnuclei +
        4.0 * ab[IDX_SiH2I] / 30.0 / Hnuclei + 4.0 * ab[IDX_SiH2II] / 30.0 /
        Hnuclei + 9.0 * ab[IDX_SiH3I] / 31.0 / Hnuclei + 9.0 * ab[IDX_SiH3II] /
        31.0 / Hnuclei + 16.0 * ab[IDX_SiH4I] / 32.0 / Hnuclei + 16.0 *
        ab[IDX_SiH4II] / 32.0 / Hnuclei + 25.0 * ab[IDX_SiH5II] / 33.0 / Hnuclei
        + 1.0 * ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_HE) = 0.0 + 4.0 * ab[IDX_HeHII] / 5.0 /
        Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_H, IDX_ELEM_N) = 0.0 + 28.0 * ab[IDX_GH2CNI] / 28.0 /
        Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 14.0 * ab[IDX_GHNCI] /
        27.0 / Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 14.0 *
        ab[IDX_GHNOI] / 31.0 / Hnuclei + 42.0 * ab[IDX_GNH3I] / 17.0 / Hnuclei +
        28.0 * ab[IDX_H2CNI] / 28.0 / Hnuclei + 28.0 * ab[IDX_H2NOII] / 32.0 /
        Hnuclei + 14.0 * ab[IDX_HCNI] / 27.0 / Hnuclei + 14.0 * ab[IDX_HCNII] /
        27.0 / Hnuclei + 28.0 * ab[IDX_HCNHII] / 28.0 / Hnuclei + 14.0 *
        ab[IDX_HNCI] / 27.0 / Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei +
        14.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0 * ab[IDX_HNOII] / 31.0 /
        Hnuclei + 28.0 * ab[IDX_N2HII] / 29.0 / Hnuclei + 14.0 * ab[IDX_NHI] /
        15.0 / Hnuclei + 14.0 * ab[IDX_NHII] / 15.0 / Hnuclei + 28.0 *
        ab[IDX_NH2I] / 16.0 / Hnuclei + 28.0 * ab[IDX_NH2II] / 16.0 / Hnuclei +
        42.0 * ab[IDX_NH3I] / 17.0 / Hnuclei + 42.0 * ab[IDX_NH3II] / 17.0 /
        Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_O) = 0.0 + 64.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 32.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei + 32.0 * ab[IDX_GH2OI]
        / 18.0 / Hnuclei + 32.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 16.0 *
        ab[IDX_GHNCOI] / 43.0 / Hnuclei + 16.0 * ab[IDX_GHNOI] / 31.0 / Hnuclei
        + 32.0 * ab[IDX_GO2HI] / 33.0 / Hnuclei + 64.0 * ab[IDX_CH3OHI] / 32.0 /
        Hnuclei + 32.0 * ab[IDX_H2COI] / 30.0 / Hnuclei + 32.0 * ab[IDX_H2COII]
        / 30.0 / Hnuclei + 32.0 * ab[IDX_H2NOII] / 32.0 / Hnuclei + 32.0 *
        ab[IDX_H2OI] / 18.0 / Hnuclei + 32.0 * ab[IDX_H2OII] / 18.0 / Hnuclei +
        32.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei + 48.0 * ab[IDX_H3COII] / 31.0 /
        Hnuclei + 48.0 * ab[IDX_H3OII] / 19.0 / Hnuclei + 16.0 * ab[IDX_HCOI] /
        29.0 / Hnuclei + 16.0 * ab[IDX_HCOII] / 29.0 / Hnuclei + 32.0 *
        ab[IDX_HCO2II] / 45.0 / Hnuclei + 16.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei
        + 16.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 * ab[IDX_HNOII] / 31.0 /
        Hnuclei + 16.0 * ab[IDX_HOCII] / 29.0 / Hnuclei + 32.0 * ab[IDX_O2HI] /
        33.0 / Hnuclei + 32.0 * ab[IDX_O2HII] / 33.0 / Hnuclei + 16.0 *
        ab[IDX_OHI] / 17.0 / Hnuclei + 16.0 * ab[IDX_OHII] / 17.0 / Hnuclei +
        16.0 * ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_H, IDX_ELEM_SI) = 0.0 + 56.0 * ab[IDX_GH2SiOI] / 46.0 /
        Hnuclei + 112.0 * ab[IDX_GSiH4I] / 32.0 / Hnuclei + 56.0 *
        ab[IDX_H2SiOI] / 46.0 / Hnuclei + 28.0 * ab[IDX_SiHI] / 29.0 / Hnuclei +
        28.0 * ab[IDX_SiHII] / 29.0 / Hnuclei + 56.0 * ab[IDX_SiH2I] / 30.0 /
        Hnuclei + 56.0 * ab[IDX_SiH2II] / 30.0 / Hnuclei + 84.0 * ab[IDX_SiH3I]
        / 31.0 / Hnuclei + 84.0 * ab[IDX_SiH3II] / 31.0 / Hnuclei + 112.0 *
        ab[IDX_SiH4I] / 32.0 / Hnuclei + 112.0 * ab[IDX_SiH4II] / 32.0 / Hnuclei
        + 140.0 * ab[IDX_SiH5II] / 33.0 / Hnuclei + 28.0 * ab[IDX_SiOHII] / 45.0
        / Hnuclei;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_H) = 0.0 + 1.0 * ab[IDX_HeHII] / 5.0 /
        Hnuclei;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_HE) = 0.0 + 4.0 * ab[IDX_HeI] / 4.0 /
        Hnuclei + 4.0 * ab[IDX_HeII] / 4.0 / Hnuclei + 4.0 * ab[IDX_HeHII] / 5.0
        / Hnuclei;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_HE, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_C) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_H) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_MG) = 0.0 + 24.0 * ab[IDX_GMgI] / 24.0 /
        Hnuclei + 24.0 * ab[IDX_MgI] / 24.0 / Hnuclei + 24.0 * ab[IDX_MgII] /
        24.0 / Hnuclei;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_O) = 0.0;
    IJth(A, IDX_ELEM_MG, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GH2CNI] / 28.0 /
        Hnuclei + 12.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 12.0 * ab[IDX_GHNCI] /
        27.0 / Hnuclei + 12.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 12.0 *
        ab[IDX_CNI] / 26.0 / Hnuclei + 12.0 * ab[IDX_CNII] / 26.0 / Hnuclei +
        12.0 * ab[IDX_H2CNI] / 28.0 / Hnuclei + 12.0 * ab[IDX_HCNI] / 27.0 /
        Hnuclei + 12.0 * ab[IDX_HCNII] / 27.0 / Hnuclei + 12.0 * ab[IDX_HCNHII]
        / 28.0 / Hnuclei + 12.0 * ab[IDX_HNCI] / 27.0 / Hnuclei + 12.0 *
        ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 * ab[IDX_OCNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_H) = 0.0 + 2.0 * ab[IDX_GH2CNI] / 28.0 /
        Hnuclei + 1.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 1.0 * ab[IDX_GHNCI] /
        27.0 / Hnuclei + 1.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 *
        ab[IDX_GHNOI] / 31.0 / Hnuclei + 3.0 * ab[IDX_GNH3I] / 17.0 / Hnuclei +
        2.0 * ab[IDX_H2CNI] / 28.0 / Hnuclei + 2.0 * ab[IDX_H2NOII] / 32.0 /
        Hnuclei + 1.0 * ab[IDX_HCNI] / 27.0 / Hnuclei + 1.0 * ab[IDX_HCNII] /
        27.0 / Hnuclei + 2.0 * ab[IDX_HCNHII] / 28.0 / Hnuclei + 1.0 *
        ab[IDX_HNCI] / 27.0 / Hnuclei + 1.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei +
        1.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 1.0 * ab[IDX_HNOII] / 31.0 /
        Hnuclei + 2.0 * ab[IDX_N2HII] / 29.0 / Hnuclei + 1.0 * ab[IDX_NHI] /
        15.0 / Hnuclei + 1.0 * ab[IDX_NHII] / 15.0 / Hnuclei + 2.0 *
        ab[IDX_NH2I] / 16.0 / Hnuclei + 2.0 * ab[IDX_NH2II] / 16.0 / Hnuclei +
        3.0 * ab[IDX_NH3I] / 17.0 / Hnuclei + 3.0 * ab[IDX_NH3II] / 17.0 /
        Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_N, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GH2CNI] / 28.0 /
        Hnuclei + 14.0 * ab[IDX_GHCNI] / 27.0 / Hnuclei + 14.0 * ab[IDX_GHNCI] /
        27.0 / Hnuclei + 14.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 14.0 *
        ab[IDX_GHNOI] / 31.0 / Hnuclei + 56.0 * ab[IDX_GN2I] / 28.0 / Hnuclei +
        14.0 * ab[IDX_GNH3I] / 17.0 / Hnuclei + 14.0 * ab[IDX_GNOI] / 30.0 /
        Hnuclei + 14.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei + 14.0 * ab[IDX_CNI] /
        26.0 / Hnuclei + 14.0 * ab[IDX_CNII] / 26.0 / Hnuclei + 14.0 *
        ab[IDX_H2CNI] / 28.0 / Hnuclei + 14.0 * ab[IDX_H2NOII] / 32.0 / Hnuclei
        + 14.0 * ab[IDX_HCNI] / 27.0 / Hnuclei + 14.0 * ab[IDX_HCNII] / 27.0 /
        Hnuclei + 14.0 * ab[IDX_HCNHII] / 28.0 / Hnuclei + 14.0 * ab[IDX_HNCI] /
        27.0 / Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei + 14.0 *
        ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0 * ab[IDX_HNOII] / 31.0 / Hnuclei +
        14.0 * ab[IDX_NI] / 14.0 / Hnuclei + 14.0 * ab[IDX_NII] / 14.0 / Hnuclei
        + 56.0 * ab[IDX_N2I] / 28.0 / Hnuclei + 56.0 * ab[IDX_N2II] / 28.0 /
        Hnuclei + 56.0 * ab[IDX_N2HII] / 29.0 / Hnuclei + 14.0 * ab[IDX_NHI] /
        15.0 / Hnuclei + 14.0 * ab[IDX_NHII] / 15.0 / Hnuclei + 14.0 *
        ab[IDX_NH2I] / 16.0 / Hnuclei + 14.0 * ab[IDX_NH2II] / 16.0 / Hnuclei +
        14.0 * ab[IDX_NH3I] / 17.0 / Hnuclei + 14.0 * ab[IDX_NH3II] / 17.0 /
        Hnuclei + 14.0 * ab[IDX_NOI] / 30.0 / Hnuclei + 14.0 * ab[IDX_NOII] /
        30.0 / Hnuclei + 14.0 * ab[IDX_NO2I] / 46.0 / Hnuclei + 14.0 *
        ab[IDX_OCNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GHNCOI] / 43.0 /
        Hnuclei + 16.0 * ab[IDX_GHNOI] / 31.0 / Hnuclei + 16.0 * ab[IDX_GNOI] /
        30.0 / Hnuclei + 32.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei + 16.0 *
        ab[IDX_H2NOII] / 32.0 / Hnuclei + 16.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei
        + 16.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 * ab[IDX_HNOII] / 31.0 /
        Hnuclei + 16.0 * ab[IDX_NOI] / 30.0 / Hnuclei + 16.0 * ab[IDX_NOII] /
        30.0 / Hnuclei + 32.0 * ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 *
        ab[IDX_OCNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_N, IDX_ELEM_SI) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 12.0 * ab[IDX_GCOI] / 28.0 / Hnuclei + 24.0 * ab[IDX_GCO2I] /
        44.0 / Hnuclei + 12.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei + 12.0 *
        ab[IDX_GHNCOI] / 43.0 / Hnuclei + 12.0 * ab[IDX_CH3OHI] / 32.0 / Hnuclei
        + 12.0 * ab[IDX_COI] / 28.0 / Hnuclei + 12.0 * ab[IDX_COII] / 28.0 /
        Hnuclei + 24.0 * ab[IDX_CO2I] / 44.0 / Hnuclei + 12.0 * ab[IDX_H2COI] /
        30.0 / Hnuclei + 12.0 * ab[IDX_H2COII] / 30.0 / Hnuclei + 12.0 *
        ab[IDX_H3COII] / 31.0 / Hnuclei + 12.0 * ab[IDX_HCOI] / 29.0 / Hnuclei +
        12.0 * ab[IDX_HCOII] / 29.0 / Hnuclei + 24.0 * ab[IDX_HCO2II] / 45.0 /
        Hnuclei + 12.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei + 12.0 * ab[IDX_HOCII] /
        29.0 / Hnuclei + 12.0 * ab[IDX_OCNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_H) = 0.0 + 4.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 2.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei + 2.0 * ab[IDX_GH2OI] /
        18.0 / Hnuclei + 2.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei + 1.0 *
        ab[IDX_GHNCOI] / 43.0 / Hnuclei + 1.0 * ab[IDX_GHNOI] / 31.0 / Hnuclei +
        2.0 * ab[IDX_GO2HI] / 33.0 / Hnuclei + 4.0 * ab[IDX_CH3OHI] / 32.0 /
        Hnuclei + 2.0 * ab[IDX_H2COI] / 30.0 / Hnuclei + 2.0 * ab[IDX_H2COII] /
        30.0 / Hnuclei + 2.0 * ab[IDX_H2NOII] / 32.0 / Hnuclei + 2.0 *
        ab[IDX_H2OI] / 18.0 / Hnuclei + 2.0 * ab[IDX_H2OII] / 18.0 / Hnuclei +
        2.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei + 3.0 * ab[IDX_H3COII] / 31.0 /
        Hnuclei + 3.0 * ab[IDX_H3OII] / 19.0 / Hnuclei + 1.0 * ab[IDX_HCOI] /
        29.0 / Hnuclei + 1.0 * ab[IDX_HCOII] / 29.0 / Hnuclei + 2.0 *
        ab[IDX_HCO2II] / 45.0 / Hnuclei + 1.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei +
        1.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 1.0 * ab[IDX_HNOII] / 31.0 /
        Hnuclei + 1.0 * ab[IDX_HOCII] / 29.0 / Hnuclei + 2.0 * ab[IDX_O2HI] /
        33.0 / Hnuclei + 2.0 * ab[IDX_O2HII] / 33.0 / Hnuclei + 1.0 *
        ab[IDX_OHI] / 17.0 / Hnuclei + 1.0 * ab[IDX_OHII] / 17.0 / Hnuclei + 1.0
        * ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_O, IDX_ELEM_N) = 0.0 + 14.0 * ab[IDX_GHNCOI] / 43.0 /
        Hnuclei + 14.0 * ab[IDX_GHNOI] / 31.0 / Hnuclei + 14.0 * ab[IDX_GNOI] /
        30.0 / Hnuclei + 28.0 * ab[IDX_GNO2I] / 46.0 / Hnuclei + 14.0 *
        ab[IDX_H2NOII] / 32.0 / Hnuclei + 14.0 * ab[IDX_HNCOI] / 43.0 / Hnuclei
        + 14.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 14.0 * ab[IDX_HNOII] / 31.0 /
        Hnuclei + 14.0 * ab[IDX_NOI] / 30.0 / Hnuclei + 14.0 * ab[IDX_NOII] /
        30.0 / Hnuclei + 28.0 * ab[IDX_NO2I] / 46.0 / Hnuclei + 14.0 *
        ab[IDX_OCNI] / 42.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GCH3OHI] / 32.0 /
        Hnuclei + 16.0 * ab[IDX_GCOI] / 28.0 / Hnuclei + 64.0 * ab[IDX_GCO2I] /
        44.0 / Hnuclei + 16.0 * ab[IDX_GH2COI] / 30.0 / Hnuclei + 16.0 *
        ab[IDX_GH2OI] / 18.0 / Hnuclei + 16.0 * ab[IDX_GH2SiOI] / 46.0 / Hnuclei
        + 16.0 * ab[IDX_GHNCOI] / 43.0 / Hnuclei + 16.0 * ab[IDX_GHNOI] / 31.0 /
        Hnuclei + 16.0 * ab[IDX_GNOI] / 30.0 / Hnuclei + 64.0 * ab[IDX_GNO2I] /
        46.0 / Hnuclei + 64.0 * ab[IDX_GO2I] / 32.0 / Hnuclei + 64.0 *
        ab[IDX_GO2HI] / 33.0 / Hnuclei + 16.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei +
        16.0 * ab[IDX_CH3OHI] / 32.0 / Hnuclei + 16.0 * ab[IDX_COI] / 28.0 /
        Hnuclei + 16.0 * ab[IDX_COII] / 28.0 / Hnuclei + 64.0 * ab[IDX_CO2I] /
        44.0 / Hnuclei + 16.0 * ab[IDX_H2COI] / 30.0 / Hnuclei + 16.0 *
        ab[IDX_H2COII] / 30.0 / Hnuclei + 16.0 * ab[IDX_H2NOII] / 32.0 / Hnuclei
        + 16.0 * ab[IDX_H2OI] / 18.0 / Hnuclei + 16.0 * ab[IDX_H2OII] / 18.0 /
        Hnuclei + 16.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei + 16.0 * ab[IDX_H3COII]
        / 31.0 / Hnuclei + 16.0 * ab[IDX_H3OII] / 19.0 / Hnuclei + 16.0 *
        ab[IDX_HCOI] / 29.0 / Hnuclei + 16.0 * ab[IDX_HCOII] / 29.0 / Hnuclei +
        64.0 * ab[IDX_HCO2II] / 45.0 / Hnuclei + 16.0 * ab[IDX_HNCOI] / 43.0 /
        Hnuclei + 16.0 * ab[IDX_HNOI] / 31.0 / Hnuclei + 16.0 * ab[IDX_HNOII] /
        31.0 / Hnuclei + 16.0 * ab[IDX_HOCII] / 29.0 / Hnuclei + 16.0 *
        ab[IDX_NOI] / 30.0 / Hnuclei + 16.0 * ab[IDX_NOII] / 30.0 / Hnuclei +
        64.0 * ab[IDX_NO2I] / 46.0 / Hnuclei + 16.0 * ab[IDX_OI] / 16.0 /
        Hnuclei + 16.0 * ab[IDX_OII] / 16.0 / Hnuclei + 64.0 * ab[IDX_O2I] /
        32.0 / Hnuclei + 64.0 * ab[IDX_O2II] / 32.0 / Hnuclei + 64.0 *
        ab[IDX_O2HI] / 33.0 / Hnuclei + 64.0 * ab[IDX_O2HII] / 33.0 / Hnuclei +
        16.0 * ab[IDX_OCNI] / 42.0 / Hnuclei + 16.0 * ab[IDX_OHI] / 17.0 /
        Hnuclei + 16.0 * ab[IDX_OHII] / 17.0 / Hnuclei + 16.0 * ab[IDX_SiOI] /
        44.0 / Hnuclei + 16.0 * ab[IDX_SiOII] / 44.0 / Hnuclei + 16.0 *
        ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_O, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GH2SiOI] / 46.0 /
        Hnuclei + 28.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei + 28.0 * ab[IDX_H2SiOI]
        / 46.0 / Hnuclei + 28.0 * ab[IDX_SiOI] / 44.0 / Hnuclei + 28.0 *
        ab[IDX_SiOII] / 44.0 / Hnuclei + 28.0 * ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_C) = 0.0 + 12.0 * ab[IDX_GSiCI] / 40.0 /
        Hnuclei + 24.0 * ab[IDX_GSiC2I] / 52.0 / Hnuclei + 36.0 * ab[IDX_GSiC3I]
        / 64.0 / Hnuclei + 12.0 * ab[IDX_SiCI] / 40.0 / Hnuclei + 12.0 *
        ab[IDX_SiCII] / 40.0 / Hnuclei + 24.0 * ab[IDX_SiC2I] / 52.0 / Hnuclei +
        24.0 * ab[IDX_SiC2II] / 52.0 / Hnuclei + 36.0 * ab[IDX_SiC3I] / 64.0 /
        Hnuclei + 36.0 * ab[IDX_SiC3II] / 64.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_H) = 0.0 + 2.0 * ab[IDX_GH2SiOI] / 46.0 /
        Hnuclei + 4.0 * ab[IDX_GSiH4I] / 32.0 / Hnuclei + 2.0 * ab[IDX_H2SiOI] /
        46.0 / Hnuclei + 1.0 * ab[IDX_SiHI] / 29.0 / Hnuclei + 1.0 *
        ab[IDX_SiHII] / 29.0 / Hnuclei + 2.0 * ab[IDX_SiH2I] / 30.0 / Hnuclei +
        2.0 * ab[IDX_SiH2II] / 30.0 / Hnuclei + 3.0 * ab[IDX_SiH3I] / 31.0 /
        Hnuclei + 3.0 * ab[IDX_SiH3II] / 31.0 / Hnuclei + 4.0 * ab[IDX_SiH4I] /
        32.0 / Hnuclei + 4.0 * ab[IDX_SiH4II] / 32.0 / Hnuclei + 5.0 *
        ab[IDX_SiH5II] / 33.0 / Hnuclei + 1.0 * ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_HE) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_MG) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_N) = 0.0;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_O) = 0.0 + 16.0 * ab[IDX_GH2SiOI] / 46.0 /
        Hnuclei + 16.0 * ab[IDX_GSiOI] / 44.0 / Hnuclei + 16.0 * ab[IDX_H2SiOI]
        / 46.0 / Hnuclei + 16.0 * ab[IDX_SiOI] / 44.0 / Hnuclei + 16.0 *
        ab[IDX_SiOII] / 44.0 / Hnuclei + 16.0 * ab[IDX_SiOHII] / 45.0 / Hnuclei;
    IJth(A, IDX_ELEM_SI, IDX_ELEM_SI) = 0.0 + 28.0 * ab[IDX_GH2SiOI] / 46.0
        / Hnuclei + 28.0 * ab[IDX_GSiCI] / 40.0 / Hnuclei + 28.0 *
        ab[IDX_GSiC2I] / 52.0 / Hnuclei + 28.0 * ab[IDX_GSiC3I] / 64.0 / Hnuclei
        + 28.0 * ab[IDX_GSiH4I] / 32.0 / Hnuclei + 28.0 * ab[IDX_GSiOI] / 44.0 /
        Hnuclei + 28.0 * ab[IDX_H2SiOI] / 46.0 / Hnuclei + 28.0 * ab[IDX_SiI] /
        28.0 / Hnuclei + 28.0 * ab[IDX_SiII] / 28.0 / Hnuclei + 28.0 *
        ab[IDX_SiCI] / 40.0 / Hnuclei + 28.0 * ab[IDX_SiCII] / 40.0 / Hnuclei +
        28.0 * ab[IDX_SiC2I] / 52.0 / Hnuclei + 28.0 * ab[IDX_SiC2II] / 52.0 /
        Hnuclei + 28.0 * ab[IDX_SiC3I] / 64.0 / Hnuclei + 28.0 * ab[IDX_SiC3II]
        / 64.0 / Hnuclei + 28.0 * ab[IDX_SiHI] / 29.0 / Hnuclei + 28.0 *
        ab[IDX_SiHII] / 29.0 / Hnuclei + 28.0 * ab[IDX_SiH2I] / 30.0 / Hnuclei +
        28.0 * ab[IDX_SiH2II] / 30.0 / Hnuclei + 28.0 * ab[IDX_SiH3I] / 31.0 /
        Hnuclei + 28.0 * ab[IDX_SiH3II] / 31.0 / Hnuclei + 28.0 * ab[IDX_SiH4I]
        / 32.0 / Hnuclei + 28.0 * ab[IDX_SiH4II] / 32.0 / Hnuclei + 28.0 *
        ab[IDX_SiH5II] / 33.0 / Hnuclei + 28.0 * ab[IDX_SiOI] / 44.0 / Hnuclei +
        28.0 * ab[IDX_SiOII] / 44.0 / Hnuclei + 28.0 * ab[IDX_SiOHII] / 45.0 /
        Hnuclei;
    
    // clang-format on

    SUNLinearSolver LS = SUNLinSol_Dense(r, A);

    int flag;
    flag = SUNLinSolSetup(LS, A);
    if (CheckFlag(&flag, "SUNLinSolSetup", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }
    flag = SUNLinSolSolve(LS, A, r, b, 0.0);
    if (CheckFlag(&flag, "SUNLinSolSolve", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    realtype *rptr = N_VGetArrayPointer(r);

    // clang-format off
    ab[IDX_GCH3OHI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0) * ab[IDX_GCH3OHI];
    ab[IDX_GCH4I] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0) * ab[IDX_GCH4I];
    ab[IDX_GCOI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 16.0 * rptr[IDX_ELEM_O] / 28.0) * ab[IDX_GCOI];
    ab[IDX_GCO2I] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 32.0 * rptr[IDX_ELEM_O] / 44.0) * ab[IDX_GCO2I];
    ab[IDX_GH2CNI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0 + 14.0 * rptr[IDX_ELEM_N] / 28.0) * ab[IDX_GH2CNI];
    ab[IDX_GH2COI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0) * ab[IDX_GH2COI];
    ab[IDX_GH2OI] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0 + 16.0 * rptr[IDX_ELEM_O] / 18.0) * ab[IDX_GH2OI];
    ab[IDX_GH2SiOI] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 28.0 * rptr[IDX_ELEM_SI] / 46.0) * ab[IDX_GH2SiOI];
    ab[IDX_GHCNI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0) * ab[IDX_GHCNI];
    ab[IDX_GHNCI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0) * ab[IDX_GHNCI];
    ab[IDX_GHNCOI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0 + 14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0) * ab[IDX_GHNCOI];
    ab[IDX_GHNOI] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0 + 14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0) * ab[IDX_GHNOI];
    ab[IDX_GMgI] = (0.0 + 24.0 * rptr[IDX_ELEM_MG] / 24.0) * ab[IDX_GMgI];
    ab[IDX_GN2I] = (0.0 + 28.0 * rptr[IDX_ELEM_N] / 28.0) * ab[IDX_GN2I];
    ab[IDX_GNH3I] = (0.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0 + 14.0 * rptr[IDX_ELEM_N] / 17.0) * ab[IDX_GNH3I];
    ab[IDX_GNOI] = (0.0 + 14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0) * ab[IDX_GNOI];
    ab[IDX_GNO2I] = (0.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0) * ab[IDX_GNO2I];
    ab[IDX_GO2I] = (0.0 + 32.0 * rptr[IDX_ELEM_O] / 32.0) * ab[IDX_GO2I];
    ab[IDX_GO2HI] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0 + 32.0 * rptr[IDX_ELEM_O] / 33.0) * ab[IDX_GO2HI];
    ab[IDX_GSiCI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0 + 28.0 * rptr[IDX_ELEM_SI] / 40.0) * ab[IDX_GSiCI];
    ab[IDX_GSiC2I] = (0.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0 + 28.0 * rptr[IDX_ELEM_SI] / 52.0) * ab[IDX_GSiC2I];
    ab[IDX_GSiC3I] = (0.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0 + 28.0 * rptr[IDX_ELEM_SI] / 64.0) * ab[IDX_GSiC3I];
    ab[IDX_GSiH4I] = (0.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0 + 28.0 * rptr[IDX_ELEM_SI] / 32.0) * ab[IDX_GSiH4I];
    ab[IDX_GSiOI] = (0.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0 + 28.0 * rptr[IDX_ELEM_SI] / 44.0) * ab[IDX_GSiOI];
    ab[IDX_CI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 12.0) * ab[IDX_CI];
    ab[IDX_CII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 12.0) * ab[IDX_CII];
    ab[IDX_CHI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0) * ab[IDX_CHI];
    ab[IDX_CHII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 13.0 + 1.0 * rptr[IDX_ELEM_H] / 13.0) * ab[IDX_CHII];
    ab[IDX_CH2I] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0) * ab[IDX_CH2I];
    ab[IDX_CH2II] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 14.0 + 2.0 * rptr[IDX_ELEM_H] / 14.0) * ab[IDX_CH2II];
    ab[IDX_CH3I] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0) * ab[IDX_CH3I];
    ab[IDX_CH3II] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 15.0 + 3.0 * rptr[IDX_ELEM_H] / 15.0) * ab[IDX_CH3II];
    ab[IDX_CH3OHI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 32.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0) * ab[IDX_CH3OHI];
    ab[IDX_CH4I] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0) * ab[IDX_CH4I];
    ab[IDX_CH4II] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 16.0 + 4.0 * rptr[IDX_ELEM_H] / 16.0) * ab[IDX_CH4II];
    ab[IDX_CNI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0 + 14.0 * rptr[IDX_ELEM_N] / 26.0) * ab[IDX_CNI];
    ab[IDX_CNII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 26.0 + 14.0 * rptr[IDX_ELEM_N] / 26.0) * ab[IDX_CNII];
    ab[IDX_COI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 16.0 * rptr[IDX_ELEM_O] / 28.0) * ab[IDX_COI];
    ab[IDX_COII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 16.0 * rptr[IDX_ELEM_O] / 28.0) * ab[IDX_COII];
    ab[IDX_CO2I] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 44.0 + 32.0 * rptr[IDX_ELEM_O] / 44.0) * ab[IDX_CO2I];
    ab[IDX_EM] = (1.0) * ab[IDX_EM];
    ab[IDX_HI] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 1.0) * ab[IDX_HI];
    ab[IDX_HII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 1.0) * ab[IDX_HII];
    ab[IDX_H2I] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 2.0) * ab[IDX_H2I];
    ab[IDX_H2II] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 2.0) * ab[IDX_H2II];
    ab[IDX_H2CNI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0 + 14.0 * rptr[IDX_ELEM_N] / 28.0) * ab[IDX_H2CNI];
    ab[IDX_H2COI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0) * ab[IDX_H2COI];
    ab[IDX_H2COII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 30.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0) * ab[IDX_H2COII];
    ab[IDX_H2NOII] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 32.0 + 14.0 * rptr[IDX_ELEM_N] / 32.0 + 16.0 * rptr[IDX_ELEM_O] / 32.0) * ab[IDX_H2NOII];
    ab[IDX_H2OI] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0 + 16.0 * rptr[IDX_ELEM_O] / 18.0) * ab[IDX_H2OI];
    ab[IDX_H2OII] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 18.0 + 16.0 * rptr[IDX_ELEM_O] / 18.0) * ab[IDX_H2OII];
    ab[IDX_H2SiOI] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 46.0 + 16.0 * rptr[IDX_ELEM_O] / 46.0 + 28.0 * rptr[IDX_ELEM_SI] / 46.0) * ab[IDX_H2SiOI];
    ab[IDX_H3II] = (0.0 + 3.0 * rptr[IDX_ELEM_H] / 3.0) * ab[IDX_H3II];
    ab[IDX_H3COII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 31.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0) * ab[IDX_H3COII];
    ab[IDX_H3OII] = (0.0 + 3.0 * rptr[IDX_ELEM_H] / 19.0 + 16.0 * rptr[IDX_ELEM_O] / 19.0) * ab[IDX_H3OII];
    ab[IDX_HCNI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0) * ab[IDX_HCNI];
    ab[IDX_HCNII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0) * ab[IDX_HCNII];
    ab[IDX_HCNHII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 28.0 + 2.0 * rptr[IDX_ELEM_H] / 28.0 + 14.0 * rptr[IDX_ELEM_N] / 28.0) * ab[IDX_HCNHII];
    ab[IDX_HCOI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0) * ab[IDX_HCOI];
    ab[IDX_HCOII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0) * ab[IDX_HCOII];
    ab[IDX_HCO2II] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 45.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0 + 32.0 * rptr[IDX_ELEM_O] / 45.0) * ab[IDX_HCO2II];
    ab[IDX_HeI] = (0.0 + 4.0 * rptr[IDX_ELEM_HE] / 4.0) * ab[IDX_HeI];
    ab[IDX_HeII] = (0.0 + 4.0 * rptr[IDX_ELEM_HE] / 4.0) * ab[IDX_HeII];
    ab[IDX_HeHII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 5.0 + 4.0 * rptr[IDX_ELEM_HE] / 5.0) * ab[IDX_HeHII];
    ab[IDX_HNCI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 27.0 + 1.0 * rptr[IDX_ELEM_H] / 27.0 + 14.0 * rptr[IDX_ELEM_N] / 27.0) * ab[IDX_HNCI];
    ab[IDX_HNCOI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 43.0 + 1.0 * rptr[IDX_ELEM_H] / 43.0 + 14.0 * rptr[IDX_ELEM_N] / 43.0 + 16.0 * rptr[IDX_ELEM_O] / 43.0) * ab[IDX_HNCOI];
    ab[IDX_HNOI] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0 + 14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0) * ab[IDX_HNOI];
    ab[IDX_HNOII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 31.0 + 14.0 * rptr[IDX_ELEM_N] / 31.0 + 16.0 * rptr[IDX_ELEM_O] / 31.0) * ab[IDX_HNOII];
    ab[IDX_HOCII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 29.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0 + 16.0 * rptr[IDX_ELEM_O] / 29.0) * ab[IDX_HOCII];
    ab[IDX_MgI] = (0.0 + 24.0 * rptr[IDX_ELEM_MG] / 24.0) * ab[IDX_MgI];
    ab[IDX_MgII] = (0.0 + 24.0 * rptr[IDX_ELEM_MG] / 24.0) * ab[IDX_MgII];
    ab[IDX_NI] = (0.0 + 14.0 * rptr[IDX_ELEM_N] / 14.0) * ab[IDX_NI];
    ab[IDX_NII] = (0.0 + 14.0 * rptr[IDX_ELEM_N] / 14.0) * ab[IDX_NII];
    ab[IDX_N2I] = (0.0 + 28.0 * rptr[IDX_ELEM_N] / 28.0) * ab[IDX_N2I];
    ab[IDX_N2II] = (0.0 + 28.0 * rptr[IDX_ELEM_N] / 28.0) * ab[IDX_N2II];
    ab[IDX_N2HII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0 + 28.0 * rptr[IDX_ELEM_N] / 29.0) * ab[IDX_N2HII];
    ab[IDX_NHI] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0 + 14.0 * rptr[IDX_ELEM_N] / 15.0) * ab[IDX_NHI];
    ab[IDX_NHII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 15.0 + 14.0 * rptr[IDX_ELEM_N] / 15.0) * ab[IDX_NHII];
    ab[IDX_NH2I] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0 + 14.0 * rptr[IDX_ELEM_N] / 16.0) * ab[IDX_NH2I];
    ab[IDX_NH2II] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 16.0 + 14.0 * rptr[IDX_ELEM_N] / 16.0) * ab[IDX_NH2II];
    ab[IDX_NH3I] = (0.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0 + 14.0 * rptr[IDX_ELEM_N] / 17.0) * ab[IDX_NH3I];
    ab[IDX_NH3II] = (0.0 + 3.0 * rptr[IDX_ELEM_H] / 17.0 + 14.0 * rptr[IDX_ELEM_N] / 17.0) * ab[IDX_NH3II];
    ab[IDX_NOI] = (0.0 + 14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0) * ab[IDX_NOI];
    ab[IDX_NOII] = (0.0 + 14.0 * rptr[IDX_ELEM_N] / 30.0 + 16.0 * rptr[IDX_ELEM_O] / 30.0) * ab[IDX_NOII];
    ab[IDX_NO2I] = (0.0 + 14.0 * rptr[IDX_ELEM_N] / 46.0 + 32.0 * rptr[IDX_ELEM_O] / 46.0) * ab[IDX_NO2I];
    ab[IDX_OI] = (0.0 + 16.0 * rptr[IDX_ELEM_O] / 16.0) * ab[IDX_OI];
    ab[IDX_OII] = (0.0 + 16.0 * rptr[IDX_ELEM_O] / 16.0) * ab[IDX_OII];
    ab[IDX_O2I] = (0.0 + 32.0 * rptr[IDX_ELEM_O] / 32.0) * ab[IDX_O2I];
    ab[IDX_O2II] = (0.0 + 32.0 * rptr[IDX_ELEM_O] / 32.0) * ab[IDX_O2II];
    ab[IDX_O2HI] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0 + 32.0 * rptr[IDX_ELEM_O] / 33.0) * ab[IDX_O2HI];
    ab[IDX_O2HII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 33.0 + 32.0 * rptr[IDX_ELEM_O] / 33.0) * ab[IDX_O2HII];
    ab[IDX_OCNI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 42.0 + 14.0 * rptr[IDX_ELEM_N] / 42.0 + 16.0 * rptr[IDX_ELEM_O] / 42.0) * ab[IDX_OCNI];
    ab[IDX_OHI] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0 + 16.0 * rptr[IDX_ELEM_O] / 17.0) * ab[IDX_OHI];
    ab[IDX_OHII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 17.0 + 16.0 * rptr[IDX_ELEM_O] / 17.0) * ab[IDX_OHII];
    ab[IDX_SiI] = (0.0 + 28.0 * rptr[IDX_ELEM_SI] / 28.0) * ab[IDX_SiI];
    ab[IDX_SiII] = (0.0 + 28.0 * rptr[IDX_ELEM_SI] / 28.0) * ab[IDX_SiII];
    ab[IDX_SiCI] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0 + 28.0 * rptr[IDX_ELEM_SI] / 40.0) * ab[IDX_SiCI];
    ab[IDX_SiCII] = (0.0 + 12.0 * rptr[IDX_ELEM_C] / 40.0 + 28.0 * rptr[IDX_ELEM_SI] / 40.0) * ab[IDX_SiCII];
    ab[IDX_SiC2I] = (0.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0 + 28.0 * rptr[IDX_ELEM_SI] / 52.0) * ab[IDX_SiC2I];
    ab[IDX_SiC2II] = (0.0 + 24.0 * rptr[IDX_ELEM_C] / 52.0 + 28.0 * rptr[IDX_ELEM_SI] / 52.0) * ab[IDX_SiC2II];
    ab[IDX_SiC3I] = (0.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0 + 28.0 * rptr[IDX_ELEM_SI] / 64.0) * ab[IDX_SiC3I];
    ab[IDX_SiC3II] = (0.0 + 36.0 * rptr[IDX_ELEM_C] / 64.0 + 28.0 * rptr[IDX_ELEM_SI] / 64.0) * ab[IDX_SiC3II];
    ab[IDX_SiHI] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0 + 28.0 * rptr[IDX_ELEM_SI] / 29.0) * ab[IDX_SiHI];
    ab[IDX_SiHII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 29.0 + 28.0 * rptr[IDX_ELEM_SI] / 29.0) * ab[IDX_SiHII];
    ab[IDX_SiH2I] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0 + 28.0 * rptr[IDX_ELEM_SI] / 30.0) * ab[IDX_SiH2I];
    ab[IDX_SiH2II] = (0.0 + 2.0 * rptr[IDX_ELEM_H] / 30.0 + 28.0 * rptr[IDX_ELEM_SI] / 30.0) * ab[IDX_SiH2II];
    ab[IDX_SiH3I] = (0.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0 + 28.0 * rptr[IDX_ELEM_SI] / 31.0) * ab[IDX_SiH3I];
    ab[IDX_SiH3II] = (0.0 + 3.0 * rptr[IDX_ELEM_H] / 31.0 + 28.0 * rptr[IDX_ELEM_SI] / 31.0) * ab[IDX_SiH3II];
    ab[IDX_SiH4I] = (0.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0 + 28.0 * rptr[IDX_ELEM_SI] / 32.0) * ab[IDX_SiH4I];
    ab[IDX_SiH4II] = (0.0 + 4.0 * rptr[IDX_ELEM_H] / 32.0 + 28.0 * rptr[IDX_ELEM_SI] / 32.0) * ab[IDX_SiH4II];
    ab[IDX_SiH5II] = (0.0 + 5.0 * rptr[IDX_ELEM_H] / 33.0 + 28.0 * rptr[IDX_ELEM_SI] / 33.0) * ab[IDX_SiH5II];
    ab[IDX_SiOI] = (0.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0 + 28.0 * rptr[IDX_ELEM_SI] / 44.0) * ab[IDX_SiOI];
    ab[IDX_SiOII] = (0.0 + 16.0 * rptr[IDX_ELEM_O] / 44.0 + 28.0 * rptr[IDX_ELEM_SI] / 44.0) * ab[IDX_SiOII];
    ab[IDX_SiOHII] = (0.0 + 1.0 * rptr[IDX_ELEM_H] / 45.0 + 16.0 * rptr[IDX_ELEM_O] / 45.0 + 28.0 * rptr[IDX_ELEM_SI] / 45.0) * ab[IDX_SiOHII];
    
    // clang-format on

    N_VDestroy(b);
    N_VDestroy(r);
    SUNMatDestroy(A);
    SUNLinSolFree(LS);

    return NAUNET_SUCCESS;
};
#endif

// To reset the size of cusparse solver
int Naunet::Reset(int nsystem, double atol, double rtol, int mxsteps) {
    n_system_ = nsystem;
    mxsteps_  = mxsteps;
    atol_     = atol;
    rtol_     = rtol;

    /* */
    if (nsystem != 1) {
        printf("This solver doesn't support nsystem > 1!");
        return NAUNET_FAIL;
    }

    // N_VFreeEmpty(cv_y_);
    N_VDestroy(cv_y_);
    SUNMatDestroy(cv_a_);
    SUNLinSolFree(cv_ls_);

    cv_y_            = N_VNewEmpty_Serial((sunindextype)NEQUATIONS);
    cv_a_            = SUNSparseMatrix(NEQUATIONS, NEQUATIONS, NNZ, CSR_MAT);
    cv_ls_           = SUNLinSol_KLU(cv_y_, cv_a_);

    /*  */

    return NAUNET_SUCCESS;
};

#ifdef IDX_ELEM_H
int Naunet::SetReferenceAbund(realtype *ref, int opt) {
    if (opt == 0) {
        for (int i = 0; i < NELEMENTS; i++) {
            ab_ref_[i] = ref[i] / ref[IDX_ELEM_H];
        }
    } else if (opt == 1) {
        double Hnuclei = GetHNuclei(ref);
        for (int i = 0; i < NELEMENTS; i++) {
            ab_ref_[i] = GetElementAbund(ref, i) / Hnuclei;
        }
    }

    return NAUNET_SUCCESS;
}
#endif

int Naunet::Solve(realtype *ab, realtype dt, NaunetData *data) {
    int cvflag;
    realtype t0 = 0.0;

    /* */

    for (int i = 0; i < NEQUATIONS; i++) {
        ab_init_[i] = ab[i];
        ab_tmp_[i]  = ab[i];
    }

    // realtype *ydata = N_VGetArrayPointer(cv_y_);
    // for (int i=0; i<NEQUATIONS; i++)
    // {
    //     ydata[i] = ab[i];
    // }
    N_VSetArrayPointer(ab, cv_y_);

    cv_mem_ = CVodeCreate(CV_BDF);

    cvflag  = CVodeSetErrFile(cv_mem_, errfp_);
    if (CheckFlag(&cvflag, "CVodeSetErrFile", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSetMaxNumSteps(cv_mem_, mxsteps_);
    if (CheckFlag(&cvflag, "CVodeSetMaxNumSteps", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeInit(cv_mem_, Fex, t0, cv_y_);
    if (CheckFlag(&cvflag, "CVodeInit", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSStolerances(cv_mem_, rtol_, atol_);
    if (CheckFlag(&cvflag, "CVodeSStolerances", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSetLinearSolver(cv_mem_, cv_ls_, cv_a_);
    if (CheckFlag(&cvflag, "CVodeSetLinearSolver", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSetJacFn(cv_mem_, Jac);
    if (CheckFlag(&cvflag, "CVodeSetJacFn", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag = CVodeSetUserData(cv_mem_, data);
    if (CheckFlag(&cvflag, "CVodeSetUserData", 1, errfp_) == NAUNET_FAIL) {
        return NAUNET_FAIL;
    }

    cvflag   = CVode(cv_mem_, dt, cv_y_, &t0, CV_NORMAL);

    // ab   = N_VGetArrayPointer(cv_y_);

    int flag = HandleError(cvflag, ab, dt, t0);

    if (flag == NAUNET_FAIL) {
        fprintf(errfp_, "Some unrecoverable error occurred. cvFlag = %d\n",
                cvflag);
        fprintf(errfp_, "Initial condition: \n");

        /* */
        fprintf(errfp_, "    data.nH = %13.7e;\n", data->nH);
        /* */
        fprintf(errfp_, "    data.Tgas = %13.7e;\n", data->Tgas);
        /* */
        fprintf(errfp_, "    data.zeta = %13.7e;\n", data->zeta);
        /* */
        fprintf(errfp_, "    data.Av = %13.7e;\n", data->Av);
        /* */
        fprintf(errfp_, "    data.omega = %13.7e;\n", data->omega);
        /* */
        fprintf(errfp_, "    data.G0 = %13.7e;\n", data->G0);
        /* */
        fprintf(errfp_, "    data.uvcreff = %13.7e;\n", data->uvcreff);
        /* */
        fprintf(errfp_, "    data.rG = %13.7e;\n", data->rG);
        /* */
        fprintf(errfp_, "    data.gdens = %13.7e;\n", data->gdens);
        /* */
        fprintf(errfp_, "    data.sites = %13.7e;\n", data->sites);
        /* */
        fprintf(errfp_, "    data.fr = %13.7e;\n", data->fr);
        /* */
        fprintf(errfp_, "    data.opt_thd = %13.7e;\n", data->opt_thd);
        /* */
        fprintf(errfp_, "    data.opt_crd = %13.7e;\n", data->opt_crd);
        /* */
        fprintf(errfp_, "    data.opt_h2d = %13.7e;\n", data->opt_h2d);
        /* */
        fprintf(errfp_, "    data.opt_uvd = %13.7e;\n", data->opt_uvd);
        /* */
        fprintf(errfp_, "    data.eb_h2d = %13.7e;\n", data->eb_h2d);
        /* */
        fprintf(errfp_, "    data.eb_crd = %13.7e;\n", data->eb_crd);
        /* */
        fprintf(errfp_, "    data.eb_uvd = %13.7e;\n", data->eb_uvd);
        /* */
        fprintf(errfp_, "    data.crdeseff = %13.7e;\n", data->crdeseff);
        /* */
        fprintf(errfp_, "    data.h2deseff = %13.7e;\n", data->h2deseff);
        /* */
        fprintf(errfp_, "    data.ksp = %13.7e;\n", data->ksp);
        /*  */

        fprintf(errfp_, "\n");

        realtype spy = 365.0 * 86400.0;

        fprintf(errfp_, "    dtyr = %13.7e;\n", dt / spy);
        fprintf(errfp_, "\n");

        for (int i = 0; i < NEQUATIONS; i++) {
            fprintf(errfp_, "    y[%d] = %13.7e;\n", i, ab_init_[i]);
        }

        for (int i = 0; i < NEQUATIONS; i++) {
            fprintf(errfp_, "    y_final[%d] = %13.7e;\n", i, ab[i]);
        }
    }

    CVodeFree(&cv_mem_);

    return flag;

    /* */
};

#ifdef PYMODULE
py::array_t<realtype> Naunet::PyWrapSolve(py::array_t<realtype> arr,
                                          realtype dt, NaunetData *data) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    Solve(ab, dt, data);

    return py::array_t<realtype>(info.shape, ab);
}
#endif