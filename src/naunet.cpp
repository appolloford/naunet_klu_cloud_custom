#include <algorithm>
#include <cvode/cvode.h>  // prototypes for CVODE fcts., consts.
/* */
#include <nvector/nvector_serial.h>      // access to serial N_Vector
#include <sunlinsol/sunlinsol_klu.h>     // access to KLU sparse direct solver
#include <sunmatrix/sunmatrix_sparse.h>  // access to sparse SUNMatrix
/* */
/*  */
#include "naunet.h"
/*  */
#include "naunet_ode.h"
/* */
#include "naunet_constants.h"
#include "naunet_utilities.h"
/* */

// check_flag function is from the cvDiurnals_ky.c example from the CVODE
// package. Check function return value...
//   opt == 0 means SUNDIALS function allocates memory so check if
//            returned NULL pointer
//   opt == 1 means SUNDIALS function returns a flag so check if
//            flag >= 0
//   opt == 2 means function allocates memory so check if returned
//            NULL pointer
static int check_flag(void *flagvalue, const char *funcname, int opt) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr,
                "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return 1;
    }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *)flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return 1;
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr,
                "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return 1;
    }

    return 0;
}

Naunet::Naunet(){};

Naunet::~Naunet(){};

int Naunet::Init(int nsystem, double atol, double rtol, int mxsteps) {
    n_system_ = nsystem;
    mxsteps_  = mxsteps;
    atol_     = atol;
    rtol_     = rtol;

    /* */
    if (nsystem != 1) {
        printf("This solver doesn't support nsystem > 1!");
        return NAUNET_FAIL;
    }

    cv_y_  = N_VNewEmpty_Serial((sunindextype)NEQUATIONS);
    cv_a_  = SUNSparseMatrix(NEQUATIONS, NEQUATIONS, NNZ, CSR_MAT);
    cv_ls_ = SUNLinSol_KLU(cv_y_, cv_a_);

    cv_mem_ = CVodeCreate(CV_BDF);

    errfp_ = fopen("naunet_error_record.txt", "a");

    int flag;

    flag = CVodeSetErrFile(cv_mem_, errfp_);
    if (check_flag(&flag, "CVodeSetErrFile", 1)) return 1;
    // flag = CVodeInit(cv_mem_, Fex, 0.0, cv_y_);
    // if (check_flag(&flag, "CVodeInit", 1)) return 1;
    flag = CVodeSetMaxNumSteps(cv_mem_, mxsteps_);
    if (check_flag(&flag, "CVodeSetMaxNumSteps", 0)) return 1;
    // flag = CVodeSStolerances(cv_mem_, rtol_, atol_);
    // if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
    // flag = CVodeSetLinearSolver(cv_mem_, cv_ls_, cv_a_);
    // if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return 1;
    // flag = CVodeSetJacFn(cv_mem_, Jac);
    // if (check_flag(&flag, "CVodeSetJacFn", 1)) return 1;

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

    // splie2(COShieldingTableX, COShieldingTableY, COShieldingTable, 6, 7, COShieldingTableD2);

    /* */

    return NAUNET_SUCCESS;
};

int Naunet::DebugInfo() {
    long int nst, nfe, nsetups, nje, netf, nge, nni, ncfn;
    int flag;

    /* */

    flag = CVodeGetNumSteps(cv_mem_, &nst);
    check_flag(&flag, "CVodeGetNumSteps", 1);
    flag = CVodeGetNumRhsEvals(cv_mem_, &nfe);
    check_flag(&flag, "CVodeGetNumRhsEvals", 1);
    flag = CVodeGetNumLinSolvSetups(cv_mem_, &nsetups);
    check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
    flag = CVodeGetNumErrTestFails(cv_mem_, &netf);
    check_flag(&flag, "CVodeGetNumErrTestFails", 1);
    flag = CVodeGetNumNonlinSolvIters(cv_mem_, &nni);
    check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
    flag = CVodeGetNumNonlinSolvConvFails(cv_mem_, &ncfn);
    check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

    flag = CVodeGetNumJacEvals(cv_mem_, &nje);
    check_flag(&flag, "CVodeGetNumJacEvals", 1);

    flag = CVodeGetNumGEvals(cv_mem_, &nge);
    check_flag(&flag, "CVodeGetNumGEvals", 1);

    printf("\nFinal Statistics:\n");
    printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nje = %ld\n", nst, nfe, nsetups, nje);
    printf("nni = %-6ld ncfn = %-6ld netf = %-6ld    nge = %ld\n \n", nni, ncfn, netf, nge);

    /*  */

    return NAUNET_SUCCESS;
}

int Naunet::Finalize() {

    /* */

    N_VDestroy(cv_y_);
    // N_VFreeEmpty(cv_y_);
    CVodeFree(&cv_mem_);
    SUNMatDestroy(cv_a_);
    SUNLinSolFree(cv_ls_);
    // delete m_data;
    fclose(errfp_);

    /*  */

    return NAUNET_SUCCESS;
};

/*  */

// int Naunet::RecursiveSolve(realtype dt, NaunetData *data) {
//     realtype t0 = 0.0;
    
//     N_VSetArrayPointer(ab, cv_y_);
//     flag = CVodeReInit(cv_mem_, t0, cv_y_);
//     if (check_flag(&flag, "CVodeReInit", 1)) return 1;

//     for (int i=0; i<10; i++) {
//         realtype tstep = pow(10.0, 0.1* log10(dt) * (i+1));
//         flag = CVode(cv_mem_, tstep, cv_y_, &t0, CV_NORMAL);

//         if(flag == -3 || flag == -4) {

//         }
//     }
// }

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

    N_VFreeEmpty(cv_y_);
    SUNMatDestroy(cv_a_);
    SUNLinSolFree(cv_ls_);
    CVodeFree(&cv_mem_);

    cv_y_  = N_VNew_Serial((sunindextype)NEQUATIONS);
    cv_a_  = SUNSparseMatrix(NEQUATIONS, NEQUATIONS, NNZ, CSR_MAT);
    cv_ls_ = SUNLinSol_KLU(cv_y_, cv_a_);

    cv_mem_ = CVodeCreate(CV_BDF);

    int flag;

    flag = CVodeSetErrFile(cv_mem_, errfp_);
    if (check_flag(&flag, "CVodeSetErrFile", 1)) return 1;
    flag = CVodeInit(cv_mem_, Fex, 0.0, cv_y_);
    if (check_flag(&flag, "CVodeInit", 1)) return 1;
    flag = CVodeSetMaxNumSteps(cv_mem_, mxsteps_);
    if (check_flag(&flag, "CVodeSetMaxNumSteps", 0)) return 1;
    flag = CVodeSStolerances(cv_mem_, rtol_, atol_);
    if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
    flag = CVodeSetLinearSolver(cv_mem_, cv_ls_, cv_a_);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return 1;
    flag = CVodeSetJacFn(cv_mem_, Jac);
    if (check_flag(&flag, "CVodeSetJacFn", 1)) return 1;

    return NAUNET_SUCCESS;
};

int Naunet::Solve(realtype *ab, realtype dt, NaunetData *data) {

    int flag = 0;

    /* */

    // realtype *ydata = N_VGetArrayPointer(cv_y_);
    // for (int i=0; i<NEQUATIONS; i++)
    // {
    //     ydata[i] = ab[i];
    // }

    realtype ab_init[NEQUATIONS];
    realtype ab_tmp[NEQUATIONS]; // Temporary state for error handling
    for (int i=0; i<NEQUATIONS; i++)
    {
        ab_init[i] = ab[i];
        ab_tmp[i] = ab[i];
    }

    N_VSetArrayPointer(ab, cv_y_);

    flag = CVodeInit(cv_mem_, Fex, 0.0, cv_y_);
    if (check_flag(&flag, "CVodeInit", 1)) return 1;
    flag = CVodeSStolerances(cv_mem_, rtol_, atol_);
    if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
    flag = CVodeSetLinearSolver(cv_mem_, cv_ls_, cv_a_);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return 1;
    flag = CVodeSetJacFn(cv_mem_, Jac);
    if (check_flag(&flag, "CVodeSetJacFn", 1)) return 1;

    flag = CVodeSetUserData(cv_mem_, data);
    if (check_flag(&flag, "CVodeSetUserData", 1)) return 1;

    realtype t0 = 0.0;


    for (int level=0; level<6; level++) {

        int nsubsteps = 1;

        // flag is 0 (CV_SUCCESS) in the first iteration
        // modify some parameters if the first iteration failed
        if (flag == -1) {
            // if error handling has been tried once, continue with the current time and state
            if (level > 1) {
                for (int i=0; i<NEQUATIONS; i++) {
                    ab_tmp[i] = ab[i];
                }
                dt -= t0;
            }
            // in case h = hmin, use smaller timesteps
            nsubsteps = level * 10;
            // flag = CVodeSetMaxNumSteps(cv_mem_, (level+1) * mxsteps_);
            // if (check_flag(&flag, "CVodeSetMaxNumSteps", 0)) return 1;
        }
        else if (flag == -2) {
            flag = CVodeSStolerances(cv_mem_, rtol_, pow(10.0, level) * atol_);
            if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
        }
        else if (flag == -3) {
            // continue with the current time and state
            for (int i=0; i<NEQUATIONS; i++) {
                ab_tmp[i] = ab[i];
            }
            dt -= t0;
        }
        else if (flag == -4) {
            // if error handling has been tried once, continue with the current time and state
            if (level > 1) {
                for (int i=0; i<NEQUATIONS; i++) {
                    ab_tmp[i] = ab[i];
                }
                dt -= t0;
            }
            // in case h = hmin, use smaller timesteps
            nsubsteps = level * 10;
        }
        else if (flag == -6) {
            // The state may have something wrong
            // Reset to the initial state and try finer steps
            for (int i=0; i<NEQUATIONS; i++) {
                ab_tmp[i] = ab_init[i];
            }
            nsubsteps = level * 10;
        }
        else if (flag < 0) {
            fprintf(errfp_, "The error cannot be recovered by Naunet! Exit from Naunet!\n");
            fprintf(errfp_, "Flag = %d, level = %d\n", flag, level);
            break;
        }

        // Restore initial conditions
        for (int i=0; i<NEQUATIONS; i++) {
            ab[i] = ab_tmp[i];
        }

        // Reinitialize
        t0 = 0.0;
        flag = CVodeReInit(cv_mem_, t0, cv_y_);
        if (check_flag(&flag, "CVodeReInit", 1)) return 1;

        for (int step = 0; step < nsubsteps; step ++) {
            realtype tout = pow(10.0, log10(dt) * (realtype)(step+1) / (realtype)nsubsteps);
            // printf("tout: %13.7e, step: %d, level: %d\n", tout, step, level);
            // realtype tcur = 0.0;
            // flag = CVodeGetCurrentTime(cv_mem_, &tcur);
            flag = CVode(cv_mem_, tout, cv_y_, &t0, CV_NORMAL);
            if (flag < 0) {
                fprintf(errfp_, "CVode failed in Naunet! Flag = %d in the %dth substep of %dth level! \n", flag, step, level);
                if (level < 5) {
                    fprintf(errfp_, "Trying to fix the error in the next level\n");
                }
                // fprintf(errfp_, "Failed to fix the error! flag = %d in the %dth substep! \n", flag, i);
                break;
            }
        }

        // if CVode succeeded, leave the loop
        if (flag >= 0) {
            if (level > 0) {
                fprintf(errfp_, "The error was successfully fixed in %dth level\n", level);
            }
            break;
        }

    }

    if (flag < 0) {
        fprintf(errfp_, "Some unrecoverable error occurred. Flag = %d\n", flag);
    }


    // ab   = N_VGetArrayPointer(cv_y_);
    // for (int i=0; i<NEQUATIONS; i++)
    // {
    //     ab[i] = ydata[i];
    // }



    /* */

    return flag;
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
