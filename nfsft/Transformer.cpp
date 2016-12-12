/*
 * Transformer.cpp
 *
 *  Created on: Nov 15, 2016
 *      Author: nigul
 */

#include <omp.h>
#include "Transformer.h"
#include <iostream>

Transformer::Transformer(int N /* bandwidth/maximum degree */, int M /* number of nodes */, bool doReconstruction, bool decompOrPower) :
	N(N),
	M(M),
	result_out("result.txt"),
	reconst_out(nullptr),
	decompOrPower(decompOrPower) {
	if (doReconstruction) {
		reconst_out = new ofstream("reconst.txt");
	}

}

Transformer::~Transformer() {
    result_out.close();
    if (reconst_out) {
		reconst_out->close();
		delete reconst_out;
    }
}

void Transformer::init() {
	//printf("num_threads = %d\n", omp_get_max_threads());
    printf("nthreads = %d\n", nfft_get_num_threads());
    /* init */
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    /* precomputation (for fast polynomial transform) */
    nfsft_precompute(N, 1000.0, 0U, 0U);

    /* Initialize transform plan using the guru interface. All input and output
     * arrays are allocated by nfsft_init_guru(). Computations are performed with
     * respect to L^2-normalized spherical harmonics Y_k^n. The array of spherical
     * Fourier coefficients is preserved during transformations. The NFFT uses a
     * cut-off parameter m = 6. See the NFFT 3 manual for details.
     */
    nfsft_init(&plan, N, M);
    //nfsft_init_guru(&plan, N, M, NFSFT_MALLOC_X | NFSFT_MALLOC_F |
    //    NFSFT_MALLOC_F_HAT | NFSFT_NORMALIZED | NFSFT_PRESERVE_F_HAT,
    //    PRE_PHI_HUT | PRE_PSI | FFTW_INIT | FFT_OUT_OF_PLACE, 6);
}

void Transformer::finalize() {
	/* Finalize the plan. */
	nfsft_finalize(&plan);

	/* Destroy data precomputed for fast polynomial transform. */
	nfsft_forget();

}

void Transformer::transform(int timeMoment) {
    /* precomputation (for NFFT, node-dependent) */
	if (timeMoment <= 0) { // first time
		nfsft_precompute_x(&plan);
	}

    /* Direct adjoint transformation, display result. */
    nfsft_adjoint_direct(&plan);

    string timeMomentStr;
    if (timeMoment >= 0) {
    	timeMomentStr = to_string(timeMoment) + " ";
    }

    //ofstream decomp_out(string("decomp") + suffixStr + ".txt");
    for (int k = 0; k <= plan.N; k++) {
    	double power = 0;
        for (int n = -k; n <= k; n++) {
        	if (decompOrPower) {
        		result_out << timeMomentStr << k << " " << n << " " << plan.f_hat[NFSFT_INDEX(k, n, &plan)][0] << " " << plan.f_hat[NFSFT_INDEX(k, n, &plan)][1] << endl;
        	} else {
        		auto re = plan.f_hat[NFSFT_INDEX(k, n, &plan)][0];
        		auto im = plan.f_hat[NFSFT_INDEX(k, n, &plan)][1];
        		power += re * re + im * im;
        	}
        }
        power /= (2 * k + 1);
    	if (!decompOrPower) {
    		result_out << timeMomentStr << k << " " << power << endl;
    	}
    }
    result_out.flush();

	//#ifdef ONLY_AZIMUTHAL_RECONST
    //	// Set all nonazimuthal waves to zero
	//	for (int k = 0; k <= plan.N; k++) {
	//		for (int n = -k+1; n <= k-1; n++) {
	//			plan.f_hat[NFSFT_INDEX(k,n,&plan)][0] = 0;
	//			plan.f_hat[NFSFT_INDEX(k,n,&plan)][1] = 0;
	//		}
	//	}
	//#endif

    if (reconst_out) {
		/* Direct transformation, display result. */
		nfsft_trafo_direct(&plan);

		//ofstream reconst_out(string("reconst") + suffixStr + ".txt");
		for (int m = 0; m < plan.M_total; m++) {
			*reconst_out << timeMomentStr << plan.x[2*m] << " " << plan.x[2*m + 1] << " " << plan.f[m][0] << " " << plan.f[m][1] << endl;
		}
		reconst_out->flush();
    }

    if (timeMoment < 0) { // only single transform
    	finalize();
    }
}


void Transformer::setX(int m, double x1, double x2) {
    plan.x[2*m] = x1;
    plan.x[2*m+1] = x2;
}

void Transformer::setY(int m, double y) {
    plan.f[m][0] = y;
    plan.f[m][1] = 0;
}
