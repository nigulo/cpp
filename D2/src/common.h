/*
 * common.h
 *
 *  Created on: May 4, 2016
 *      Author: nigul
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <ctime>
#include <string>

#ifndef _NOMPI
#include "mpi.h"
#endif
#ifdef _OPENACC
#include <openacc.h>
#else
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

#define DIFF_NORMS_FILE_PREFIX "diffnorms"
#define DIFF_NORMS_FILE_SUFFIX ".csv"
#define DIFF_NORMS_FILE (DIFF_NORMS_FILE_PREFIX DIFF_NORMS_FILE_SUFFIX)
#define PARAMETERS_FILE_PREFIX "parameters"
#define PARAMETERS_FILE_SUFFIX ".txt"
#define PARAMETERS_FILE (PARAMETERS_FILE_PREFIX PARAMETERS_FILE_SUFFIX)

using namespace std;

int GetProcId();
int GetNumProc();
time_t GetCurrentTime();

const string& GetParamFileName();

#endif /* COMMON_H_ */
