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

#define TAG_TTY 1
#define TAG_TTA 2
#define TAG_VAR 3
#define TAG_IND1 4
#define TAG_IND2 5
#define TAG_LOG_LEN 6
#define TAG_LOG 7


using namespace std;

int GetProcId();
int GetNumProc();
time_t GetCurrentTime();

const string& GetParamFileName();

// Synchronized logging
void log(const string& str);

#endif /* COMMON_H_ */
