/*
 * mpiutils.h
 *
 *  Created on: 1 Mar 2017
 *      Author: olspern1
 */

#ifndef MPIUTILS_H_
#define MPIUTILS_H_

#include <string>

using namespace std;

void mpiInit(int argc, char *argv[], int tagLogLen = 1, int tagLog = 2);
void mpiFinalize();

int getProcId();
int getNumProc();

// Synchronized logging
void sendLog(const string& str);
void recvLog();


#endif /* MPIUTILS_H_ */
