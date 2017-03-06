/*
 * mpiutils.h
 *
 *  Created on: 1 Mar 2017
 *      Author: olspern1
 */

#ifndef MPIUTILS_H_
#define MPIUTILS_H_

#include <string>
#include <memory>
#include <fstream>

using namespace std;

void mpiInit(int argc, char *argv[], int tagLogLen = 1, int tagLog = 2, int tagStrLen = 3, int tagStr = 4);
void mpiFinalize();
void mpiBarrier();

int getProcId();
int getNumProc();

// Synchronized logging
void sendLog(const string& str);
void recvLog();


class FileWriter {
public:
	FileWriter(const string& fileName = "");
	virtual ~FileWriter();
	void write(const string& str = "");

private:
	unique_ptr<std::ofstream> output;
};

#endif /* MPIUTILS_H_ */
