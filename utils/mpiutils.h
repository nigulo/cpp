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

enum MessageTag {
	tagLogLen = 1,
	tagLog = 2,
	tagStrLen = 3,
	tagStr = 4,
	tagId = 5
};


void mpiInit(int argc, char *argv[]);
void mpiFinalize();
void mpiBarrier();

int getProcId();
int getNumProc();

// Synchronized logging
void sendLog(const string& str);
void recvLog();


class FileWriter {
public:
	FileWriter(const string& fileName, int id = 0);
	virtual ~FileWriter();
	void write(const string& str = "");

private:
	const int id; // For double checking if the message sender and receiver are correct
	unique_ptr<std::ofstream> output;
};

#endif /* MPIUTILS_H_ */
