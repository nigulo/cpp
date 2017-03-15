/*
 * mpiutils.cpp
 *
 *  Created on: 1 Mar 2017
 *      Author: olspern1
 */

#include "mpiutils.h"
#ifdef _MPI
#include "mpi.h"
#endif

#include <iostream>
#include <cassert>

int procId;
int numProc;

void mpiInit(int argc, char *argv[]) {
#ifdef _MPI
	MPI::Init(argc, argv);
	numProc = MPI::COMM_WORLD.Get_size();
	procId = MPI::COMM_WORLD.Get_rank();
#else
	numProc = 1;
	procId = 0;
#endif

}

void mpiFinalize() {
#ifdef _MPI
	MPI::Finalize();
#endif
}

void mpiBarrier() {
#ifdef _MPI
	MPI::COMM_WORLD.Barrier();
#endif
}

int getProcId() {
	return procId;
}

int getNumProc() {
	return numProc;
}

void sendLog(const string& str) {
	if (getProcId() > 0) {
#ifdef _MPI
		int len = str.length() + 1;
		MPI::COMM_WORLD.Send(&len, 1, MPI::INT, 0, tagLogLen);
		MPI::COMM_WORLD.Send(str.c_str(), len, MPI::CHAR, 0, tagLog);
		MPI::COMM_WORLD.Barrier();
#endif
	} else {
		cout << "PROC" << getProcId() << ": " << str;
	}
}

void recvLog() {
#ifdef _MPI
	if (getProcId() == 0) {
		for (int i = 1; i < getNumProc(); i++) {
			int len;
			MPI::Status status;
			MPI::COMM_WORLD.Recv(&len, 1,  MPI::INT, MPI_ANY_SOURCE, tagLogLen, status);
			assert(status.Get_error() == MPI::SUCCESS);
			char strRecv[len];
			int source = status.Get_source();
			MPI::COMM_WORLD.Recv(strRecv, len,  MPI::CHAR, source, tagLog, status);
			assert(status.Get_error() == MPI::SUCCESS);
			cout << "PROC" << source << ": " << strRecv;
		}
		MPI::COMM_WORLD.Barrier();
	}
#endif
}


FileWriter::FileWriter(const string& fileName, int id) : id(id) {
	if (getProcId() == 0 && !fileName.empty()) {
		output = unique_ptr<std::ofstream>(new std::ofstream(fileName, ios_base::app));
	}
}

FileWriter::~FileWriter() {
}

void FileWriter::write(const string& str) {
	if (getProcId() > 0) {
#ifdef _MPI
		int len = str.length() + 1;
		MPI::COMM_WORLD.Send(&id, 1, MPI::INT, 0, tagId);
		MPI::COMM_WORLD.Send(&len, 1, MPI::INT, 0, tagStrLen);
		MPI::COMM_WORLD.Send(str.c_str(), len, MPI::CHAR, 0, tagStr);
		MPI::COMM_WORLD.Barrier();
#endif
	} else {
		if (str.length() > 0 && output.get()) {
			*output << str;
			output->flush();
		}
#ifdef _MPI
		for (int i = 1; i < getNumProc(); i++) {
			int idRecv;
			int len;
			MPI::Status status;
			MPI::COMM_WORLD.Recv(&idRecv, 1, MPI::INT, MPI_ANY_SOURCE, tagId, status);
			assert(id == idRecv);
			MPI::COMM_WORLD.Recv(&len, 1,  MPI::INT, MPI_ANY_SOURCE, tagStrLen, status);
			assert(status.Get_error() == MPI::SUCCESS);
			char strRecv[len];
			int source = status.Get_source();
			MPI::COMM_WORLD.Recv(strRecv, len,  MPI::CHAR, source, tagStr, status);
			assert(status.Get_error() == MPI::SUCCESS);
			if (len > 1 && output.get()) {
				*output << strRecv;
				output->flush();
			}
		}
		MPI::COMM_WORLD.Barrier();
#endif
	}
}
