/*
 * Transformer.h
 *
 *  Created on: Nov 15, 2016
 *      Author: nigul
 */

#ifndef TRANSFORMER_H_
#define TRANSFORMER_H_

#include "nfft3.h" /* NFFT3 header */
#include <fstream>

using namespace std;

class Transformer {
public:
	Transformer(int N /* bandwidth/maximum degree */, int M /* number of nodes */, bool doReconstruction = true);
	virtual ~Transformer();

	void init();
	void transform(int timeMoment = -1);
	void finalize();

	void setX(int m, double x1, double x2);
	void setY(int m, double y);

	int getN() const {
		return N;
	}

	int getM() const {
		return M;
	}

private:
	const int N;
	const int M;
	nfsft_plan plan; /* transform plan */

	ofstream decomp_out;
	ofstream* reconst_out;
};

#endif /* TRANSFORMER_H_ */
