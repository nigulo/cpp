/*
 * HarmonicEnsemble.h
 *
 *  Created on: Sep 9, 2015
 *      Author: nigul
 */

#ifndef HARMONICENSEMBLE_H_
#define HARMONICENSEMBLE_H_

#include <random>

using namespace std;

class HarmonicEnsemble {
public:
	HarmonicEnsemble(size_t size,
			double freqMean, double freqStdDev,
			double ampMean, double ampStdDev,
			double durationMean, double durationStdDev) :
		size(size),
		freqMean(freqMean), freqStdDev(freqStdDev),
		ampMean(ampMean), ampStdDev(ampStdDev),
		durationMean(durationMean), durationStdDev(durationStdDev),
		gen(rd()) {
	}

	virtual ~HarmonicEnsemble();

	const size_t size;
	const double freqMean;
	const double freqStdDev;
	const double ampMean;
	const double ampStdDev;
	const double durationMean;
	const double durationStdDev;

private:

    random_device rd;
    mt19937 gen;

};



#endif /* HARMONICENSEMBLE_H_ */
