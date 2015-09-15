/*
 * HarmonicEnsemble.h
 *
 *  Created on: Sep 9, 2015
 *      Author: nigul
 */

#ifndef HARMONICENSEMBLE_H_
#define HARMONICENSEMBLE_H_

#include <random>
#include "Harmonic.h"

using namespace std;

class HarmonicEnsemble {
public:
	HarmonicEnsemble(size_t size,
			double freqMean, double freqStdDev,
			double ampMax, double ampMaxFreq,
			double durationMean,
			double durationStdDev,
			double timeStepMean,
			double timeStepStdDev) :
		size(size),
		time(0),
		gen(rd()),
		freqMean(freqMean),
		freqStdDev(freqStdDev),
		ampMax(ampMax),
		ampMaxFreq(ampMaxFreq),
		freqDist(freqMean, freqStdDev),
		durationDist(durationMean, durationStdDev),
		phaseDist(0, 1),
		timeStepDist(timeStepMean, timeStepStdDev)
		{
	}

	virtual ~HarmonicEnsemble() {}
	pair<double, double> NextStep();



private:

	const size_t size;

	double time;

    random_device rd;
    mt19937 gen;

	const double freqMean;
	const double freqStdDev;
	const double ampMax;
	const double ampMaxFreq;
	normal_distribution<> freqDist;
	normal_distribution<> durationDist;
	uniform_real_distribution<> phaseDist;
	normal_distribution<> timeStepDist;

    vector<pair<Harmonic, double /*endTime*/>> ensemble;

};



#endif /* HARMONICENSEMBLE_H_ */
