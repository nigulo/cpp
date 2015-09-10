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
			double ampMean, double ampStdDev,
			double durationMean,
			double durationStdDev,
			double timeStepMean,
			double timeStepStdDev) :
		size(size),
		time(0),
		gen(rd()),
		freqDist(freqMean, freqStdDev),
		ampDist(ampMean, ampStdDev),
		durationDist(durationMean, durationStdDev),
		phaseDist(0, 1),
		timeStepDist(timeStepMean, timeStepStdDev)
		{
	}

	virtual ~HarmonicEnsemble() {}
	pair<double, double> NextStep();


	const size_t size;

	double time;

private:

    random_device rd;
    mt19937 gen;

	normal_distribution<> freqDist;
	normal_distribution<> ampDist;
	normal_distribution<> durationDist;
	uniform_real_distribution<> phaseDist;
	normal_distribution<> timeStepDist;

    vector<pair<Harmonic, double /*endTime*/>> ensemble;

};



#endif /* HARMONICENSEMBLE_H_ */
