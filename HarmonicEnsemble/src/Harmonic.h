/*
 * Harmonic.h
 *
 *  Created on: Sep 9, 2015
 *      Author: nigul
 */

#ifndef HARMONIC_H_
#define HARMONIC_H_

#include <math.h>
#include <assert.h>

class Harmonic {
public:
	Harmonic(double freq, double amp, double phase) :
		freq(freq), amp(amp), phase(phase) {
	}

	virtual ~Harmonic();

	const double freq;
	const double amp;
	const double phase;

	double GetValue(double time) {
		return amp * sin((freq * time + phase) * 2 * M_PI);
	}

};

#endif /* HARMONIC_H_ */
