#ifndef ANALYTICSIGNAL_H_
#define ANALYTICSIGNAL_H_

#include "TimeSeries.h"
#include <string>

using namespace std;

class AnalyticSignal {
public:
	AnalyticSignal();
	virtual ~AnalyticSignal();
	static double /*meanEnergy*/ calculate(const TimeSeries& ts, int modeNo, const string& prefix);
};

#endif /* ANALYTICSIGNAL_H_ */
