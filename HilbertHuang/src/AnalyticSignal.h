#ifndef ANALYTICSIGNAL_H_
#define ANALYTICSIGNAL_H_

#include <vector>
#include <string>

using namespace std;

class AnalyticSignal {
public:
	AnalyticSignal();
	virtual ~AnalyticSignal();
	static double /*meanEnergy*/ calculate(const vector<double>& xs, const vector<double>& realSignal, int modeNo, const string& prefix);
};

#endif /* ANALYTICSIGNAL_H_ */
