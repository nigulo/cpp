#ifndef HILBERTHUANG_H_
#define HILBERTHUANG_H_

#include "TimeSeries.h"
#include <vector>
#include <string>
#include <memory>

using namespace std;

class HilbertHuang {

public:
	HilbertHuang(const vector<double>& xs, const vector<double>& ys, const string& prefix);
	virtual ~HilbertHuang();
	void calculate();
	const vector<unique_ptr<TimeSeries>>& getImfs() const {
		return imfs;
	}
private:
	pair<int /*numZeroCrossings*/, pair<double, double> /*extremaStart, extremaEnd*/> imfStep(TimeSeries& imf,
			pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>>& stepExtrema);
	bool /*found*/ imf();
private:
	TimeSeries ts;
	vector<unique_ptr<TimeSeries>> imfs;
	const string prefix;
};



#endif /* HILBERTHUANG_H_ */
