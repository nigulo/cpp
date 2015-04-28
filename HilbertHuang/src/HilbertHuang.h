#ifndef HILBERTHUANG_H_
#define HILBERTHUANG_H_

#include "TimeSeries.h"
#include <vector>
#include <string>
#include <memory>

using namespace std;

class HilbertHuang {

public:
	HilbertHuang(TimeSeries* ts);
	HilbertHuang(const vector<double>& xs, const vector<double>& ys);
	virtual ~HilbertHuang();
	void calculate();

	const vector<unique_ptr<TimeSeries>>& getImfs() const {
		return imfs;
	}

private:
	bool /*found*/ imf();
private:
	unique_ptr<TimeSeries> ts;
	vector<unique_ptr<TimeSeries>> imfs;

};



#endif /* HILBERTHUANG_H_ */
