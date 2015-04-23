#ifndef HILBERTHUANG_H_
#define HILBERTHUANG_H_

#include <vector>
#include <string>

using namespace std;

class HilbertHuang {

public:
	HilbertHuang(const vector<double>& xs, const vector<double>& ys, const string& prefix);
	virtual ~HilbertHuang();
	void calculate();
	const vector<pair<const vector<double>* /*xs*/, const vector<double>* /*ys*/>>& getImfs() const {
		return imfs;
	}
private:
	pair<int /*numZeroCrossings*/, pair<double, double> /*extremaStart, extremaEnd*/> imfStep(vector<double>& imf,
			pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>>& stepExtrema);
	bool /*found*/ imf(vector<double>& dat);
private:
	vector<double> xs;
	vector<double> ys;
	vector<pair<const vector<double>* /*xs*/, const vector<double>* /*ys*/>> imfs;
	const string prefix;
	double xRange;

};



#endif /* HILBERTHUANG_H_ */
