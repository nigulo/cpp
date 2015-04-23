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
private:
	pair<int /*numZeroCrossings*/, pair<double, double> /*extremaStart, extremaEnd*/> imfStep(vector<double>& imf,
			pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>>& stepExtrema);
	pair<const vector<double>* /*imf*/, double /*avgFreq*/> imf(int modeNo, vector<double>& dat);
private:
	vector<double> xs;
	vector<double> ys;
	const string prefix;
	double xStep;
	double xRange;

};



#endif /* HILBERTHUANG_H_ */
