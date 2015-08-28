#ifndef D2_H_
#define D2_H_

#include "DataLoader.h"
#include <vector>
#include <random>

using namespace std;

enum Mode {
	Box,
	Gauss,
	GaussWithCosine
};

class D2 {
private:

	const unsigned coherenceGrid = 200;
	const unsigned phaseBins = 50;
	const double deltaPhi = 0.05;
	const double epsilon = 0.1;

	double epslim, eps, ln2, lnp;

	DataLoader& mrDataLoader;
	const double minCoherence;
	const double maxCoherence;
	const Mode mode;
	const bool normalize;
	const bool relative;

	const double tScale;
	const vector<double> varScales;

	vector<double> ty;
	vector<double> td;
	vector<int> ta;

    unsigned lp;
    unsigned k;
    unsigned m;
    double a;
    double b;
    double dmin;
    double dmax;
    double wmin;
    double delta;
    double step;
    // For bootstrap resampling
    // Seed with a real random value, if available
    random_device rd;

public:
    D2(DataLoader& rDataLoader, double minPeriod, double maxPeriod,
    		double minCoherence, double maxCoherence,
			Mode mode, bool normalize, bool relative,
			double tScale, const vector<double>& varScales);
    void CalcDiffNorms();
    void LoadDiffNorms();
    void Compute2DSpectrum();

private:
    double Criterion(double d, double w);

    // The norm of the difference of two datasets
    double DiffNorm(const real y1[], const real y2[]);
};



#endif /* D2_H_ */
