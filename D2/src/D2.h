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
    const unsigned numFreqs = 100;
	const unsigned phaseBins = 50;
	const double epsilon = 0.1;

	double epslim, eps, ln2, lnp;

	DataLoader* mpDataLoader;
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

    unsigned numCoherences;
    unsigned numCoherenceBins;
    double a;
    double b;
    double dmin;
    double dmax;
    double dmaxUnscaled;
    double wmin;
    double coherenceBinSize;
    double freqStep;
    // For bootstrap resampling
    // Seed with a real random value, if available
    random_device rd;
    default_random_engine e1;

public:
    D2(DataLoader* pDataLoader, double minPeriod, double maxPeriod,
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
    bool ProcessPage(DataLoader& dl1, DataLoader& dl2, vector<double>& tty, vector<int>& tta);
};



#endif /* D2_H_ */
