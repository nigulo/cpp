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

class D2Minimum {
public:
	D2Minimum(double coherenceLength, double frequency, double value) : coherenceLength(coherenceLength), frequency(frequency), value(value) {}

	double coherenceLength;
	double frequency;
	double value;
	vector<double> bootstrapValues;
};

class D2 {
public:
	// Essentially input parameters to constructor
	DataLoader* mpDataLoader;
	double minCoherence;
	double maxCoherence;
	Mode mode;
	bool normalize;
	bool relative;
	bool differential;

	double tScale;
	double startTime;
	vector<double> varScales;
	vector<pair<double, double>> varRanges;
	bool removeSpurious;
	int bootstrapSize;
	bool confIntOrSignificance = true;
	bool saveDiffNorms;
	bool saveParameters;
private:

	double maxX = -1; // little hack

	const unsigned coherenceGrid = 200;
    const unsigned numFreqs = 200;
	const unsigned phaseBins = 50;
	const double epsilon = 0.1;

	double epslim, eps, ln2, lnp;

	// Square differences, coherence bin lengths and counts
	// Index 0 represents actual data, the rest is bootstrap data
	vector<vector<double>> ty;
	vector<vector<double>> td;
	vector<vector<int>> ta;
	double varSum = 0;

    unsigned numCoherences;
    unsigned numCoherenceBins;
    double a;
    double b;
    double dmin;
    double dmax;
    double dbase;
    double dmaxUnscaled;
    double wmin;
    double coherenceBinSize;
    double freqStep;
	vector<D2Minimum> allMinima;

	// For bootstrap resampling
    // Seed with a real random value, if available
    random_device rd;
    default_random_engine e1;

public:
    D2(DataLoader* pDataLoader, double minPeriod, double maxPeriod,
    		double minCoherence, double maxCoherence,
			Mode mode, bool normalize, bool relative,
			double tScale, double startTime, const vector<double>& varScales,
			const vector<pair<double, double>>& varRanges, bool removeSpurious,
			int bootstrapSize, bool saveDiffNorms, bool saveParameters);
    void CalcDiffNorms(int filePathIndex);
    void LoadDiffNorms(int filePathIndex);
    const vector<D2Minimum>& Compute2DSpectrum(const string& outputFilePrefix);
    void Bootstrap();


private:
    double Criterion(int bootstrapIndex, double d, double w);

    // The norm of the difference of two datasets
    double DiffNorm(const real y1[], const real y2[]);
    bool ProcessPage(DataLoader& dl1, DataLoader& dl2, double* tty, int* tta);
    void VarCalculation(double* ySum, double* y2Sum);
};



#endif /* D2_H_ */
