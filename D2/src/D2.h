#ifndef D2_H_
#define D2_H_

#include "DataLoader.h"
#include <vector>
#include <random>

using namespace std;

enum Mode {
	Box,
	Gauss,
	GaussCosine
};

class D2SpecLine {
public:
	D2SpecLine() : frequency(0), tyv(0), tav(0) {
		value = 0;
	}

	D2SpecLine(double frequency, double tyv, double tav, double varSum) : frequency(frequency), tyv(tyv), tav(tav) {
		value = 0.5 * tyv / tav / varSum;
	}

	double frequency;
	double tyv;
	double tav;
	double value;
};

class D2Minimum {
public:
	D2Minimum(double coherenceLength, double frequency, double value, double error) :
		coherenceLength(coherenceLength), frequency(frequency), value(value), error(error) {}

	double coherenceLength;
	double frequency;
	double value;
	double error;
};

class D2 {
public:
	// Essentially input parameters to constructor
	DataLoader* mpDataLoader;
	double minCoherence;
	double maxCoherence;
    int numFreqs;
	Mode mode;
	bool normalize;
	bool relative;
	bool differential;

	// Unfinished implementation (works only in case of even sampling and 1 page)
    double smoothWindow;

	double tScale;
	double startTime;
	double endTime;
	vector<double> varScales;
	vector<pair<double, double>> varRanges;
	bool removeSpurious;
	int bootstrapSize;
	bool confIntOrSignificance = true;
	bool saveDiffNorms;
	bool saveParameters;
private:

	//double maxX = -1; // little hack

	const int coherenceGrid = 200;
	const int phaseBins = 50;
	const double epsilon = 0.1;

	double epslim, eps, ln2, lnp;

	// Square differences, coherence bin lengths and counts
	// Index 0 represents actual data, the rest is bootstrap data
	vector<vector<double>> ty;
	vector<vector<double>> td;
	vector<vector<int>> ta;
	double varSum = 0;

    int numCoherences;
    int numCoherenceBins;
    double dmin;
    double dmax;
    double dbase;
    double dmaxUnscaled;
    double wmax;
    double wmin;
    double coherenceBinSize;
    double freqStep;
	vector<D2Minimum> allMinima;

	// For bootstrap resampling
    // Seed with a real random value, if available
    random_device rd;
    default_random_engine e1;

    int smoothWindowSize;

public:
    D2(DataLoader* pDataLoader, double duration, 
    		double minPeriod, double maxPeriod,
    		double minCoherence, double maxCoherence, int numFreqs,
			Mode mode, bool normalize, bool relative,
			double tScale, double startTime, const vector<double>& varScales,
			const vector<pair<double, double>>& varRanges, bool removeSpurious,
			int bootstrapSize, bool saveDiffNorms, bool saveParameters);
    void CalcDiffNorms();
    void LoadDiffNorms();
    const vector<D2Minimum> Compute2DSpectrum(int bootstrapIndex, const string& outputFilePrefix);
    void Bootstrap(const string& outputFilePrefix);


private:
    pair<double, double> Criterion(int bootstrapIndex, double d, double w);

    // The norm of the difference of two datasets
    double DiffNorm(const real y1[], const real y2[]) const;
    bool ProcessPage(DataLoader& dl1, DataLoader& dl2, double* tty, int* tta);
    void VarCalculation(double* ySum, double* y2Sum) const;
    void RemoveSpurious(vector<D2SpecLine>& minima) const;
    pair<double, double> GetIndexes(int pageSize, int* bsIndexes, int bootstrapIndex, int i) const;
    void Smooth();
};



#endif /* D2_H_ */
