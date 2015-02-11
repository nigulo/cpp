#ifndef D2_H_
#define D2_H_

#include <vector>
#include <random>

using namespace std;

enum Mode {
	Box,
	Gauss
};

class D2 {
	private:

	const Mode mode = Gauss;
	const double minPeriod;
	const double maxPeriod;
	const int coherenceGrid = 200;
	const double startCoherence = 0;
	const double minCoherence;
	const double maxCoherence;
	const int phaseBins = 50;
	const double deltaPhi = 0.05;
	const double epsilon = 0.1;

	double epslim, eps, ln2, lnp;
	const vector<double>& x, y;
    vector<double> ty, ta, td, cum;

    int l;
    int lp;
    int k;
    int m;
    double a;
    double b;
    double dbase;
    double dmin;
    double dmax;
    double wmin;
    double delta;
    double deltac;
    double step;
    // For bootstrap resampling
    // Seed with a real random value, if available
    random_device rd;

	public:
    D2(const vector<double>& x, const vector<double>& y, double minPeriod = 2, double maxPeriod = 10, double minCoherence = 0, double maxCoherence = 60);
    void Compute2DSpectrum(bool bootstrap = false);

	private:
    double Criterion(double d, double w);
};



#endif /* D2_H_ */
