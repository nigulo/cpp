#include "D2.h"
#include "BinaryDataLoader.h"
#include "TextDataLoader.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <math.h>
#include <sstream>
#include <memory>
#include <boost/algorithm/string.hpp>
#include "mpi.h"

using namespace std;

int procId;
int numProc;

int main(int argc, char *argv[]) {

	MPI::Init (argc, argv);
	numProc = MPI::COMM_WORLD.Get_size();
	procId = MPI::COMM_WORLD.Get_rank();

	string dirPrefix = argv[1];
	string fileName = argv[2];
	string path;
	if (numProc > 1) {
		path = dirPrefix + to_string(procId) + "/" + fileName;
	} else {
		path = dirPrefix + "/" + fileName;
	}
	int i = 2;
	bool binary = argc > ++i ? atoi(argv[i]) : 0;
	unsigned bufferSize = argc > ++i ? atoi(argv[i]) : 0;
	unsigned dim = argc > ++i ? atoi(argv[i]) : 1;
	unsigned totalNumVars = argc > ++i ? atoi(argv[i]) : 1;

	string strVarIndices = argc > ++i ? argv[i] : "0";
	vector<string> varIndicesStr;
	vector<unsigned> varIndices;
	boost::split(varIndicesStr, strVarIndices, boost::is_any_of(","), boost::token_compress_on);
	for (vector<string>::iterator it = varIndicesStr.begin() ; it != varIndicesStr.end(); ++it) {
		if ((*it).length() != 0) {
			varIndices.push_back(stoi(*it));
		}
	}

	double minPeriod = argc > ++i ? atof(argv[i]) : 2;
	double maxPeriod = argc > ++i ? atof(argv[i]) : 10;
	double minCoherence = argc > ++i ? atof(argv[i]) : 3;
	double maxCoherence = argc > ++i ? atof(argv[i]) : 30;
	double tScale = argc > ++i ? atof(argv[i]) : 1;
	DataLoader* dl;
	if (binary) {
		dl = new BinaryDataLoader(path, bufferSize, dim, totalNumVars, varIndices);
	} else {
		dl = new TextDataLoader(path, bufferSize, dim, totalNumVars, varIndices);
	}
	D2 d2(*dl, minPeriod, maxPeriod, minCoherence, maxCoherence, tScale);
	d2.Compute2DSpectrum();
	delete dl;
	MPI::Finalize();
	return 0;
}

#define square(x) ((x) * (x))

D2::D2(DataLoader& rDataLoader, double minPeriod, double maxPeriod, double minCoherence, double maxCoherence, double tScale) :
		mrDataLoader(rDataLoader),
		minCoherence(minCoherence),
		maxCoherence(maxCoherence),
		tScale(tScale) {

	double wmax = 1.0 / minPeriod;
	wmin = 1.0 / maxPeriod;

	dmin = minCoherence * (relative ? minPeriod : 1);
	dmax = maxCoherence * (relative ? maxPeriod : 1);

	if (dmax < dmin || minPeriod > maxPeriod) {
		throw "Check Arguments";
	}
	k = dmax > dmin ? coherenceGrid : 1;
	m = round(phaseBins * (dmax - dmin) * wmax);
	a = (m - 1.0) / (dmax - dmin);
	b = -dmin * a;
	delta = (dmax - dmin) / (m - 1);

	lp = round((wmax - wmin) * (dmax - dmin) / deltaPhi);
	step = (wmax - wmin) / (lp - 1);
	eps = epsilon;
	epslim = 1.0 - eps;
	ln2 = sqrt(log(2.0));
	lnp = ln2 / eps;
}

double D2::Criterion(double d, double w) {

	unsigned j;
	double tyv, tav, dd, ww, wp, ph;

	tyv = 0.0;
	tav = 0.0;
	switch (mode) {
	case Box:
		for (j = 0; j < td.size(); j++) {// to jj-1 do begin
			dd = td[j];
			if (dd <= d) {
				ph = dd * w - floor(dd * w);//Frac(dd*w);
				if (ph < 0.0) {
					ph = ph+1;
				}
				if (ph<eps || ph>epslim) {
					tyv = tyv+ty[j];
					tav = tav+ta[j];
				}
			}
		}
		break;
	case Gauss: //This is important, in td[] are precomputed sums of squares and counts.
	case GaussWithCosine:
		for (j = 0; j < td.size(); j++) {// to jj-1 do begin
			dd = td[j];
			if (d > 0.0) {
				ww = exp(-square(ln2 * dd / d));
			} else {
				ww = 0.0;
			}
			ph=dd * w - floor(dd * w);//Frac(dd*w);
			if (ph < 0.0) {
				ph = ph + 1;
			}
			if (ph>0.5) {
				ph = 1.0 - ph;
			}
			bool closeInPhase = true;
			if (mode == Gauss) {
				closeInPhase = ph < eps || ph > epslim;
				wp = exp(-square(lnp*ph));
			} else {
				if (ph == 0.5) {
					wp = 0;
				} else if (ph == 0) {
					wp = 1;
				} else {
					wp = 0.5 * (cos (0.5 * M_PI / ph) + 1);
				}
				if (std::isnan(wp)) {
					wp = 0;
					cout << "wp is still nan" << endl;
				}
			}
			if (closeInPhase) {
				tyv=tyv + ww * wp * ty[j];
				tav=tav + ww * wp * ta[j];
			}
		}
		break;
	}
	if (tav > 0.0) {
		return 0.5 * tyv / tav;
	} else {
		return 0.0;
	}
}

void MapTo01D(vector<double>& cum) {

	unique_ptr<double> min;
	unique_ptr<double> max;

	for (auto& val : cum) {
		if (!min.get() || val < *min.get()) {
			min.reset(new double(val));
		}
		if (!max.get() || val > *max.get()) {
			max.reset(new double(val));
		}
	}
	if (max.get() && min.get() && *max.get() > *min.get()) {
		double range = *max.get() - *min.get();
		for (auto& val: cum) {
			val = (val - *min.get()) / range;
		}

	} else {
		cout << "Cannot normalize" << endl;
	}

}

// Currently implemented as Frobenius norm
double D2::DiffNorm(const real y1[], const real y2[]) {
	double norm = 0;
	for (unsigned i : mrDataLoader.GetVarIndices()) {
		norm += square(y1[i] - y2[i]);
	}
	return norm;
}

void D2::Compute2DSpectrum() {

    cout << "lp= " << lp << endl;
    cout << "k= " << k << endl;
    cout << "m= " << m << endl;
    cout << "a= " << a << endl;
    cout << "b= " << b << endl;
    cout << "dmin= " << dmin << endl;
    cout << "dmax= " << dmax << endl;
    cout << "wmin= " << wmin << endl;
    cout << "delta= " << delta << endl;
    cout << "step= " << step << endl;

	cum.assign(lp, 0);
	vector<vector<double>*> ydiffs(m, 0);
	vector<double> tty(m, 0);
	vector<double> tta(m, 0);

	// Now comes precomputation of differences and counts. They are accumulated in two grids.
	unsigned i, j;
	while (mrDataLoader.Next()) {
		cout << "A page: " << mrDataLoader.GetPage() << endl;
		auto dl2Ptr = mrDataLoader.Clone();
		DataLoader* dl2 = dl2Ptr.get();
		do {
			cout << "    B page: " << dl2->GetPage() << endl;
			for (i = 0; i < mrDataLoader.GetPageSize(); i++) {
				for (j = 0; j < dl2->GetPageSize(); j++) {
					//if (i == 999) {
					//	cout << "pageSize: " << mrDataLoader.GetPageSize() << " " << dl2->GetPageSize() << endl;
					//	cout << "        j1: " << j << " " << dl2->GetX(j) << " " << mrDataLoader.GetX(i) << endl;
					//}
					if (j <= i && mrDataLoader.GetPage() == dl2->GetPage()) {
						continue;
					}
					real d = (dl2->GetX(j) - mrDataLoader.GetX(i)) * tScale;
					if (d >= dmin && d <= dmax) {
						int kk = round(a * d + b);
						tty[kk] += DiffNorm(dl2->GetY(j), mrDataLoader.GetY(i));
						tta[kk] += 1.0;
					}
				}
			}
		} while (dl2->Next());
	}
	cout << "Siin" << endl;

	if (procId > 0) {
		MPI::COMM_WORLD.Send(tty.data(), tty.size(), MPI::DOUBLE, 0, 1);
	} else {
		for (int i = 1; i < numProc; i++) {
			double received[m];
			MPI::Status status;
			MPI::COMM_WORLD.Recv (received, m,  MPI::DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, status);
			for (unsigned j = 0; j < m; j++) {
				tty[j] += received[j];
				tta[j] += 1.0;
			}
		}
		// How many time differences was actually used?
		j=0;
		for (i = 0; i < m; i++) {
			if (tta[i] > 0.5) {
				j++;
			}
		}
		cout << "j=" << j << endl;
		ta.assign(j, 0);
		ty.assign(j, 0);
		td.assign(j, 0);
		cout << "td.size()=" << td.size() << endl;
		double d = dmin;

		// Build final grids for periodicity search.

		j = 0;
		for (i = 0; i < m; i++) {
			if (tta[i] > 0.5) {
				ta[j] = tta[i];
				ty[j] = tty[i];
				td[j] = d;
				j++;
			}
			d = d + delta;
		}
		ofstream output("phasedisp.csv");
		ofstream output_min("phasedisp_min.csv");
		ofstream output_max("phasedisp_max.csv");

		// Basic cycle with printing for GnuPlot

		double deltac = maxCoherence > minCoherence ? (maxCoherence - minCoherence) / (k - 1) : 0;
		for (i = 0; i < k; i++) {
			d = minCoherence + i * deltac;
			for (j = 0; j < lp; j++) {
				double w = wmin + j * step;
				double d1 = d;
				if (relative) {
					d1 = d / w;
				}
				double res=Criterion(d1, w);
				cum[j] = res;
			}

			// Spectrum in cum can be normalized

			if (true) {
				MapTo01D(cum);
			}

			vector<int> minima(0);
			unsigned dk = lp / 20;
			for (j = 0; j < lp; j++) {
				output << d << " " << (wmin + j * step) << " " << cum[j] << endl;
				if (d == minCoherence) {
					output_min << (wmin + j * step) << " " << cum[j] << endl;
				} else if (d == maxCoherence) {
					output_max << (wmin + j * step) << " " << cum[j] << endl;
				}

				if (j > dk - 1 && j < lp - dk - 1) {
					bool isMinimum = true;
					for (unsigned k1 = j - dk; k1 <= j + dk; k1++) {
						if (cum[j] > cum[k1]) {
							isMinimum = false;
							break;
						}
					}
					if (isMinimum) {
						vector<int>::iterator it = minima.begin();
						for (; it != minima.end(); ++it) {
							if (cum[j] < cum[(*it)]) {
								break;
							}
						}
						minima.insert(it, j);
					}
				}
			}
			for (unsigned k1 = 0; k1 < minima.size(); k1++) {
				if (k1 > 0) {
					//cout << " ";
				}
				//cout << wmin + minima[k1] * step;
			}
			//cout << endl;
		}
		output.close();
		output_min.close();
		output_max.close();
	}
}

