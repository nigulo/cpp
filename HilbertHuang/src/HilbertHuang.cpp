#include "HilbertHuang.h"
#include "AnalyticSignal.h"
#include <iostream>
#include <math.h>
#include <sstream>
#include <fstream>
#include <tuple>
#include "spline.h"

HilbertHuang::HilbertHuang(const vector<double>& xs, const vector<double>& ys, const string& prefix) :
		xs(xs), ys(ys), prefix(prefix) {
	xStep = xs[1] - xs[0]; // Assuming even sampling
	xRange = *(xs.end() - 1) - *(xs.begin());
}

HilbertHuang::~HilbertHuang() {
}


pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>> findExtrema(const vector<double>& xs, const vector<double>& ys) {
	vector<double>* minima_x = new vector<double>();
	vector<double>* minima_y = new vector<double>();
	vector<double>* maxima_x = new vector<double>();
	vector<double>* maxima_y = new vector<double>();
	int j = 0;
	for (unsigned i = 1; i < ys.size() -  1; i++) {
		double y1 = ys[i];
		double y2 = ys[i + 1];
		if (y1 != y2) {
			double y0 = ys[j]; // in a regular case j = i - 1
			if (y1 - y0 < 0 && y1 - y2 < 0 ){
				minima_y->push_back(y1);
				minima_x->push_back(xs[i]);
			} else if (y1 - y0 > 0 && y1 - y2 > 0 ) {
				maxima_y->push_back(y1);
				maxima_x->push_back(xs[i]);
			}
			j = i;
		}
	}
	return {{minima_x, minima_y}, {maxima_x, maxima_y}};
}
/*
const vector<double>* findZeroCrossings(const vector<double>& xs, const vector<double>& ys) {
	vector<double>* zeroCrossings = new vector<double>();
	for (int i = 0; i < ys.size() - 1; i++) {
		double y1 = ys[i];
        if (y1 == 0) {
        	zeroCrossings->push_back(xs[i]);
        } else {
    		double y2 = ys[i + 1];
        	if ((y1 > 0 && y2 <= 0) || (y1 < 0 && y2 >= 0)) {
            	zeroCrossings->push_back(abs(y1) < abs(y2) ? xs[i] : xs[i + 1]);
        	}
        }
	}
	return zeroCrossings;
}
*/

int findNumZeroCrossings(const vector<double>& xs, const vector<double>& ys) {
	int numZeroCrossings = 0;
	for (unsigned i = 0; i < ys.size() - 1; i++) {
		double y1 = ys[i];
        if (y1 == 0) {
        	numZeroCrossings++;
        } else {
    		double y2 = ys[i + 1];
        	if ((y1 > 0 && y2 <= 0) || (y1 < 0 && y2 >= 0)) {
            	numZeroCrossings++;
        	}
        }
	}
	return numZeroCrossings;
}

pair<int, pair<double, double>> HilbertHuang::imfStep(vector<double>& imf, pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>>& stepExtrema) {
	tk::spline lowerEnv;
	lowerEnv.set_points(*stepExtrema.first.first, *stepExtrema.first.second);
	tk::spline upperEnv;
	upperEnv.set_points(*stepExtrema.second.first, *stepExtrema.second.second);
	//vector<double>* dat1 = new vector<double>(dat.size());
	for (unsigned i = 0; i < imf.size(); i++) {
		double x = xs[i];
		imf[i] = imf[i] - (lowerEnv(x) + upperEnv(x)) / 2;
	}
	delete stepExtrema.first.first;
	delete stepExtrema.first.second;
	delete stepExtrema.second.first;
	delete stepExtrema.second.second;
	auto newExtrema = findExtrema(xs, imf);
	int numExtrema = newExtrema.first.first->size() + newExtrema.second.first->size();
	int numZeroCrossings = findNumZeroCrossings(xs, imf);
    //cout << endl << "NE: " << numExtrema << ", NZC: " << numZeroCrossings;
	if (abs(numExtrema - numZeroCrossings) <= 1) {
		double extremaStart = max(*(newExtrema.first.first->begin()), *(newExtrema.second.first->begin()));
		double extremaEnd = min(*(newExtrema.first.first->end() - 1), *(newExtrema.second.first->end() - 1));
		delete newExtrema.first.first;
		delete newExtrema.first.second;
		delete newExtrema.second.first;
		delete newExtrema.second.second;
		return {numZeroCrossings, {extremaStart, extremaEnd}};
	}
	return imfStep(imf, newExtrema);
}

pair<const vector<double>* /*imf*/, double /*avgFreq*/> HilbertHuang::imf(int modeNo, vector<double>& dat) {
	assert(xs.size() == dat.size());
	auto extrema = findExtrema(xs, dat);
	int numExtrema = min(extrema.first.first->size(), extrema.second.first->size());
	if (numExtrema <= 3) {
		delete extrema.first.first;
		delete extrema.first.second;
		delete extrema.second.first;
		delete extrema.second.second;
		return {&dat, 0};
	}
	cout << "Extracting mode " << modeNo << " (" << numExtrema << ") ...";
	cout.flush();
	vector<double>* imf = new vector<double>(dat);
	auto imfResult = imfStep(*imf, extrema);
    int numZeroCrossings = imfResult.first;
    auto extremaStart = imfResult.second.first;
    auto extremaEnd = imfResult.second.second;
    cout << " done." << endl;

    auto xsIter = xs.begin();
    auto datIter = dat.begin();
    auto imfIter = imf->begin();
    for (;xsIter != xs.end();) {
    	double x = *xsIter;
    	if (x < extremaStart || x > extremaEnd) {
    		xsIter = xs.erase(xsIter);
    		datIter = dat.erase(datIter);
    		imfIter = imf->erase(imfIter);
    	} else {
    		*datIter -= *imfIter;
    		xsIter++;
    		datIter++;
    		imfIter++;
    	}
    }
	xRange = *(xs.end() - 1) - *(xs.begin());

    //for (unsigned i = 0; i < dat.size(); i++) {
   	//	dat[i] -= (*imf)[i];
    //}
    return {imf, 0.5 * numZeroCrossings / xRange};
}

void HilbertHuang::calculate() {
	stringstream logText;
	for (int modeNo = 1;; modeNo++) {
		auto imfAndFreq = imf(modeNo, ys);
		if (imfAndFreq.second == 0) {
			break;
		}
		double meanEnergy = AnalyticSignal::calculate(xs, *imfAndFreq.first, modeNo, prefix);
        logText << modeNo << ": " << imfAndFreq.second << " " << meanEnergy << endl;
	}
	ofstream logStream(prefix + ".log");
	logStream << logText.str();
	logStream.close();
	cout << logText.str();
}

