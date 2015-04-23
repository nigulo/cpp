#include "HilbertHuang.h"
#include "TimeSeries.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include <tuple>
#include "spline.h"

HilbertHuang::HilbertHuang(const vector<double>& xs, const vector<double>& ys, const string& prefix) :
		ts(xs, ys), prefix(prefix) {
}

HilbertHuang::~HilbertHuang() {
}



pair<int, pair<double, double>> HilbertHuang::imfStep(TimeSeries& imf, pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>>& stepExtrema) {
	tk::spline lowerEnv;
	lowerEnv.set_points(*stepExtrema.first.first, *stepExtrema.first.second);
	tk::spline upperEnv;
	upperEnv.set_points(*stepExtrema.second.first, *stepExtrema.second.second);
	//vector<double>* dat1 = new vector<double>(dat.size());
	for (imf.begin(); imf.hasNext(); imf.next()) {
		double x = imf.getX();
		imf.setY(imf.getY() - (lowerEnv(x) + upperEnv(x)) / 2);
	}
	delete stepExtrema.first.first;
	delete stepExtrema.first.second;
	delete stepExtrema.second.first;
	delete stepExtrema.second.second;
	auto newExtrema = ts.findExtrema();
	int numExtrema = newExtrema.first.first->size() + newExtrema.second.first->size();
	int numZeroCrossings = ts.findNumZeroCrossings();
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

bool HilbertHuang::imf() {
	unsigned modeNo = imfs.size() + 1;
	auto extrema = ts.findExtrema();
	int numExtrema = min(extrema.first.first->size(), extrema.second.first->size());
	if (numExtrema <= 3) {
		delete extrema.first.first;
		delete extrema.first.second;
		delete extrema.second.first;
		delete extrema.second.second;
		return false;
	}
	cout << "Extracting mode " << modeNo << " (" << numExtrema << ") ...";
	cout.flush();
	TimeSeries* imf = new TimeSeries(ts.getXs(), ts.getYs());
	//vector<double>* imf = new vector<double>(ts.getYs());
	auto imfResult = imfStep(*imf, extrema);
    auto extremaStart = imfResult.second.first;
    auto extremaEnd = imfResult.second.second;
    cout << " done." << endl;

    for (ts.begin(), imf->begin(); ts.hasNext();) {
    	double x = ts.getX();
    	if (x < extremaStart || x > extremaEnd) {
    		ts.erase();
    		imf->erase();
    	} else {
    		ts.setY(ts.getY() - imf->getY());
    		ts.next();
    		imf->next();
    	}
    }

    imfs.push_back(unique_ptr<TimeSeries>(imf));
    return true;
}

void HilbertHuang::calculate() {
	while(imf());
}

