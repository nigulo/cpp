#include "HilbertHuang.h"
#include "TimeSeries.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include <tuple>
#include <algorithm>

#include "spline.h"

HilbertHuang::HilbertHuang(TimeSeries* ts) : ts(unique_ptr<TimeSeries>(ts)) {
}

HilbertHuang::HilbertHuang(const vector<double>& xs, const vector<double>& ys) :
		ts(new TimeSeries(xs, ys)) {
}

HilbertHuang::~HilbertHuang() {
}



pair<int, pair<double, double>> HilbertHuang::imfStep(TimeSeries& imf, pair<unique_ptr<const TimeSeries>, unique_ptr<const TimeSeries>>& stepExtrema) {
	tk::spline lowerEnv;
	lowerEnv.set_points(stepExtrema.first->getXs(), stepExtrema.first->getYs());
	tk::spline upperEnv;
	upperEnv.set_points(stepExtrema.second->getXs(), stepExtrema.second->getYs());
	for (imf.begin(); imf.hasNext(); imf.next()) {
		double x = imf.getX();
		imf.setY(imf.getY() - (lowerEnv(x) + upperEnv(x)) / 2);
	}
	auto newExtrema = imf.findExtrema();
	unsigned numExtrema = newExtrema.first->size() + newExtrema.second->size();
	unsigned numZeroCrossings = imf.findNumZeroCrossings();
    //cout << endl << "NE: " << numExtrema << ", NZC: " << numZeroCrossings;
	if (abs(numExtrema - numZeroCrossings) <= 1) {
		double extremaStart = max(*(newExtrema.first->getXs().begin()), *(newExtrema.second->getXs().begin()));
		double extremaEnd = min(*(newExtrema.first->getXs().end() - 1), *(newExtrema.second->getXs().end() - 1));
		return {numZeroCrossings, {extremaStart, extremaEnd}};
	}
	return imfStep(imf, newExtrema);
}

bool HilbertHuang::imf() {
	unsigned modeNo = imfs.size() + 1;
	auto extrema = ts->findExtrema();
	int numExtrema = min(extrema.first->size(), extrema.second->size());
	if (numExtrema <= 3) {
		return false;
	}
	cout << "Extracting mode " << modeNo << " (" << numExtrema << ") ...";
	cout.flush();
	TimeSeries* imf = new TimeSeries(*ts);
	auto imfResult = imfStep(*imf, extrema);
    auto extremaStart = imfResult.second.first;
    auto extremaEnd = imfResult.second.second;
    cout << " done." << endl;

    for (ts->begin(), imf->begin(); ts->hasNext();) {
    	double x = ts->getX();
    	if (x < extremaStart || x > extremaEnd) {
    		ts->erase();
    		imf->erase();
    	} else {
    		ts->setY(ts->getY() - imf->getY());
    		ts->next();
    		imf->next();
    	}
    }

    imfs.push_back(unique_ptr<TimeSeries>(imf));
    return true;
}

void HilbertHuang::calculate() {
	while(imf());
}

