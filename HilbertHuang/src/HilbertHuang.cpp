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
	double maxEnvMean = 0;
	double sum = 0;
	double sumSquares = 0;
	unsigned i = 0;
	for (imf.begin(); imf.hasNext(); imf.next()) {
		double x = imf.getX();
		double y = imf.getY();
		double lower = lowerEnv(x);
		double upper = upperEnv(x);
		// End-point corrections
		if (i == 0 || i == imf.size() - 1) {
			if (upper < y) {
				upper = y;
			}
			if (lower > y) {
				lower = y;
			}
		}
		i++;
		double envMean = (lower + upper) / 2;
		if (abs(envMean) > maxEnvMean) {
			maxEnvMean = abs(envMean);
		}
		imf.setY(y - envMean);
		sum += y;
		sumSquares += y * y;
	}
	int n = imf.size();
	double mean = sum / n;
	double stDev = sqrt(sumSquares / n - mean * mean);
	auto newExtrema = imf.findExtrema();
	int numExtrema = newExtrema.first->size() + newExtrema.second->size();
	int numZeroCrossings = imf.findNumZeroCrossings();
    //cout << endl << "NE: " << numExtrema << ", NZC: " << numZeroCrossings << ", MEM: " << maxEnvMean << ", stDev: " << stDev;
	if (maxEnvMean < 0.05 * stDev && abs(numExtrema - numZeroCrossings) <= 1) {
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
	if (numExtrema <= 2) {
		return false;
	}
	cout << "Extracting mode " << modeNo << " (" << numExtrema << ") ...";
	cout.flush();
	TimeSeries* imf = new TimeSeries(*ts);
	auto imfResult = imfStep(*imf, extrema);
    auto extremaStart = imfResult.second.first;
    auto extremaEnd = imfResult.second.second;
    cout << " done." << endl;

    *ts - *imf;

    /*for (ts->begin(), imf->begin(); ts->hasNext();) {
    	double x = ts->getX();
    	if (x < extremaStart || x > extremaEnd) {
    		ts->erase();
    		imf->erase();
    	} else {
    		ts->setY(ts->getY() - imf->getY());
    		ts->next();
    		imf->next();
    	}
    }*/

    /*for (imf->begin(); imf->hasNext();) {
    	double x = imf->getX();
    	if (x < extremaStart || x > extremaEnd) {
    		imf->erase();
    	} else {
    		imf->next();
    	}
    }*/

    imfs.push_back(unique_ptr<TimeSeries>(imf));
    return true;
}

void HilbertHuang::calculate() {
	while(imf());
}

