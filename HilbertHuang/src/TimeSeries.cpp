/*
 * TimeSeries.cpp
 *
 *  Created on: Apr 23, 2015
 *      Author: nigul
 */

#include "TimeSeries.h"
#include <cassert>

TimeSeries::TimeSeries(const vector<double>& xs, const vector<double>& ys) :
	xs(xs), ys(ys) {
	assert(xs.size() == ys.size());
}

TimeSeries::~TimeSeries() {
}

pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>> TimeSeries::findExtrema() const {
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

unsigned TimeSeries::findNumZeroCrossings() const {
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

void TimeSeries::begin() {
	xsIter = xs.begin();
	ysIter = ys.begin();
}

bool TimeSeries::hasNext() {
	return (xsIter == xs.end());
}

bool TimeSeries::next() {
	xsIter++;
	ysIter++;
	return (xsIter == xs.end());
}

double TimeSeries::getX() {
	return *xsIter;
}

double TimeSeries::getY() {
	return *ysIter;
}

void TimeSeries::setY(double y) {
	*ysIter = y;
}

TimeSeries& TimeSeries::operator+(const TimeSeries& ts) {
	for (unsigned i = 0, j = 0; i < xs.size() && j < ts.xs.size();) {
		double x1 = xs[i];
		double x2 = ts.xs[i];
		if (x1 > x2) {
			i++;
		} else if (x2 > x1) {
			j++;
		} else {
			ys[i] += ts.ys[j];
			i++;
			j++;
		}
	}
	return *this;
}

TimeSeries& TimeSeries::operator-(const TimeSeries& ts) {
	for (unsigned i = 0, j = 0; i < xs.size() && j < ts.xs.size();) {
		double x1 = xs[i];
		double x2 = ts.xs[i];
		if (x1 > x2) {
			i++;
		} else if (x2 > x1) {
			j++;
		} else {
			ys[i] -= ts.ys[j];
			i++;
			j++;
		}
	}
	return *this;
}

TimeSeries& TimeSeries::operator*(double k) {
	for (auto i = ys.begin(); i != ys.end(); i++) {
		*i *= k;
	}
	return *this;
}

TimeSeries& TimeSeries::operator/(double k) {
	for (auto i = ys.begin(); i != ys.end(); i++) {
		*i /= k;
	}
	return *this;
}

void TimeSeries::erase() {
	xsIter = xs.erase(xsIter);
	ysIter = ys.erase(ysIter);
}
