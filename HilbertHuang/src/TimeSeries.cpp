/*
 * TimeSeries.cpp
 *
 *  Created on: Apr 23, 2015
 *      Author: nigul
 */

#include "TimeSeries.h"
#include <cassert>
#include <iostream>

TimeSeries::TimeSeries() {
}

TimeSeries::TimeSeries(const vector<double>& xs, const vector<double>& ys) :
	xs(xs), ys(ys) {
	assert(xs.size() == ys.size());
}

TimeSeries::~TimeSeries() {
}

void TimeSeries::add(double x, double y) {
	xs.push_back(x);
	ys.push_back(y);
}

pair<double /*mean*/, double /*variance*/> TimeSeries::meanVariance() const {
	assert(ys.size() > 0);
	double sum = 0;
	double sumSquares = 0;
	for(auto val = ys.begin(); val != ys.end(); val++) {
		sum += (*val);
		sumSquares += (*val) * (*val);
	}
	int n = ys.size();
	double mean = sum / n;
	return {mean,  sumSquares / n - mean * mean};
}

pair<unique_ptr<const TimeSeries>, unique_ptr<const TimeSeries>> TimeSeries::findExtrema() const {
	TimeSeries* minima = new TimeSeries();
	TimeSeries* maxima = new TimeSeries();
	int j = 0;
	for (unsigned i = 1; i < ys.size() -  1; i++) {
		double y1 = ys[i];
		double y2 = ys[i + 1];
		if (y1 != y2) {
			double y0 = ys[j]; // in a regular case j = i - 1
			if (y1 - y0 < 0 && y1 - y2 < 0 ){
				minima->add(xs[i], y1);
			} else if (y1 - y0 > 0 && y1 - y2 > 0 ) {
				maxima->add(xs[i], y1);
			}
			j = i;
		}
	}
	return {unique_ptr<TimeSeries>(minima), unique_ptr<TimeSeries>(maxima)};
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
	unsigned numZeroCrossings = 0;
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
	return (xsIter != xs.end());
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
		double x2 = ts.xs[j];
		if (x1 < x2) {
			i++;
		} else if (x2 < x1) {
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
		double x2 = ts.xs[j];
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

TimeSeries& TimeSeries::operator+(double k) {
	for (auto i = ys.begin(); i != ys.end(); i++) {
		*i += k;
	}
	return *this;
}

TimeSeries& TimeSeries::operator-(double k) {
	for (auto i = ys.begin(); i != ys.end(); i++) {
		*i -= k;
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
