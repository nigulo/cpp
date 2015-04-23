/*
 * TimeSeries.h
 *
 *  Created on: Apr 23, 2015
 *      Author: nigul
 */

#ifndef SRC_TIMESERIES_H_
#define SRC_TIMESERIES_H_

#include <vector>

using namespace std;

class TimeSeries {
public:
	TimeSeries(const vector<double>& xs, const vector<double>& ys);
	virtual ~TimeSeries();

	pair<pair<const vector<double>*, const vector<double>*>, pair<const vector<double>*, const vector<double>*>> findExtrema() const;
	unsigned findNumZeroCrossings() const;

	const vector<double>& getXs() const {
		return xs;
	}

	const vector<double>& getYs() const {
		return ys;
	}

	TimeSeries& operator+(const TimeSeries& ts);
	TimeSeries& operator-(const TimeSeries& ts);
	TimeSeries& operator*(double k);
	TimeSeries& operator/(double k);

	void begin();
	bool hasNext();
	bool next();
	double getX();
	double getY();
	void setY(double y);
	void erase();

private:
	vector<double> xs;
	vector<double> ys;

	vector<double>::iterator xsIter;
	vector<double>::iterator ysIter;
};

#endif /* SRC_TIMESERIES_H_ */
