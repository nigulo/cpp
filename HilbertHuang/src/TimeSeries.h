/*
 * TimeSeries.h
 *
 *  Created on: Apr 23, 2015
 *      Author: nigul
 */

#ifndef SRC_TIMESERIES_H_
#define SRC_TIMESERIES_H_

#include <vector>
#include <memory>

using namespace std;

class TimeSeries {
public:
	TimeSeries();
	TimeSeries(const TimeSeries& ts) :
		xs(ts.xs),
		ys(ts.ys),
		xsIter(ts.xsIter),
		ysIter(ts.ysIter) {
	}

	TimeSeries(const vector<double>& xs, const vector<double>& ys);
	virtual ~TimeSeries();

	void operator=(const TimeSeries& ts) {
		xs = ts.xs;
		ys = ts.ys;
		xsIter = ts.xsIter;
		ysIter = ts.ysIter;
	}

	void add(double x, double y);

	pair<double /*mean*/, double /*variance*/> meanVariance() const;
	pair<unique_ptr<const TimeSeries>, unique_ptr<const TimeSeries>> findExtrema() const;
	unsigned findNumZeroCrossings() const;

	unsigned size() const {
		return xs.size();
	}

	const vector<double>& getXs() const {
		return xs;
	}

	const vector<double>& getYs() const {
		return ys;
	}

	TimeSeries& operator+(const TimeSeries& ts);
	TimeSeries& operator-(const TimeSeries& ts);
	TimeSeries& operator+(double k);
	TimeSeries& operator-(double k);
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
