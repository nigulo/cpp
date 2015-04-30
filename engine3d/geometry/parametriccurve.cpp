#include "parametriccurve.h"

using namespace engine3d;

ParametricCurve::ParametricCurve()
{
}

ParametricCurve::~ParametricCurve()
{
}

void ParametricCurve::SetPoints(const vector<Vector>& rPoints) {
    for (int i = 0; i < rPoints.size(); i++) {
        mPoints.push_back(rPoints[i]);
    }
}
