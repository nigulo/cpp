#include "beziercurve.h"
#include <math.h>

using namespace engine3d;

BezierCurve::BezierCurve(unsigned int order) :
    mOrder(order)
{

}

BezierCurve::~BezierCurve()
{
}

Vector BezierCurve::GetPoint(double parameter) const
{
    assert(parameter >= GetParameterMin() && parameter < GetParameterMax());
    int startingPoint = 0;
//    parameter *= mPoints.Size() / (mOrder + 1);
//    while (parameter >= 1) {
//        parameter -= 1;
//        startingPoint += mOrder;
//    }
    
    
    int k, kn, nn, nkn;
    double blend, muk, munk;
    Vector point(0.0, 0.0, 0.0);
    
    muk = 1;
    munk = pow(1 - parameter, (double) mPoints.size() - 1);
    
    for (k = startingPoint; k < startingPoint + mPoints.size(); k++) {
        nn = mPoints.size() - 1;
        kn = k;
        nkn = mPoints.size() - 1 - k;
        blend = muk * munk;
        muk *= parameter;
        munk /= (1 - parameter);
        while (nn >= 1) {
            blend *= nn;
            nn--;
            if (kn > 1) {
                blend /= (double) kn;
                kn--;
            }
            if (nkn > 1) {
                blend /= (double) nkn;
                nkn--;
            }
        }
        point += mPoints[k] * blend;
    }

    return point;
}
