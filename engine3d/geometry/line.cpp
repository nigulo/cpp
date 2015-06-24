#include "line.h"
#include "segment.h"

using namespace engine3d;

Line::Line(const Vector& rPoint1, const Vector& rPoint2) :
    mPoint1(rPoint1),
    mPoint2(rPoint2),
    mDirection(rPoint2 - rPoint1)
{
}

Line::~Line()
{
}

bool Line::Contains(const Vector& rPoint) const
{
    const Vector& diff = (rPoint - mPoint1);
    const Vector& unitDir = mDirection / mDirection.Length();
    if (unitDir * diff.DotProduct(unitDir) == diff) {
        return true;
    }
    return false;
}

double Line::GetDistance(const Vector& rPoint) const
{
    const Vector& diff = (rPoint - mPoint1);
    const Vector& unitDir = mDirection / mDirection.Length();
    return (diff - unitDir * (diff).DotProduct(unitDir)).Length();
}

Vector Line::GetDistanceVector(const Vector& rPoint) const
{
    const Vector& diff = (rPoint - mPoint1);
    const Vector& unitDir = mDirection / mDirection.Length();
    return diff - unitDir * (diff).DotProduct(unitDir);
}

double Line::GetSquaredDistance(const Line& rLine) const
{
    double A = mDirection.Length2();
    double B = 2 * (mDirection.DotProduct(mPoint1) - mDirection.DotProduct(rLine.mPoint1));
    double C = 2 * mDirection.DotProduct(rLine.mDirection);
    double D = 2 * (rLine.mDirection.DotProduct(rLine.mPoint1) - rLine.mDirection.DotProduct(mPoint1));
    double E = rLine.mDirection.Length2();
    double F = mPoint1.Length2() + rLine.mPoint1.Length2();
    return ((B * C * D) + (B * B * E) + (C * C * F) + A * (D * D - 4 * E * F)) / (C * C - 4 * A * E);
}

const Pointer<Segment*> Line::GetDistance(const Line& rLine) const
{
    double A = mDirection.Length2();
    double B = 2 * (mDirection.DotProduct(mPoint1) - mDirection.DotProduct(rLine.mPoint1));
    double C = 2 * mDirection.DotProduct(rLine.mDirection);
    double D = 2 * (rLine.mDirection.DotProduct(rLine.mPoint1) - rLine.mDirection.DotProduct(mPoint1));
    double E = rLine.mDirection.Length2();
    double F = mPoint1.Length2() + rLine.mPoint1.Length2();
    double s = (2 * A * D + B * C) / (C * C - 4 * A * E);
    double t = (C * s - B) / (2 * A);
    return new Segment(Vector(mPoint1 + mDirection * t), Vector(rLine.mPoint1 + rLine.mDirection * s));
}

bool Line::Crosses(const Line& rLine) const
{
    return GetSquaredDistance(rLine) == 0;
}

string Line::ToString() const {
	string ret_val = (IsSegment() ? "Segment " : "Line ") + mPoint1.ToString() + ", " + mPoint2.ToString();
	return ret_val;
}
