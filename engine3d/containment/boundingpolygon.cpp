#include "boundingpolygon.h"
#include "engine3d/geometry/segment.h"

using namespace engine3d;

BoundingPolygon::BoundingPolygon(Vector vertices[], int number)
{
	for (int i = 0; i < number; i++) {
        mVertices.push_back(vertices[i]);
        mTransformedVertices.push_back(vertices[i]);
    }
}

BoundingPolygon* BoundingPolygon::Clone() const 
{
    BoundingPolygon* p_bound = new BoundingPolygon();
	for (int i = 0; i < mVertices.size(); i++) {
        //p_bound->AddVertex(mVertices[i]);
        p_bound->mVertices.push_back(mVertices[i]);
        p_bound->mTransformedVertices.push_back(mTransformedVertices[i]);
    }
    return p_bound;
}

BoundingPolygon::~BoundingPolygon()
{
	//for (int i = 0; i < mNumber; i++) {
    //    delete mpVertices[i];
    //}
    //delete []mpVertices;
    //mVertices.Clear();
}

Vector&  BoundingPolygon::AddVertex(const Vector& rVertex) {
    mVertices.push_back(rVertex);
    mTransformedVertices.push_back(rVertex);
    //Debug(string("vertex added to bound: ") + to_string(rVertex[0]) + ", " + to_string(rVertex[1]) + ", " + to_string(rVertex[2]));
    return mVertices.back();
}

int BoundingPolygon::WhichSide(const Plane& rPlane) const
{
    if (mVertices.size() <= 0) {
        return Plane::PLACEMENT_UNKNOWN;
    }
    Vector v = mTransformedVertices[0];//Spatial::mTransformation.Transform(mVertices[0]);
    int placement = rPlane.WhichSide(v);
	for (int i = 1; i < mVertices.size(); i++) {
        v = mTransformedVertices[i];//Spatial::mTransformation.Transform(mVertices[i]);
        if (placement != rPlane.WhichSide(v)) {
            return Plane::PLACEMENT_COINCIDE;
        }
    }
    return placement;
    
}

/**
 * Checks if this polygon intersects with the given line (segment)
 * Every three vertices of this polygon form a plane
 * which is tested against line intersection. If at least one
 * intersecting plane exists the polygon as a whole intersects
 * with the line.
 **/
unique_ptr<Vector> BoundingPolygon::Intersects(const Line& rLine) const
{
    //Debug("polygon intersect 1");
    for (unsigned i = 0; i < mTransformedVertices.size(); i++) {
        for (unsigned j = i + 1; j < mTransformedVertices.size(); j++) {
            for (unsigned k = j + 1; k < mTransformedVertices.size(); k++) {
        
                //Debug("polygon intersect 2");
                const Vector& r_point1 = mTransformedVertices[i];//Spatial::mTransformation.Transform(mVertices[i]);
                const Vector& r_point2 = mTransformedVertices[j];//Spatial::mTransformation.Transform(mVertices[j]);
                const Vector& r_point3 = mTransformedVertices[k];//Spatial::mTransformation.Transform(mVertices[k]);
                Plane p(r_point1, r_point2, r_point3);
                const Vector& r_normal = p.GetNormal();
                double dot2 = r_normal.DotProduct(rLine.mDirection);
                if (dot2 == 0) { 
                    // line is parallel to the plane
                    if (p.GetDistance(rLine.mPoint1) == 0) {
                        // the line is on the plane
                        if (Segment(r_point1, r_point2).Crosses(rLine) ||
                            Segment(r_point2, r_point3).Crosses(rLine) ||
                            Segment(r_point3, r_point1).Crosses(rLine)
                            ) {
                            Debug("Parallel intersection");
                            // TODO Not correct value is returned
                            return unique_ptr<Vector>(new Vector(rLine.mPoint1));
                        }
                    }
                    else {
                        continue;
                    }
                }
                else {
                    Debug(string("polygon intersect 3: ") + rLine.ToString() + " "  + p.ToString());
                    // line intersects with plane
                    if (!rLine.IsSegment() || p.WhichSide(rLine.mPoint1) != p.WhichSide(rLine.mPoint2)) {
                        // line or segment intersects with plane
                        double dot1 = r_normal.DotProduct(rLine.mPoint1 - p.GetPoint());
                        
                        Vector intersectionPoint = rLine.mPoint1 - rLine.mDirection * (dot1 / dot2);
                        Debug(string("intersection point: ") + intersectionPoint.ToString());
                        
                        //Vector normal1 = Line(r_point2, r_point1).GetDistanceVector(intersectionPoint);
                        //Vector normal2 = Line(r_point3, r_point2).GetDistanceVector(intersectionPoint);
                        //Vector normal3 = Line(r_point1, r_point3).GetDistanceVector(intersectionPoint);
                        
                        
                        
                        Vector v1 = intersectionPoint - r_point1;
                        Vector v2 = intersectionPoint - r_point2;
                        Vector v3 = intersectionPoint - r_point3;

                        Debug(string("v1: ") + v1.ToString());
                        Debug(string("v2: ") + v2.ToString());
                        Debug(string("v3: ") + v3.ToString());

                        Debug(string("2 - 1: ") + (r_point2 - r_point1).ToString());
                        Debug(string("3 - 2: ") + (r_point3 - r_point2).ToString());
                        Debug(string("1 - 3: ") + (r_point1 - r_point3).ToString());
                        
                        Vector cross1 = v1.CrossProduct(r_point2 - r_point1);
                        Vector cross2 = v2.CrossProduct(r_point3 - r_point2);
                        Vector cross3 = v3.CrossProduct(r_point1 - r_point3);

                        Debug(string("cross1: ") + (cross1).ToString());
                        Debug(string("cross2: ") + (cross2).ToString());
                        Debug(string("cross3: ") + (cross3).ToString());
                        
                        double dott1 = cross1.DotProduct(cross2);
                        double dott2 = cross2.DotProduct(cross3);
                        double dott3 = cross3.DotProduct(cross1);
                        
                        if ((dott1 >= 0 && dott2 >= 0 && dott3 >= 0) ||
                            (dott1 < 0 && dott2 < 0 && dott3 < 0)) {
                                // intersection point is inside the triangle
                            	return unique_ptr<Vector>(new Vector(intersectionPoint));
                        }
                        else {
                            // intersection point is outside of the triangle
                            continue;
                        }
                    }
                    else {
                        // segment doesn't intersect with plane
                        continue;
                    }
                }
            }
        }
    }
    return unique_ptr<Vector>();
}

unique_ptr<Vector> BoundingPolygon::Collides(const BoundingVolume& rOtherBound) const
{
    Vector center;
    for (unsigned i = 0; i < mVertices.size(); i++) {
        const Vector& old_pos = mTransformedVertices[i];//mTransformation.Transform(mVertices[i]);
        Vector new_pos = GetTransformation().Transform(mVertices[i]);
        //Debug(String("Polygon old pos: ") + old_pos.ToString());
        //Debug(String("Polygon new pos: ") + new_pos.ToString());
        Segment movement_line(old_pos, new_pos);
        unique_ptr<Vector> intersection_point = rOtherBound.Intersects(movement_line);
        if (intersection_point.get()) {
            // movement line of the vertex intersects
            return intersection_point;
        }
        center += mVertices[i];
    }
    if (mVertices.size() > 1) {
        center /= mVertices.size();
        Vector old_center = GetOldTransformation().Transform(center);
        Vector new_center = GetTransformation().Transform(center);
        Segment movement_line(old_center, new_center);
        Debug(string("BoundingPolygon::Collides old_center=") + old_center.ToString() + ", new_center=" + new_center.ToString());
        unique_ptr<Vector> intersection_point = rOtherBound.Intersects(movement_line);
        if (intersection_point.get()) {
            // movement line of the center of polygon intersects
            return intersection_point;
        }
    }
    return unique_ptr<Vector>();
}

void BoundingPolygon::Transform() 
{
    Spatial::Transform();
	for (unsigned i = 0; i < mVertices.size(); i++) {
        mTransformedVertices[i] = GetOldTransformation().Transform(mVertices[i]);
        //Object::Dbg("BoundingPolygon::Transform " + to_string(i) + ": " + mTransformedVertices[i].ToString());
    }
    
}

