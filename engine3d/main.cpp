/**************************
 * Includes
 *
 **************************/

#include <math.h>
#include <gl.h>
#include <freeglut.h>
#include <fstream>

#include "base/string.h"
#include "engine3d/geometry/vector.h"
#include "engine3d/meshes/vertex.h"
#include "engine3d/geometry/triangle.h"
#include "engine3d/meshes/triangles.h"
#include "engine3d/scenegraph/node.h"
#include "engine3d/attributes/color.h"
#include "engine3d/attributes/texture.h"
#include "engine3d/meshes/sphere.h"
#include "engine3d/scenegraph/camera.h"
#include "engine3d/geometry/transformation.h"
#include "engine3d/scenegraph/viewport.h"
#include "engine3d/controllers/mousecontroller.h"
#include "engine3d/scenegraph/sceneloader.h"
#include "base/thread.h"
#include "engine3d/controllers/transformationinterpolator.h"
#include "engine3d/geometry/beziercurve.h"
#include "engine3d/containment/boundingsphere.h"
#include "engine3d/containment/boundingpolygon.h"

using namespace engine3d;
using namespace base;

/**************************
 * Function Declarations
 *
 **************************/

MouseController* pMouseController;
double theta = 0;
Scene* scene;
Camera camera;//(90.0f, 1.0f, 0.5f, 10.0f);
bool doubleBuffer = true;
long tickCount = 0;

////////////////////////////
Vertex v1(-0.1, -0.1, 2);
Vertex v2(-0.1, 0.1, 2);
Vertex v3(0.1, 0.1, 2);
Vertex v4(0.1, -0.1, 2);
Triangles tr;
TriangleStrip trs;

Node n("W");
////////////////////////////

/**************************
 * WinMain
 *
 **************************/
// test
void motionFunc(int x, int y) {
    Object::Dbg("motionFunc 1");
    pMouseController->OnMove(x, y);
    glutPostRedisplay();
    Object::Dbg("motionFunc 2");
}

void mouseFunc(int button, int state, int x, int y) {
    Object::Dbg("mouseFunc 1");
    pMouseController->OnButton(button, state, x, y);
    glutPostRedisplay();
    Object::Dbg("mouseFunc 2");
}

void renderFunc() {
    Object::Dbg("renderFunc 1");
	scene->Render();
	if (doubleBuffer) {
		glutSwapBuffers();
	} else {
		glFlush();
	}
    Object::Dbg("renderFunc 2");
}

void init() {
    Object::Dbg("main -4");
    BoundingPolygon* p_bp = new BoundingPolygon();
    p_bp->AddVertex(Vector(0, 0, 0.5));
    p_bp->AddVertex(Vector(0, 0, 0));
    camera.SetCollisionBound(p_bp);//new BoundingSphere(Vector(0, 0, 0), 0.5));

    scene = new Scene();
    scene->SetCamera(&camera);

    SceneLoader sl(*scene);
    sl.Load();
    //Object::Dbg("main -2");
    //Object::Dbg("main -1");

    //String textureFile("ConcreteWall.bmp");
    //Texture* p_tex = new Texture(textureFile);
    //Object::Dbg("main -0.5");
    //Sphere* p_s = new Sphere(0.5, 10, 10);
    //p_s->SetTexture(p_tex);
    //p_s->SetTransformation(Transformation(Vector(0, 0, 5000)));
    //n.AddChild(p_s);
    //scene->SetNode(&n);
    //scene->GetNode()->AddChild(p_s);

    //Sphere s1(0.5, 20, true);
    //s1.SetTexture(&tex);
    //s1.SetTransformation(Transformation(Vector(0, 0, 5)));
    //s1.TEST_WORLD_COORDS = true;
    //s1.SetDistant();
    
    Object::Dbg("main 0");
    
//    Sphere s2(0.5, 20, true);
//    s2.SetTexture(&tex);
//    s2.SetTransformation(Transformation(Vector(0, 0, 10)));
//    Sphere s3(0.5, 20, true);
//    s3.SetTexture(&tex);
//    s3.SetTransformation(Transformation(Vector(0, 0, 15)));
//    Sphere s4(0.5, 20, true);
//    s4.SetTexture(&tex);
//    s4.SetTransformation(Transformation(Vector(0, 0, 20)));
//    Sphere s5(0.5, 20, true);
//    s5.SetTexture(&tex);
//    s5.SetTransformation(Transformation(Vector(0, 0, 25)));
//    Sphere s6(0.5, 20, true);
//    s6.SetTexture(&tex);
//    s6.SetTransformation(Transformation(Vector(0, 0, 30)));
//    Sphere s7(0.5, 20, true);
//    s7.SetTexture(&tex);
//    s7.SetTransformation(Transformation(Vector(0, 0, 35)));
//    Sphere s8(0.5, 20, true);
//    s8.SetTexture(&tex);
//    s8.SetTransformation(Transformation(Vector(0, 0, 40)));

    Object::Dbg("main 1");
    
    v1.SetColor(Color(1, 0, 0));
    v2.SetColor(Color(0, 1, 0));
    v3.SetColor(Color(0, 0, 1));
    v4.SetColor(Color(0, 1, 1));
    tr.Add(v1, v2, v3);

    //trs.AddVertex(v1);
    //trs.AddVertex(v2);
    //trs.AddVertex(v3);
    //trs.AddVertex(v4);
    //trs.AddVertex(v1);

    Object::Dbg("main 2");
    
    n.AddChild(&tr);
    //n.AddChild(p_s);
    //scene->SetNode(&n);

    Object::Dbg("main 3");

    pMouseController = new MouseController(*scene);
    pMouseController->AddTarget(&scene->GetCamera());
    
    Object::Dbg("main 4");
    /*
    TransformationInterpolator ti(*scene, scene->GetTime(), -1, 30);
    ArrayList<Vector> points;
    points.Add(Vector(-6, 0, 0));
    points.Add(Vector(-5, 0, 0));
    points.Add(Vector(0, 0, -6));
    points.Add(Vector(0, 0, -5));
    points.Add(Vector(6, 0, 0));
    points.Add(Vector(5, 0, 0));
    points.Add(Vector(0, 0, 6));
    points.Add(Vector(0, 0, 5));
    points.Add(Vector(-6, 0, 0));
    //points.Add(Vector(2, 0, 0));
    //points.Add(Vector(3, 0, 0));
    //points.Add(Vector(4, 0, 0));
    BezierCurve bc(8);
    bc.SetPoints(points);
    ti.SetTrajectory(&bc);
    
    Object::Dbg("main 5");
    Sphere s(0.5, 20, 20, true);
    scene->GetNode()->AddChild(&s);
    //s.SetCollisionBound(new BoundingSphere(Vector(), 0.5));
    ti.AddTarget(&s);
    scene->AddController(&ti);
    */
//    Triangles t;
//    t.Add(Vertex(0, -3, 4), Vertex(0, 2, 5), Vertex(0, 2, 2));
//    BoundingPolygon* bp = new BoundingPolygon();
//    bp->AddVertex(Vector(0, -3, 4));
//    bp->AddVertex(Vector(0, 2, 5));
//    bp->AddVertex(Vector(0, 2, 2));
//    t.SetCollisionBound(bp);
//    scene->GetNode()->AddChild(&t);

    Object::Dbg("main 6");
}

int main (int argc, char* argv[]) {

    glutInit(&argc, argv);
    glutInitDisplayMode(doubleBuffer ? GLUT_DOUBLE : GLUT_SINGLE);
    glutInitWindowSize(500,500);
    glutInitWindowPosition(100,100);
    glutCreateWindow("OpenGL - First window demo");
    init();
    glutDisplayFunc(renderFunc);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);
    //glutPassiveMotionFunc(motionFunc);
    glutMainLoop();
    delete scene;
    return 0;
}
