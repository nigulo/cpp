/**************************
 * Includes
 *
 **************************/

#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <fstream>
#include <iostream>

#include "engine3d/geometry/vector.h"
#include "engine3d/meshes/vertex.h"
#include "engine3d/geometry/triangle.h"
#include "engine3d/meshes/triangles.h"
#include "engine3d/scenegraph/node.h"
#include "engine3d/attributes/color.h"
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
#include "engine3d/projection/perspective.h"

using namespace std;

using namespace engine3d;
using namespace base;

MouseController* pMouseController;
double theta = 0;
Scene* pScene = nullptr;
bool doubleBuffer = true;
long tickCount = 0;

// This is needed only to overcome some unknown bug
#include <pthread.h>
void junk() {
  int i;
  i=pthread_getconcurrency();
};

////////////////////////////
//Vertex v1(-0.1, -0.1, 2);
//Vertex v2(-0.1, 0.1, 2);
//Vertex v3(0.1, 0.1, 2);
//Vertex v4(0.1, -0.1, 2);
//Triangles tr;

//Node n("W");
////////////////////////////

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

void keyboardFunc(unsigned char key, int x, int y) {
    if (key == 'r' || key == 'R') {
        // Do stuff
    }
}

void renderFunc() {
    Object::Dbg("renderFunc 1");
	pScene->Render();
	if (doubleBuffer) {
	    Object::Dbg("renderFunc 1.5");
		glutSwapBuffers();
	} else {
		glFlush();
	}
    Object::Dbg("renderFunc 2");
}

void init() {
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		Object::Dbg("glewInit failed");
		exit(1);
	}
	if (!GLEW_VERSION_2_1) { // check that the machine supports the 2.1 API.
		Object::Dbg("!GLEW_VERSION_2_1");
		exit(1);
	}
    Object::Dbg("main -4");
    //BoundingPolygon* p_bp = new BoundingPolygon();
    //p_bp->AddVertex(Vector(0, 0, 0.5));
    //p_bp->AddVertex(Vector(0, 0, 0));
    //camera.SetCollisionBound(p_bp);//new BoundingSphere(Vector(0, 0, 0), 0.5));


    SceneLoader sl;
    pScene = sl.Load();
    Perspective* p_projection = new Perspective(pScene->GetProgram());
    Camera* p_camera = new Camera(pScene->GetProgram(), p_projection);
    pScene->SetCamera(p_camera);
    //Object::Dbg("main -2");
    //Object::Dbg("main -1");

    //string textureFile("ConcreteWall.png");
    //Texture* p_tex = new Texture(textureFile);
    //Object::Dbg("main -0.5");
    //Sphere* p_s = new Sphere(0.5, 10, 10);
    //p_s->SetTexture(p_tex);
    //Transformation t;
    //t.SetRotation(Vector(1, 1, 0), -M_PI / 4);
    //t.SetTranslation(Vector(2, 2, 2));
    //p_s->SetTransformation(t);
    //n.AddChild(p_s);
    //scene->SetNode(&n);

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
    
    //v1.SetColor(Color(1, 0, 0));
    //v2.SetColor(Color(0, 1, 0));
    //v3.SetColor(Color(0, 0, 1));
    //v4.SetColor(Color(0, 1, 1));
    //tr.Add(v1, v2, v3);

    Object::Dbg("main 2");
    
    //n.AddChild(&tr);
    //n.AddChild(p_s);
    //scene->SetNode(&n);

    Object::Dbg("main 3");

    pMouseController = new MouseController(*pScene);
    pMouseController->AddTarget(&pScene->GetCamera());
    
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

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    //glEnable(GL_TEXTURE_2D);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    Object::Dbg("main 6");
}

void update() {
    int milliseconds = glutGet(GLUT_ELAPSED_TIME);
    GLfloat seconds = (GLfloat)milliseconds * (1.0f/1000.0f);
    // Do stuff
    //glutPostRedisplay();
}

static void reshape(int w, int h) {
    //g_resources.window_size[0] = w;
    //g_resources.window_size[1] = h;
    //update_p_matrix(g_resources.p_matrix, w, h);
    //projection.Update(w, h);
    glViewport(0, 0, w, h);
}


int main (int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | (doubleBuffer ? GLUT_DOUBLE : GLUT_SINGLE));
    glutInitWindowSize(640, 480);
    glutInitWindowPosition(100,100);
    glutCreateWindow("OpenGL - First window demo");
    glutDisplayFunc(renderFunc);
    glutIdleFunc(update);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);
    //glutPassiveMotionFunc(motionFunc);
    glutKeyboardFunc(keyboardFunc);
    glutReshapeFunc(reshape);

    init();

    /*
    glewInit();
    if (!GLEW_VERSION_2_0) {
        fprintf(stderr, "OpenGL 2.0 not available\n");
        return 1;
    }

    init_gl_state();
    if (!make_resources()) {
        fprintf(stderr, "Failed to load resources\n");
        return 1;
    }
	*/
    glutMainLoop();
    delete pScene;
    return 0;
}
