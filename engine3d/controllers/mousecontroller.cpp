#include "mousecontroller.h" // class's header file
#include "engine3d/scenegraph/viewport.h" // class's header file
#include "engine3d/scenegraph/scene.h" // class's header file
#include <math.h>
#include <GL/glut.h>

using namespace engine3d;


MouseController::MouseController(const Scene& rScene) : 
                                        TransformationController(rScene, true),
                                        x(-1),
                                        y(-1),
                                        mpButton(nullptr),
										mpModifier(nullptr)
{
}

MouseController::~MouseController()
{
}

void MouseController::OnMove(int xPos, int yPos)
{
    const Viewport& r_viewport = mrScene.GetViewport();
    //int xCenter = r_viewport.GetXC();
    //int yCenter = r_viewport.GetYC();
    if (x < 0) {
        x = xPos;
    }
    if (y < 0) {
        y = yPos;
    }
    double dx = ((double) xPos - x) / r_viewport.GetWidth() * 2 * M_PI;
    double dy = ((double) yPos - y) / r_viewport.GetHeight() * 2 * M_PI;
    if (mpButton && *mpButton ==  GLUT_LEFT_BUTTON) {
    	if (mpModifier && *mpModifier == GLUT_ACTIVE_SHIFT) {
    		Vector trVector(-dx, -dy, 0);
    		Translate(trVector);

    	} else {
            double rotAngle = sqrt(dx * dx + dy * dy);
            if (rotAngle > 0) {
    			Vector rotVector(dy, -dx, 0);
    			Rotate(rotVector, rotAngle);
            }
    	}
    }
    else if (mpButton && *mpButton == GLUT_RIGHT_BUTTON) {
        double rotAngle = fabs(dx);
        if (rotAngle > 0) {
            Vector rotVector(0, 0, dx);
            Rotate(rotVector, rotAngle);
        }
        if (dy != 0) {
            Vector trVector(0, 0, -dy);
            Translate(trVector);
        }
    }
    x = xPos;
    y = yPos;
}

void MouseController::OnButton(int button, int state, int xPos, int yPos) {
	if (state == GLUT_DOWN) {
		this->mpButton = new int(button);
	    x = xPos;
	    y = yPos;
    	this->mpModifier=new int(glutGetModifiers());
	} else {
		delete mpButton;
		this->mpButton = nullptr;
		delete mpModifier;
		this->mpModifier = nullptr;
	}
}
