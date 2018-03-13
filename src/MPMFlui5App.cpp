#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Rand.h"
#include "cinder/gl/gl.h"
#include "cinder/Utilities.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//#include "Simulator.h"
#include "SimulatorCUDA.cuh"
#include "Common.cuh"
#include <vector>

using namespace ci;
using namespace ci::app;
using namespace std;


//extern int* cuda_main(void);

class MPMFlui5App : public App {
	SimulatorCUDA s;
	int n;
	GLfloat*            vertices;
	ColorA*             colors;

	// Buffer holding raw particle data on GPU, written to every update().
	gl::VboRef			mParticleVbo;
	// Batch for rendering particles with  shader.
	gl::BatchRef		mParticleBatch;
	gl::GlslProgRef     mGlsl;
public:
	void setup() override;
	void mouseDown(MouseEvent event) override;
	void update() override;
	void draw() override;
};

void MPMFlui5App::setup()
{	
	s.initializeGrid(400, 200);
	
	s.addParticles(10240*20);
	
	
	s.scale = 3.0f;
	n = s.particles.size();
	console() << "jumlah partikel  " << n << endl;


	mParticleVbo = gl::Vbo::create(GL_ARRAY_BUFFER, s.particles, GL_STREAM_DRAW);
	cudaGraphicsGLRegisterBuffer(&(s.cuda_vbo_resource), mParticleVbo.get()->getId(), cudaGraphicsMapFlagsNone);
	gpuErrchk(cudaPeekAtLastError());

	// Describe particle semantics for GPU.
	geom::BufferLayout particleLayout;
	particleLayout.append(geom::Attrib::POSITION, 3, sizeof(Particle), offsetof(Particle, pos));
	particleLayout.append(geom::Attrib::CUSTOM_9, 3, sizeof(Particle), offsetof(Particle, trail));
	particleLayout.append(geom::Attrib::COLOR, 4, sizeof(Particle), offsetof(Particle, color));

	// Create mesh by pairing our particle layout with our particle Vbo.
	// A VboMesh is an array of layout + vbo pairs
	auto mesh = gl::VboMesh::create(s.particles.size(), GL_POINTS, { { particleLayout, mParticleVbo } });

	try {
		mGlsl = gl::GlslProg::create(gl::GlslProg::Format()
			.vertex(
				CI_GLSL(150,
					uniform     mat4    ciModelViewProjection;
					in          vec4    ciPosition;
					in          vec4    ciColor;
					in          vec4    trailPosition;

					out         vec4    vColor;
					out         vec4    trailPos;

					void main(void) {
						gl_Position = ciModelViewProjection * ciPosition;
						trailPos = ciModelViewProjection * trailPosition;
						vColor = ciColor;
					}
				)
			)
			.geometry(

				CI_GLSL(150,
					layout(points) in;
					layout(points, max_vertices = 3) out;

					in vec4 trailPos[];
					in vec4 vColor[]; // Output from vertex shader for each vertex

					out vec4 gColor; // Output to fragment shader

					void main()
					{
						gColor = vColor[0];

						gl_Position = gl_in[0].gl_Position;
						EmitVertex();
						gl_Position = trailPos[0];
						EmitVertex();

						EndPrimitive();
					}
				)
			)
			.fragment(
				CI_GLSL(150,
					in       vec4 gColor;
					out      vec4 oColor;

					void main(void) {
						oColor = gColor;
					}
				)
			));
	}
	catch (gl::GlslProgCompileExc ex) {
		cout << ex.what() << endl;
		quit();
	}

	gl::Batch::AttributeMapping mapping({ { geom::Attrib::CUSTOM_9, "trailPosition" } });
	mParticleBatch = gl::Batch::create(mesh, mGlsl, mapping);
	gl::pointSize(1.0f); 
}

void MPMFlui5App::mouseDown( MouseEvent event )
{
}

void MPMFlui5App::update()
{
	s.updateCUDA();
}

void MPMFlui5App::draw()
{
	
	// clear out the window with black
	gl::clear(Color(0, 0, 0));
	gl::setMatricesWindowPersp(getWindowSize());
	gl::enableDepthRead();
	gl::enableDepthWrite();

	mParticleBatch->draw();

	gl::drawString(toString(static_cast<int>(getAverageFps())) + " fps", vec2(32.0f, 52.0f));
	gl::drawString(toString(static_cast<int>(s.particles.size())) + " Partciles", vec2(32.0f, 75.0f));
}


CINDER_APP(MPMFlui5App, RendererGl, [](App::Settings *settings) {
	settings->setWindowSize(1200, 600);
	settings->setMultiTouchEnabled(false);
	settings->setFrameRate(1000000.0f);
})