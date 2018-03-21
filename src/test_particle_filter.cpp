#include "particle_filter.h"
#include "gmock/gmock.h"
#include <random>
#include <numeric>


#include <iostream>

using namespace testing;
using namespace std;

class UnitializedParticleFilter : public Test {
public:
	ParticleFilter pf;
};

TEST_F(UnitializedParticleFilter, OnCreationNumberParticlesIsZero) {
	ASSERT_THAT(pf.particles.size(), Eq(0));
}

TEST_F(UnitializedParticleFilter, AfterInitializationAddsNumParticlesToParticleVector) {
	double x=0;
	double y=0;
	double theta=0;
	double std[3] = {1,2,3};
	pf.init(x, y, theta, std, 100);
	int num_particles = pf.length();

	ASSERT_NE(pf.particles.size(), 0);
	ASSERT_THAT(pf.particles.size(), Eq(num_particles));
}


class ParticleFilterWithTwoParticles : public Test {
public:
	ParticleFilter pf;
	double nu;
	double yaw_dot;

	void SetUp() override{
		double x1 = 102;
		double y1 = 65;
		double theta1 = 5 * M_PI / 8.0;
		pf.set_num_particles(2);
		Particle p1;
		ParticleFilter::set_particle_location(1, x1, y1, theta1, 1, p1);
		pf.particles.push_back(p1);
	
		double x2 = 202;
		double y2 = 65;
		double theta2 = 13 * M_PI / 8.0;
		Particle p2;
		ParticleFilter::set_particle_location(2, x1, y1, theta2, 1, p2);
		pf.particles.push_back(p2);

		nu = 110;
		yaw_dot = M_PI / 8.0;
	}

};

TEST_F(ParticleFilterWithTwoParticles, DeterministicPredictionPassesTrivialExample) {
	double expected_x(97.592046082729);
	double expected_y(75.077419972153);
	double expected_theta(2.00276531666349);
	double dt(0.1);

	ParticleFilter::prediction_deterministic(pf.particles.at(0),nu, yaw_dot, dt);

	EXPECT_NEAR(pf.particles.at(0).x, expected_x, 1e-6);
	EXPECT_NEAR(pf.particles.at(0).y, expected_y, 1e-6);
	EXPECT_NEAR(pf.particles.at(0).theta, expected_theta, 1e-6);

}

TEST_F(ParticleFilterWithTwoParticles, ResampleKeepsOriginalNumberElements) {
	 pf.resample();

	ASSERT_THAT(pf.particles.size(), Eq(pf.length()));
	
}


class InitializedParticleFilter : public Test {
public:
	ParticleFilter pf;
	
	void SetUp() {
		double x=1.0;
		double y=2.0;
		double theta=M_PI/8.0;
		double std[3] = {0.3,0.3,M_PI/32.0};

		pf.init(x, y, theta, std, 1);
	}
};



class ParticleFilterWithTwoParticlesAndTwoLandmarks : public ParticleFilterWithTwoParticles {
public:

	Map m;
	double std_landmark[2];
	vector<LandmarkObs> landmark_obs;
	void SetUp() override{

		double x1 = 102;
		double y1 = 65;
		double theta1 = 5 * M_PI / 8.0;
		pf.set_num_particles(2);
		Particle p1;
		ParticleFilter::set_particle_location(1, x1, y1, theta1, 1, p1);
		pf.particles.push_back(p1);
	
		double x2 = 202;
		double y2 = 65;
		double theta2 = 13 * M_PI / 8.0;
		Particle p2;
		ParticleFilter::set_particle_location(2, x1, y1, theta2, 1, p2);
		pf.particles.push_back(p2);

		nu = 110;
		yaw_dot = M_PI / 8.0;

		// two landmarks set so that are where particle
		// 1 would suggest they are. 
		m.landmark_list.push_back({ 1, 75, 160 });
		m.landmark_list.push_back({ 2, 115, -95 });
		landmark_obs = { {1, 100, -10}, {2, -150, 50}};
		std_landmark[0] = 0.3;
		std_landmark[1] = 0.3;

	}


};


TEST_F(ParticleFilterWithTwoParticlesAndTwoLandmarks, UpdateWeightsGiveParticle2EssentiallyZeroWeight) {
	pf.updateWeights(500, std_landmark, landmark_obs, m);
	ASSERT_THAT(pf.particles.at(0).weight, Ne(0));
	ASSERT_THAT(pf.particles.at(1).weight, Eq(0));
}

int main(int argc, char** argv) {

	testing::InitGoogleMock(&argc, argv);
	return RUN_ALL_TESTS();
}
