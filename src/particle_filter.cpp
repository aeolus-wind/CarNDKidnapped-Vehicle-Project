/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>
#include "helper_functions.h"

#include "particle_filter.h"

using namespace std;

int ParticleFilter::length() {
	return num_particles;
}

void ParticleFilter::set_num_particles(int n) {
	num_particles = n;
}

void ParticleFilter::set_particle_location(const int id, const double x, const double y, 
	const double theta, const double weight, Particle &p) {
	p.id = id;
	p.x = x;
	p.y = y;
	p.theta = theta;
	
	p.weight = weight;
	if (weight > 1) {
		cout << "input weight must be <= 1" << endl; //really, it only matters that the weight is initilaized to something reasonable
		return;
	}
	
}

void ParticleFilter::init(double x, double y, double theta, double std[], int n) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	set_num_particles(n);
	default_random_engine gen;

	normal_distribution<double> distribution_x(x, std[0]);
	normal_distribution<double> distribution_y(y, std[1]);
	normal_distribution<double> distribution_theta(theta, std[2]);
	
	double initial_w = 1;

	for (int i = 0; i < num_particles; i++) {
		int id = i;
		double rand_x = distribution_x(gen);
		double rand_y = distribution_y(gen);
		double rand_theta = distribution_theta(gen);
		
		Particle p;
		set_particle_location(id, rand_x, rand_y, rand_theta, initial_w, p);
				

		particles.push_back(p);
	}
}

double ParticleFilter::gauss_2d(double sq_res_x, double sq_res_y, double std_x, double std_y) {
	return 1 / (2 * M_PI*std_x*std_y)*exp(-(sq_res_x / (2 * std_x*std_x) + sq_res_y / (2 * std_y*std_y)));
}



void ParticleFilter::prediction_deterministic(Particle& p, double velocity, double yaw_rate, double dt) {
	if (essentially_equal(yaw_rate, 0.0, 1e-9)) {
		p.x += velocity*cos(p.theta)*dt;
		p.y += velocity*sin(p.theta)*dt;
	}
	else {
		double nu_div_yaw_dot = velocity / yaw_rate;
		double theta_prime = p.theta + yaw_rate * dt;
		p.x += nu_div_yaw_dot * (sin(theta_prime) - sin(p.theta));
		p.y += nu_div_yaw_dot * (cos(p.theta) - cos(theta_prime));
		p.theta = theta_prime;
		normalize_theta(p.theta);
	}

}



void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	normal_distribution<double> distribution_x(0, std_pos[0]);
	normal_distribution<double> distribution_y(0, std_pos[1]);
	normal_distribution<double> distribution_theta(0, std_pos[2]);

	
	for (int i = 0; i < num_particles; i++) {
		prediction_deterministic(particles.at(i), velocity, yaw_rate, delta_t);
		particles.at(i).x += distribution_x(gen);
		particles.at(i).y += distribution_y(gen);
		particles.at(i).theta += distribution_theta(gen);
		normalize_theta(particles.at(i).theta);
	}

}

std::vector<pair<double, double>> ParticleFilter::dataAssociation(const Map map_landmarks, std::vector<LandmarkObs>& observations, Particle& p) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	//returns vector of tuples of squared errors to be used in gaussian

	//take observations and convert to difference between observations and prediction in the map frame
	// assuming that a particle is in a certain direction. 
	
	int id_min;
	vector<pair<double, double>> square_residuals;
	double x_res_min;
	double x_min;
	double y_res_min;
	double y_min;
	double dist_min;
	vector<int> associations;
	vector<double> sense_x;
	vector<double> sense_y;

	for (int j = 0; j < observations.size(); j++) {
		int current_min = map_landmarks.landmark_list.at(0).id_i;
		LandmarkObs ObservationToCompare = observations.at(j);
		x_res_min = pow(map_landmarks.landmark_list.at(0).x_f - ObservationToCompare.x, 2);
		y_res_min = pow(map_landmarks.landmark_list.at(0).y_f - ObservationToCompare.y, 2);
		dist_min = x_res_min + y_res_min;

		id_min = map_landmarks.landmark_list.at(0).id_i;
		x_min = map_landmarks.landmark_list.at(0).x_f;
		y_min = map_landmarks.landmark_list.at(0).y_f;

		//iterate over all landmarks to find association of observation with landmark
		for (int i = 1; i < map_landmarks.landmark_list.size(); i++) {
			double current_x_res = pow(map_landmarks.landmark_list.at(i).x_f - ObservationToCompare.x, 2);
			double current_y_res = pow(map_landmarks.landmark_list.at(i).y_f - ObservationToCompare.y, 2);
			double current_difference = current_x_res + current_y_res;
			if (current_difference < dist_min) {
				dist_min = current_difference;
				x_res_min = current_x_res;
				y_res_min = current_y_res;
				id_min = map_landmarks.landmark_list.at(i).id_i;
				x_min = ObservationToCompare.x;
				y_min = ObservationToCompare.y;
			}
					
		}
		// at this stage, will have the minimal pair for a a landmark
		// so observation has been matched
		// can reuse all values
		pair<double, double> res_pair{ x_res_min, y_res_min };
		square_residuals.push_back(res_pair);
		associations.push_back(id_min);
		sense_x.push_back(x_min);
		sense_y.push_back(y_min);	
		
		

	}
	SetAssociations(p, associations, sense_x, sense_y);
	return square_residuals;
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//give each particle a weight
	//then draw from the weights
	//don't even need to renormalize?
	
	for (int i = 0; i < num_particles; i++) {
		Particle current_p = particles.at(i);
		double theta_p = current_p.theta; 
		double x_p = current_p.x;
		double y_p = current_p.y;

		double cos_theta_p = cos(theta_p);
		double sin_theta_p = sin(theta_p);

		vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); j++) {
			double x_o = observations.at(j).x;
			double y_o = observations.at(j).y;

			double x_m = cos_theta_p * x_o - sin_theta_p * y_o + x_p;
			double y_m = sin_theta_p * x_o + cos_theta_p * y_o + y_p;

			LandmarkObs obs_trans{ observations.at(j).id, x_m, y_m };
			transformed_observations.push_back(obs_trans);
		}

		vector<pair<double, double>> square_residuals = dataAssociation(map_landmarks, transformed_observations, particles.at(i));
		double accum = 1;
		for (int j = 0; j < square_residuals.size(); j++) {
			accum *= gauss_2d(square_residuals.at(j).first, square_residuals.at(j).second, std_landmark[0], std_landmark[1]);
			
		}
		particles.at(i).weight = accum;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd;
    std::mt19937 gen(rd());
	std::vector<int> weights;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(i);
	}
	if (num_particles <= 1) {
		cout << "must be more than 1 particle to resample" << endl;
		return;
	}

	discrete_distribution<int> discrete(weights.begin(), weights.end());

	int draw_index = discrete(gen);
	double max_weight = 0;

	for (int i = 0; i < num_particles; i++) {
		if (particles.at(i).weight > max_weight)
			max_weight = particles.at(i).weight;
	}
	
	uniform_real_distribution<double> distribution(0.0, 2*max_weight);
	double beta=0;
	beta = distribution(gen);
	vector<Particle> resample;
	for (int i = 0; i < num_particles; i++) {
		beta += distribution(gen);
		while (beta > particles.at(draw_index).weight) {
			beta -= particles.at(draw_index).weight;
			draw_index++;
			draw_index =draw_index % num_particles;
		}

		resample.push_back(particles.at(draw_index));

	}
	particles = resample;
	
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
	
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
