//========================================================================
//  This software is free: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License Version 3,
//  as published by the Free Software Foundation.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  Version 3 in the file COPYING that came with this distribution.
//  If not, see <http://www.gnu.org/licenses/>.
//========================================================================
/*!
\file    particle-filter.cc
\brief   Particle Filter Starter Code
\author  Joydeep Biswas, (C) 2019
*/
//========================================================================

#include <algorithm>
#include <cmath>
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Geometry"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "shared/math/geometry.h"
#include "shared/math/line2d.h"
#include "shared/math/math_util.h"
#include "shared/util/timer.h"

#include "config_reader/config_reader.h"
#include "particle_filter.h"

#include "vector_map/vector_map.h"

using geometry::line2f;
using std::cout;
using std::endl;
using std::string;
using std::swap;
using std::vector;
using Eigen::Vector2f;
using Eigen::Vector2i;
using vector_map::VectorMap;

DEFINE_double(num_particles, 50, "Number of particles");

namespace particle_filter {

config_reader::ConfigReader config_reader_({"config/particle_filter.lua"});

ParticleFilter::ParticleFilter() :
    prev_odom_loc_(0, 0),
    prev_odom_angle_(0),
    odom_initialized_(false) {}

void ParticleFilter::GetParticles(vector<Particle>* particles) const {
  *particles = particles_;
}

Eigen::Vector2f ParticleFilter::LaserScanToPoint(float angle, float distance) {
    Eigen::Vector2f point;
    point[0] = laser_x_offset + distance * cos(angle);
    point[1] = distance * sin(angle);

    return point;
}

void ParticleFilter::GetPredictedPointCloud(const Vector2f& loc,
                                            const float angle,
                                            int num_ranges,
                                            float range_min,
                                            float range_max,
                                            float angle_min,
                                            float angle_max,
                                            vector<Vector2f>* scan_ptr) {
  vector<Vector2f>& scan = *scan_ptr;
  // Compute what the predicted point cloud would be, if the car was at the pose
  // loc, angle, with the sensor characteristics defined by the provided
  // parameters.
  // This is NOT the motion model predict step: it is the prediction of the
  // expected observations, to be used for the update step.

  // Note: The returned values must be set using the `scan` variable:
  scan.resize(num_ranges);

  float angle_delta = (angle_max - angle_min) / num_ranges;
  float theta = angle_min;

  // Create the laser scan points
  for (size_t i = 0; i < scan.size(); ++i) {
    // Initialize with max range
    Vector2f point = LaserScanToPoint(range_max, theta);

    // compare with lines on map
    // Optimize this with AABB if time permits
    for (size_t i = 0; i < map_.lines.size(); ++i) {
      const line2f map_line = map_.lines[i];
      line2f scan_line(0, 0, point.x(), point.y());

      bool intersects = map_line.Intersects(scan_line);

      Vector2f intersection_point;
      intersects = map_line.Intersection(scan_line, &intersection_point);

      if (intersects) {
        // Find the closest intersection that is outside the min range
        if (intersection_point.norm() < point.norm() && intersection_point.norm() > range_min) {
          point = intersection_point;
        }
      }
    }

    scan[i] = point;
    theta += angle_delta;
  }
}

void ParticleFilter::Update(const vector<float>& ranges,
                            float range_min,
                            float range_max,
                            float angle_min,
                            float angle_max,
                            Particle* p_ptr) {
  // Implement the update step of the particle filter here.
  // You will have to use the `GetPredictedPointCloud` to predict the expected
  // observations for each particle, and assign weights to the particles based
  // on the observation likelihood computed by relating the observation to the
  // predicted point cloud.

  // Particle p = *p_ptr;

  // // Get the scan that would be expected if the robot truly is at this location
  // vector<Vector2f> predicted_scan;
  // GetPredictedPointCloud(p.loc, p.angle, num_scans_predicted, range_min, range_max, angle_min, angle_max, &predicted_scan);
  
  // // double likelihood = 1.0;

  // double log_likelihood = 0;

  // float angle_delta = (angle_max - angle_min) / ranges.size();
  // float angle = angle_min;

  // // Calculate for each point in point cloud
  // for (unsigned index = 0; index < ranges.size(); index++) {
  //   Vector2f true_point = LaserScanToPoint(angle, ranges[index]);
  //   Vector2f predicted_point = predicted_scan[index];

  //   log_likelihood += pow(true_point.norm() - predicted_point.norm(), 2) / pow(update_variance, 2);

  //   // Slide deck 7: 31-32
  //   // Original:
  //   // double term = pow(exp(pow(true_point.norm() - predicted_point.norm(),2) / (pow(update_variance,2) * -2)), gamma);
  //   // likelihood *= term;
  //   angle += angle_delta;
  // }

  // p_ptr->weight = gamma * log_likelihood;

}

void ParticleFilter::NormalizeWeights() {
  // This maybe needs to be readjusted bc log likelihood
  double weight_sum = 0;
  for (auto p : particles_) {
    weight_sum += p.weight;
  }

  for (auto p : particles_) {
    p.weight = p.weight / weight_sum;
  }
}

int ParticleFilter::SearchBins(vector<float>& bins, float sample) {
  // Does a binary search for the bin that the sample falls in
  int upper = bins.size();
  int lower = 0;
  
  int index = (lower + upper) / 2;
  while (lower < upper) {
    index = (lower + upper) / 2;
    if (sample <= bins[index] && (index == 0 || sample > bins[index-1])) {
      return index;
    }

    if (sample <= bins[index]) {
      upper = index;
    } 
    else {
      lower = index +1;
    }
  }

  printf("forgot how to data structures");
  return index;
}

void ParticleFilter::Resample() {
  // // Resample the particles, proportional to their weights.
  // // The current particles are in the `particles_` variable. 
  // // Create a variable to store the new particles, and when done, replace the
  // // old set of particles:
  // NormalizeWeights();
  // vector<Particle> new_particles;

  // vector<float> bins;
  // float running_sum = 0;
  // for (unsigned i = 0; i < particles_.size(); i++) {
  //   running_sum += particles_[i].weight;
  //   bins.push_back(running_sum);
  // }

  // for (unsigned i = 0; i < particles_.size(); i++) {
  //   float sample = rng_.UniformRandom(0, 1);
  //   int index = SearchBins(bins, sample);
  //   new_particles.push_back(particles_[index]);
  // }

  // particles_ = new_particles;
}

void ParticleFilter::ObserveLaser(const vector<float>& ranges,
                                  float range_min,
                                  float range_max,
                                  float angle_min,
                                  float angle_max) {
  // A new laser scan observation is available (in the laser frame)
  // Call the Update and Resample steps as necessary.

  num_scans_predicted = ranges.size();

  // Update the weights of the particles
  for (auto p_ptr : particles_) {
    Update(ranges, range_min, range_max, angle_min, angle_max, &p_ptr);
  }

  Resample();
}

void ParticleFilter::Predict(const Vector2f& odom_loc,
                             const float odom_angle) {
  // Implement the predict step of the particle filter here.
  // A new odometry value is available (in the odom frame)
  // Implement the motion model predict step here, to propagate the particles
  // forward based on odometry.


  // You will need to use the Gaussian random number generator provided. For
  // example, to generate a random number from a Gaussian with mean 0, and
  // standard deviation 2:
  // printf("Random number drawn from Gaussian distribution with 0 mean and "
  //        "standard deviation of 2 : %f\n", x);
  if (prev_odom_loc[0] == (float)-1000) {
    // Todo do you need deep copy?
    prev_odom_loc[0] = odom_loc[0];
    prev_odom_loc[1] = odom_loc[1];
    prev_odom_angle = odom_angle;
  } else {
    for (unsigned int i = 0; i < particles_.size(); i++)
    {
      float delta_x = odom_loc[0] - prev_odom_loc[0];
      float delta_y = odom_loc[1] - prev_odom_loc[1];
      // printf("dX: %f, dY: %f \n", delta_x,delta_y);

      float trans_x = delta_x * cos(prev_odom_angle) - delta_y * sin(prev_odom_angle);
      float trans_y = delta_x * sin(prev_odom_angle) + delta_y * cos(prev_odom_angle);

      Particle particle = particles_[i];
      float next_x = rng_.Gaussian(particle.loc[0] + trans_x, k * delta_x);
      float next_y = rng_.Gaussian(particle.loc[1] + trans_y, k * delta_y);
      float next_theta = rng_.Gaussian(particle.angle + odom_angle - prev_odom_angle, k * (odom_angle - prev_odom_angle));
      particles_[i].loc[0] = next_x;
      particles_[i].loc[1] = next_y;
      // printf("X: %f, Y: %f \n", next_x,next_y);
      particles_[i].angle = next_theta;
      particles_[i].weight = 1;
    }
    prev_odom_loc[0] = odom_loc[0];
    prev_odom_loc[1] = odom_loc[1];
    prev_odom_angle = odom_angle;
  }
}

void ParticleFilter::Initialize(const string& map_file,
                                const Vector2f& loc,
                                const float angle) {
  // The "set_pose" button on the GUI was clicked, or an initialization message
  // was received from the log. Initialize the particles accordingly, e.g. with
  // some distribution around the provided location and angle.
  map_.Load(map_file);

  particles_.clear();
  double weight = 1.0 / num_initial_particles;
  for (int i = 0; i < num_initial_particles; i++) {
    float x = rng_.Gaussian(loc(0), initial_std_x);
    float y = rng_.Gaussian(loc(1), initial_std_y);
    float theta = rng_.Gaussian(angle, initial_std_theta);

    particles_.emplace_back(x, y, theta, weight);
  }
}

void ParticleFilter::GetLocation(Eigen::Vector2f* loc_ptr, 
                                 float* angle_ptr) const {
  Vector2f& loc = *loc_ptr;
  float& angle = *angle_ptr;
  // Compute the best estimate of the robot's location based on the current set
  // of particles. The computed values must be set to the `loc` and `angle`
  // variables to return them. Modify the following assignments:
  loc = Vector2f(0, 0);
  angle = 0;
}


}  // namespace particle_filter
