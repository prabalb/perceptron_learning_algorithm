/*
 * Perceptron Learning Algorithm
 * Author: Prabal Basu (A02049867)
 * Contact: prabalb@aggiemail.usu.edu
 */

#include "./pla.h"

#include <sys/time.h>

#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <cassert>

Perceptron::Perceptron(std::string& trainFileName) {
  if(allTds.size() == 0 && weights.size() == 0) {
    populateTrainingDataSet(trainFileName);
    initializeWeights();
  }
}

// parse the input file to populate training instances info
void Perceptron::populateTrainingDataSet(std::string& trainingFileName) {
  std::ifstream file;
  file.open(trainingFileName.c_str());
  if (file.is_open()) {
    while (!file.eof())
    {
      char buf[100];
      file.getline(buf, 100);
      
      const char* token[4] = {};
      token[0] = strtok(buf, " ");
      if(!token[0]) continue; // blank line
      double x1 = atof(token[0]);

      token[1] = strtok(0, " "); 
      if(!token[1]) {
        std::cout << "Something is wrong with the format of the training file ..." << std::endl;
        exit(0);
      }
      double x2 = atof(token[1]);

      token[2] = strtok(0, " "); 
      if(!token[2]) {
        std::cout << "Something is wrong with the format of the training file ..." << std::endl;
        exit(0);
      }
      int output = atoi(token[2]);

      allTds.push_back(std::make_tuple(x1, x2, output));
    }

    file.close();
    return;
  }
  std::cout << "Could not open the training file specified, please try again ..." << std::endl;
  exit(0);
}

// total number of instances in the training data set
int Perceptron::getSizeOfTrainingDataSet() {
  return allTds.size();
}

void Perceptron::populateTrainingDataSet() {
  assert(allTds.size() > 0);
  tds = allTds;
}

// main routine to train the perceptron
void Perceptron::trainPerceptron() {
  double overall_avg_error = 0.0;
  double error = 1000.0;
  int num_iteration = 1;
  std::vector<double> predictedOutputs;
  while (error >= ERROR_TOLERENCE && num_iteration < ITERATION_LIMIT) {
    predictedOutputs.clear();
    for(size_t pos = 0; pos < tds.size(); pos++) {
      double inputToActivationFunc = weights[0] * std::get<0>(tds[pos]) + weights[1] * std::get<1>(tds[pos]) + weights[2];
      double predictedOutput = activationFunc(inputToActivationFunc);
      predictedOutputs.push_back(predictedOutput);
      double actualOutput = std::get<2>(tds[pos]);
      updateWeights(0, std::get<0>(tds[pos]), actualOutput, predictedOutput);
      updateWeights(1, std::get<1>(tds[pos]), actualOutput, predictedOutput);
      updateWeights(2, 1, actualOutput, predictedOutput);
    }
    error = calculateAvgSquaredError(predictedOutputs);
    overall_avg_error += error;
#ifdef COLLECT_STAT
    if(num_iteration % EPOCH == 0) {
      std::cout << error << std::endl;
    }
#endif
    num_iteration++;   
  }
  overall_avg_error /= num_iteration;
  //std::cout << "Residual Average error: " << overall_avg_error << std::endl;
}

// initialize weights randomly between [-0.2, 0.2]
void Perceptron::initializeWeights() {
  for(int pos = 0; pos < 3; pos++) {
    int val = rand() % 5000 - 2000;
    weights.push_back((double)val/10000);
  }
}

// Delta rule to update the weights
// delta_w = -1 * learning_rate * difference_b/w_actual_and_predicted_output * derivative_of_sigmoid_activation_func * input
void Perceptron::updateWeights(int weightIndex, 
                               double input, 
                               double actualOutput, 
                               double predictedOutput) {

  double deltaWeight = -1 * LEARNING_RATE * (predictedOutput - actualOutput) * predictedOutput * (1 - predictedOutput) * input;
  weights[weightIndex] += deltaWeight;
}

// activation function (sigmoid)
double Perceptron::activationFunc(double input) {
  double output = 1 / (1 + exp (-1 * input));
  return output;
}

// calculation of average of the squared error over the training set
double Perceptron::calculateAvgSquaredError(std::vector<double>& predictedOutput) {
  assert(predictedOutput.size() == tds.size());

  double avgSquaredError = 0.;
  for(size_t pos = 0; pos < tds.size(); pos++) {
    double modifiedPredictedOutput = predictedOutput[pos] <= 0.5 ? 0 : 1;
    //double modifiedPredictedOutput = predictedOutput[pos];
    double error = modifiedPredictedOutput - std::get<2>(tds[pos]);
    double sqError = error * error;
    avgSquaredError += sqError;
  }
  avgSquaredError /= tds.size();
  return avgSquaredError;
}

// perform 10-fold cross validation
void Perceptron::perform10FoldXValidation() {
  std::cout << "Performing 10-fold cross-validation ..." << std::endl;
  const int num_split = 10;
  std::cout << "Following result is for " << num_split << " splits ..." << std::endl;
  std::cout << "Test Run   |   Accuracy on Test Set (%)" << std::endl;
  double avg_accuracy = 0.0;
  int counter = 1;
  while(counter <= num_split) {
    resetDB();
    crossSplitDataSet(counter);
    initializeWeights();
    trainPerceptron();
    double accuracy = reportAccuracy(counter);
    avg_accuracy += accuracy;
    std::string format = (counter == 10) ? "         |   " : "          |   ";
    std::cout << counter << format << accuracy << std::endl;
    counter++;
  }
  avg_accuracy /= num_split;

  std::cout << "\nAverage accuracy(%): " << avg_accuracy << std::endl;
}

// split the entire data set into 10 parts for 10-fold cross-validation
void Perceptron::crossSplitDataSet(int counter) {
  assert(tds.size() == 0 && testSet.size() == 0);
  int total_data_set_size = allTds.size();
  int startPos = (counter - 1) * (total_data_set_size / 10);
  int endPos = (counter == 10) ? (startPos + floor(float(total_data_set_size) / 10) - 1 + (total_data_set_size % 10))
                               : (startPos + floor(float(total_data_set_size) / 10) - 1);

  for(int pos = startPos; pos <= endPos; pos++) {
    testSet.push_back(allTds[pos]);
  }

  tds = allTds;
  for(int pos = startPos; pos <= endPos; pos++) {
    tds.erase(allTds.begin() + pos);
  }
}

// reset training, test and weights data structures
void Perceptron::resetDB() {
  tds.clear();
  testSet.clear();
  weights.clear();
}

// calculate prediction accuracy of the perceptron
double Perceptron::reportAccuracy(int counter) {
  assert(testSet.size() > 0);
  int num_correct_pred = 0;
  for(size_t i = 0; i < testSet.size(); i++) {
    if(!isCorrectlyPredicted(testSet[i])) continue;
    num_correct_pred++;
  }
  return (double) num_correct_pred * 100 / testSet.size();
}

// check if the predicted output is correct
bool Perceptron::isCorrectlyPredicted(std::tuple<double, double, int>& input, bool printPredictedOutput) {
  double inputToActivationFunc = weights[0] * std::get<0>(input) + weights[1] * std::get<1>(input) + weights[2];
  double predictedOutput = activationFunc(inputToActivationFunc);
  predictedOutput = predictedOutput <= 0.5 ? 0 : 1;
  if(printPredictedOutput) {
    std::cout << "Predicted Output: " << predictedOutput << std::endl;
  }
  if(predictedOutput == std::get<2>(input)) return true;
  return false;
}

double Perceptron::get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time, NULL)) {
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

double Perceptron::get_cpu_time()
{
  return (double)clock() / CLOCKS_PER_SEC;
}
