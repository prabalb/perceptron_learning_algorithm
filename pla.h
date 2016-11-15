/*
 * Perceptron Learning Algorithm
 * Author: Prabal Basu (A02049867)
 * Contact: prabalb@aggiemail.usu.edu
 */

#include <iostream>

#include <vector>
#include <tuple>
#include <string>

using namespace std;

#define LEARNING_RATE 0.001
#define ITERATION_LIMIT 100000
#define ERROR_TOLERENCE 0.0001
#define EPOCH 1000

//#define COLLECT_STAT

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

typedef std::vector<std::tuple<double, double, int> > DataSet;

class Perceptron {

  private:
    DataSet allTds; // all training instances
    DataSet tds;
    DataSet testSet; // instances used for testing the Perceptron
    std::vector<double> weights; // weights associated with the inputs and the bias

    Perceptron(std::string& trainFileName);
    void operator=(Perceptron&);
    Perceptron(const Perceptron&);

  public:
    static Perceptron& getPerceptron(std::string& trainFileName) {
      static Perceptron pt(trainFileName);
      return pt;
    }

    bool isCorrectlyPredicted(std::tuple<double, double, int>& input, bool printPredictedOutput=false);

    int getSizeOfTrainingDataSet();

    void populateTrainingDataSet(std::string& trainingFileName);
    void populateTrainingDataSet();
    void trainPerceptron();
    void initializeWeights();
    void updateWeights(int weightIndex, double input, double actualOutput, double predictedOutput);
    void perform10FoldXValidation();
    void crossSplitDataSet(int counter);
    void resetDB();

    double activationFunc(double input);
    double calculateAvgSquaredError(std::vector<double>& predictedOutput);
    double reportAccuracy(int counter);

    double get_wall_time();
    double get_cpu_time();
};

#endif // PERCEPTRON_H
