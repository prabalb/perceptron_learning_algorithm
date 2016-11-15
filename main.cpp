/*
 * Perceptron Learning Algorithm
 * Author: Prabal Basu (A02049867)
 * Contact: prabalb@aggiemail.usu.edu
 */

#include "./pla.h"

#include <stdlib.h>
#include <time.h>

int main() {
  std::string trainFileName;
  std::cout << "Enter the training file name: ";
  std::cin >> trainFileName;

  srand(time(NULL));
  Perceptron& ptron = Perceptron::getPerceptron(trainFileName);

  std::string skip10FoldXValidation("N");
  std::cout << "Do you want to skip the 10-fold cross-validation [Y/N]: ";
  std::cin >> skip10FoldXValidation;
  if(skip10FoldXValidation.compare("N") == 0) {
    if(ptron.getSizeOfTrainingDataSet() >= 10) {
      double wall0 = ptron.get_wall_time();
      double cpu0  = ptron.get_cpu_time();

      ptron.perform10FoldXValidation();

      double wall1 = ptron.get_wall_time();
      double cpu1  = ptron.get_cpu_time();

      std::cout << "\nTime(s) : 10-fold Cross-Validation : Wall Time : " << wall1 - wall0 << ", CPU Time : " << cpu1  - cpu0 << "\n\n";
    } else {
      std::cout << "Cannot perform 10-fold cross-validation, as the number of training instances is less than 10 ..." << std::endl;
    }
  }

  std::string skipManualTesting("N");
  std::cout << "Do you want to skip training the perceptron using all training instances and manually check its accuracy [Y/N]: ";
  std::cin >> skipManualTesting;
  if(skipManualTesting.compare("N") == 0) {
    std::cout << "Training the Perceptron using all the instances ..." << std::endl;
    ptron.populateTrainingDataSet();
    ptron.trainPerceptron();

    while(1) {
      double x1 = 0, x2 = 0;
      std::cout << "Enter test input 1: ";
      std::cin >> x1;
      std::cout << "Enter test input 2: ";
      std::cin >> x2;
      std::tuple<double, double, int> entry = std::make_tuple(x1, x2, 9999);
      ptron.isCorrectlyPredicted(entry, true);

      std::string doMoreTesting("N");
      std::cout << "Do you want to perform more manual testing [Y/N]: ";
      std::cin >> doMoreTesting;
      if(doMoreTesting.compare("N") == 0) break;
    }
  }

  std::cout << "That's all folks!" << std::endl;
  return 0;
}
