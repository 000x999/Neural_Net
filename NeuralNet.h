#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>

struct Neuron;
typedef std::vector<Neuron> Layer;

struct Connections {
    double weight;
    double deltaWeight;
};

struct TrainingData {
private:
    std::ifstream _dataFile;
public:
    TrainingData(const std::string FileName) {
        _dataFile.open(FileName.c_str());
        if (!_dataFile.is_open()) {
            throw std::runtime_error("Unable to open file: " + FileName);
        }
    }
    bool IsEof(void) const { return _dataFile.eof(); }
    void GetTopology(std::vector<unsigned>& topology) {
        std::string line;
        std::string label;
        std::getline(_dataFile, line);
        std::stringstream ss(line);
        ss >> label;
        if (label != "topology:") {
            std::cerr << "Error: Expected 'topology:', found '" << label << "'" << std::endl;
            return;
        }
        unsigned n;
        while (ss >> n) {
            topology.push_back(n);
        }
        std::cout << "Debug: Topology read: ";
        for (unsigned val : topology) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    unsigned GetNextInputs(std::vector<double>& InputVals) {
        InputVals.clear();
        std::string line;
        std::string label;
        while (std::getline(_dataFile, line)) {
            std::stringstream ss(line);
            ss >> label;
            if (label == "in:") {
                double OneVal;
                while (ss >> OneVal) {
                    InputVals.push_back(OneVal);
                }
                std::cout << "Debug: Number of input values read: " << InputVals.size() << std::endl;
                return InputVals.size();
            }
        }
        std::cerr << "Error: Couldn't find 'in:' line" << std::endl;
        return 0;
    }
    unsigned GetTargetOutputs(std::vector<double>& TargetVals) {
        TargetVals.clear();
        std::string line;
        std::string label;
        while (std::getline(_dataFile, line)) {
            std::stringstream ss(line);
            ss >> label;
            if (label == "out:") {
                double OneVal;
                while (ss >> OneVal) {
                    TargetVals.push_back(OneVal);
                }
                std::cout << "Debug: Number of target values read: " << TargetVals.size() << std::endl;
                return TargetVals.size();
            }
        }
        std::cerr << "Error: Couldn't find 'out:' line" << std::endl;
        return 0;
    }
};

struct Neuron {
public:
    Neuron(uint16_t numOutputs, uint16_t myIndex)
        : _myIndex(myIndex)
    {
        for (uint16_t c = 0; c < numOutputs; ++c) {
            _outputWeights.push_back(Connections());
            _outputWeights.back().weight = RandomWeight();
        }
    }
    void SetOutputVal(double val) { _outputVal = val; }
    double GetOutputVal() const { return _outputVal; }
    void FeedNeurons(const Layer& prevLayer) {
        double sum = 0.0;
        for (uint16_t n = 0; n < prevLayer.size(); ++n) {
            sum += prevLayer[n].GetOutputVal() *
                prevLayer[n]._outputWeights[_myIndex].weight;
        }
        _outputVal = Neuron::TransferFunction(sum);
        std::cout << "Debug: FeedNeurons result: " << _outputVal << std::endl;
    }
    void CalcOutputGradients(double targetVal)
    {
        double delta = targetVal - _outputVal;
        _gradient = delta * Neuron::TransferFunctionDerivative(_outputVal);
        std::cout << "Debug: CalcOutputGradients - targetVal: " << targetVal
            << ", _outputVal: " << _outputVal
            << ", delta: " << delta
            << ", _gradient: " << _gradient << std::endl;
    }

    void CalcHiddenGradients(const Layer& nextLayer)
    {
        double dow = SumDOW(nextLayer);
        _gradient = dow * Neuron::TransferFunctionDerivative(_outputVal);
        std::cout << "Debug: CalcHiddenGradients - dow: " << dow
            << ", _outputVal: " << _outputVal
            << ", _gradient: " << _gradient << std::endl;
    }
    void UpdateWeights(Layer& prevLayer)
    {
        const double eta = 0.10;  // learning rate
        const double alpha = 0.70; // momentum

        for (uint16_t n = 0; n < prevLayer.size(); ++n) {
            Neuron& neuron = prevLayer[n];
            double oldDeltaWeight = neuron._outputWeights[_myIndex].deltaWeight;
            double newDeltaWeight =
                eta * neuron.GetOutputVal() * _gradient
                + alpha * oldDeltaWeight;

            neuron._outputWeights[_myIndex].deltaWeight = newDeltaWeight;
            neuron._outputWeights[_myIndex].weight += newDeltaWeight;

            std::cout << "Debug: UpdateWeights - oldDeltaWeight: " << oldDeltaWeight
                << ", newDeltaWeight: " << newDeltaWeight
                << ", new weight: " << neuron._outputWeights[_myIndex].weight
                << ", gradient: " << _gradient << std::endl;
        }
    }

private:
    static double TransferFunction(double x) { return tanh(x); }
    double TransferFunctionDerivative(double x) {
        double t = tanh(x);
        return 1.0 - t * t;
    }
    static double RandomWeight() { return rand() / double(RAND_MAX); }
    double SumDOW(const Layer& nextLayer) const
    {
        double sum = 0.0;
        for (uint16_t n = 0; n < nextLayer.size() - 1; ++n) {
            sum += _outputWeights[n].weight * nextLayer[n]._gradient;
        }
        std::cout << "Debug: SumDOW result: " << sum << std::endl;
        return sum;
    }
    double _outputVal;
    std::vector<Connections> _outputWeights;
    uint16_t _myIndex;
    double _gradient;
};

class Net {
private:
    std::vector<double> _inputVals;
    std::vector<double> _targetVals;
    std::vector<double> _resultVals;
    std::vector<unsigned> _topology;
    std::vector<Layer> _layers; //[layerNum][neuronNum]
    double _error;
    double _recentAvgError;
    double _recentAvgSmoothingFactor = 100.0;
public:
    Net(const std::vector<unsigned>& topology);
    Net() = default;
    void FeedForward(const std::vector<double>& inputVals);
    void BackProp(const std::vector<double>& targetVals);
    void GetResults(std::vector<double>& resultVals) const;
    void ShowVectorVals(std::string label, std::vector<double>& inputVector);
};