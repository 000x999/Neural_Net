#include "NeuralNet.h"

Net::Net(const std::vector<unsigned>& topology) {
    uint16_t numLayers = static_cast<uint16_t>(topology.size());
    for (uint16_t layerNum = 0; layerNum < numLayers; ++layerNum) {
        _layers.push_back(Layer());
        uint16_t numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        for (uint16_t neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            _layers.back().push_back(Neuron(numOutputs, neuronNum));
        }

        _layers.back().back().SetOutputVal(1.0);
    }
}

void Net::FeedForward(const std::vector<double>& inputVals) {
    assert(inputVals.size() == _layers[0].size() - 1);

    for (uint16_t i = 0; i < inputVals.size(); ++i) {
        _layers[0][i].SetOutputVal(inputVals[i]);
    }

    for (uint16_t layerNum = 1; layerNum < _layers.size(); ++layerNum) {
        Layer& prevLayer = _layers[layerNum - 1];
        for (uint16_t n = 0; n < _layers[layerNum].size() - 1; ++n) {
            _layers[layerNum][n].FeedNeurons(prevLayer);
        }
    }
}

void Net::BackProp(const std::vector<double>& targetVals) {
    Layer& outputLayer = _layers.back();
    _error = 0.0;

    for (uint16_t n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].GetOutputVal();
        _error += delta * delta;
    }
    _error /= outputLayer.size() - 1;
    _error = sqrt(_error);

    std::cout << "Debug: Overall net error: " << _error << std::endl;

    for (uint16_t n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].CalcOutputGradients(targetVals[n]);
    }

    for (uint16_t layerNum = _layers.size() - 2; layerNum > 0; --layerNum) {
        Layer& hiddenLayer = _layers[layerNum];
        Layer& nextLayer = _layers[layerNum + 1];

        for (uint16_t n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].CalcHiddenGradients(nextLayer);
        }
    }

    for (uint16_t layerNum = _layers.size() - 1; layerNum > 0; --layerNum) {
        Layer& layer = _layers[layerNum];
        Layer& prevLayer = _layers[layerNum - 1];

        for (uint16_t n = 0; n < layer.size() - 1; ++n) {
            layer[n].UpdateWeights(prevLayer);
        }
    }
}

void Net::GetResults(std::vector<double>& resultVals) const {
    resultVals.clear();
    for (uint16_t n = 0; n < _layers.back().size() - 1; ++n) {
        resultVals.push_back(_layers.back()[n].GetOutputVal());
    }
}

void Net::ShowVectorVals(std::string label, std::vector<double>& v) {
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}