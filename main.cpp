#include <iostream>
#include <fstream>
#include <streambuf>
#include "NeuralNet.h"
#include "DualOutput.h"



int main() {

    std::ofstream ofs("TrainingData\\Outs.txt", std::ofstream::out);
    ofs.close();
    // Redirect stdout to Outs.txt
    std::ofstream out("TrainingData\\Outs.txt");
    dual_output dualOut(std::cout, out);
    // Redirect std::cout to dual_output
    std::streambuf* coutbuf = std::cout.rdbuf(); // Save old buf
    std::cout.rdbuf(dualOut.rdbuf());


    try{
	TrainingData data("TrainingData\\DataFile.txt"); 
	std::vector<unsigned> topology;
	data.GetTopology(topology);
	Net NeuralNet(topology);

	std::vector<double> InputVals;
	std::vector<double>TargetVals;
	std::vector <double> ResultVals;
	uint16_t TrainingPass = 0; 

    while (!data.IsEof()) {
        ++TrainingPass;
        std::cout << "Pass " << TrainingPass << std::endl;

        if (data.GetNextInputs(InputVals) != topology[0]) {
            std::cout << "End of training data" << std::endl;
            break;
        }

        std::cout << "Debug: InputVals size: " << InputVals.size() << std::endl;
        NeuralNet.ShowVectorVals(": Inputs:", InputVals);
        NeuralNet.FeedForward(InputVals);

        NeuralNet.GetResults(ResultVals);
        if (ResultVals.empty()) {
            std::cerr << "Error: No results produced by the network" << std::endl;
            break;
        }
        NeuralNet.ShowVectorVals("Outputs: ", ResultVals);

        if (data.GetTargetOutputs(TargetVals) != topology.back()) {
            std::cerr << "Mismatch in number of target values" << std::endl;
            std::cerr << "Expected " << topology.back() << ", got " << TargetVals.size() << std::endl;
            break;
        }

        std::cout << "Debug: TargetVals size: " << TargetVals.size() << std::endl;
        NeuralNet.ShowVectorVals("TargetVals: ", TargetVals);
        NeuralNet.BackProp(TargetVals);

        //std::cout << "NeuralNet average error: " << NeuralNet.GetRecentAvgError() << std::endl;

    } 

    }catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

	std::cout << "Training completed" << std::endl;
    //reset buff
    std::cout.rdbuf(coutbuf);
	return 0;
}