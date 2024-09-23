# C++ Neural Network with Training Visualizer
<img align="right" width="1280" src="https://github.com/000x999/000x999_gifs/blob/main/ShorterVers_1.gif">
A C++ implementation of a Neural Network, along with a real-time training visualizer and data generation tool. The project aims to visualize the error, gradient, and weight values as the neural network trains over iterations.

## Features
- **Neural Network in C++**: A simple feedforward neural network with training using backpropagation and activation functions.
- **Real-Time Training Visualizer**: Visualizes error, gradients, and weights over iterations using a Python script.
- **Random Training Data Generator**: A Python script to generate random training data in the required format.
- **Terminal Output Logging**: Logs all output to `Outs.txt` for further analysis while still displaying on the terminal.

## How It Works:
- [A Comprehensive and in depth analysis of my Neural Net implementation is available in a PDF format written in LaTeX.](https://github.com/000x999/Neural_Net/blob/main/Comprehensive_Analasys_NeuralNetwork_000x999.pdf).
- To use the [Training visualizer](https://github.com/000x999/Neural_Net/blob/main/ErrorPlotter.py) simply generate training data using the [Training data generation tool](https://github.com/000x999/Neural_Net/blob/main/TrainingDataOuput.py), manually build the Neural Net or open it using Visual studio,
run the Training visualizer and run the Neural Net, this will initiate the training process using the newly generated data set and the training visualizer will update every 100ms as the Neural Net runs through the entire data set.
- The [Training data generation tool](https://github.com/000x999/Neural_Net/blob/main/TrainingDataOuput.py) allows you to generate **N** amount of data lines, allows you to edit the topology of the Neural Net as well as the range of the data values.
## Requirements
- **C++11 and later**: For building and running the neural network.
- **Python 3.x**: Required for the visualizer and data generation scripts.
- **Matplotlib & Numpy**: For visualizing the training process.


## Installation
**Note:**
- The Neural Network can be built and ran entirely in Visual Studio.
- Build instructions for anyone who doesn't use Visual studio can be found below including installing python, matplotlib and numpy using the Python installer, Homebrew and apt for Windows, MacOS and linux respectively. Other build instructions and compiler flags are included as well.
### Installing Python

#### Windows
1. Download the latest Python installer from [python.org](https://www.python.org/downloads/)
2. Run the installer and check "Add Python to PATH" during setup.
3. Verify installation by running in Command Prompt:
    ```bash
    python --version
    ```

#### macOS
1. Open Terminal and install Python using Homebrew:
    ```bash
    brew install python
    ```

2. Verify installation:
    ```bash
    python3 --version
    ```

#### Linux (Debian/Ubuntu)
1. Open Terminal and run:
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```

2. Verify installation:
    ```bash
    python3 --version
    ```

---

### Installing Required Python Libraries

Once Python is installed, install the required libraries (matplotlib and numpy):

```bash
pip install matplotlib numpy
```

### Clone the Repository and Build
1. Clone the repository:
    ```bash
    git clone https://github.com/000x999/Neural_Net
    cd Neural_Net
    ```
2. Compile and run the program:
    ```bash
    g++ -o Neural_Net NeuralNet.cpp main.cpp
    ./Neural_Net
    ```
