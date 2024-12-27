import random

def generate_training_data(num_samples, file_path):
    with open(file_path, 'w') as file:
        file.write("topology: 7 14 1\n") 
        for _ in range(num_samples):
            inputs = [round(random.random(), 3) for _ in range(7)]
            output = 1
            file.write(f"in: {' '.join(map(str, inputs))}\n")
            file.write(f"out: {output}\n")


file_path = 'TrainingData\\DataFile.txt'
generate_training_data(3500, file_path)
