import random

def generate_training_data(num_samples, file_path):
    with open(file_path, 'w') as file:
        file.write("topology: 5 20 15 1\n") 
        for _ in range(num_samples):
            inputs = [round(random.random(), 6) for _ in range(5)]
            output = random.randint(0, 1)
            file.write(f"in: {' '.join(map(str, inputs))}\n")
            file.write(f"out: {output}\n")


file_path = 'TrainingData\\DataFile.txt'
generate_training_data(10000, file_path)
