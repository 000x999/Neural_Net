import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib.animation as animation

errors = []
gradients = []
weights = []
iterations = []

error_pattern = re.compile(r'Debug: Overall net error: ([\d\.\-e]+)')
gradient_pattern = re.compile(r'_gradient: ([\d\.\-e]+)')
weight_pattern = re.compile(r'new weight: ([\d\.\-e]+)')
iteration_pattern = re.compile(r'Pass (\d+)')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

def animate(i):

    ax1.clear()
    ax2.clear()
    ax3.clear()

    errors.clear()
    gradients.clear()
    weights.clear()
    iterations.clear()

    current_gradients = []
    current_weights = []
    current_iteration = None

    with open('TrainingData\\Outs.txt', 'r') as file:
        lines = file.readlines()

    # Parse the log file
    for line in lines:
        if "Pass" in line:
            match = iteration_pattern.search(line)
            if match:
                current_iteration = int(match.group(1))
                iterations.append(current_iteration)

        elif "Overall net error" in line:
            match = error_pattern.search(line)
            if match:
                error_value = float(match.group(1))
                errors.append((current_iteration, error_value))

                if current_gradients:
                    gradients.append((current_iteration, np.mean(current_gradients)))
                    current_gradients = []

                if current_weights:
                    weights.append((current_iteration, np.mean(current_weights)))
                    current_weights = []

        elif "_gradient:" in line:
            match = gradient_pattern.search(line)
            if match:
                gradient_value = float(match.group(1))
                current_gradients.append(gradient_value)

        elif "new weight:" in line:
            match = weight_pattern.search(line)
            if match:
                weight_value = float(match.group(1))
                current_weights.append(weight_value)

    iterations_sorted = list(set(iterations))
    iterations_sorted.sort()

    errors_dict = dict(errors)
    gradients_dict = dict(gradients)
    weights_dict = dict(weights)

    error_values = [errors_dict.get(i, None) for i in iterations_sorted]
    gradient_values = [gradients_dict.get(i, None) for i in iterations_sorted]
    weight_values = [weights_dict.get(i, None) for i in iterations_sorted]

    ax1.plot(iterations_sorted, error_values, label='Error')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error')
    ax1.set_title('Training Error over Iterations', fontsize=12)
    ax1.legend(loc='upper right')

    ax2.plot(iterations_sorted, gradient_values, label='Gradient', color='orange')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient')
    ax2.set_title('Gradient Values over Iterations', fontsize=12)
    ax2.legend(loc='upper right')

    ax3.plot(iterations_sorted, weight_values, label='Weight', color='green')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Weight')
    ax3.set_title('Weight Values over Iterations', fontsize=12)
    ax3.legend(loc='upper right')

    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()

    fig.subplots_adjust(hspace=0.5)

ani = animation.FuncAnimation(fig, animate, interval=100)

# Ensure proper layout
plt.tight_layout()

plt.show()
