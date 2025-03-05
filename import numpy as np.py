import numpy as np

input_data = np.array([[0.05, 0.10]])

weights = np.array([0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55])


target_output = np.array([[0.01, 0.99]])

learning_rate = 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

hidden_input = np.array([
    input_data[0, 0] * weights[0] + input_data[0, 1] * weights[2],
    input_data[0, 0] * weights[1] + input_data[0, 1] * weights[3]
])


hidden_output = sigmoid(hidden_input)


output_input = np.array([
    hidden_output[0] * weights[4] + hidden_output[1] * weights[6],
    hidden_output[0] * weights[5] + hidden_output[1] * weights[7]
])

final_output = sigmoid(output_input)


output_error = final_output - target_output[0]
output_delta = output_error * sigmoid_derivative(final_output)


weights[4] -= learning_rate * output_delta[0] * hidden_output[0]
weights[5] -= learning_rate * output_delta[1] * hidden_output[0]
weights[6] -= learning_rate * output_delta[0] * hidden_output[1]
weights[7] -= learning_rate * output_delta[1] * hidden_output[1]


hidden_error = np.array([
    output_delta[0] * weights[4] + output_delta[1] * weights[5],
    output_delta[0] * weights[6] + output_delta[1] * weights[7]
])

hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
weights[0] -= learning_rate * hidden_delta[0] * input_data[0, 0]
weights[1] -= learning_rate * hidden_delta[1] * input_data[0, 0]
weights[2] -= learning_rate * hidden_delta[0] * input_data[0, 1]
weights[3] -= learning_rate * hidden_delta[1] * input_data[0, 1]


for i, weight in enumerate(weights):
    print(f"Updated weight {i + 1}: {weight}")