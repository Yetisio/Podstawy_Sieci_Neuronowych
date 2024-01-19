import numpy as np
import matplotlib.pyplot as plt

# Dane z tabeli
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
h = np.array([1.0, 1.32, 1.6, 1.54, 1.41, 1.01, 0.6, 0.42, 0.2, 0.51, 0.8])

# Normalizacja danych
t_normalized = t / np.max(t)
h_normalized = h / np.max(h)

# Generowanie próbek co 6 minut
t_samples = np.arange(0, 10, 0.1)
t_samples_normalized = t_samples / np.max(t_samples)

# Inicjalizacja wag i biasów
input_size = 1
hidden_size_1 = 8  # liczba węzłów w pierwszej warstwie ukrytej
hidden_size_2 = 5  # liczba węzłów w drugiej warstwie ukrytej
hidden_size_3 = 3  # liczba węzłów w trzeciej warstwie ukrytej
hidden_size_4 = 2  # liczba węzłów w czwartej warstwie ukrytej
output_size = 1

np.random.seed(42)  # Aby uzyskać powtarzalne wyniki
weights_input_hidden_1 = np.random.rand(input_size, hidden_size_1)
bias_hidden_1 = np.zeros((1, hidden_size_1))

weights_hidden_1_hidden_2 = np.random.rand(hidden_size_1, hidden_size_2)
bias_hidden_2 = np.zeros((1, hidden_size_2))

weights_hidden_2_hidden_3 = np.random.rand(hidden_size_2, hidden_size_3)
bias_hidden_3 = np.zeros((1, hidden_size_3))

weights_hidden_3_hidden_4 = np.random.rand(hidden_size_3, hidden_size_4)
bias_hidden_4 = np.zeros((1, hidden_size_4))

weights_hidden_4_output = np.random.rand(hidden_size_4, output_size)
bias_output = np.zeros((1, output_size))

# Parametry uczenia
learning_rate = 0.09
epochs = 100000

# Funkcje aktywacji sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Trening sieci neuronowej
for epoch in range(epochs):
    # Propagacja w przód
    hidden_layer_input_1 = np.dot(t_normalized.reshape(-1, 1), weights_input_hidden_1) + bias_hidden_1
    hidden_layer_output_1 = sigmoid(hidden_layer_input_1)

    hidden_layer_input_2 = np.dot(hidden_layer_output_1, weights_hidden_1_hidden_2) + bias_hidden_2
    hidden_layer_output_2 = sigmoid(hidden_layer_input_2)

    hidden_layer_input_3 = np.dot(hidden_layer_output_2, weights_hidden_2_hidden_3) + bias_hidden_3
    hidden_layer_output_3 = sigmoid(hidden_layer_input_3)

    hidden_layer_input_4 = np.dot(hidden_layer_output_3, weights_hidden_3_hidden_4) + bias_hidden_4
    hidden_layer_output_4 = sigmoid(hidden_layer_input_4)

    output_layer_input = np.dot(hidden_layer_output_4, weights_hidden_4_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Obliczanie błędu
    error = h_normalized.reshape(-1, 1) - predicted_output

    # Propagacja wsteczna
    output_delta = error * sigmoid_derivative(predicted_output)
    hidden_error_4 = output_delta.dot(weights_hidden_4_output.T)
    hidden_delta_4 = hidden_error_4 * sigmoid_derivative(hidden_layer_output_4)

    hidden_error_3 = hidden_delta_4.dot(weights_hidden_3_hidden_4.T)
    hidden_delta_3 = hidden_error_3 * sigmoid_derivative(hidden_layer_output_3)

    hidden_error_2 = hidden_delta_3.dot(weights_hidden_2_hidden_3.T)
    hidden_delta_2 = hidden_error_2 * sigmoid_derivative(hidden_layer_output_2)

    hidden_error_1 = hidden_delta_2.dot(weights_hidden_1_hidden_2.T)
    hidden_delta_1 = hidden_error_1 * sigmoid_derivative(hidden_layer_output_1)

    # Aktualizacja wag i biasów
    weights_hidden_4_output += hidden_layer_output_4.T.dot(output_delta) * learning_rate
    weights_hidden_3_hidden_4 += hidden_layer_output_3.T.dot(hidden_delta_4) * learning_rate
    weights_hidden_2_hidden_3 += hidden_layer_output_2.T.dot(hidden_delta_3) * learning_rate
    weights_hidden_1_hidden_2 += hidden_layer_output_1.T.dot(hidden_delta_2) * learning_rate
    weights_input_hidden_1 += t_normalized.reshape(-1, 1).T.dot(hidden_delta_1) * learning_rate

    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden_4 += np.sum(hidden_delta_4, axis=0, keepdims=True) * learning_rate
    bias_hidden_3 += np.sum(hidden_delta_3, axis=0, keepdims=True) * learning_rate
    bias_hidden_2 += np.sum(hidden_delta_2, axis=0, keepdims=True) * learning_rate
    bias_hidden_1 += np.sum(hidden_delta_1, axis=0, keepdims=True) * learning_rate

# Aproksymacja danych testowych
hidden_layer_output_1 = sigmoid(np.dot(t_samples_normalized.reshape(-1, 1), weights_input_hidden_1) + bias_hidden_1)
hidden_layer_output_2 = sigmoid(np.dot(hidden_layer_output_1, weights_hidden_1_hidden_2) + bias_hidden_2)
hidden_layer_output_3 = sigmoid(np.dot(hidden_layer_output_2, weights_hidden_2_hidden_3) + bias_hidden_3)
hidden_layer_output_4 = sigmoid(np.dot(hidden_layer_output_3, weights_hidden_3_hidden_4) + bias_hidden_4)
predicted_normalized = sigmoid(np.dot(hidden_layer_output_4, weights_hidden_4_output) + bias_output)

# Denormalizacja danych
predicted_output = predicted_normalized * np.max(h)

# Średniokwadratowy błąd aproksymacji
mse = np.mean((h - predicted_output[:len(h)])**2)
print(f"Średniokwadratowy błąd aproksymacji: {mse}")

# Denormalizacja przewidzianych wartości przewidywanych dla predykcji probek 6min
predicted_output_samples = predicted_normalized * np.max(h)

# Wykres dla aproksymacji danych rzeczywistych
plt.figure(figsize=(10, 6))

# Pierwszy wykres - Aproksymacja z punktami rzeczywistymi
plt.scatter(t, h, label='Dane rzeczywiste')
plt.plot(t_samples, predicted_output[:len(t_samples)], label='Aproksymacja sieci neuronowej', color='red')
plt.xlabel('Czas [h]')
plt.ylabel('Poziom wody [m]')
plt.title('Aproksymacja z punktami rzeczywistymi')
plt.legend()

# Drugi wykres - Predykcje dla próbek 6-minutowych
plt.figure(figsize=(10, 6))
plt.scatter(t, h, label='Dane rzeczywiste')
plt.scatter(t_samples, predicted_output_samples, label='Predykcje dla próbek 6-minutowych', color='green', marker='x')
plt.xlabel('Czas [h]')
plt.ylabel('Poziom wody [m]')
plt.title('Predykcje dla próbek 6-minutowych')
plt.legend()

# Trzeci wykres - Łączący obie linie
plt.figure(figsize=(10, 6))
plt.scatter(t, h, label='Dane rzeczywiste')
plt.plot(t_samples, predicted_output[:len(t_samples)], label='Aproksymacja sieci neuronowej', color='red')
plt.scatter(t_samples, predicted_output_samples, label='Predykcje dla próbek 6-minutowych', color='green', marker='x')
plt.xlabel('Czas [h]')
plt.ylabel('Poziom wody [m]')
plt.title('Aproksymacja + Predykcje')
plt.legend()

plt.show()
