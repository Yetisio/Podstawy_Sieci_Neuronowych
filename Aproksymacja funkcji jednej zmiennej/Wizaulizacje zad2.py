import numpy as np
import matplotlib.pyplot as plt

# Dane z tabeli
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
h = np.array([1.0, 1.32, 1.6, 1.54, 1.41, 1.01, 0.6, 0.42, 0.2, 0.51, 0.8])

mse_values = []  # Lista do przechowywania błędów
weights_history = []  # Lista do przechowywania historii wag

# Normalizacja danych
t_normalized = t / np.max(t)
h_normalized = h / np.max(h)

# Generowanie próbek co 6 minut
t_samples = np.arange(0, 10, 0.1)
t_samples_normalized = t_samples / np.max(t_samples)

# Inicjalizacja wag i biasów
input_size = 1
hidden_size = 10  # liczba węzłów aproksymacyjnych
output_size = 1

np.random.seed(42)  # Aby uzyskać powtarzalne wyniki
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))

weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Parametry uczenia
learning_rate = 0.09
epochs = 100000

# Funkcje aktywacji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Trening sieci neuronowej
for epoch in range(epochs):
    # Propagacja w przód
    hidden_layer_input = np.dot(t_normalized.reshape(-1, 1), weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Obliczanie błędu
    error = h_normalized.reshape(-1, 1) - predicted_output

    # Propagacja wsteczna
    output_delta = error * sigmoid_derivative(predicted_output)
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Aktualizacja wag i biasów
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += t_normalized.reshape(-1, 1).T.dot(hidden_delta) * learning_rate

    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

     # Po obliczeniu błędu na końcu każdej epoki
    mse = np.mean((h_normalized.reshape(-1, 1) - predicted_output)**2)
    mse_values.append(mse)

    # Zapisywanie historii wag
    current_weights = np.concatenate([weights_input_hidden.flatten(), weights_hidden_output.flatten()])
    weights_history.append(current_weights)

# Aproksymacja danych testowych
predicted_normalized = sigmoid(np.dot(sigmoid(np.dot(t_samples_normalized.reshape(-1, 1), weights_input_hidden) + bias_hidden), weights_hidden_output) + bias_output)

# Denormalizacja danych
predicted_output = predicted_normalized * np.max(h)

# Średniokwadratowy błąd aproksymacji
mse = np.mean((h - predicted_output[:len(h)])**2)
print(f"Średniokwadratowy błąd aproksymacji: {mse}")

# Denormalizacja przewidzianych wartości przewidywanych dla predykcji probek 6min
predicted_output_samples = predicted_normalized * np.max(h)


# Wykres dla zbioru uczącego i wyników klasyfikacji
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(t, h, label='Dane rzeczywiste')
plt.xlabel('Czas [h]')
plt.ylabel('Poziom wody [m]')
plt.title('Zbiór uczący')

plt.subplot(1, 2, 2)
plt.scatter(t_samples, predicted_output[:len(t_samples)], label='Predykcje')
plt.xlabel('Czas [h]')
plt.ylabel('Poziom wody [m]')
plt.title('Wyniki klasyfikacji')

plt.tight_layout()
plt.show()

# Wizualizacja błędów w trakcie procesu uczenia się
plt.figure(figsize=(10, 5))

plt.plot(range(epochs), mse_values, label='Błąd średniokwadratowy')
plt.xlabel('Epoka')
plt.ylabel('Błąd')
plt.title('Błędy w trakcie uczenia się')
plt.legend()
plt.show()

# Wizualizacja wartości wag w kolejnych iteracjach uczenia
plt.figure(figsize=(16, 10))
num_weights = input_size * hidden_size + hidden_size * output_size

rows = 5
cols = 4

for i in range(num_weights):
    plt.subplot(rows, cols, i + 1)
    plt.plot(range(epochs), [weights[i] for weights in weights_history], label=f'Waga {i+1}')
    plt.xlabel('Epoka')
    plt.ylabel('Wartość wagi')
    plt.title(f'Waga {i+1}')

plt.tight_layout()
plt.show()