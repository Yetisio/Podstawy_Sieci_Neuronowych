from data import get_mnist
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# Załadowanie MNIST_DATASET
images, labels = get_mnist()

# Podział danych na zbiór uczący, walidacyjny i testujący
images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
images_val, images_test, labels_val, labels_test = train_test_split(images_temp, labels_temp, test_size=0.5, random_state=42)

# Reshape dla zgodności z kodem treningowym
images_train = images_train.reshape(images_train.shape[0], -1, 1)
images_val = images_val.reshape(images_val.shape[0], -1, 1)
images_test = images_test.reshape(images_test.shape[0], -1, 1)

# Wagi i biasy sieci neuronowej
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))   # weights from input layer to hidden layer
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))    # weights from hidden layer to output layer
b_i_h = np.zeros((20, 1))                         # biases for the hidden layer
b_h_o = np.zeros((10, 1))                         # biases for the output layer

# Parametry uczenia
learn_rate = 0.05
nr_correct = 0
epochs = 5

error_list = [] 

for epoch in range(epochs):
    total_error = 0
    
    for img, l in zip(images_train, labels_train):
        img.shape += (1,)
        l.shape += (1,)
        
        img = img.reshape(784, 1)  # Zmiana kształtu do kompatybilności

        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        
        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))
        total_error += e
    
        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        
        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Obliczenie średniego błędu dla epoki
    avg_error = total_error / len(images_train)
    error_list.append(avg_error)
    
    # Pokazanie poziomu przyswojenia wiedzy w treningach w %
    accuracy = (nr_correct / len(images_train)) * 100
    print(f"Przyswojenie wiedzy poprzez trening nr {epoch+1}: {round(accuracy, 2)}%, Błąd: {avg_error[0]}")
    nr_correct = 0


# Testowanie modelu na danych walidacyjnych
nr_correct_val = 0
for img_val, l_val in zip(images_val, labels_val):
    img_val.shape += (1,)
    l_val.shape += (1,)
    img_val = img_val.reshape(784, 1)
    # Forward propagation input -> hidden
    h_pre_val = b_i_h + w_i_h @ img_val
    h_val = 1 / (1 + np.exp(-h_pre_val))
    
    # Forward propagation hidden -> output
    o_pre_val = b_h_o + w_h_o @ h_val
    o_val = 1 / (1 + np.exp(-o_pre_val))

    nr_correct_val += int(np.argmax(o_val) == np.argmax(l_val))

# Pokazanie poziomu przyswojenia wiedzy na danych walidacyjnych
accuracy_val = (nr_correct_val / len(images_val)) * 100
print(f"Dokładność na danych walidacyjnych: {round(accuracy_val, 2)}%")

# Testowanie modelu na danych testowych
nr_correct_test = 0
for img_test, l_test in zip(images_test, labels_test):
    img_test.shape += (1,)
    l_test.shape += (1,)
    img_test = img_test.reshape(784, 1)
    # Forward propagation input -> hidden
    h_pre_test = b_i_h + w_i_h @ img_test
    h_test = 1 / (1 + np.exp(-h_pre_test))
    
    # Forward propagation hidden -> output
    o_pre_test = b_h_o + w_h_o @ h_test
    o_test = 1 / (1 + np.exp(-o_pre_test))

    nr_correct_test += int(np.argmax(o_test) == np.argmax(l_test))

# Pokazanie poziomu przyswojenia wiedzy na danych testowych
accuracy_test = (nr_correct_test / len(images_test)) * 100
print(f"Dokładność na danych testowych: {round(accuracy_test, 2)}%")

# Wykres błędu dla każdej epoki
plt.plot(range(1, epochs + 1), error_list, marker='o')
plt.title('Wykres błędu dla każdej epoki')
plt.xlabel('Epoka')
plt.ylabel('Błąd')
plt.show()
