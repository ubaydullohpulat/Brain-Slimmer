import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

#chart var
compression_rates = []
accuracies = []

(train_X, train_y), (test_X, test_y) = mnist.load_data() # load main dataset


# printing chape of data set (aka matrix sizes)
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


# Flatten images from 28x28 to 784 and normalize to 0-1
train_X = train_X.reshape(train_X.shape[0], 784) / 255.0
test_X = test_X.reshape(test_X.shape[0], 784) / 255.0

# One-hot encode labels
def one_hot(y, num_classes=10):
    result = np.zeros((y.shape[0], num_classes))
    result[np.arange(y.shape[0]), y] = 1
    return result

train_y = one_hot(train_y)
test_y = one_hot(test_y)

def init_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(pred, true):
    # avoid log(0)
    eps = 1e-12
    pred = np.clip(pred, eps, 1. - eps)
    return -np.mean(np.sum(true * np.log(pred), axis=1))

def forward_pass(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def backward_pass(X, y_true, z1, a1, a2, W2):
    m = X.shape[0]

    dz2 = a2 - y_true  # shape: (batch, 10)
    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = (dz2 @ W2.T) * relu_derivative(z1)
    dW1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def train(X, y, epochs=10, lr=0.1, hidden_size=128, batch_size=64):
    input_size = X.shape[1]
    output_size = y.shape[1]
    W1, b1, W2, b2 = init_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            z1, a1, z2, a2 = forward_pass(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_pass(X_batch, y_batch, z1, a1, a2, W2)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # evaluate after each epoch
        _, _, _, a2 = forward_pass(X, W1, b1, W2, b2)
        loss = cross_entropy(a2, y)
        acc = np.mean(np.argmax(a2, axis=1) == np.argmax(y, axis=1))
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")

    return W1, b1, W2, b2


def test(X, y, W1, b1, W2, b2):
    _, _, _, a2 = forward_pass(X, W1, b1, W2, b2)
    predictions = np.argmax(a2, axis=1)
    labels = np.argmax(y, axis=1)
    acc = np.mean(predictions == labels)
    print(f"Test Accuracy: {acc*100:.2f}%")






# ----------------- training & saving model start
# W1, b1, W2, b2 = train(train_X, train_y, epochs=15, lr=0.1)

# np.savez('mnist_model_save.npz', W1=W1, b1=b1, W2=W2, b2=b2)
# print("Model saved to mnist_model_save.npz")
# ----------------- training & saving model end






# ----------------- Load model weights start
data = np.load('mnist_model_save.npz')
# Extract variables
W1 = data['W1']
b1 = data['b1']
W2 = data['W2']
b2 = data['b2']
print("Model loaded successfully.")
# ----------------- Load model weights end


# test(test_X, test_y, W1, b1, W2, b2) # general test


def trace_activation_path(target_digit, test_X, test_y, W1, b1, W2, b2, top_k=5):
    """
    For each test sample of `target_digit`, shows which hidden neurons most contributed to the output.

    Args:
        target_digit (int): The digit to analyze (0–9)
        test_X (ndarray): Input test data
        test_y (ndarray): One-hot encoded test labels
        W1, b1, W2, b2: Network parameters
        top_k (int): Number of top contributors to show
    """
    import matplotlib.pyplot as plt

    labels = np.argmax(test_y, axis=1)
    indices = np.where(labels == target_digit)[0]

    print(f"Tracing activation paths for digit {target_digit} ({len(indices)} samples):")

    for sample_id in indices[:3]:  # Limit for brevity; remove `[:3]` to analyze all
        x_sample = test_X[sample_id:sample_id+1]
        y_true = np.argmax(test_y[sample_id])

        z1, a1, z2, a2 = forward_pass(x_sample, W1, b1, W2, b2)
        y_pred = np.argmax(a2)

        print(f"\nSample ID {sample_id} — True: {y_true}, Predicted: {y_pred}")

        # Get contribution of each hidden unit to the predicted output neuron
        w_to_output = W2[:, y_pred]
        contributions = a1.flatten() * w_to_output

        # Top contributing neurons
        top_indices = np.argsort(contributions)[-top_k:][::-1]
        print(f"Top {top_k} contributing hidden units:")

        for idx in top_indices:
            print(f"  Neuron {idx} → Activation: {a1[0, idx]:.4f}, Weight to output: {w_to_output[idx]:.4f}, Contribution: {contributions[idx]:.4f}")

        # Visualize weights of top neurons
        for i, idx in enumerate(top_indices):
            plt.subplot(1, top_k, i+1)
            plt.imshow(W1[:, idx].reshape(28, 28), cmap='gray')
            plt.title(f'Unit {idx}')
            plt.axis('off')

        plt.suptitle(f"Top {top_k} Hidden Units for Sample {sample_id}")
        plt.show()


def get_max_activations(X, W1, b1):
    """
    Computes the maximum activation for each hidden unit over the dataset.
    """
    batch_size = 128
    max_activations = np.zeros(W1.shape[1])

    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i+batch_size]
        z1 = X_batch @ W1 + b1
        a1 = np.maximum(0, z1)  # ReLU

        max_batch = np.max(a1, axis=0)
        max_activations = np.maximum(max_activations, max_batch)

    return max_activations



def get_active_neuron_indices(max_activations, threshold=0.2):
    """
    Returns indices of neurons that activated above threshold at least once.
    """
    return np.where(max_activations > threshold)[0]



def prune_network(W1, b1, W2, b2, keep_indices):
    """
    Creates a pruned version of the network, keeping only selected neurons.
    """
    # W1: (784, 128) → keep selected columns (neurons)
    W1_pruned = W1[:, keep_indices]
    b1_pruned = b1[:, keep_indices]

    # W2: (128, 10) → keep selected rows (neurons)
    W2_pruned = W2[keep_indices, :]

    return W1_pruned, b1_pruned, W2_pruned, b2



def forward_pass_pruned(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def test_pruned(X, y, W1, b1, W2, b2):
    _, _, _, a2 = forward_pass_pruned(X, W1, b1, W2, b2)
    predictions = np.argmax(a2, axis=1)
    true_labels = np.argmax(y, axis=1)
    acc = np.mean(predictions == true_labels)
    print(f"Accuracy after pruning: {acc * 100:.2f}%")


def test_pruned_for_digit(test_X, test_y, W1, b1, W2, b2, target_digit, return_accuracy=False):
    """
    Evaluates a pruned model specialized to recognize a single digit.
    It is successful if:
      - input is digit N → predicts digit N ✅
      - input is NOT digit N → predicts NOT digit N ✅
    """
    _, _, _, a2 = forward_pass_pruned(test_X, W1, b1, W2, b2)
    y_pred = np.argmax(a2, axis=1)
    y_true = np.argmax(test_y, axis=1)

    success = 0
    total = len(test_X)

    for i in range(total):
        pred = y_pred[i]
        actual = y_true[i]

        if actual == target_digit and pred == target_digit:
            success += 1  # correctly identified target digit
        elif actual != target_digit and pred != target_digit:
            success += 1  # correctly rejected other digits
        # otherwise it's a failure

    accuracy = success / total
    print(f"Specialized Digit Evaluation for '{target_digit}':")
    print(f"  Success rate: {accuracy * 100:.2f}% ({success}/{total})")
    print(f"  Failure rate: {(1 - accuracy) * 100:.2f}% ({total - success}/{total})")

    if return_accuracy:
        return accuracy


def main_text(label=1):
    # Filter test samples where label
    digitN_indices = np.where(np.argmax(test_y, axis=1) == label)[0]
    test_X_N = test_X[digitN_indices]
    test_y_N = test_y[digitN_indices]

    # Test only on digit 9
    test(test_X_N, test_y_N, W1, b1, W2, b2)

    # trace_activation_path(label, test_X, test_y, W1, b1, W2, b2, top_k=5)

    # 1. Get max activations over the dataset
    max_acts = get_max_activations(test_X_N, W1, b1)

    # 2. Identify which neurons to keep
    keep_indices = get_active_neuron_indices(max_acts, threshold=2)
    print(f"Keeping {len(keep_indices)} of {W1.shape[1]} hidden units.")

    # 3. Prune the model
    W1_pruned, b1_pruned, W2_pruned, b2_pruned = prune_network(W1, b1, W2, b2, keep_indices)

    # 4. Evaluate new (smaller) network
    test_pruned(test_X, test_y, W1_pruned, b1_pruned, W2_pruned, b2_pruned)


    # 5. Evaluates a pruned model specialized to recognize a single digit.
    test_pruned_for_digit(test_X, test_y, W1_pruned, b1_pruned, W2_pruned, b2_pruned, target_digit=label)


def main(label=1):
    digitN_indices = np.where(np.argmax(test_y, axis=1) == label)[0]
    test_X_N = test_X[digitN_indices]
    test_y_N = test_y[digitN_indices]

    # 1. Get max activations for this digit
    max_acts = get_max_activations(test_X_N, W1, b1)

    # 2. Identify neurons to keep
    keep_indices = get_active_neuron_indices(max_acts, threshold=2)
    compression_rate = len(keep_indices) / W1.shape[1]

    # 3. Prune the model
    W1_pruned, b1_pruned, W2_pruned, b2_pruned = prune_network(W1, b1, W2, b2, keep_indices)

    # 4. Evaluate specialized accuracy
    accuracy = test_pruned_for_digit(
        test_X, test_y,
        W1_pruned, b1_pruned, W2_pruned, b2_pruned,
        target_digit=label,
        return_accuracy=True  # Add this flag in the function
    )

    return compression_rate, accuracy



def find_best_threshold_for_digit(
    digit, test_X, test_y, W1, b1, W2, b2,
    min_threshold=0.0, max_threshold=5.0, step=0.1,
    target_accuracy=0.95
):
    digitN_indices = np.where(np.argmax(test_y, axis=1) == digit)[0]
    test_X_N = test_X[digitN_indices]

    best_threshold = None
    best_accuracy = 0
    best_keep_count = 0

    for threshold in np.arange(min_threshold, max_threshold, step):
        # 1. Get activations and prune
        max_acts = get_max_activations(test_X_N, W1, b1)
        keep_indices = get_active_neuron_indices(max_acts, threshold=threshold)

        # Skip if too few neurons remain
        if len(keep_indices) < 1:
            continue

        # 2. Prune the network
        W1_p, b1_p, W2_p, b2_p = prune_network(W1, b1, W2, b2, keep_indices)

        # 3. Test
        acc = test_pruned_for_digit(
            test_X, test_y, W1_p, b1_p, W2_p, b2_p,
            target_digit=digit, return_accuracy=True
        )

        if acc >= target_accuracy:
            best_threshold = threshold
            best_accuracy = acc
            best_keep_count = len(keep_indices)
        else:
            break  # Stop when accuracy drops below target

    return best_threshold, best_accuracy, best_keep_count


target_acc = 0.99  # specialized accuracy

for digit in range(10):
    print(f"\n--- Finding threshold for digit {digit} ---")
    th, acc, kept = find_best_threshold_for_digit(
        digit,
        test_X, test_y,
        W1, b1, W2, b2,
        target_accuracy=target_acc
    )

    if th is not None:
        print(f"✅ Best threshold: {th:.2f} | Accuracy: {acc:.2%} | Kept neurons: {kept}")
        compression_rates.append(1 - kept / W1.shape[1])  # compression %
        accuracies.append(acc)
    else:
        print(f"❌ No threshold found that keeps accuracy ≥ {target_acc * 100}%")
        compression_rates.append(0.0)
        accuracies.append(0.0)


# Plot
digits = list(range(10))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(digits, [round(1-c, 2) for c in compression_rates], color='orange')
plt.title("Pruned Compression Rate")
plt.xlabel("Digit")
plt.ylabel("Compression (removed %)")

plt.subplot(1, 2, 2)
plt.plot(digits, [round(a * 100, 2) for a in accuracies], marker='o', color='green')
plt.title("Accuracy After Pruning")
plt.xlabel("Digit")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()




