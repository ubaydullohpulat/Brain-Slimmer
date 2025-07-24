# Brain-Slimmer
A simple framework for tracing neuron activations and pruning unused parts of a neural network while preserving accuracy. Includes dynamic threshold selection based on desired performance.

## Key Idea
Neural networks are often over-parameterized. This project tracks neuron activations during inference and removes those that are rarely used. We prune the model based on a dynamic threshold to meet a target accuracy.

> **"Cut your brain in half — and still solve math problems just fine."**

---

## Features

- Train a simple feed-forward neural network on MNIST
- Trace activations during inference
- Prune neurons that are activated less than a threshold
- Automatically find the threshold that meets your desired test accuracy
- Visualize pruning vs. performance

---

## Why It Matters

Modern AI is exploding in size — from billions to trillions of parameters — driven by the AGI race. But many real-world applications (especially in healthcare, finance, and private enterprise) don’t need giant models. They need:

- Accuracy  
- Privacy  
- Efficiency  

This project is a lightweight approach for those use cases.

---

## Example Output

The following charts show:
1. **How much of the network was removed**
2. **How accuracy behaves after pruning**

<img width="1280" height="651" alt="image" src="https://github.com/user-attachments/assets/fdce0121-e2eb-4818-917e-1f9dd33bf4cc" />

License
This project is licensed under the GPL-3.0 license — free to use and modify (subproducts should be opensource).

Let's Collaborate
Are you working on model compression, privacy-first AI, or deploying neural networks in sensitive domains?
Let’s connect. Reach out if you want to share ideas or explore use cases together.
