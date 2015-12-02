NeuralNet
=========

This is a simple to use Neural Network library in Swift 2.

Installation
===========
TBD

Usage
=====

The following code train a network solving XOR problem

```swift
import NeuralNet

let net = NeuralNetwork(sizes: [2, 2, 2])
let table = [
    ([0.0, 0.0], [1.0, 0.0]),
    ([0.0, 1.0], [0.0, 1.0]),
    ([1.0, 0.0], [0.0, 1.0]),
    ([1.0, 1.0], [1.0, 0.0])
]

net.train(table, epochs: 4000, miniBatchSize: 2, eta: 3.0)

assert(maxi(net.feedforward([1.0, 0.0])) == 1)
assert(maxi(net.feedforward([1.0, 1.0])) == 0)

```
