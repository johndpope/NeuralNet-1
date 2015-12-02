NeuralNet
=========

This is a simple to use Neural Network library in Swift 2.

Installation
===========
TBD

Usage
=====

The following code create a network can solve XOR logic problem by just repeatedly showing it XOR logic table.

```swift
import NeuralNet

let net = NeuralNetwork(sizes: [2, 2, 1])
let table = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0])
]

net.train(table, epochs: 4000, miniBatchSize: 2, eta: 4.0)

net.feedforward([0.0, 0.0])  // -> [0.000658463610889205]
net.feedforward([1.0, 0.0])  // -> [0.999297972978209]
net.feedforward([0.0, 1.0])  // -> [0.999298128871601]
net.feedforward([1.0, 1.0])  // -> [0.00107466760274599]
```
