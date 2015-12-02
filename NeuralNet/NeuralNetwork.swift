//
//  NeuralNetwork.swift
//  NeuralNet
//
//  Created by pwang on 11/25/15.
//  Copyright © 2015 wpc. All rights reserved.
//

import Foundation
import Surge

public typealias NeuralNetworkData = [([Double], [Double])]

public class NeuralNetwork {
    let numberOfLayers: Int
    let sizes: [Int]
    var biases: [[Double]]
    var weights: [Matrix<Double>]

    public init(sizes: [Int]) {
        self.numberOfLayers = sizes.count
        self.sizes = sizes
        self.biases = sizes[1..<sizes.count].map(randomVector)
        self.weights = zip(sizes[0..<sizes.count-1], sizes[1..<sizes.count]).map({from, to in
            return Matrix((0..<to).map { _ in randomVector(from) })
        })
    }

    public func feedforward(input: [Double]) -> [Double] {
        var active = input
        for layer in (0..<sizes.count-1) {
            let w = weights[layer]
            let b = biases[layer]
            active = sigmoid(add(dot(w, active), b))
        }

        return active;
    }

    public func train(trainingData: NeuralNetworkData, epochs: Int, miniBatchSize: Int, eta: Double?=0.5, testData: NeuralNetworkData? = nil) {
        let n = Double(trainingData.count)
        for epoch in 0..<epochs {
            let data = shuffle(trainingData);
            for batch in chunk(data, size: miniBatchSize) {
                updateMiniBatch(batch, eta: eta!, trainingDataSize: n)
            }

            if testData != nil {
                print("Epoch \(epoch):  sum-error = \(evaluateTotalError(testData!))")
            }
        }

    }

    public func evaluate(data: NeuralNetworkData) -> Int {
        var success = 0
        for (input, label) in data {
            if (maxi(feedforward(input)) == maxi(label)) {
                success += 1
            }
        }
        return success
    }

    public func evaluateTotalError(data: NeuralNetworkData) -> Double {
        var result = 0.0
        for (input, label) in data {
            result += cost(feedforward(input), desired: label)
        }
        return result
    }


    func sigmoid(z:[Double]) -> [Double] {
        let one = ones(z.count)
        return div(one, add(one, exp(neg(z))))
    }


    func updateMiniBatch(miniBatch: NeuralNetworkData, eta: Double, trainingDataSize: Double){
        var nablaB = self.biases.map  { vector($0.count, 0) }
        var nablaW = self.weights.map { Matrix(rows: $0.rows, columns: $0.columns, repeatedValue: 0) }
        for (input, labelOutput) in miniBatch {
            let (deltaNablaB, deltaNablaW) = backProp(input, label: labelOutput)
            nablaB = zip(nablaB, deltaNablaB).map { add($0, $1) }
            nablaW = zip(nablaW, deltaNablaW).map { add($0, $1) }
        }

        self.biases = zip(self.biases, nablaB).map { b, nb in
            add(b, nb.map { -1 * eta / trainingDataSize * $0 })
        }

        self.weights = zip(self.weights, nablaW).map { w, nw in
            add(w, mul(-1 * eta / trainingDataSize, nw))
        }
    }

    func backProp(input: [Double], label: [Double]) -> ([[Double]], [Matrix<Double>]) {
        var deltaNablaB = self.biases.map  { vector($0.count, 0) }
        var deltaNablaW = self.weights.map { Matrix(rows: $0.rows, columns: $0.columns, repeatedValue: 0) }

        // forward pass to generate all activations and weighted input

        var activation = input
        var activations: [[Double]] = [input]
        var zs: [[Double]] = []

        for (b, w) in zip(self.biases, self.weights) {
            let z = add(dot(w, activation), b)
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        }

        // backword pass to back propgate delta
        let delta: [Double] = costDelta(activations.last!, desired: label)

        deltaNablaB[deltaNablaB.count - 1] = delta;
        deltaNablaW[deltaNablaW.count - 1] = mul(toMatrix(delta), transpose(toMatrix(activations[activations.count - 2])))

        for l in 2..<self.numberOfLayers {
            let z = zs[zs.count - l]
            let sp = sigmoidPrime(z)

            let delta = mul(dot(transpose(self.weights[self.weights.count - l + 1]), delta), sp)
            deltaNablaB[deltaNablaB.count - l] = delta
            deltaNablaW[deltaNablaW.count - l] = mul(toMatrix(delta), transpose(toMatrix(activations[activations.count - l - 1])))
        }


        return (deltaNablaB, deltaNablaW)
    }

    // use cross-entropy cost function
    // CS = -∑x[ylna+(1−y)ln(1−a)]
    func cost(output: [Double], desired: [Double]) -> Double {
        let one = ones(output.count)
        return sum(neg(add(mul(desired, log(output)), mul(add(one, neg(desired)), log(add(one, neg(output)))))))
    }

    func costDelta(output:[Double], desired:[Double]) -> [Double] {
        return add(output, neg(desired))
    }

    func sigmoidPrime(z: [Double]) -> [Double] {
        return mul(sigmoid(z), add(ones(z.count), neg(sigmoid(z))))
    }
}


/*
 Uniform shuffle: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
*/
func shuffle(data: NeuralNetworkData) -> NeuralNetworkData  {
    var list = Array(data)

    if list.count < 2 { return list }

    for i in 0..<list.count - 1 {
        let j = Int(arc4random_uniform(UInt32(list.count - i))) + i
        guard i != j else { continue }
        swap(&list[i], &list[j])
    }

    return list
}

/*
 Chunk data into batches
*/
func chunk(data: NeuralNetworkData, size: Int) -> [NeuralNetworkData] {
    let s =  0.stride(to: data.count, by: size)
    return s.map { Array(data[$0..<$0.advancedBy(size, limit: data.count)]) }
}


/*
Random number generated by a standard normal distribution with mean=0 std=1.
From: http://mathworld.wolfram.com/Box-MullerTransformation.html
*/
func normalDistributionRand() -> Double {
    let u1 = Double(arc4random()) / Double(UINT32_MAX)
    let u2 = Double(arc4random()) / Double(UINT32_MAX)
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2)
}

func randomVector(dimentions: Int) -> [Double] {
    return (0..<dimentions).map { _ in return normalDistributionRand() }
}

func vector(dimentions: Int, _ value: Double) -> [Double] {
    return [Double](count: dimentions, repeatedValue: value)
}


func toMatrix(content: [[Double]]) -> Matrix<Double> {
    return Matrix(content)
}

func toMatrix(vector: [Double]) -> Matrix<Double> {
    return Matrix(vector.map { [$0] })
}

func toVector(matrix: Matrix<Double>) -> [Double] {
    assert(matrix.columns == 1)
    return matrix.enumerate().map { _, row in
        return row.first!
    }
}

func dot(matrix: Matrix<Double>, _ vector: [Double]) -> [Double] {
    return toVector(mul(matrix, toMatrix(vector)))
}

func ones(size: Int) -> [Double] {
    return vector(size, 1.0)
}
