//
//  NeuralNetworkTests.swift
//  NeuralNet
//
//  Created by pwang on 11/30/15.
//  Copyright © 2015 wpc. All rights reserved.
//

import XCTest
import Surge
@testable import NeuralNet

class NeuralNetworkTests: XCTestCase {

    func testWeightsAndBiasInitializedWithRightDimentions() {
        let net = NeuralNetwork(sizes: [2, 3, 1])
        XCTAssertEqual(2, net.biases.count)
        XCTAssertEqual(2, net.weights.count)
        XCTAssertEqual(3, net.biases[0].count)
        XCTAssertEqual(3, net.weights[0].rows)
        XCTAssertEqual(2, net.weights[0].columns)

        XCTAssertEqual(1, net.biases[1].count)
        XCTAssertEqual(1, net.weights[1].rows)
        XCTAssertEqual(3, net.weights[1].columns)
    }

    func testFeedForward() {
        let net = NeuralNetwork(sizes: [2,2])
        net.weights = [Matrix([[ 0.2001109 , -0.36950856], [ 0.87468356, -0.09459014]])]
        net.biases = [[-0.46867192, -1.7646126]]
        let output = net.feedforward([1.0, 2.0])
        XCTAssertEqualWithAccuracy(0.267, output[0], accuracy: 0.001)
        XCTAssertEqualWithAccuracy(0.254, output[1], accuracy: 0.001)
    }

    func testEvaluate() {
        let or = NeuralNetwork(sizes: [2,2])
        or.weights = [Matrix([[ 0.0 , 0.0], [ 1.0, 1.0]])]
        or.biases = [[0.5, 0.0]]
        XCTAssertEqual(1, maxi(or.feedforward([1.0, 0.0])))
        XCTAssertEqual(1, maxi(or.feedforward([0.0, 1.0])))
        XCTAssertEqual(1, maxi(or.feedforward([1.0, 1.0])))
        XCTAssertEqual(0, maxi(or.feedforward([0.0, 0.0])))

        let data = [
            ([0.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0], [0.0, 1.0]),
            ([1.0, 1.0], [1.0, 0.0])
        ]

        XCTAssertEqual(4, or.evaluate(data))
    }

    func testTrainSimpleOneNeuronNetwork() {
        let net = determinize(NeuralNetwork(sizes: [1, 1]))
        net.train([([1.0], [0.0])], epochs: 300, miniBatchSize: 1)
        XCTAssertEqualWithAccuracy(0.0, net.feedforward([1.0])[0], accuracy: 0.01)
    }


    func testLearningANDLogic() {
        let net = determinize(NeuralNetwork(sizes: [2, 2]))
        let table = [
            ([0.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [1.0, 0.0]),
            ([1.0, 0.0], [1.0, 0.0]),
            ([1.0, 1.0], [0.0, 1.0])
        ]

        net.train(table, epochs: 100, miniBatchSize: 2)

        XCTAssertEqual(table.count, net.evaluate(table))
    }

    func testLearningORLogic() {
        let net = determinize(NeuralNetwork(sizes: [2, 2]))
        let table = [
            ([0.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0], [0.0, 1.0])
        ]

        net.train(table, epochs: 100, miniBatchSize: 2)

        XCTAssertEqual(table.count, net.evaluate(table))
    }


    func testLearningXORLogicWith2OutputNeron() {
        let net = determinize(NeuralNetwork(sizes: [2, 2, 2]))
        let table = [
            ([0.0, 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 1.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 1.0], [1.0, 0.0])
        ]

        net.train(table, epochs: 300, miniBatchSize: 4, eta: 4.0)

        XCTAssertEqual(table.count, net.evaluate(table))
    }

    func testLearningXORLogicWith1OuputNeron() {
        let net = determinize(NeuralNetwork(sizes: [2, 2, 1]))
        let table = [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0])
        ]

        net.train(table, epochs: 300, miniBatchSize: 4, eta: 4.0)
        XCTAssertEqualWithAccuracy(1.0, net.feedforward([1.0, 0.0])[0], accuracy: 0.1)
        XCTAssertEqualWithAccuracy(1.0, net.feedforward([0.0, 1.0])[0], accuracy: 0.1)
        XCTAssertEqualWithAccuracy(0.0, net.feedforward([0.0, 0.0])[0], accuracy: 0.1)
        XCTAssertEqualWithAccuracy(0.0, net.feedforward([1.0, 1.0])[0], accuracy: 0.1)
    }

    func testCallbackOnEachEpoch() {
        let net = determinize(NeuralNetwork(sizes: [1, 1, 1, 1]))
        var epoches: [Int] = []
        net.train([([1.0], [1.0])], epochs: 30, miniBatchSize: 1, eachEpoch: {
            epoches.append($0)
        })

        XCTAssertEqual(Array(0..<30), epoches)
    }



    // fill initial weights and biases with pregenerated value, determinize test result
    func determinize(net: NeuralNetwork) -> NeuralNetwork {
        let seeds = [0.2001109 , -0.36950856,  0.87468356, -0.09459014, -0.46867192, -1.7646126]
        let result = NeuralNetwork(sizes: net.sizes)
        result.biases = result.biases.map {
            var bs = [Double](count: $0.count, repeatedValue: 0.0)
            for j in 0..<bs.count {
                bs[j] = seeds[j % seeds.count]
            }
            return bs
        }

        result.weights = result.weights.map {
            var w = Matrix(rows: $0.rows, columns: $0.columns, repeatedValue: 0.0)
            for i in 0..<w.rows {
                for j in 0..<w.columns {
                    w[i, j] = seeds[ (i*j) % seeds.count]
                }
            }
            return w
        }

        return result
    }
}