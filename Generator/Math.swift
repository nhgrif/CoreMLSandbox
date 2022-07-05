//
//  Math.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 27/06/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import Accelerate
import CoreML

///
/// From M.I. Hollemans
///
/// https://github.com/hollance/CoreMLHelpers
///
struct Math {
    
    /**
     Returns the index and value of the largest element in the array.
     
     - Parameters:
     - ptr: Pointer to the first element in memory.
     - count: How many elements to look at.
     - stride: The distance between two elements in memory.
     */
    static func argmax(_ ptr: UnsafePointer<Float>, count: Int, stride: Int = 1) -> (Int, Float) {
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }
    
    /**
     Returns the index and value of the largest element in the array.
     - Parameters:
     - ptr: Pointer to the first element in memory.
     - count: How many elements to look at.
     - stride: The distance between two elements in memory.
     */
    static func argmax(_ ptr: UnsafePointer<Double>, count: Int, stride: Int = 1) -> (Int, Double) {
        var maxValue: Double = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxviD(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }
    
    
    /// MLMultiArray helper.
    /// Works in our specific use case.
    static func argmax(_ multiArray: MLMultiArray) -> (Int, Double) {
        assert(multiArray.dataType == .double)
        let ptr = UnsafeMutablePointer<Double>(OpaquePointer(multiArray.dataPointer))
        return Math.argmax(ptr, count: multiArray.count)
    }
    
    /// MLMultiArray helper.
    /// Works in our specific use case.
    static func argmax32(_ multiArray: MLMultiArray) -> (Int, Float) {
        assert(multiArray.dataType == .float32)
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(multiArray.dataPointer))
        let count = multiArray.count
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0
        vDSP_maxvi(ptr, vDSP_Stride(1), &maxValue, &maxIndex, vDSP_Length(count))
        return (Int(maxIndex), maxValue)
    }
    
    /// Top-K.
    /// Select the k most-probable elements indices from `arr`
    /// and return both the indices (from the original array)
    /// and their softmaxed probabilities.
    ///
    static func topK(arr: [Double], k: Int) -> (indexes: [Int], probs: [Double]) {
        let x = Array(arr.enumerated().map { ($0, $1) }
            .sorted(by: { a, b -> Bool in a.1 > b.1 })
            .prefix(through: min(k, arr.count) - 1))
        let indexes = x.map { $0.0 }
        let logits = x.map { $0.1 }
        let probs = softmax(logits)
        return (indexes: indexes, probs: probs)
    }
    
    static func topP(arr: [Double], p: Double) -> (indexes: [Int], probs: [Double]) {
        var accumulatedProbability = 0.0
        
        var indexes = [Int]()
        indexes.reserveCapacity(arr.count)
        var probabilities = [Double]()
        probabilities.reserveCapacity(arr.count)
        
        let softmaxed = softmax(arr)
        
        let x = softmaxed.lazy.enumerated()
            .map { ($0, $1) }
            .sorted { $0.1 > $1.1 }

        for (index, probability) in x {
            indexes.append(index)
            probabilities.append(probability)
            accumulatedProbability += probability
            if accumulatedProbability > p { break }
        }
        
        return (indexes: indexes, probs: probabilities)
    }
    
    static func selectRandom(from probabilities: [Double], withinTopP probability: Double) -> Int {
        var accumulatedProbability = 0.0
        
        let curated = softmax(probabilities).enumerated()
            .map { ($0, $1) }
            .sorted { $0.1 > $1.1 }
        
        let randomProbability = randomDouble(min: 0, max: probability)
        for (index, probability) in curated {
            accumulatedProbability += probability
            if accumulatedProbability > randomProbability { return index }
        }

        return 0
    }
    
    private static func randomDouble(min: Double, max: Double) -> Double {
        let seed = Double(arc4random()) / 0xFFFFFFFF
        return seed * (max - min) + min
    }
    
    /// Multinomial sampling from an array of probs. Works well with topK
    static func sample(indexes: [Int], probs: [Double]) -> Int {
        let i = randomNumber(probabilities: probs)
        return indexes[i]
    }
    
    /**
     Computes the "softmax" function over an array.
     Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/
     This is what softmax looks like in "pseudocode" (actually using Python
     and numpy):
     x -= np.max(x)
     exp_scores = np.exp(x)
     softmax = exp_scores / np.sum(exp_scores)
     First we shift the values of x so that the highest value in the array is 0.
     This ensures numerical stability with the exponents, so they don't blow up.
     */
    static func softmax(_ x: [Double]) -> [Double] {
        var x = x
        let len = vDSP_Length(x.count)
        
        // Find the maximum value in the input array.
        var max = 0.0
        vDSP_maxvD(x, 1, &max, len)
        
        // Subtract the maximum from all the elements in the array.
        // Now the highest value in the array is 0.
        max = -max
        vDSP_vsaddD(x, 1, &max, &x, 1, len)
        
        // Exponentiate all the elements in the array.
        var count = Int32(x.count)
        vvexp(&x, x, &count)
        
        // Compute the sum of all exponentiated values.
        var sum = 0.0
        vDSP_sveD(x, 1, &sum, len)
        
        // Divide each element by the sum. This normalizes the array contents
        // so that they all add up to 1.
        vDSP_vsdivD(x, 1, &sum, &x, 1, len)
        
        return x
    }
    
    /// Multinomial sampling
    ///
    /// From https://stackoverflow.com/questions/30309556/generate-random-numbers-with-a-given-distribution
    ///
    static func randomNumber(probabilities: [Double]) -> Int {
        // Sum of all probabilities (so that we don't have to require that the sum is 1.0):
        let sum = probabilities.reduce(0, +)
        // Random number in the range 0.0 <= rnd < sum :
        let rnd = sum * Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max)
        // Find the first interval of accumulated probabilities into which `rnd` falls:
        var accum = 0.0
        for (i, p) in probabilities.enumerated() {
            accum += p
            if rnd < accum {
                return i
            }
        }
        // This point might be reached due to floating point inaccuracies:
        return (probabilities.count - 1)
    }
}
