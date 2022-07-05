//
//  GPT2.swift
//  CoreMLGPT2
//
//  Created by Julien Chaumond on 19/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import CoreML
import Combine

public func generate(from prompt: String, tokens: Int = 512) -> AnyPublisher<String,Never> {
    let subject = PassthroughSubject<String,Never>()
    
    Task {
        let generator = GPT2(strategy: .topP(0.8))
        _ = generator.generate(text: prompt, nTokens: tokens) { token in
            subject.send(token)
            return token != "<|endoftext|>"
        }
    }
    
    return subject.eraseToAnyPublisher()
}

public func generate(from prompt: String, untilToken endToken: String, strategy: DecodingStrategy) -> String {
    let generator = GPT2(strategy: strategy)
    return generator.generate(text: prompt, nTokens: 150) { newestToken in
        return newestToken != endToken
    }
}

public enum DecodingStrategy {
    /// At each time step, we select the most likely next token
    case greedy
    /// Sample only from the top-k most-probable tokens (k is a hyper-parameter).
    case topK(Int)
    /// Sample from the top tokens with a cumulative probability just above a threshold (nucleus/top-p).
    case topP(Double)
    case topP2(Double)
}

class GPT2 {
    

    
    private let model = try! gpt2_64_12_2(configuration: .init())
    let tokenizer = GPT2Tokenizer()
    let seqLen = 64
    private let strategy: DecodingStrategy
    
    init(strategy: DecodingStrategy = .greedy) {
        self.strategy = strategy
    }
    
    /// Main prediction loop:
    /// Predict next token from array of previous tokens.
    /// - featurization
    /// - model inference
    /// - Decoding according to the model's `strategy`
    func predict(tokens: [Int]) -> Int {
        let maxTokens = (tokens.count > seqLen)
        ? Array(tokens[..<seqLen])
        : tokens
        
        /// Pad input_ids on the right, up to `seqLen`:
        let input_ids = MLMultiArray.from(
            maxTokens + Array(repeating: 0, count: seqLen - maxTokens.count)
        )
        let position_ids = MLMultiArray.from(
            Array(0..<seqLen)
        )
        
        let output = try! model.prediction(input_ids: input_ids, position_ids: position_ids)
        
        let outputLogits = MLMultiArray.slice(
            output.output_logits,
            indexing: [.select(0), .select(maxTokens.count - 1), .slice, .select(0), .select(0)]
        )
        
        switch strategy {
        case .greedy:
            let nextToken = Math.argmax(outputLogits)
            return nextToken.0
        case .topK(let k):
            let logits = MLMultiArray.toDoubleArray(outputLogits)
            let topk = Math.topK(arr: logits, k: k)
            let sampleIndex = Math.sample(indexes: topk.indexes, probs: topk.probs)
            return sampleIndex
        case .topP(let p):
            let logits = MLMultiArray.toDoubleArray(outputLogits)
            let topP = Math.topP(arr: logits, p: p)
            let sampleIndex = Math.sample(indexes: topP.indexes, probs: topP.probs)
            return sampleIndex
        case .topP2(let p):
            let logits = MLMultiArray.toDoubleArray(outputLogits)
            return Math.selectRandom(from: logits, withinTopP: p)
        }
    }
    
    
    /// Main generation loop.
    ///
    /// Will generate next `nTokens` (defaults to 10).
    /// Calls an incremental `callback` for each new token, then returns the generated string at the end.
    ///
    func generate(text: String, nTokens: Int = 10, callback: (String) -> Bool) -> String {
        var tokens = tokenizer.encode(text: text)
        var newTokens: [Int] = []
        for _ in 0..<nTokens {
            let nextToken = predict(tokens: tokens)
            
            tokens.append(nextToken)
            newTokens.append(nextToken)
            if !callback(
                tokenizer.decode(tokens: [nextToken])
            ) { break }
        }
        return tokenizer.decode(tokens: tokens)
    }
}
