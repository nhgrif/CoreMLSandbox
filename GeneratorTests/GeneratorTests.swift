//
//  GeneratorTests.swift
//  GeneratorTests
//
//  Created by Nick Griffith on 6/13/22.
//

import XCTest
import Generator

final class GeneratorTests: XCTestCase {
    func testGenerator() throws {
        let prompt = "<|name|> Fireball <|spell|>"
        let tokensToGenerate = 64
        
        let expectation = expectation(description: "Generator expectation")
        var generatedTokens = [String]()
        let subscription = generate(from: prompt, tokens: tokensToGenerate).sink { newToken in
            defer { if newToken == "<|endoftext|>" { expectation.fulfill() } }
            print("[newToken]: \(newToken)")
            generatedTokens.append(newToken)
        }
        wait(for: [expectation], timeout: 600)
        let generated = generatedTokens.joined(separator: "")
        print("Generated text: \(prompt)\(generated)")
        subscription.cancel()
    }
    
    func testPerformanceTopP() throws {
        let prompt = "<|name|> Fireball <|spell|>"
        let endToken = "<|endoftext|>"
        measure {
            print("\(#function): \(generate(from: prompt, untilToken: endToken, strategy: .topP(0.8)))")
        }
    }
    
    func testPerformanceTopP2() throws {
        let prompt = "<|name|> Fireball <|spell|>"
        let endToken = "<|endoftext|>"
        measure {
            print("\(#function): \(generate(from: prompt, untilToken: endToken, strategy: .topP2(0.8)))")
        }
    }
    
    func testPerformanceTopK() throws {
        let prompt = "<|name|> Fireball <|spell|>"
        let endToken = "<|endoftext|>"
        measure {
            print("\(#function): \(generate(from: prompt, untilToken: endToken, strategy: .topK(100)))")
        }
    }
    
    func testPerformanceGreedy() throws {
        let prompt = "<|name|> Fireball <|spell|>"
        let endToken = "<|endoftext|>"
        measure {
            print("\(#function): \(generate(from: prompt, untilToken: endToken, strategy: .greedy))")
        }
    }
}
