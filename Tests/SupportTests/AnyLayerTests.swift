// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import XCTest
import ModelSupport
import TensorFlow

final class AnyLayerTests: XCTestCase {
    func testGradients() {
        let originalLayer = Dense<Float>(inputSize: 1, outputSize: 1)
        let anyLayer = AnyLayer<Tensor<Float>, Tensor<Float>, Float>(originalLayer)

        let originalGradient = gradient(at: originalLayer, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        let anyLayerGradient = gradient(at: anyLayer, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        XCTAssertEqual(originalGradient, anyLayerGradient.base as! Dense<Float>.TangentVector)
    }

    func testTangentOperations() {
        let originalLayer = Dense<Float>(inputSize: 1, outputSize: 1)
        let anyLayer = AnyLayer<Tensor<Float>, Tensor<Float>, Float>(originalLayer)

        let originalGradient = gradient(at: originalLayer, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        let anyLayerGradient = gradient(at: anyLayer, in: { layer in
            return (layer(Tensor([[1.0]])) - Tensor([2.0])).squared().mean()
        })

        let transformedOriginal = Dense<Float>.TangentVector.one + originalGradient

        let transformedAny = AnyLayer<Tensor<Float>, Tensor<Float>, Float>.TangentVector.one + anyLayerGradient

        XCTAssertEqual(transformedOriginal, transformedAny.base as! Dense<Float>.TangentVector)
    }
    
    static var allTests = [
        ("testGradients", testGradients),
        ("testTangentOperations", testTangentOperations),
    ]
}
