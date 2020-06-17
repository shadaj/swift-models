// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import TensorFlow
import LayerInit

// Original Paper:
// "Gradient-Based Learning Applied to Document Recognition"
// Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
// http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
//
// Note: this implementation connects all the feature maps in the second convolutional layer.
// Additionally, ReLU is used instead of sigmoid activations.

public struct AutoLeNet: AutoModule {
    public lazy var initializeLayer = {
        return AutoConv2D<Float>(filterShape: (5, 5), outputChannels: 6, padding: .same, activation: relu)
            .then(AutoAvgPool2D(poolSize: (2, 2), strides: (2, 2)))
            .then(AutoConv2D(filterShape: (5, 5), outputChannels: 16, activation: relu))
            .then(AutoAvgPool2D(poolSize: (2, 2), strides: (2, 2)))
            .then(AutoFlatten())
            .then(AutoDense(outputSize: 120, activation: relu))
            .then(AutoDense(outputSize: 84, activation: relu))
            .then(AutoDense(outputSize: 10))
    }()
}

public struct LeNet: Layer {
    var underlying = AutoLeNet().buildModel(inputShape: (28, 28, 1))

    public init() {}

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return underlying(input)
    }
}
