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
// "Very Deep Convolutional Networks for Large-Scale Image Recognition"
// Karen Simonyan, Andrew Zisserman
// https://arxiv.org/abs/1409.1556

public struct AutoVGGBlock: AutoModule {
    let featureCounts: (Int, Int, Int, Int)
    let blockCount: Int

    public init(featureCounts: (Int, Int, Int, Int), blockCount: Int) {
        self.featureCounts = featureCounts
        self.blockCount = blockCount
    }

    public typealias LayerType = AutoSequenced<AutoSequencedMany<AutoConv2D<Float>>, AutoMaxPool2D<Float>>
    public lazy var initializeLayer: LayerType = {
        var blocks = [AutoConv2D<Float>(filterShape: (3, 3),
            outputChannels: featureCounts.1,
            padding: .same,
            activation: relu)]
        
        for _ in 1..<blockCount {
            blocks += [AutoConv2D<Float>(filterShape: (3, 3),
                outputChannels: featureCounts.3,
                padding: .same,
                activation: relu)]
        }
        
        return AutoSequencedMany(layers: blocks)
            .then(AutoMaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2)))
    }()
}

public struct AutoVGG16: AutoModule {
    let classCount: Int

    public init(classCount: Int = 1000) {
        self.classCount = classCount
    }

    public lazy var initializeLayer = {
        return AutoVGGBlock(featureCounts: (3, 64, 64, 64), blockCount: 2)
            .then(AutoVGGBlock(featureCounts: (64, 128, 128, 128), blockCount: 2))
            .then(AutoVGGBlock(featureCounts: (128, 256, 256, 256), blockCount: 3))
            .then(AutoVGGBlock(featureCounts: (256, 512, 512, 512), blockCount: 3))
            .then(AutoVGGBlock(featureCounts: (512, 512, 512, 512), blockCount: 3))
            .then(AutoFlatten())
            .then(AutoDense(outputSize: 4096, activation: relu))
            .then(AutoDense(outputSize: 4096, activation: relu))
            .then(AutoDense(outputSize: classCount))
    }()
}

public struct VGG16: Layer {
    var underlying: BuiltAutoLayer<AutoVGG16.InstanceType>

    public init(classCount: Int = 1000) {
        underlying = AutoVGG16(classCount: classCount).buildModel(inputShape: (224, 224, 3))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        underlying(input)
    }
}

public struct AutoVGG19: AutoModule {
    let classCount: Int

    public init(classCount: Int = 1000) {
        self.classCount = classCount
    }

    public lazy var initializeLayer = {
        return AutoVGGBlock(featureCounts: (3, 64, 64, 64), blockCount: 2)
            .then(AutoVGGBlock(featureCounts: (64, 128, 128, 128), blockCount: 2))
            .then(AutoVGGBlock(featureCounts: (128, 256, 256, 256), blockCount: 4))
            .then(AutoVGGBlock(featureCounts: (256, 512, 512, 512), blockCount: 4))
            .then(AutoVGGBlock(featureCounts: (512, 512, 512, 512), blockCount: 4))
            .then(AutoFlatten())
            .then(AutoDense(outputSize: 4096, activation: relu))
            .then(AutoDense(outputSize: 4096, activation: relu))
            .then(AutoDense(outputSize: classCount))
    }()
}

public struct VGG19: Layer {
    var underlying: BuiltAutoLayer<AutoVGG19.InstanceType>

    public init(classCount: Int = 1000) {
        underlying = AutoVGG19(classCount: classCount).buildModel(inputShape: (224, 224, 3))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        underlying(input)
    }
}
