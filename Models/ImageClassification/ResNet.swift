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
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// This uses shortcut layers to connect residual blocks
// (aka Option (B) in https://arxiv.org/abs/1812.01187).
//
// The structure of this implementation was inspired by the Flax ResNet example:
// https://github.com/google/flax/blob/master/examples/imagenet/models.py

extension FunctionalLayer {
    public func convBN(filterShape: (Int, Int), outputChannels: Int, strides: (Int, Int) = (1, 1), padding: Padding = .valid) -> FunctionalLayer {
        return self
            .conv2D(filterShape: filterShape, outputChannels: outputChannels, strides: strides, padding: padding, useBias: false)
            .batchNorm(momentum: 0.9, epsilon: 1e-5)
    }
}

public func residualBlock(input: FunctionalLayer, inputFilters: Int, filters: Int, strides: (Int, Int), useLaterStride: Bool, isBasic: Bool) -> FunctionalLayer {
    let outFilters = filters * (isBasic ? 1 : 4)

    let needsProjection = (inputFilters != outFilters) || (strides.0 != 1)
    let residual = needsProjection ?
        input.convBN(filterShape: (1, 1), outputChannels: outFilters, strides: strides) :
        input

    let convsApplied: FunctionalLayer
    if isBasic {
        convsApplied = input
            .convBN(filterShape: (3, 3), outputChannels: filters, strides: strides, padding: .same).relu()
            .convBN(filterShape: (3, 3), outputChannels: outFilters, padding: .same)
    } else {
        let earlyConvsApplied: FunctionalLayer
        if useLaterStride {
            // Configure for ResNet V1.5 (the more common implementation).
            earlyConvsApplied = input
                .convBN(filterShape: (1, 1), outputChannels: filters).relu()
                .convBN(filterShape: (3, 3), outputChannels: filters, strides: strides, padding: .same).relu()
        } else {
            // Configure for ResNet V1 (the paper implementation).
            earlyConvsApplied = input
                .convBN(filterShape: (1, 1), outputChannels: filters, strides: strides).relu()
                .convBN(filterShape: (3, 3), outputChannels: filters, padding: .same).relu()
        }
        
        convsApplied = earlyConvsApplied.convBN(filterShape: (1, 1), outputChannels: outFilters)
    }

    return (residual + convsApplied).relu()
}

public func resNet(_ input: FunctionalLayer, classCount: Int, depth: ResNet.Depth, downsamplingInFirstStage: Bool = true, useLaterStride: Bool = true) -> FunctionalLayer {
    let inputFilters: Int
    let initialTransformed: FunctionalLayer
        
    if downsamplingInFirstStage {
        inputFilters = 64
        initialTransformed = input
            .convBN(filterShape: (7, 7), outputChannels: inputFilters, strides: (2, 2), padding: .same)
            .relu()
            .maxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .same)
    } else {
        inputFilters = 16
        initialTransformed = input
            .convBN(filterShape: (3, 3), outputChannels: inputFilters, padding: .same)
            .relu()
    }

    var throughBlocks = initialTransformed
    var lastInputFilterCount = inputFilters
    for (blockSizeIndex, blockSize) in depth.layerBlockSizes.enumerated() {
        for blockIndex in 0..<blockSize {
            let strides = ((blockSizeIndex > 0) && (blockIndex == 0)) ? (2, 2) : (1, 1)
            let filters = inputFilters * Int(pow(2.0, Double(blockSizeIndex)))
            throughBlocks = residualBlock(
                input: throughBlocks,
                inputFilters: lastInputFilterCount, filters: filters, strides: strides,
                useLaterStride: useLaterStride, isBasic: depth.usesBasicBlocks
            )

            lastInputFilterCount = filters * (depth.usesBasicBlocks ? 1 : 4)
        }
    }

    return throughBlocks
        .globalAvgPool2D()
        .flatten()
        .dense(outputSize: classCount)
}

/// An implementation of the ResNet v1 and v1.5 architectures, at various depths.
public struct ResNet: Layer {
    public var underlying: ComposedLayer

    /// Initializes a new ResNet v1 or v1.5 network model.
    ///
    /// - Parameters:
    ///   - classCount: The number of classes the network will be or has been trained to identify.
    ///   - depth: A specific depth for the network, chosen from the enumerated values in 
    ///     ResNet.Depth.
    ///   - downsamplingInFirstStage: Whether or not to downsample by a total of 4X among the first
    ///     two layers. For ImageNet-sized images, this should be true, but for smaller images like
    ///     CIFAR-10, this probably should be false for best results.
    ///   - inputFilters: The number of filters at the first convolution.
    ///   - useLaterStride: If false, the stride within the residual block is placed at the position
    ///     specified in He, et al., corresponding to ResNet v1. If true, the stride is moved to the
    ///     3x3 convolution, corresponding to the v1.5 variant of the architecture. 
    public init(
        classCount: Int, depth: Depth, downsamplingInFirstStage: Bool = true,
        useLaterStride: Bool = true
    ) {
        underlying = resNet(
            InputFunctionalLayer(shape: [1, 1, 3]),
            classCount: classCount, depth: depth, downsamplingInFirstStage: downsamplingInFirstStage,
            useLaterStride: useLaterStride
        ).build()
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return underlying(input)
    }
}

extension ResNet {
    public enum Depth {
        case resNet18
        case resNet34
        case resNet50
        case resNet56
        case resNet101
        case resNet152

        var usesBasicBlocks: Bool {
            switch self {
            case .resNet18, .resNet34, .resNet56: return true
            default: return false
            }
        }

        var layerBlockSizes: [Int] {
            switch self {
            case .resNet18: return [2, 2, 2, 2]
            case .resNet34: return [3, 4, 6, 3]
            case .resNet50: return [3, 4, 6, 3]
            case .resNet56: return [9, 9, 9]
            case .resNet101: return [3, 4, 23, 3]
            case .resNet152: return [3, 8, 36, 3]
            }
        }
    }
}
