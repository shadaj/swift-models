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

public struct AutoConvBN: AutoModule {
    let filterShape: (Int, Int)
    let outputChannels: (Int)
    let strides: (Int, Int)
    let padding: Padding

    public init(
        filterShape: (Int, Int),
        outputChannels: Int,
        strides: (Int, Int) = (1, 1),
        padding: Padding = .valid
    ) {
        self.filterShape = filterShape
        self.outputChannels = outputChannels
        self.strides = strides
        self.padding = padding
    }

    public lazy var initializeLayer = {
        return AutoConv2D<Float>(filterShape: filterShape, outputChannels: outputChannels, strides: strides, padding: padding, useBias: false)
            .then(AutoBatchNorm(momentum: 0.9, epsilon: 1e-5))
    }()
}

public struct AutoResidualBlock: AutoModule {
    let inputFilters: Int
    let filters: Int
    let strides: (Int, Int)
    let useLaterStride: Bool
    let isBasic: Bool

    public init(
        inputFilters: Int, filters: Int, strides: (Int, Int), useLaterStride: Bool, isBasic: Bool
    ) {
        self.inputFilters = inputFilters
        self.filters = filters
        self.strides = strides
        self.useLaterStride = useLaterStride
        self.isBasic = isBasic
    }

    public typealias ConvPlusResidual = AutoSplitMerge<
        AutoSequencedMany<AutoConvBN>,
        AutoSequenced<
            AutoSequencedMany<
                AutoSequenced<
                    AutoConvBN,
                    AutoFunction<Tensor<Float>, Tensor<Float>, (Int, Int, Int), (Int, Int, Int)
                >
            >>,
            AutoConvBN
        >,
        Tensor<Float>, (Int, Int, Int)
    >
    
    public typealias LayerType = AutoSequenced<ConvPlusResidual, AutoFunction<Tensor<Float>, Tensor<Float>, (Int, Int, Int), (Int, Int, Int)>>

    public lazy var initializeLayer: LayerType = {
        let outFilters = filters * (isBasic ? 1 : 4)
        let needsProjection = (inputFilters != outFilters) || (strides.0 != 1)

        let projection = needsProjection
            ? AutoConvBN(filterShape: (1, 1), outputChannels: outFilters, strides: strides)
            : AutoConvBN(filterShape: (1, 1), outputChannels: 1)

        let residual = AutoSequencedMany(layers: needsProjection ? [projection]: [])

        var earlyConvs: [AutoConvBN] = []
        let lastConv: AutoConvBN
        if isBasic {
            earlyConvs.append(
                AutoConvBN(filterShape: (3, 3), outputChannels: filters, strides: strides, padding: .same))
            lastConv = AutoConvBN(filterShape: (3, 3), outputChannels: outFilters, padding: .same)
        } else {
            if useLaterStride {
                // Configure for ResNet V1.5 (the more common implementation).
                earlyConvs.append(AutoConvBN(filterShape: (1, 1), outputChannels: filters))
                earlyConvs.append(
                    AutoConvBN(filterShape: (3, 3), outputChannels: filters, strides: strides, padding: .same))
            } else {
                // Configure for ResNet V1 (the paper implementation).
                earlyConvs.append(
                    AutoConvBN(filterShape: (1, 1), outputChannels: filters, strides: strides))
                earlyConvs.append(AutoConvBN(filterShape: (3, 3), outputChannels: filters, padding: .same))
            }
            lastConv = AutoConvBN(filterShape: (1, 1), outputChannels: outFilters)
        }

        let earlyConvsWithRelu = earlyConvs.map({ (conv) in
            conv.then(AutoFunction(fnShape: { $0 }, fn: { relu($0) }))
        })

        let lastConvResult = AutoSequencedMany(layers: earlyConvsWithRelu).then(lastConv)

        let convPlusResidual = AutoSplitMerge(
            layer1: residual,
            layer2: lastConvResult,
            mergeOutputShape: { (l1, l2) in l1 }, mergeFn: { $0 + $1 })

        return convPlusResidual.then(AutoFunction(fnShape: { $0 }, fn: { relu($0) }))
    }()
}

/// An implementation of the ResNet v1 and v1.5 architectures, at various depths.
public struct AutoResNet: AutoModule {
    let classCount: Int
    let depth: ResNet.Depth
    let downsamplingInFirstStage: Bool
    let useLaterStride: Bool

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
    public init(classCount: Int, depth: ResNet.Depth, downsamplingInFirstStage: Bool = true, useLaterStride: Bool = true) {
        self.classCount = classCount
        self.depth = depth
        self.downsamplingInFirstStage = downsamplingInFirstStage
        self.useLaterStride = useLaterStride
    }

    public typealias LayerType = AutoSequenced<AutoSequenced<AutoSequenced<AutoSequenced<AutoSequenced<AutoConvBN, AutoFunction<Tensor<Float>, Tensor<Float>, AutoConv2D<Float>.OutputShape, AutoMaxPool2D<Float>.InputShape>>, AutoMaxPool2D<Float>>, AutoSequencedMany<AutoResidualBlock>>, AutoGlobalAvgPool2D<Float>>, AutoDense<Float>>

    public lazy var initializeLayer: LayerType = {
        let initialLayer: AutoConvBN
        let maxPool: AutoMaxPool2D<Float>

        let inputFilters: Int

        if downsamplingInFirstStage {
            inputFilters = 64
            initialLayer = AutoConvBN(
                filterShape: (7, 7), outputChannels: inputFilters, strides: (2, 2), padding: .same)
            maxPool = AutoMaxPool2D(poolSize: (3, 3), strides: (2, 2), padding: .same)
        } else {
            inputFilters = 16
            initialLayer = AutoConvBN(
                filterShape: (3, 3), outputChannels: inputFilters, padding: .same)
            maxPool = AutoMaxPool2D(poolSize: (1, 1), strides: (1, 1))  // no-op
        }

        var residualBlocks: [AutoResidualBlock] = []
        var lastInputFilterCount = inputFilters
        for (blockSizeIndex, blockSize) in depth.layerBlockSizes.enumerated() {
            for blockIndex in 0..<blockSize {
                let strides = ((blockSizeIndex > 0) && (blockIndex == 0)) ? (2, 2) : (1, 1)
                let filters = inputFilters * Int(pow(2.0, Double(blockSizeIndex)))
                let residualBlock = AutoResidualBlock(
                    inputFilters: lastInputFilterCount, filters: filters, strides: strides,
                    useLaterStride: useLaterStride, isBasic: depth.usesBasicBlocks)
                lastInputFilterCount = filters * (depth.usesBasicBlocks ? 1 : 4)
                residualBlocks.append(residualBlock)
            }
        }

        return initialLayer
            .then(AutoFunction(fnShape: { $0 }, fn: { (prev: Tensor<Float>) in relu(prev) }))
            .then(maxPool)
            .then(AutoSequencedMany(layers: residualBlocks))
            .then(AutoGlobalAvgPool2D())
            .then(AutoDense(outputSize: classCount))
    }()
}

public struct ResNet: Layer {
    public var underlying: BuiltAutoLayer<AutoResNet.InstanceType>
    
    public init(
        classCount: Int, depth: Depth, downsamplingInFirstStage: Bool = true,
        useLaterStride: Bool = true
    ) {
        underlying = AutoResNet(
            classCount: classCount, depth: depth,
            downsamplingInFirstStage: downsamplingInFirstStage,
            useLaterStride: useLaterStride
        ).buildModel(inputShape: (1, 1, 3))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        underlying(input)
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
