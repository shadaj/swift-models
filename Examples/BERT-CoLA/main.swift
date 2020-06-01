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

import Datasets
import Foundation
import ModelSupport
import TensorFlow
import TextModels
import x10_optimizers_optimizer

// TODO(shadaj): Device.defaultXLA causes CUDA OOM
let device = Device(kind: .GPU, ordinal: 0, backend: .XLA)

let bertPretrained = BERT.PreTrainedModel.bertBase(cased: false, multilingual: false)
let workspaceURL = URL(
    fileURLWithPath: "bert_models", isDirectory: true,
    relativeTo: URL(
        fileURLWithPath: NSTemporaryDirectory(),
        isDirectory: true))
let bert = try BERT.PreTrainedModel.load(bertPretrained)(from: workspaceURL)
var bertClassifier = BERTClassifier(bert: bert, classCount: 1)
bertClassifier.move(to: device)

// Regarding the batch size, note that the way batching is performed currently is that we bucket
// input sequences based on their length (e.g., first bucket contains sequences of length 1 to 10,
// second 11 to 20, etc.). We then keep processing examples in the input data pipeline until a
// bucket contains enough sequences to form a batch. The batch size specified in the task
// constructor specifies the *total number of tokens in the batch* and not the total number of
// sequences. So, if the batch size is set to 1024, the first bucket (i.e., lengths 1 to 10)
// will need 1024 / 10 = 102 examples to form a batch (every sentence in the bucket is padded
// to the max length of the bucket). This kind of bucketing is common practice with NLP models and
// it is done to improve memory usage and computational efficiency when dealing with sequences of
// varied lengths. Note that this is not used in the original BERT implementation released by
// Google and so the batch size setting here is expected to differ from that one.
let maxSequenceLength = 128
let batchSize = 1024

var cola = try CoLA(
  taskDirectoryURL: workspaceURL,
  maxSequenceLength: maxSequenceLength,
  batchSize: batchSize,
  entropy: SystemRandomNumberGenerator()
) { (example: CoLAExample) -> LabeledTextBatch in
  let textBatch = bertClassifier.bert.preprocess(
    sequences: [example.sentence],
    maxSequenceLength: maxSequenceLength)
  return (data: textBatch, label: Tensor<Int32>(example.isAcceptable! ? 1 : 0))
}

print("Dataset acquired.")

public func makeWeightDecayedAdam(
  learningRate: Float = 0.01,
  beta1: Float = 0.9,
  beta2: Float = 0.999,
  weightDecayRate: Float = 0.01,
  epsilon: Float = 1e-6
) -> ParameterGroupOptimizer {
  var b = ParameterGroupOptimizerBuilder()
  let lr = b.makeParameter("learningRate", learningRate)
  let beta1 = b.makeParameter("beta1", beta1)
  let beta2 = b.makeParameter("beta2", beta2)
  let wd = b.makeParameter("weightDecay", weightDecayRate)

  let firstMoment = b[state: "firstMoment"]
  let secondMoment = b[state: "secondMoment"]

  b.appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
    optState[state, firstMoment] =
      state[beta1] * optState[state, firstMoment] + state.grad * (1 - state[beta1])
  }

  b.appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
    optState[state, secondMoment] =
      state[beta2] * optState[state, secondMoment] + state.grad .* state.grad * (1 - state[beta2])
  }

  b.appendCallback { (state: inout OptimizerWeightStepState, optState: inout OptimizerState) in
    let denominator = sqrt(optState[state, secondMoment]).adding(epsilon)
    let update = optState[state, firstMoment] ./ denominator + state.weight * state[wd]
    state.step = -state[lr] * update
  }

  return b.makeOptimizer()
}

let beta1: Float = 0.9
let beta2: Float = 0.999
let useBiasCorrection = true

var optimizer = x10_optimizers_optimizer.GeneralOptimizer(
    for: bertClassifier,
    TensorVisitorPlan(bertClassifier.differentiableVectorView),
    defaultOptimizer: makeWeightDecayedAdam(
      learningRate: 2e-5,
      beta1: beta1,
      beta2: beta2
    )
)

// TOOD(shadaj): 1068 is a magic number for the number of batches in each epoch, where does this come from?
// TODO(shadaj): learning rate originally went to zero very early, changed to decay until the last step
var scheduledLearningRate = LinearlyDecayedParameter(
  baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter<Float>(2e-5),
      warmUpStepCount: 10,
      warmUpOffset: 0),
  slope: -(2e-5 / (1068 * 3)),  // The LR decays linearly to zero in 3*1068 steps.
  startStep: 10
)

print("Training BERT for the CoLA task!")
for (epoch, epochBatches) in cola.trainingEpochs.prefix(3).enumerated() {
    print("[Epoch \(epoch + 1)]")
    Context.local.learningPhase = .training
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0

    for batch in epochBatches {
        let (eagerDocuments, eagerLabels) = (batch.data, Tensor<Float>(batch.label))
        let documents = TextBatch(
          tokenIds: Tensor(copying: eagerDocuments.tokenIds, to: device),
          tokenTypeIds: Tensor(copying: eagerDocuments.tokenTypeIds, to: device),
          mask: Tensor(copying: eagerDocuments.mask, to: device)
        )
        let labels = Tensor(copying: eagerLabels, to: device)
        var (loss, gradients) = valueWithGradient(at: bertClassifier) { model -> Tensor<Float> in
            let logits = model(documents)
            return sigmoidCrossEntropy(
                logits: logits.squeezingShape(at: -1),
                labels: labels,
                reduction: { $0.mean() })
        }

        trainingLossSum += loss.scalarized()
        trainingBatchCount += 1
        gradients.clipByGlobalNorm(clipNorm: 1)

        let step = optimizer.step + 1 // for scheduled rates and bias correction, steps start at 1
        optimizer.learningRate = scheduledLearningRate(forStep: UInt64(step))
        if useBiasCorrection {
          let step = Float(step)
          optimizer.learningRate *= sqrtf(1 - powf(beta2, step)) / (1 - powf(beta1, step))
        }

        optimizer.update(&bertClassifier, along: gradients)
        LazyTensorBarrier()

        print(
            """
              Training loss: \(trainingLossSum / Float(trainingBatchCount))
            """
        )
    }

    Context.local.learningPhase = .inference
    var devLossSum: Float = 0
    var devBatchCount = 0
    var devPredictedLabels = [Bool]()
    var devGroundTruth = [Bool]()
    for batch in cola.validationBatches {
        let (eagerDocuments, eagerLabels) = (batch.data, Tensor<Float>(batch.label))
        let documents = TextBatch(
          tokenIds: Tensor(copying: eagerDocuments.tokenIds, to: device),
          tokenTypeIds: Tensor(copying: eagerDocuments.tokenTypeIds, to: device),
          mask: Tensor(copying: eagerDocuments.mask, to: device)
        )
        let labels = Tensor(copying: eagerLabels, to: device)
        let logits = bertClassifier(documents)
        let loss = sigmoidCrossEntropy(
            logits: logits.squeezingShape(at: -1),
            labels: labels,
            reduction: { $0.mean() }
        )
        devLossSum += loss.scalarized()
        devBatchCount += 1

        let predictedLabels = sigmoid(logits.squeezingShape(at: -1)) .>= 0.5
        devPredictedLabels.append(contentsOf: predictedLabels.scalars)
        devGroundTruth.append(contentsOf: labels.scalars.map { $0 == 1 })
    }

    let mcc = matthewsCorrelationCoefficient(
        predictions: devPredictedLabels,
        groundTruth: devGroundTruth)

    print(
        """
          MCC: \(mcc)
          Eval loss: \(devLossSum / Float(devBatchCount))
        """
    )
}
