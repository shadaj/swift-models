import Benchmark
import TensorFlow
import _Differentiation

var baselineWeights = [
    NoOpTensor(shape: []),
    NoOpTensor(shape: []),
    NoOpTensor(shape: []),
    NoOpTensor(shape: []),
    NoOpTensor(shape: []),
    NoOpTensor(shape: []),
    NoOpTensor(shape: []),
    NoOpTensor(shape: [])
]

@differentiable
public func leNetBaseline(input: NoOpTensor, weightArray: [NoOpTensor]) -> NoOpTensor {
    return input
      .transformByFakeWeights(weights: weightArray[0])
      .transformByFakeWeights(weights: weightArray[1])
      .transformByFakeWeights(weights: weightArray[2])
      .transformByFakeWeights(weights: weightArray[3])
      .transformByFakeWeights(weights: weightArray[4])
      .transformByFakeWeights(weights: weightArray[5])
      .transformByFakeWeights(weights: weightArray[6])
      .transformByFakeWeights(weights: weightArray[7])
}

var classifier =
  input(shape: [28, 28, 1])
    .conv2D(filterShape: (5, 5), outputChannels: 6, padding: .same, activation: relu)
    .avgPool2D(poolSize: (2, 2), strides: (2, 2))
    .conv2D(filterShape: (5, 5), outputChannels: 16, activation: relu)
    .avgPool2D(poolSize: (2, 2), strides: (2, 2))
    .flatten()
    .dense(outputSize: 120, activation: relu)
    .dense(outputSize: 84, activation: relu)
    .dense(outputSize: 10)
    .build()

benchmark("inference - baseline") {
    leNetBaseline(input: NoOpTensor(shape: [28, 28, 1]), weightArray: baselineWeights)
}

benchmark("inference - dynamic") {
    classifier(NoOpTensor(shape: [28, 28, 1]))
}

benchmark("training - baseline") {
    let (_, pullback) = valueWithPullback(at: baselineWeights, in: { leNetBaseline(input: NoOpTensor(shape: [28, 28, 1]), weightArray: $0) })
    baselineWeights.move(along: pullback(0.0))
}

benchmark("training - dynamic") {
    let (_, pullback) = valueWithPullback(at: classifier, in: { $0(NoOpTensor(shape: [28, 28, 1])) })
    classifier.move(along: pullback(0.0))
}

Benchmark.main()
