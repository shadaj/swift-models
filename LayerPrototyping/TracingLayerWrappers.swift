import TensorFlow

/// A specification of the shape of the input to a traced graph.
public class InputTracingLayer: TracingLayer {
    let _outputShape: [Int]

    public init(shape: [Int]) {
        self._outputShape = shape
    }

    public override var outputShape: [Int] {
        return _outputShape
    }

    override func makeWeights() -> AnyDifferentiable {
        return AnyDifferentiable(NoOpTensor(shape: [])) // TODO
    }

    public override var dependencies: [TracingLayer] {
        return []
    }

    override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([NoOpTensor], AnyDifferentiable) -> NoOpTensor {
        let inputIndex = dependencyIndices[0]
        return { (dependencySource: [NoOpTensor], weights: AnyDifferentiable) in
            return dependencySource[inputIndex]
        }
    }
}

public protocol LayerImpl {
    associatedtype Weights: Differentiable & EuclideanDifferentiable & KeyPathIterable

    @differentiable
    func callAsFunction(weights: Weights, input: NoOpTensor) -> NoOpTensor
}

/// A specification for a layer that passes a single dependency's result through a classic layer
public class TracingLayerWrapper<Impl: LayerImpl>: TracingLayer
where Impl.Weights.TangentVector: VectorProtocol, Impl.Weights.TangentVector.VectorSpaceScalar == Float {
    let impl: Impl
    let weights: Impl.Weights
    let dependency: TracingLayer
    let _outputShape: [Int]

    public init(dependency: TracingLayer, impl: Impl, weights: Impl.Weights, outputShape: [Int]) {
        self.impl = impl
        self.dependency = dependency
        self.weights = weights
        self._outputShape = outputShape
    }

    public override var outputShape: [Int] {
        return _outputShape
    }

    override func makeWeights() -> AnyDifferentiable {
        return AnyDifferentiable(weights)
    }

    public override var dependencies: [TracingLayer] {
        return [dependency]
    }

    override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([NoOpTensor], AnyDifferentiable) -> NoOpTensor {
        let prevIndex = dependencyIndices[0]
        return { (outputs: [NoOpTensor], erasedWeights: AnyDifferentiable) -> NoOpTensor in
            return self.impl(
                weights: DynamicWeightHelper.castDifferentiably(
                    erasedWeights, to: Impl.Weights.self
                ),
                input: outputs[prevIndex]
            )
        }
    }
}

/// A tracing layer which combines the results of two dependencies with a custom function
public class MergeTracingLayer: TracingLayer {
    let mergeFn: @differentiable (NoOpTensor, NoOpTensor) -> NoOpTensor
    let dependency1: TracingLayer
    let dependency2: TracingLayer
    
    let _outputShape: [Int]

    public init(
        dependency1: TracingLayer, dependency2: TracingLayer,
        mergeFn: @escaping @differentiable (NoOpTensor, NoOpTensor) -> NoOpTensor,
        outputShape: [Int]
    ) {
        self.dependency1 = dependency1
        self.dependency2 = dependency2
        self.mergeFn = mergeFn
        self._outputShape = outputShape
    }

    public override var outputShape: [Int] {
        return _outputShape
    }

    override func makeWeights() -> AnyDifferentiable {
        return AnyDifferentiable(NoOpTensor(shape: [0])) // TODO
    }

    public override var dependencies: [TracingLayer] {
        return [dependency1, dependency2]
    }

    override func buildLayerApplication(dependencyIndices: [Int])
        -> @differentiable ([NoOpTensor], AnyDifferentiable) -> NoOpTensor {
        let prev1Index = dependencyIndices[0]
        let prev2Index = dependencyIndices[1]
        return { outputs, _ in
            return self.mergeFn(outputs[prev1Index], outputs[prev2Index])
        }
    }
}
