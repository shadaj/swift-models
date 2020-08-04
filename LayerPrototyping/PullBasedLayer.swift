// does not compile yet due to AD bug (fix will exist soon)
protocol MyProtocol {
    @differentiable(wrt: weight)
    func execute(weight: Float, outputs: inout Float)
}

struct Impl: MyProtocol {
    func execute(weight: Float, outputs: inout Float) {
        outputs = weight
    }
}


public class ReferenceProvider: Hashable {
    public static func ==(lhs: ReferenceProvider, rhs: ReferenceProvider) -> Bool {
        return lhs === rhs
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

public protocol PullBasedLayer {
    /// The shape of the tensor emitted by this layer
    var outputShape: [Int] { get }

    /// Constructs the initial weights wrapped into a type-erased container
    func makeWeights() -> AnyDifferentiable

    @differentiable(wrt: weights)
    func execute(input: NoOpTensor, weightIndices: [ReferenceProvider : Int], weights: [AnyDifferentiable], outputIndices:  [ReferenceProvider: Int], outputs: inout [NoOpTensor]) -> ()
}

public struct InputPullLayer: PullBasedLayer {
    let _outputShape: [Int]
    let selfReferenceProvider = ReferenceProvider()

    public init(shape: [Int]) {
        self._outputShape = shape
    }

    public var outputShape: [Int] {
        return _outputShape
    }

    public func makeWeights() -> AnyDifferentiable {
        return AnyDifferentiable(NoOpTensor(shape: [])) // TODO
    }

    @differentiable(wrt: weights)
    public func execute(input: NoOpTensor, weightIndices: [ReferenceProvider : Int], weights: [AnyDifferentiable], outputIndices: [ReferenceProvider: Int], outputs: inout [NoOpTensor]) -> () {
        outputs[outputIndices[selfReferenceProvider]!] = input
        return
    }
}

public class PullLayerWrapper<Impl: LayerImpl>: PullBasedLayer
where Impl.Weights.TangentVector: VectorProtocol, Impl.Weights.TangentVector.VectorSpaceScalar == Float {
    let impl: Impl
    let weights: Impl.Weights
    let dependency: PullBasedLayer
    let _outputShape: [Int]

    public init(dependency: PullBasedLayer, impl: Impl, weights: Impl.Weights, outputShape: [Int]) {
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

    @differentiable(wrt: weights)
    override func execute(input: NoOpTensor, weightIndices: [PullBasedLayer : Int], weights: [AnyDifferentiable], outputIndices:  [PullBasedLayer: Int], outputs: inout [NoOpTensor]) {
        dependency.execute(input: input, weightIndices: weightIndices, weights: weights, outputIndices: outputIndices, outputs: &outputs)

        let selfWeights: Impl.Weights = DynamicWeightHelper.castDifferentiably(
            weights[weightIndices[self]!], to: Impl.Weights.self
        )
        
        let selfResult: NoOpTensor = self.impl.callAsFunction(
            weights: selfWeights,
            input: outputs[outputIndices[dependency]!]
        )

        outputs[outputIndices[self]!] = selfResult
    }
}
