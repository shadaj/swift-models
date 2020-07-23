public struct NoOpTensor: EuclideanDifferentiable, KeyPathIterable {
    @noDerivative let shape: [Int]; // TODO(shadaj): fixed size array
    public typealias TangentVector = Float

    @inline(never)
    public init(shape: [Int]) {
        self.shape = shape
    }

    @differentiable
    @inline(never)
    public func transformByFakeWeights(weights: NoOpTensor) -> NoOpTensor {
        return NoOpTensor(shape: weights.shape)
    }

    @inline(never)
    public mutating func move(along direction: Float) {}

    @inline(never)
    public var differentiableVectorView: Float = 0.0
}
