public struct NoOpTensor: EuclideanDifferentiable, KeyPathIterable {
    @noDerivative let shape: [Int]; // TODO(shadaj): fixed size array
    public typealias TangentVector = Float

    public init(shape: [Int]) {
        self.shape = shape
    }

    @differentiable
    public func transformByFakeWeights(weights: NoOpTensor) -> NoOpTensor {
        return NoOpTensor(shape: weights.shape)
    }

    public mutating func move(along direction: Float) {}

    public var differentiableVectorView: Float = 0.0
}
