import TensorFlow
import _Differentiation

struct DynamicWeightHelper {
  @differentiable
  static func castDifferentiably<W: EuclideanDifferentiable & KeyPathIterable>(
    _ weights: AnyDifferentiable,
    to: W.Type
  ) -> W where W.TangentVector: VectorProtocol, W.TangentVector.VectorSpaceScalar == Float {
    // TODO(shadaj): unsafeDowncast
    return weights.base as! W
  }

  @derivative(of: castDifferentiably)
  static func dCastDifferentiably<W: EuclideanDifferentiable & KeyPathIterable>(
    _ weights: AnyDifferentiable,
    to: W.Type
  ) -> (
    value: W,
    pullback: (W.TangentVector) -> AnyDifferentiable.TangentVector
  ) where W.TangentVector: VectorProtocol, W.TangentVector.VectorSpaceScalar == Float {
    let underlyingWeights = weights.base as! W
    return (value: underlyingWeights, pullback: { v in
      return AnyLayerTangentVector(v)
    })
  }
}

public struct ComposedLayer: Differentiable {
  typealias CallFunction = @differentiable (NoOpTensor, [AnyDifferentiable]) -> NoOpTensor

  var weights: [AnyDifferentiable] = []
  @noDerivative let callFunction: CallFunction

  init(weights: [AnyDifferentiable], callFunction: @escaping CallFunction) {
    self.weights = weights
    self.callFunction = callFunction
  }

  @differentiable
  public func callAsFunction(_ input: NoOpTensor) -> NoOpTensor {
    return callFunction(input, weights)
  }
}
