import TensorFlow

// workaround for https://bugs.swift.org/browse/TF-1078
public extension Array {
  /// A functional version of `Array.subscript.modify`.
  /// Differentiation does yet not support `Array.subscript.modify` because
  /// it is a coroutine.
  @differentiable(where Element: Differentiable)
  mutating func updated(at index: Int, with newValue: Element) {
    self[index] = newValue
  }
}

public extension Array where Element: Differentiable {
  @derivative(of: updated)
  mutating func vjpUpdated(at index: Int, with newValue: Element)
    -> (value: Void, pullback: (inout TangentVector) -> (Element.TangentVector)) {
    self.updated(at: index, with: newValue)
    return ((), { v in
       let dElement = v[index]
      if index < v.base.count{
          v.base[index] = .zero
      }
      return dElement
    })
  }
}

/// A specification for a layer and all its dependencies.
public class TracingLayer : Hashable {
    /// The shape of the tensor emitted by this layer
    var outputShape: [Int] {
        fatalError("Must implement")
    }

    /// Constructs the initial weights wrapped into a type-erased container
    func makeWeights() -> AnyDifferentiable {
        fatalError("Must implement")
    }

    /// Gets the list of immediate dependencies of the current layer, whose outputs are used in the current layer's computation
    var dependencies: [TracingLayer] {
        fatalError("Must implement")
    }

    /// Returns a closure which executes the layer by pulling inputs from a dependency source and calling the classic layer
    /// - dependencyIndices: the indices of the cache at which the layer's dependencies lie; each index in the array corresponds
    ///   to the layer at the same index in getDependencies()
    func buildLayerApplication(dependencyIndices: [Int]) // TODO(shadaj): layerApplication
        -> @differentiable (_ dependencySource: [NoOpTensor], _ weights: AnyDifferentiable) -> NoOpTensor {
        fatalError("Must implement")
    }

    public static func ==(lhs: TracingLayer, rhs: TracingLayer) -> Bool {
        return lhs === rhs
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
}

extension TracingLayer {
    /// Constructs an instance of the layer graph specified by `self`.
    public func build() -> ComposedLayer {
        // first, explore the graph to locate all layers and precompute values for topological sort
        var allLayers: [TracingLayer] = []
        var toVisit: [TracingLayer] = [self] // TODO(shadaj): should be a queue
        var unresolvedDependenciesPerLayer: [TracingLayer:Int] = [:]
        var inputLayer: TracingLayer? = nil
        
        var dependents: [TracingLayer: [TracingLayer]] = [:]
        var dependentsCount: [TracingLayer : Int] = [:]

        while toVisit.count > 0 {
            let next = toVisit.removeFirst()
            if (!allLayers.contains(next)) {
                allLayers.append(next)
                
                let dependencies = next.dependencies
                unresolvedDependenciesPerLayer[next] = dependencies.count
                
                if dependencies.count > 0 {
                    for dependency in dependencies {
                        if dependents[dependency] == nil {
                            dependents[dependency] = []
                            dependentsCount[dependency] = 0
                        }

                        dependents[dependency]!.append(next)
                        dependentsCount[dependency]! += 1
                        toVisit.append(dependency)
                    }
                } else {
                    // we've found the input layer, which has no dependencies
                    inputLayer = next
                }
            }
        }

        // compute topological sort
        var allDependenciesMet: [TracingLayer] = [inputLayer!]
        var layerComputeOrder: [TracingLayer] = []
        var weightsBuilt: [AnyDifferentiable] = []
        var layerToIndex: [TracingLayer : Int] = [:]

        while allDependenciesMet.count > 0 {
            let next = allDependenciesMet.removeFirst()
            layerComputeOrder.append(next)
            weightsBuilt.append(next.makeWeights())
            layerToIndex[next] = weightsBuilt.count - 1
            for dependent in dependents[next, default: []] {
                unresolvedDependenciesPerLayer[dependent]! -= 1
                if unresolvedDependenciesPerLayer[dependent]! == 0 {
                    allDependenciesMet.append(dependent)
                }
            }
        }

        // build out the function that executes all layers in the order determined by the topological sort
        var accumulatedFunction: @differentiable (inout [NoOpTensor], [AnyDifferentiable]) -> Void = {_,_ in}

        var lastIndex = 0
        var maxIndex = 0
        var allocatedIndices: [TracingLayer : Int] = [:]
        var openSlots: [Int] = []

        for (layerIndex, layer) in layerComputeOrder.enumerated() {
            var dependencyIndices: [Int] = []
            for dependency in layer.dependencies {
                let previouslyAllocated = allocatedIndices[dependency]!
                dependencyIndices.append(previouslyAllocated)
                
                dependentsCount[dependency]! -= 1
                if dependentsCount[dependency] == 0 {
                    // we read the dependencies before we write, so it's safe to output to a slot of a dependency
                    openSlots.append(previouslyAllocated)
                }
            }
            
            if dependencyIndices.count == 0 { // input layer
                dependencyIndices.append(0)
                openSlots.append(0) // the input value is only used once, in the single input layer
            }

            let layerCaller = layer.buildLayerApplication(dependencyIndices: dependencyIndices)

            let prevAccumulated = accumulatedFunction
            let allocatedSlot = openSlots.count > 0 ? openSlots.removeFirst() : maxIndex + 1

            if allocatedSlot > maxIndex {
                assert(allocatedSlot == maxIndex + 1)
                accumulatedFunction = { (outputs: inout [NoOpTensor], weights: [AnyDifferentiable]) in
                    prevAccumulated(&outputs, weights)
                    outputs.append(layerCaller(outputs, weights[layerIndex]))
                }
            } else {
                accumulatedFunction = { (outputs: inout [NoOpTensor], weights: [AnyDifferentiable]) in
                    prevAccumulated(&outputs, weights)
                    outputs.updated(at: allocatedSlot, with: layerCaller(outputs, weights[layerIndex]))
                    // faster: outputs = [underlyingFunction(outputs, layers[index])]
                }
            }

            allocatedIndices[layer] = allocatedSlot
            lastIndex = allocatedSlot
            maxIndex = max(maxIndex, allocatedSlot)
        }

        return ComposedLayer(
            weights: weightsBuilt,
            callFunction: { (input: NoOpTensor, weights: [AnyDifferentiable]) in
                var outputs: [NoOpTensor] = [input]
                accumulatedFunction(&outputs, weights)
                return outputs[lastIndex]
            }
        )
    }
}
