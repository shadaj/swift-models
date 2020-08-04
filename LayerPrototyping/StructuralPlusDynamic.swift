// sketch, does not compile
protocol StagedModel {
    var input: TracingLayer { get }
}

public func dynamicLayer(func: (TracingLayer) -> TracingLayer) -> TracingLayer {
    fatalError("")
}

struct MyModel: StagedModel {
    public var initialDense = dense(input, outputSize: 10)
    
    public var middleLayers = dynamicLayer { inputLayer in
        return initialDense
            .dense(outputSize: 1)
            .dense(outputSize: 10) + inputLayer
    }

    public var finalLayer = middleLayers.dense(outputSize: 1)
}
