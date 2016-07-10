import Foundation
extension Tensor {
    public var randomize: Tensor {
        return Tensor(shape: shape, elements: elements.map({
        	(x: Float) -> Float in return Float(drand48())
        }))
    }
}