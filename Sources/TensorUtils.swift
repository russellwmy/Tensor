import Foundation
extension Tensor {
    public var random: Tensor {
        return Tensor(shape: shape, elements: elements.map { drand48() })
    }
}