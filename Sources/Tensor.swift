// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py

import Foundation

public struct Tensor {
    public typealias Element = Float
    public let shape: Shape
    public private(set) var elements: [Element]
    public init(shape: Shape, elements: [Element]) {
        let volume = shape.volume
        precondition(elements.count >= volume, "`elements.count` must be greater than or equal to `shape.volume`: elements.count = \(elements.count), shape.volume = \(shape.volume)")
        self.shape = shape
        self.elements = (elements.count == volume) ? elements : Array(elements[0..<volume])
    }
}

extension Tensor { // Additional Initializers
    public init(shape: Shape, element: Element = 0.0) {
        self.init(shape: shape, elements: [Element](repeating: element, count: shape.volume))
    }
}

extension Tensor {
    public func reshape(shape: Shape) -> Tensor {
        return Tensor(shape: shape, elements: elements)
    }
}

extension Tensor { // like CollentionType
    internal func index(indices: [Int]) -> Int {
        assert(indices.count == shape.dimensions.count, "`indices.count` must be \(shape.dimensions.count): \(indices.count)")
        return zip(shape.dimensions, indices).reduce(0) {
            assert(0 <= $1.1 && $1.1 < $1.0.value, "Illegal index: indices = \(indices), shape = \(shape)")
            return $0 * $1.0.value + $1.1
        }
    }
    
    public subscript(indices: Int...) -> Element {
        get {
            return elements[index(indices: indices)]
        }
        set {
            elements[index(indices: indices)] = newValue
        }
    }
    
    public var volume: Int {
        return shape.volume
    }
}

extension Tensor: Sequence {
    public func makeIterator() -> IndexingIterator<[Element]> {
        return elements.makeIterator()
    }
}

extension Tensor: Equatable {}
public func ==(left: Tensor, right: Tensor) -> Bool {
    return left.shape == right.shape && left.elements == right.elements
}

internal func commutativeBinaryOperation(_ left: Tensor, _ right: Tensor, operation: (Float, Float) -> Float) -> Tensor {
    let lSize = left.shape.dimensions.count
    let rSize = right.shape.dimensions.count
    
    if lSize == rSize {
        precondition(left.shape == right.shape, "Incompatible shapes of tensors: left.shape = \(left.shape), right.shape = \(right.shape)")
        return Tensor(shape: left.shape, elements: zipMap(left.elements, right.elements, operation: operation))
    }
    
    let a: Tensor
    let b: Tensor
    if lSize < rSize {
        a = right
        b = left
    } else {
        a = left
        b = right
    }
    assert(hasSuffix(array: a.shape.dimensions, suffix: b.shape.dimensions), "Incompatible shapes of tensors: left.shape = \(left.shape), right.shape = \(right.shape)")
    
    return Tensor(shape: a.shape, elements: zipMapRepeat(a.elements, b.elements, operation: operation))
}

internal func noncommutativeBinaryOperation(_ left: Tensor, _ right: Tensor, operation: (Float, Float) -> Float) -> Tensor {
    let lSize = left.shape.dimensions.count
    let rSize = right.shape.dimensions.count
    
    if lSize == rSize {
        precondition(left.shape == right.shape, "Incompatible shapes of tensors: left.shape = \(left.shape), right.shape = \(right.shape)")
        return Tensor(shape: left.shape, elements: zipMap(left.elements, right.elements, operation: operation))
    } else if lSize < rSize {
        precondition(hasSuffix(array: right.shape.dimensions, suffix: left.shape.dimensions), "Incompatible shapes of tensors: left.shape = \(left.shape), right.shape = \(right.shape)")
        return Tensor(shape: right.shape, elements: zipMapRepeat(right.elements, left.elements, operation: { operation($1, $0) }))
    } else {
        precondition(hasSuffix(array: left.shape.dimensions, suffix: right.shape.dimensions), "Incompatible shapes of tensors: left.shape = \(left.shape), right.shape = \(right.shape)")
        return Tensor(shape: left.shape, elements: zipMapRepeat(left.elements, right.elements, operation: operation))
    }
}

public func +(left: Tensor, right: Tensor) -> Tensor {
    return commutativeBinaryOperation(left, right, operation: +)
}

public func -(left: Tensor, right: Tensor) -> Tensor {
    return noncommutativeBinaryOperation(left, right, operation: -)
}

public func *(left: Tensor, right: Tensor) -> Tensor {
    return commutativeBinaryOperation(left, right, operation: *)
}

public func /(left: Tensor, right: Tensor) -> Tensor {
    return noncommutativeBinaryOperation(left, right, operation: /)
}

public func *(left: Tensor, right: Float) -> Tensor {
    return Tensor(shape: left.shape, elements: left.elements.map { $0 * right })
}

public func *(left: Float, right: Tensor) -> Tensor {
    return Tensor(shape: right.shape, elements: right.elements.map { left * $0 })
}

public func /(left: Tensor, right: Float) -> Tensor {
    return Tensor(shape: left.shape, elements: left.elements.map { $0 / right })
}

public func /(left: Float, right: Tensor) -> Tensor {
    return Tensor(shape: right.shape, elements: right.elements.map { left / $0 })
}

extension Tensor { // Matrix
    public func matmul(tensor: Tensor) -> Tensor {
        precondition(shape.dimensions.count == 2, "This tensor is not a matrix: shape = \(shape)")
        precondition(tensor.shape.dimensions.count == 2, "The given tensor is not a matrix: shape = \(tensor.shape)")
        precondition(tensor.shape.dimensions[0] == shape.dimensions[1], "Incompatible shapes of matrices: self.shape = \(shape), tensor.shape = \(tensor.shape)")
        
        let n = shape.dimensions[1].value
        
        let numRows = shape.dimensions[0]
        let numCols = tensor.shape.dimensions[1]
        
        let leftHead = UnsafeMutablePointer<Float>(self.elements)
        let rightHead = UnsafeMutablePointer<Float>(tensor.elements)
        
        let elements = [Float](repeating: 0.0, count: (numCols * numRows).value)
        for r in 0..<numRows.value {
            for i in 0..<n {
                var pointer = UnsafeMutablePointer<Float>(elements) + r * numCols.value
                let left = leftHead[r * n + i]
                var rightPointer = rightHead + i * numCols.value
                for _ in 0..<numCols.value {
                    pointer.pointee += left * rightPointer.pointee
                    pointer += 1
                    rightPointer += 1
                }
            }
        }
        
        return Tensor(shape: [numRows, numCols], elements: elements)
    }
}