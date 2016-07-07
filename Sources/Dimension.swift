// Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/tensor_shape.py

public struct Dimension {
    public let value: Int
    public init(_ value: Int) {
        guard value >= 0 else { fatalError("`value` must be greater than or equal to 0: \(value)") }
        self.value = value
    }
}

extension Dimension: IntegerLiteralConvertible {
    public init(integerLiteral value: Int) {
        self.init(value)
    }
}

extension Dimension: CustomStringConvertible {
    public var description: String {
        return value.description
    }
}

extension Dimension { 
    public func is_compatible_with(other: Dimension) -> Bool {
        return self.value == 0
            || other.value == 0 
            || self.value == other.value
    }
}

extension Dimension { 
    public func assert_is_compatible_with(other: Dimension) {
        if !self.is_compatible_with(other: other) {
            fatalError("Dimensions \(self.value) and \(other.value) are not compatible")
        }
    }
}

extension Dimension { 
    public func merge_with(other: Dimension) -> Dimension {
        if (self.value == 0) {
            return Dimension(other.value)
        } else {
            return Dimension(self.value)
        }
    }
}

public func +(left: Dimension, right: Dimension) -> Dimension {
    return Dimension(left.value + right.value)
}

public func -(left: Dimension, right: Dimension) -> Dimension {
    return Dimension(left.value - right.value)
}

public func *(left: Dimension, right: Dimension) -> Dimension {
    return Dimension(left.value * right.value)
}

public func /(left: Dimension, right: Dimension) -> Dimension {
    return Dimension(left.value / right.value)
}

public func %(left: Dimension, right: Dimension) -> Dimension {
    return Dimension(left.value % right.value)
}

extension Dimension: Equatable {}
public func ==(left: Dimension, right: Dimension) -> Bool {
    return left.value == right.value
}

public func <(left: Dimension, right: Dimension) -> Bool {
    return left.value < right.value
}

public func <=(left: Dimension, right: Dimension) -> Bool {
    return left.value <= right.value
}

public func >(left: Dimension, right: Dimension) -> Bool {
    return left.value > right.value
}

public func >=(left: Dimension, right: Dimension) -> Bool {
    return left.value >= right.value
}