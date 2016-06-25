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

extension Dimension: Equatable {}
public func ==(left: Dimension, right: Dimension) -> Bool {
    return left.value == right.value
}

extension Dimension: CustomStringConvertible {
    public var description: String {
        return value.description
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