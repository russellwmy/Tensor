// Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/tensor_shape.py

public struct Shape {
    public let dimensions: [Dimension]
    public var volume: Int {
        return dimensions.reduce(1) { $0 * $1.value }
    }
    public init(_ dimensions: [Dimension]) {
        self.dimensions = dimensions
    }
}

extension Shape: ArrayLiteralConvertible {
    public init(arrayLiteral elements: Dimension...) {
        self.init(elements)
    }
}

extension Shape: CustomStringConvertible {
    public var description: String {
        return dimensions.description
    }
}

extension Shape {
    public var ndims: Int {
        return dimensions.count
    }
}
extension Shape {
    public var num_elements: Int {
        return dimensions.reduce(0) {$0 + $1.value}
    }
}

extension Shape { 
    public func merge_with(other: Shape) -> Shape {
        if (self.dimensions.count == 0) {
            return other
        } else {
            self.assert_same_rank(other: other)
            var new_dims = [Dimension] ();
            for (index, dim) in self.dimensions.enumerated() {
                new_dims.append(dim.merge_with(other: other.dimensions[index]))
            }
            return Shape(new_dims)
        }
    }
}

extension Shape { 
    public func concatenate(other: Shape) -> Shape {
        return Shape(self.dimensions + other.dimensions)
    }
}

extension Shape {
    public func assert_same_rank(other: Shape) {
        if self.ndims != other.ndims {
            fatalError ("Shapes \(self) and \(other) must have the same rank")
        }
    }
}

extension Shape {
    public func assert_has_rank(rank: Int) {
        if self.ndims != rank {
            fatalError ("Shape \(self) must have rank \(rank)")
        }
    }
}

extension Shape {
    public func with_rank(rank: Int) -> Shape {
        return self.merge_with(other: Shape([Dimension](repeating: Dimension(0), count: rank)))
    }
}

extension Shape {
    public func with_rank_at_least(rank: Int) -> Shape {
        if self.ndims < rank {
            fatalError ("Shape \(self) must have rank at least \(rank)")
        } else{
            return self
        }
    }
}

extension Shape {
    public func with_rank_at_most(rank: Int) -> Shape{
        if self.ndims > rank {
            fatalError ("Shape \(self) must have rank at most \(rank)")
        } else{
            return self
        }
    }
}

extension Shape {
    public func is_compatible_with(other: Shape) -> Bool {
        if self.ndims > 0 && other.ndims > 0 {
            if self.ndims != other.ndims {
                return false
            }
            for (x_dim, y_dim) in zip(self.dimensions, other.dimensions) {
                if !x_dim.is_compatible_with(other: y_dim) {
                    return false
                }
            }
        }
        return true
    }
}

extension Shape {
    public func assert_is_compatible_with(other: Shape) {
        if !self.is_compatible_with(other: other) {
            fatalError("Shapes \(self) and \(other) are incompatible")
        }
    }
}

extension Shape {
    public func is_fully_defined() -> Bool{
        var is_fully_defined = self.dimensions.count > 0;
        for dim in self.dimensions {
            is_fully_defined = is_fully_defined && dim.value != 0
        }
        return is_fully_defined
    }
}

extension Shape {
    public func assert_is_fully_defined() {
        if !self.is_fully_defined() {
            fatalError("Shapes \(self) is not fully defined")
        }
    }
}


extension Shape: Equatable {}
public func ==(left: Shape, right: Shape) -> Bool {
    return left.dimensions == right.dimensions
}
