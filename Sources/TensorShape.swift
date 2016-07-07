// Reference: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/tensor_shape.py

public struct TensorShape {
    public let dimensions: [Dimension]
    public var volume: Int {
        return dimensions.reduce(1) { $0 * $1.value }
    }
    public init(_ dimensions: [Dimension]) {
        self.dimensions = dimensions
    }
}

extension TensorShape: ArrayLiteralConvertible {
    public init(arrayLiteral elements: Dimension...) {
        self.init(elements)
    }
}

extension TensorShape: CustomStringConvertible {
    public var description: String {
        return dimensions.description
    }
}

extension TensorShape {
    public var ndims: Int {
        return dimensions.count
    }
}
extension TensorShape {
    public var num_elements: Int {
        return dimensions.reduce(0) {$0 + $1.value}
    }
}

extension TensorShape { 
    public func merge_with(other: TensorShape) -> TensorShape {
        if (self.dimensions.count == 0) {
            return other
        } else {
            self.assert_same_rank(other: other)
            var new_dims = [Dimension] ();
            for (index, dim) in self.dimensions.enumerated() {
                new_dims.append(dim.merge_with(other: other.dimensions[index]))
            }
            return TensorShape(new_dims)
        }
    }
}

extension TensorShape { 
    public func concatenate(other: TensorShape) -> TensorShape {
        return TensorShape(self.dimensions + other.dimensions)
    }
}

extension TensorShape {
    public func assert_same_rank(other: TensorShape) {
        if self.ndims != other.ndims {
            fatalError ("Shapes \(self) and \(other) must have the same rank")
        }
    }
}

extension TensorShape {
    public func assert_has_rank(rank: Int) {
        if self.ndims != rank {
            fatalError ("Shape \(self) must have rank \(rank)")
        }
    }
}

extension TensorShape {
    public func with_rank(rank: Int) -> TensorShape {
        return self.merge_with(other: TensorShape([Dimension](repeating: Dimension(0), count: rank)))
    }
}

extension TensorShape {
    public func with_rank_at_least(rank: Int) -> TensorShape {
        if self.ndims < rank {
            fatalError ("Shape \(self) must have rank at least \(rank)")
        } else{
            return self
        }
    }
}

extension TensorShape {
    public func with_rank_at_most(rank: Int) -> TensorShape{
        if self.ndims > rank {
            fatalError ("Shape \(self) must have rank at most \(rank)")
        } else{
            return self
        }
    }
}

extension TensorShape {
    public func is_compatible_with(other: TensorShape) -> Bool {
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

extension TensorShape {
    public func assert_is_compatible_with(other: TensorShape) {
        if !self.is_compatible_with(other: other) {
            fatalError("Shapes \(self) and \(other) are incompatible")
        }
    }
}

extension TensorShape {
    public func is_fully_defined() -> Bool{
        var is_fully_defined = self.dimensions.count > 0;
        for dim in self.dimensions {
            is_fully_defined = is_fully_defined && dim.value != 0
        }
        return is_fully_defined
    }
}

extension TensorShape {
    public func assert_is_fully_defined() {
        if !self.is_fully_defined() {
            fatalError("Shapes \(self) is not fully defined")
        }
    }
}


extension TensorShape: Equatable {}
public func ==(left: TensorShape, right: TensorShape) -> Bool {
    return left.dimensions == right.dimensions
}
