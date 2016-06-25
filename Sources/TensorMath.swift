import Foundation

infix operator ** {}

public func **(left: Tensor, right: Tensor) -> Tensor {
    return noncommutativeBinaryOperation(left, right, operation: powf)
}

public func **(left: Tensor, right: Tensor.Element) -> Tensor {
    return Tensor(shape: left.shape, elements: left.elements.map { powf($0, right) })
}

public func **(left: Tensor.Element, right: Tensor) -> Tensor {
    return Tensor(shape: right.shape, elements: right.elements.map { powf(left, $0) })
}

extension Tensor {
    public var sin: Tensor {
        return Tensor(shape: shape, elements: elements.map(sinf))
    }

    public var cos: Tensor {
        return Tensor(shape: shape, elements: elements.map(cosf))
    }

    public var tan: Tensor {
        return Tensor(shape: shape, elements: elements.map(tanf))
    }

    public var asin: Tensor {
        return Tensor(shape: shape, elements: elements.map(asinf))
    }
    
    public var acos: Tensor {
        return Tensor(shape: shape, elements: elements.map(acosf))
    }
    
    public var atan: Tensor {
        return Tensor(shape: shape, elements: elements.map(atanf))
    }
    
    public var sinh: Tensor {
        return Tensor(shape: shape, elements: elements.map(sinhf))
    }
    
    public var cosh: Tensor {
        return Tensor(shape: shape, elements: elements.map(coshf))
    }
    
    public var tanh: Tensor {
        return Tensor(shape: shape, elements: elements.map(tanhf))
    }
    
    public var exp: Tensor {
        return Tensor(shape: shape, elements: elements.map(expf))
    }
    
    public var log: Tensor {
        return Tensor(shape: shape, elements: elements.map(logf))
    }
    
    public var sqrt: Tensor {
        return Tensor(shape: shape, elements: elements.map(sqrtf))
    }
    
    public var cbrt: Tensor {
        return Tensor(shape: shape, elements: elements.map(cbrtf))
    }
}

extension Tensor {
    public var sigmoid: Tensor {
        return Tensor(shape: shape, elements: elements.map { 1.0 / expf(-$0) })
    }
}