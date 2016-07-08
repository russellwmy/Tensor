import XCTest
@testable import Tensor

class TensorTests: XCTestCase {
	func testInitTensor () {
		let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
		XCTAssertTrue( t[1, 1] == 5 )
		let zeros = Tensor(shape: [2, 3])
		XCTAssertTrue( zeros[1, 1] == 0 )
		let ones = Tensor(shape: [2, 3], element: 1)
		XCTAssertTrue( ones[1, 1] == 1 )
	}

	func testDimension() {
		let d1 = Dimension(10)
		let d2 = Dimension(12)
		XCTAssertTrue( d1 < d2 )

		let s1 = TensorShape( [d1])
		let s2 = TensorShape( [d2])
		XCTAssertTrue( s1 != s2 )
		
		let d3 = Dimension(12)
		let s3 = TensorShape( [d3])
		XCTAssertTrue( s2 == s3 )

	}
}