//******************************************************************************
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation
import SwiftRT

func mandelbrotSet(
    iterations: Int,
    tolerance: Float,
    range: ComplexRange,
    size: ImageSize
) -> Tensor2 {
    let size2 = (r: size[0], c: size[1])

    let rFirst = Complex<Float>(range.start.real, 0), rLast = Complex<Float>(range.end.real, 0)
    let iFirst = Complex<Float>(0, range.start.imaginary), iLast = Complex<Float>(0, range.end.imaginary)

    // repeat rows of real range, columns of imaginary range, and combine
    let Xr = repeating(array(from: rFirst, to: rLast, (1, size2.c)), size2)
    let Xi = repeating(array(from: iFirst, to: iLast, (size2.r, 1)), size2)
    let X = Xr + Xi

    var divergence = full(size, iterations)
    var Z = X

    print("rows: \(size[0]), cols: \(size[1]), iterations: \(iterations)")
    let start = Date()
    // pmap(Z, X, &divergence) { Z, X, divergence in
    //     for i in 1..<iterations {
    //         divergence[abs(Z) .> tolerance] = min(divergence, i)
    //         Z = multiply(Z, Z, add: X)
    //     }
    // }
    
//    pmap(Z, X, &divergence, boundBy: .compute) {
//        mandelbrotKernel(Z: $0, X: $1, divergence: &$2, tolerance, iterations)
//    }

    print("elapsed \(String(format: "%.3f", Date().timeIntervalSince(start))) seconds")
    return divergence
}

@inlinable public func mandelbrotKernel<E>(
    Z: TensorR2<Complex<E>>,
    X: TensorR2<Complex<E>>,
    divergence: inout TensorR2<E>,
    _ tolerance: E,
    _ iterations: Int
) {
    let message =
        "mandelbrot(Z: \(Z.name), divergence: \(divergence.name), X: \(X.name), " +
        "tolerance: \(tolerance), iterations: \(iterations))"

    kernel(Z, &divergence, message) {
        var Z = $0, d = $1
        for i in 0..<iterations {
//            Z = Z * Z + X
            Z = Z * Z + Complex(1.0, 0.0)
            if abs(Z) > tolerance { d = min(d, E.Value(exactly: i)!) }
        }
        return d
    }
}
