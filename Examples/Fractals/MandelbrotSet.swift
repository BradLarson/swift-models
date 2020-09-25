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

    for i in 1..<iterations {
        divergence[abs(Z) .> tolerance] = min(divergence, i)
        Z = multiply(Z, Z, add: X)
    }
    
    print("elapsed \(String(format: "%.3f", Date().timeIntervalSince(start))) seconds")
    return divergence
}
