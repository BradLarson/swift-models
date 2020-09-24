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
    let X = array(from: range.start, to: range.end, size)
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
