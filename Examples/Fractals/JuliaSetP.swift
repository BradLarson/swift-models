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

func pmapJuliaSet(
    iterations: Int,
    constant C: Complex<Float>,
    tolerance: Float,
    range: ComplexRange,
    size: ImageSize
) -> Tensor2 {
    var Z = array(from: range.start, to: range.end, size)
    var divergence = full(size, iterations)
    
    measureTime {
//        pmap(&Z, &divergence) { Z, divergence in
//            print("\(Context.currentQueue.name)", Z.storageBase, divergence.storageBase)
//            for i in 0..<1 {
//                Z = multiply(Z, Z, add: C)
//                divergence[abs(Z) .> tolerance] = min(divergence, i)
//            }
//        }
    }
    return divergence
}
