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
import ArgumentParser
import Numerics

struct ComplexRange {
    let start: Complex<Float>
    let end: Complex<Float>

    init(_ start: Complex<Float>, _ end: Complex<Float>) {
        self.start = start
        self.end = end
    }

    var imaginaryReversed: ComplexRange {
        ComplexRange(Complex(start.real, end.imaginary),
                      Complex(end.real, start.imaginary))
    }
}

extension ComplexRange: ExpressibleByArgument {
    init?(argument: String) {
        let subArguments = argument.split(separator: ",")
            .compactMap { Float(String($0)) }
        guard subArguments.count >= 4 else { return nil }
        
        start = Complex(subArguments[0], subArguments[2])
        end = Complex(subArguments[1], subArguments[3])
    }
    
    var defaultValueDescription: String {
        "\(self.start.real),\(self.end.real)," +
            "\(self.start.imaginary),\(self.end.imaginary)"
    }
}

extension Complex: ExpressibleByArgument where RealType: BinaryFloatingPoint {
    public init?(argument: String) {
        let subArguments: [RealType] = argument.split(separator: ",")
            .compactMap {
                if let value = Float(String($0)) {
                    return RealType(value)
                } else {
                    return nil
                }
            }
        guard subArguments.count >= 2 else { return nil }
        self = Complex(subArguments[0], subArguments[1])
    }
    
    public var defaultValueDescription: String {
        "\(real),\(imaginary)"
    }
}

