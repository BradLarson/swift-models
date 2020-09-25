// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import ArgumentParser
import ModelSupport
import SwiftRT

public typealias ImageSize = Shape2

extension ImageSize: ExpressibleByArgument {
    public init(rows: Int, cols: Int) {
        self.init(rows, cols)
    }
    
    public init?(argument: String) {
        let subArguments = argument.split(separator: ",").compactMap { Int(String($0)) }
        guard subArguments.count >= 2 else { return nil }
        self.init(rows: subArguments[0], cols: subArguments[1])
    }
    
    public var defaultValueDescription: String {
        "\(self[0]) \(self[1])"
    }
}

fileprivate func prismColor(_ value: Float, iterations: Int) -> [Float] {
    guard value < Float(iterations) else { return [0.0, 0.0, 0.0, 255.0] }
    
    let normalizedValue = value / Float(iterations)
    
    // Values drawn from Matplotlib: https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_cm.py
    let red = (0.75 * .sin((normalizedValue * 20.9 + 0.25) * Float.pi) + 0.67) * 255
    let green = (0.75 * .sin((normalizedValue * 20.9 - 0.25) * Float.pi) + 0.33) * 255
    let blue = (-1.1 * .sin((normalizedValue * 20.9) * Float.pi)) * 255
    let alpha: Float = 255.0
    return [red, green, blue, alpha]
}

func saveFractalImage(_ divergenceGrid: Tensor2, iterations: Int, fileName: String) throws {
    let colorValues: [Float] = divergenceGrid.flatArray.reduce(into: []) {
        $0 += prismColor($1, iterations: iterations)
    }
    let gridShape = divergenceGrid.shape
    let colorImage = array(colorValues, (gridShape[0], gridShape[1], 4))
    
    try saveImage(
        colorImage, shape: (gridShape[0], gridShape[1]),
        colorspace: .rgba, directory: "./", name: fileName,
        format: .png)
}
