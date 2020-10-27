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

import Foundation
import ModelSupport
import TensorFlow

let stepCount = 100
let gridSize = 512
let particleCount = 1024
let senseAngle = 0.20 * Float.pi
let senseDistance: Float = 4.0
let evaporationRate: Float = 0.95
let moveAngle = 0.1 * Float.pi
let moveStep: Float = 2.0
let channelSize = 1

let device = Device.defaultTFEager
// let device = Device.defaultXLA

var grid = Tensor<Float>(zeros: [2, gridSize, gridSize], on: device)
var positions = Tensor<Float>(randomUniform: [particleCount, 2], on: device) * Float(gridSize)
var headings = Tensor<Float>(randomUniform: [particleCount], on: device) * 2.0 * Float.pi
let gridShape = Tensor<Int32>(shape: [2], scalars: [Int32(gridSize), Int32(gridSize)])
let scatterValues = Tensor<Float>(ones: [particleCount], on: device)

extension Tensor where Scalar: Numeric {
  func mask(condition: (Tensor) -> Tensor<Bool>) -> Tensor {
    let satisfied = condition(self)
    return Tensor(zerosLike: self)
      .replacing(with: Tensor(onesLike: self), where: satisfied)
  }
}

func angleToVector(_ angle: Tensor<Float>) -> Tensor<Float> {
  return Tensor(stacking: [cos(angle), sin(angle)], alongAxis: -1)
}

func step(phase: Int) {
  var currentGrid = grid[phase]
  // Perceive
  let senseDirection = headings.expandingShape(at: 1).broadcasted(to: [particleCount, 3]) + Tensor<Float>([-moveAngle, 0.0, moveAngle], on: device)
  let sensingOffset = angleToVector(senseDirection) * senseDistance
  let sensingPosition = positions.expandingShape(at: 1) + sensingOffset
  // TODO: This wrapping around negative values needs to be fixed.
  let sensingIndices = abs(Tensor<Int32>(sensingPosition)) % gridShape
  let sensedValues = currentGrid.expandingShape(at: 2).dimensionGathering(atIndices: sensingIndices).squeezingShape(at: 2)
  
  // Move
  let lowValues = sensedValues.argmin(squeezingAxis: -1)
  let highValues = sensedValues.argmax(squeezingAxis: -1)
  let middleMask = lowValues.mask { $0 .== 1 }
  let middleDistribution = Tensor<Float>(randomUniform: [particleCount], on: device)
  let randomTurn = middleDistribution.mask { $0 .< 0.1 } * Tensor<Float>(middleMask)
  let turn = Tensor<Float>(highValues - 1) * Tensor<Float>(1 - middleMask) + randomTurn
  headings += (turn * moveAngle)
  positions += angleToVector(headings) * moveStep
  
  // Deposit
  // TODO: This wrapping around negative values needs to be fixed.
  let depositIndices = abs(Tensor<Int32>(positions)) % gridShape
  let deposits = scatterValues.dimensionScattering(atIndices: depositIndices, shape: gridShape)
  currentGrid += deposits
  
  // Diffuse
  currentGrid = currentGrid.expandingShape(at: 0).expandingShape(at: 3)
  currentGrid = currentGrid.padded(forSizes: [(0, 0), (1, 1), (1, 1), (0, 0)], mode: .reflect)
  currentGrid = avgPool2D(currentGrid, filterSize: (1, 3, 3, 1), strides: (1, 1, 1, 1), padding: .valid)
  currentGrid = currentGrid * evaporationRate
  grid[1 - phase] = currentGrid.squeezingShape(at: 3).squeezingShape(at: 0)
}

let start = Date()

var steps: [Tensor<Float>] = []
for stepIndex in 0..<stepCount {
  print("Step: \(stepIndex)")
  step(phase: stepIndex % 2)
  LazyTensorBarrier()
  steps.append(grid[0].expandingShape(at: 2) * 255.0)
}

print("Total calculation time: \(String(format: "%.4f", Date().timeIntervalSince(start))) seconds")


try steps.saveAnimatedImage(directory: "output", name: "physarum", delay: 1)
