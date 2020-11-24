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

extension Tensor where Scalar: Numeric {
  func mask(condition: (Tensor) -> Tensor<Bool>) -> Tensor {
    let satisfied = condition(self)
    return Tensor(zerosLike: self)
      .replacing(with: Tensor(onesLike: self), where: satisfied)
  }
}

func angleToVector(_ angle: Tensor<Float>) -> Tensor<Float> {
  // Note: the following is a workaround for a zero derivative shape problem.
  let result = Tensor(stacking: [cos(angle), sin(angle)], alongAxis: -1)
  let shape = result.shape
  return result.withDerivative {
    if $0 == Tensor(0) { $0 = Tensor(zeros: shape) }
  }
}

protocol TurnRule {
  @differentiable
  func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float>
}

struct LearnedTurnRule: Layer, TurnRule {
  var fc1 = Dense<Float>(inputSize: 3, outputSize: 16, activation: relu)
  var fc2 = Dense<Float>(inputSize: 16, outputSize: 16, activation: relu)
  var fc3 = Dense<Float>(inputSize: 16, outputSize: 1, activation: identity, useBias: false)

  @differentiable
  func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return input.sequenced(through: fc1, fc2, fc3).squeezingShape(at: 1)
  }
}

struct HeuristicTurnRule: ParameterlessLayer, TurnRule {
  public typealias TangentVector = EmptyTangentVector

  @differentiable
  func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    return withoutDerivative(at: input) { _ in
      let lowValues = input.argmin(squeezingAxis: -1)
      let highValues = input.argmax(squeezingAxis: -1)
      let middleMask = lowValues.mask { $0 .== 1 }
      let middleDistribution = Tensor<Float>(randomUniform: [particleCount], on: device)
      let randomTurn = middleDistribution.mask { $0 .< 0.1 } * Tensor<Float>(middleMask)
      return Tensor<Float>(highValues - 1) * Tensor<Float>(1 - middleMask) + randomTurn
    }
  }
}

let gridShape = Tensor<Int32>(shape: [2], scalars: [Int32(gridSize), Int32(gridSize)], on: device)
let scatterValues = Tensor<Float>(ones: [particleCount], on: device)

func initializeSimulation() -> (grid: Tensor<Float>, positions: Tensor<Float>, headings: Tensor<Float>) {
  return (grid: Tensor<Float>(zeros: [gridSize, gridSize], on: device),
          positions: Tensor<Float>(randomUniform: [particleCount, 2], on: device) * Float(gridSize),
          headings: Tensor<Float>(randomUniform: [particleCount], on: device) * 2.0 * Float.pi)
}

@differentiable
func simulationStep(
  turnRule: LearnedTurnRule, inputGrid: Tensor<Float>, inputPositions: Tensor<Float>, inputHeadings: Tensor<Float>
) -> (grid: Tensor<Float>, positions: Tensor<Float>, headings: Tensor<Float>) {
  var currentGrid = inputGrid
  var positions = inputPositions
  var headings = inputHeadings
  
  // Perceive
  let senseDirection = headings.expandingShape(at: 1).broadcasted(to: [particleCount, 3])
    + Tensor<Float>([-moveAngle, 0.0, moveAngle], on: device)
  let sensingOffset = angleToVector(senseDirection) * senseDistance
  let sensingPosition = positions.expandingShape(at: 1) + sensingOffset
  // TODO: This wrapping around negative values needs to be fixed.
  let sensingIndices = withoutDerivative(at: turnRule) {_ in
    abs(Tensor<Int32>(sensingPosition))
    % (gridShape.expandingShape(at: 0).expandingShape(at: 0))
  }
  let perceptions = currentGrid.expandingShape(at: 2)
    .dimensionGathering(atIndices: sensingIndices).squeezingShape(at: 2)
  
  // Move
  let turn = turnRule(perceptions)
  headings = headings + (turn * moveAngle)
  positions = positions + angleToVector(headings) * moveStep
  
  // Deposit
  // TODO: This wrapping around negative values needs to be fixed.
  // TODO: XLA errors out with "Invalid argument: Automatic shape inference not supported: s32[1024,2] and s32[1,1,2]"
  // if the manual shape expansion isn't present here.
  let depositIndices = withoutDerivative(at: turnRule) {_ in
    abs(Tensor<Int32>(positions)) % (gridShape.expandingShape(at: 0))
  }
  let deposits = scatterValues.dimensionScattering(atIndices: depositIndices, shape: gridShape)
  currentGrid = currentGrid + deposits
  
  // Diffuse
  currentGrid = currentGrid.expandingShape(at: 0).expandingShape(at: 3)
  currentGrid = currentGrid.padded(forSizes: [(0, 0), (1, 1), (1, 1), (0, 0)], mode: .reflect)
  currentGrid = avgPool2D(currentGrid, filterSize: (1, 3, 3, 1), strides: (1, 1, 1, 1), padding: .valid)
  currentGrid = currentGrid * evaporationRate
  LazyTensorBarrier()

  return (grid: currentGrid.squeezingShape(at: 3).squeezingShape(at: 0), positions: positions, headings: headings)
}

func trainPhysarumSimulation(iterations: Int, stepCount: Int, turnRule: inout LearnedTurnRule) {
  var optimizer = SGD(for: turnRule, learningRate: 2e-3)
  optimizer = SGD(copying: optimizer, to: device)

  for iteration in 0..<iterations {
    let (loss, ruleGradient) = valueWithGradient(at: turnRule) { model -> Tensor<Float> in
      var (grid, positions, headings) = initializeSimulation()
      for _ in 0..<stepCount {
        (grid, positions, headings) = simulationStep(
          turnRule: model, inputGrid: grid, inputPositions: positions, inputHeadings: headings)
      }
      print("Variance: \(grid.variance())")
      return -grid.variance()
    }
    print("Iteration: \(iteration), loss: \(loss)")
    print("Gradient: \(ruleGradient)")
    optimizer.update(&turnRule, along: ruleGradient)
    LazyTensorBarrier()
  }
}

func capturePhysarumSimulation(stepCount: Int, turnRule: LearnedTurnRule, name: String) throws {
  var (grid, positions, headings) = initializeSimulation()

  var steps: [Tensor<Float>] = []
  for _ in 0..<stepCount {
    (grid, positions, headings) = simulationStep(
      turnRule: turnRule, inputGrid: grid, inputPositions: positions, inputHeadings: headings)

    steps.append(grid.expandingShape(at: 2) * 255.0)
  }

  try steps.saveAnimatedImage(directory: "output", name: name, delay: 1)
}


var learningRule = LearnedTurnRule()
learningRule.move(to: device)

//let turnRule = HeuristicTurnRule()

let start = Date()
trainPhysarumSimulation(iterations: 100, stepCount: 200, turnRule: &learningRule)
print("Total calculation time: \(String(format: "%.4f", Date().timeIntervalSince(start))) seconds")

print("Saving animation...")
try! capturePhysarumSimulation(stepCount: 100, turnRule: learningRule, name: "learned_physarum")
print("Animation saved.")

