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
import ArgumentParser
import SwiftRT

struct FractalCommand: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "Fractals",
        abstract: """
      Computes fractals of a variety of types and writes an image from the results.
      """,
        subcommands: [
            JuliaSubcommand.self,
            MandelbrotSubcommand.self,
        ])
}

func measureTime(_ body: () -> Void) {
    let start = Date()
    body()
    print("elapsed \(String(format: "%.3f", Date().timeIntervalSince(start))) seconds")
}

extension FractalCommand {
    struct Parameters: ParsableArguments {
        @Flag(help: "Use CPU")
        var cpu: Bool = false
        
        @Flag(help: "Use GPU")
        var gpu: Bool = false
        
        @Flag(help: "Use pmap to distribute the work across devices")
        var pmap: Bool = false

        @Option(help: "Number of iterations to run.")
        var iterations: Int?
        
        @Option(help: "The region of complex numbers to operate over.")
        var region: ComplexRange?
        
        @Option(help: "Tolerance threshold to mark divergence.")
        var tolerance: Float?
        
        @Option(help: "Output image file.")
        var outputFile: String?
        
        @Option(help: "Output image rows, cols")
        var ImageSize: ImageSize?
        
        func validate() throws {
            guard !(cpu && gpu) else {
                throw ValidationError(
                    "Can't specify both --cpu and --gpu backends.")
            }
        }
    }
}

extension FractalCommand {
    struct JuliaSubcommand: ParsableCommand {
        static var configuration = CommandConfiguration(
            commandName: "JuliaSet",
            abstract: "Calculate and save an image of the Julia set.")
        
        @OptionGroup()
        var parameters: FractalCommand.Parameters
        
        @Option(help: "Complex constant.")
        var constant: Complex<Float>?
        
        func run() throws {
            // select the device
//            Context.log.level = .diagnostic
            if parameters.gpu { use(device: 1) }            

//            if parameters.pmap {
//                Context.cpuQueueCount = ProcessInfo().activeProcessorCount
//            }
            
            let region = parameters.region ??
                ComplexRange(Complex<Float>(-1.7, -1.7), Complex<Float>(1.7, 1.7))
            let iterations = parameters.iterations ?? 200
            let size = parameters.ImageSize ?? ImageSize(rows: 1030, cols: 1030)
            
            //print("ImageSize(r: \(size[0]), c: \(size[1])) iterations: \(iterations) queue: \(Context.currentQueue.name)")

            let divergenceGrid: Tensor2
            if parameters.pmap {
                divergenceGrid = pmapJuliaSet(
                    iterations: iterations,
                    constant: constant ?? Complex<Float>(-0.8, 0.156),
                    tolerance: parameters.tolerance ?? 4.0,
                    range: region.imaginaryReversed,
                    size: size)
            } else {
                divergenceGrid = juliaSet(
                    iterations: iterations,
                    constant: constant ?? Complex<Float>(-0.8, 0.156),
                    tolerance: parameters.tolerance ?? 4.0,
                    range: region.imaginaryReversed,
                    size: size)
            }
            
            do {
                try saveFractalImage(
                    divergenceGrid,
                    iterations: parameters.iterations ?? 200,
                    fileName: parameters.outputFile ?? "julia")
            } catch {
                print("Error saving fractal image: \(error)")
            }
        }
    }
}

extension FractalCommand {
    struct MandelbrotSubcommand: ParsableCommand {
        static var configuration = CommandConfiguration(
            commandName: "MandelbrotSet",
            abstract: "Calculate and save an image of the Mandelbrot set.")
        
        @OptionGroup()
        var parameters: FractalCommand.Parameters
        
        func run() throws {
            if parameters.gpu { use(device: 1) }
            
//            if parameters.pmap {
//                Context.cpuQueueCount = ProcessInfo().activeProcessorCount
//            }

            let region = parameters.region ??
                ComplexRange(Complex<Float>(-2.0, -1.3), Complex<Float>(1.0, 1.3))

            let divergenceGrid = mandelbrotSet(
                iterations: parameters.iterations ?? 200,
                tolerance: parameters.tolerance ?? 2.0,
                range: region.imaginaryReversed,
                size: parameters.ImageSize ?? ImageSize(rows: 1024, cols: 1024))
            
            do {
                try saveFractalImage(
                    divergenceGrid,
                    iterations: parameters.iterations ?? 200,
                    fileName: parameters.outputFile ?? "mandelbrot")
            } catch {
                print("Error saving fractal image: \(error)")
            }
        }
    }
}

FractalCommand.main()
