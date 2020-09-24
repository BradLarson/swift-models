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
@_implementationOnly import STBImage
import SwiftRT

// Image loading and saving is inspired by t-ae's Swim library: https://github.com/t-ae/swim
// and uses the stb_image single-file C headers from https://github.com/nothings/stb .

public struct Image {
    public enum ByteOrdering {
        case bgr
        case rgb
    }

    public enum Colorspace {
        case rgb
        case rgba
        case grayscale
    }
  
    public enum Format {
        case jpeg(quality: Float)
        case png
    }
  
    let imageData: TensorR3<Float>

    public init(_ tensor: TensorR3<Float>) {
        self.imageData = tensor
    }

    public init(contentsOf url: URL, byteOrdering: ByteOrdering = .rgb) {
        if byteOrdering == .bgr {
            // TODO: Add BGR byte reordering.
            fatalError("BGR byte ordering is currently unsupported.")
        } else {
            guard FileManager.default.fileExists(atPath: url.path) else {
                // TODO: Proper error propagation for this.
                fatalError("File does not exist at: \(url.path).")
            }
            
            var width: Int32 = 0
            var height: Int32 = 0
            var bpp: Int32 = 0
            guard let bytes = stbi_load(url.path, &width, &height, &bpp, 0) else {
                // TODO: Proper error propagation for this.
                fatalError("Unable to read image at: \(url.path).")
            }

            let data = [UInt8](UnsafeBufferPointer(start: bytes, count: Int(width * height * bpp)))
            stbi_image_free(bytes)
            // TODO: Storage internally as UInt8
            let loadedTensor = array(data.map { Float($0) }, (Int(width), Int(height), Int(bpp)))
            
            // TODO: Grayscale conversion to RGB channels
//            if bpp == 1 {
//                loadedTensor = loadedTensor.broadcasted(to: [Int(height), Int(width), 3])
//            }
            self.imageData = loadedTensor
        }
    }

    public func save(to url: URL, colorspace: Colorspace = .rgb, format: Format = .jpeg(quality: 95)) {
        let bpp: Int32

        switch colorspace {
        case .grayscale:
            bpp = 1
            // Broadcast across channels.
        case .rgb:
            bpp = 3
        case .rgba:
            bpp = 4
        }

        let width = Int32(imageData.shape[0])
        let height = Int32(imageData.shape[1])
        
        let outputImageData: [UInt8] = imageData.flatArray.map { UInt8(max(0.0, min($0, 255.0))) }
        outputImageData.withUnsafeBufferPointer { bytes in
            switch format {
            case let .jpeg(quality):
                let status = stbi_write_jpg(
                    url.path, width, height, bpp, bytes.baseAddress!, Int32(round(quality)))
                guard status != 0 else {
                    // TODO: Proper error propagation for this.
                    fatalError("Unable to save image to: \(url.path).")
                }
            case .png:
                let status = stbi_write_png(
                    url.path, width, height, bpp, bytes.baseAddress!, 0)
                guard status != 0 else {
                    // TODO: Proper error propagation for this.
                    fatalError("Unable to save image to: \(url.path).")
                }
            }
        }
    }

    /*
    public func resized(to size: (Int, Int)) -> Image {
        switch self.imageData {
        case let .uint8(data):
            let resizedImage = resize(images: Tensor<Float>(data), size: size, method: .bilinear)
            return Image(Tensor<UInt8>(resizedImage))
        case let .float(data):
            let resizedImage = resize(images: data, size: size, method: .bilinear)
            return Image(resizedImage)
        }
    }
  
    func premultiply(_ input: Tensor<Float>) -> Tensor<Float> {
        let alphaChannel = input.slice(
            lowerBounds: [0, 0, 3], sizes: [input.shape[0], input.shape[1], 1])
        let colorComponents = input.slice(
            lowerBounds: [0, 0, 0], sizes: [input.shape[0], input.shape[1], 3])
        let adjustedColorComponents = colorComponents * alphaChannel / 255.0
        return Tensor(concatenating: [adjustedColorComponents, alphaChannel], alongAxis: 2)
    }
    
    public func premultipliedAlpha() -> Image {
        switch self.imageData {
        case let .uint8(data):
            guard data.shape[2] == 4  else { return self }
            return Image(premultiply(Tensor<Float>(data)))
        case let .float(data):
            guard data.shape[2] == 4  else { return self }
            return Image(premultiply(data))
        }
    }
 */
}

public func saveImage(
    _ tensor: TensorR3<Float>, shape: (Int, Int)? = nil, size: (Int, Int)? = nil,
    colorspace: Image.Colorspace = .rgb, directory: String, name: String,
    format: Image.Format = .jpeg(quality: 95)
) throws {
    try createDirectoryIfMissing(at: directory)

    // TODO: Handle resizing, layer RGB on white background, etc.
    let image = Image(tensor)
    
    let fileExtension: String
    switch format {
    case .jpeg: fileExtension = "jpg"
    case .png: fileExtension = "png"
    }
    let outputURL = URL(fileURLWithPath: "\(directory)/\(name).\(fileExtension)")
    image.save(to: outputURL, colorspace: colorspace, format: format)
}
