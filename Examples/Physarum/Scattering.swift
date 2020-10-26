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

import TensorFlow

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @differentiable(wrt: self)
  public func dimensionScattering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>, shape: Tensor<Index>
  ) -> Tensor {
    return _Raw.scatterNd(indices: indices, updates: self, shape: shape)
  }

  /// Derivative of `_Raw.scatterNd`.
  ///
  /// Ported from TensorFlow Python reference implementation:
  /// https://github.com/tensorflow/tensorflow/blob/r2.2/tensorflow/python/ops/array_grad.py#L1093-L1097
  @inlinable
  @derivative(of: dimensionScattering)
  func _vjpDimensionScattering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>, shape: Tensor<Index>
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = _Raw.scatterNd(indices: indices, updates: self, shape: shape)
    return (
      value,
      { v in
        let dparams = _Raw.gatherNd(params: v, indices: indices)
        return dparams
      }
    )
  }
}
