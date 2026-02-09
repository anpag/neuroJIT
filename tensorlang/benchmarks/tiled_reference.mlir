module {
  func.func @matmul_tiled(%A: tensor<256x256xf32>, %B: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    
    %cst_0 = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<256x256xf32>
    %C_init = linalg.fill ins(%cst_0 : f32) outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>

    // Tile sizes: 64x64x64
    // Loops: i, j, k
    %C_final = scf.for %i = %c0 to %c256 step %c64 iter_args(%arg_C = %C_init) -> (tensor<256x256xf32>) {
      %ret_i = scf.for %j = %c0 to %c256 step %c64 iter_args(%arg_C_i = %arg_C) -> (tensor<256x256xf32>) {
        %ret_j = scf.for %k = %c0 to %c256 step %c64 iter_args(%arg_C_j = %arg_C_i) -> (tensor<256x256xf32>) {
          
          // Extract slices
          %A_sub = tensor.extract_slice %A[%i, %k] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
          %B_sub = tensor.extract_slice %B[%k, %j] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
          %C_sub_init = tensor.extract_slice %arg_C_j[%i, %j] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
          
          // Matmul on tile
          %C_sub_res = linalg.matmul ins(%A_sub, %B_sub : tensor<64x64xf32>, tensor<64x64xf32>) outs(%C_sub_init : tensor<64x64xf32>) -> tensor<64x64xf32>
          
          // Insert slice back
          %C_next = tensor.insert_slice %C_sub_res into %arg_C_j[%i, %j] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<256x256xf32>
          
          scf.yield %C_next : tensor<256x256xf32>
        }
        scf.yield %ret_j : tensor<256x256xf32>
      }
      scf.yield %ret_i : tensor<256x256xf32>
    }
    return %C_final : tensor<256x256xf32>
  }
}
