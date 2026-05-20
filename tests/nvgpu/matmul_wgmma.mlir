// CHECK: NvGPU analysis for @gemm_warp_specialized_kernel
// CHECK: synthesized cost-estimate function:
// CHECK: func.func private @gemm_warp_specialized_kernel_nvgpu_cost(
// CHECK-SAME: %warpgroup_mma_cost: f64
// CHECK-SAME: %warpgroup_mma_store_cost: f64
// CHECK-SAME: %wgmma_wait_cost: f64
// CHECK-SAME: %setmaxregister_cost: f64
// CHECK-SAME: %barrier_cost: f64
// CHECK-SAME: %memref_load_cost: f64
// CHECK-SAME: %memref_store_cost: f64
// CHECK-SAME: %mbarrier_init_cost: f64
// CHECK-SAME: %mbarrier_arrive_cost: f64
// CHECK-SAME: %mbarrier_arrive_expect_tx_cost: f64
// CHECK-SAME: %mbarrier_try_wait_parity_cost: f64
// CHECK-SAME: %tma_prefetch_descriptor_cost: f64
// CHECK-SAME: %warpgroup_generate_descriptor_cost: f64
// CHECK: %[[INIT:.*]] = arith.addf %mbarrier_init_cost, %mbarrier_init_cost : f64
// CHECK: %[[INIT_LOOP:.*]] = arith.mulf %[[INIT]], %cst_1 : f64
// CHECK: %[[PREFETCH0:.*]] = arith.addf %[[INIT_LOOP]], %tma_prefetch_descriptor_cost : f64
// CHECK: %[[PREFETCH1:.*]] = arith.addf %[[PREFETCH0]], %tma_prefetch_descriptor_cost : f64
// CHECK: %[[WAIT_EXPECT:.*]] = arith.addf %mbarrier_try_wait_parity_cost, %mbarrier_arrive_expect_tx_cost : f64
// CHECK: %[[RES_L1:.*]] = arith.mulf %residency_p1, %residency_l1_cost : f64
// CHECK: %[[RES_GLOBAL:.*]] = arith.mulf %{{.*}}, %residency_global_cost : f64
// CHECK: %[[RES_SUM:.*]] = arith.addf %{{.*}}, %[[RES_GLOBAL]] : f64
// CHECK: %[[WITH_RES:.*]] = arith.addf %{{.*}}, %[[RES_SUM]] : f64
// CHECK: %[[SETMAX:.*]] = arith.addf %setmaxregister_cost, %{{.*}} : f64
// CHECK: %[[DESC0:.*]] = arith.addf %{{.*}}, %warpgroup_generate_descriptor_cost : f64
// CHECK: %[[DESC1:.*]] = arith.addf %[[DESC0]], %warpgroup_generate_descriptor_cost : f64
// CHECK: %[[MMA:.*]] = arith.addf %[[DESC1]], %warpgroup_mma_cost : f64
// CHECK: %[[ARRIVE:.*]] = arith.addf %[[MMA]], %mbarrier_arrive_cost : f64
// CHECK: %[[WG_WAIT:.*]] = arith.addf %{{.*}}, %wgmma_wait_cost : f64
// CHECK: %[[WG_STORE:.*]] = arith.addf %{{.*}}, %warpgroup_mma_store_cost : f64
// CHECK: %[[BARRIER:.*]] = arith.addf %[[WG_STORE]], %barrier_cost : f64
// CHECK: %[[STORE:.*]] = arith.addf %memref_load_cost, %memref_store_cost : f64
// CHECK: return %{{.*}} : f64

module attributes {gpu.container_module} {
  func.func @gemm_warp_specialized(%arg0: memref<512x1024xf16>, %arg1: memref<1024x256xf16>, %arg2: memref<512x256xf32>) attributes {llvm.emit_c_interface} {
    %0 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%0] () : memref<512x1024xf16>
    %memref_0, %asyncToken_1 = gpu.alloc async [%asyncToken] () : memref<1024x256xf16>
    %memref_2, %asyncToken_3 = gpu.alloc async [%asyncToken_1] () : memref<512x256xf32>
    %1 = gpu.memcpy async [%asyncToken_3] %memref, %arg0 : memref<512x1024xf16>, memref<512x1024xf16>
    %2 = gpu.memcpy async [%1] %memref_0, %arg1 : memref<1024x256xf16>, memref<1024x256xf16>
    %3 = gpu.wait async [%2]
    %cast = memref.cast %memref : memref<512x1024xf16> to memref<*xf16>
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %4 = nvgpu.tma.create.descriptor %cast box[%c128, %c64] : memref<*xf16> -> <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    %cast_4 = memref.cast %memref_0 : memref<1024x256xf16> to memref<*xf16>
    %c64_5 = arith.constant 64 : index
    %c64_6 = arith.constant 64 : index
    %5 = nvgpu.tma.create.descriptor %cast_4 box[%c64_5, %c64_6] : memref<*xf16> -> <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
    %c229376_i32 = arith.constant 229376 : i32
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c1_7 = arith.constant 1 : index
    %c1_8 = arith.constant 1 : index
    gpu.launch_func  @gemm_warp_specialized_kernel::@gemm_warp_specialized_kernel blocks in (%c4, %c2, %c1) threads in (%c256, %c1_7, %c1_8)  dynamic_shared_memory_size %c229376_i32 args(%4 : !nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, %5 : !nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, %memref_2 : memref<512x256xf32>)
    %6 = gpu.memcpy async [%3] %arg2, %memref_2 : memref<512x256xf32>, memref<512x256xf32>
    %7 = gpu.wait async [%6]
    return
  }
  gpu.module @gemm_warp_specialized_kernel {
    gpu.func @gemm_warp_specialized_kernel(%arg0: !nvgpu.tensormap.descriptor<tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, %arg1: !nvgpu.tensormap.descriptor<tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, %arg2: memref<512x256xf32>) kernel attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 4, 2, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %thread_id_x_0 = gpu.thread_id  x
      %c128 = arith.constant 128 : index
      %0 = arith.remui %thread_id_x_0, %c128 : index
      %c0 = arith.constant 0 : index
      %1 = arith.cmpi eq, %0, %c0 : index
      %c128_1 = arith.constant 128 : index
      %2 = arith.divui %thread_id_x_0, %c128_1 : index
      %c1 = arith.constant 1 : index
      %3 = arith.cmpi eq, %2, %c1 : index
      %thread_id_x_2 = gpu.thread_id  x
      %c128_3 = arith.constant 128 : index
      %4 = arith.remui %thread_id_x_2, %c128_3 : index
      %c0_4 = arith.constant 0 : index
      %5 = arith.cmpi eq, %4, %c0_4 : index
      %c128_5 = arith.constant 128 : index
      %6 = arith.divui %thread_id_x_2, %c128_5 : index
      %c0_6 = arith.constant 0 : index
      %7 = arith.cmpi eq, %6, %c0_6 : index
      %thread_id_x_7 = gpu.thread_id  x
      %8 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
      %9 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
      %c0_8 = arith.constant 0 : index
      %10 = arith.cmpi eq, %thread_id_x_7, %c0_8 : index
      scf.if %10 {
        %c0_9 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg3 = %c0_9 to %c7 step %c1_10 {
          %c1_11 = arith.constant 1 : index
          nvgpu.mbarrier.init %8[%arg3], %c1_11 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
          %c1_12 = arith.constant 1 : index
          nvgpu.mbarrier.init %9[%arg3], %c1_12 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
        }
        nvgpu.tma.prefetch.descriptor %arg0 : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
        nvgpu.tma.prefetch.descriptor %arg1 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
      }
      scf.if %3 {
        nvvm.setmaxregister  decrease 40
        %true = arith.constant true
        %c0_9 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        %11 = scf.for %arg3 = %c0_9 to %c16 step %c1_10 iter_args(%arg4 = %true) -> (i1) {
          %c7 = arith.constant 7 : index
          %12 = arith.remui %arg3, %c7 : index
          %c10000000 = arith.constant 10000000 : index
          nvgpu.mbarrier.try_wait.parity %8[%12], %arg4, %c10000000 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
          %c6 = arith.constant 6 : index
          %13 = arith.cmpi eq, %12, %c6 : index
          %true_11 = arith.constant true
          %14 = arith.xori %arg4, %true_11 : i1
          %15 = arith.select %13, %14, %arg4 : i1
          %block_id_x_12 = gpu.block_id  x
          %block_id_y_13 = gpu.block_id  y
          %c128_14 = arith.constant 128 : index
          %16 = arith.muli %block_id_x_12, %c128_14 : index
          %c128_15 = arith.constant 128 : index
          %17 = arith.muli %block_id_y_13, %c128_15 : index
          %thread_id_x_16 = gpu.thread_id  x
          %c16384 = arith.constant 16384 : index
          %18 = arith.muli %12, %c16384 : index
          %c16384_17 = arith.constant 16384 : index
          %19 = arith.muli %12, %c16384_17 : index
          %c114688 = arith.constant 114688 : index
          %20 = arith.addi %19, %c114688 : index
          %c8192 = arith.constant 8192 : index
          %21 = arith.addi %20, %c8192 : index
          %22 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view = memref.view %22[%18][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
          %23 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view_18 = memref.view %23[%20][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
          %24 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view_19 = memref.view %24[%21][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
          %c32768 = arith.constant 32768 : index
          nvgpu.mbarrier.arrive.expect_tx %9[%12], %c32768, predicate = %1 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
          %c128_20 = arith.constant 128 : index
          %25 = arith.remui %thread_id_x_16, %c128_20 : index
          %c0_21 = arith.constant 0 : index
          %26 = arith.cmpi eq, %25, %c0_21 : index
          %c64 = arith.constant 64 : index
          %27 = arith.muli %arg3, %c64 : index
          nvgpu.tma.async.load %arg0[%27, %16], %9[%12] to %view, predicate = %26 : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<128x64xf16, #gpu.address_space<workgroup>>
          nvgpu.tma.async.load %arg1[%17, %27], %9[%12] to %view_18, predicate = %26 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
          %c64_22 = arith.constant 64 : index
          %28 = arith.addi %17, %c64_22 : index
          nvgpu.tma.async.load %arg1[%28, %27], %9[%12] to %view_19, predicate = %26 : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> memref<64x64xf16, #gpu.address_space<workgroup>>
          scf.yield %15 : i1
        }
      }
      scf.if %7 {
        nvvm.setmaxregister  increase 232
        %false = arith.constant false
        %11 = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>
        %c0_9 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        %12:2 = scf.for %arg3 = %c0_9 to %c16 step %c1_10 iter_args(%arg4 = %11, %arg5 = %false) -> (!nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, i1) {
          %c7 = arith.constant 7 : index
          %16 = arith.remui %arg3, %c7 : index
          %c10000000 = arith.constant 10000000 : index
          nvgpu.mbarrier.try_wait.parity %9[%16], %arg5, %c10000000 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7>
          %c16384 = arith.constant 16384 : index
          %17 = arith.muli %16, %c16384 : index
          %c114688 = arith.constant 114688 : index
          %18 = arith.addi %17, %c114688 : index
          %19 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view_20 = memref.view %19[%17][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
          %20 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
          %view_21 = memref.view %20[%18][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>
          %21 = nvgpu.warpgroup.generate.descriptor %view_20, %arg0 : memref<128x64xf16, #gpu.address_space<workgroup>>, <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>
          %22 = nvgpu.warpgroup.generate.descriptor %view_21, %arg1 : memref<64x128xf16, #gpu.address_space<workgroup>>, <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>
          %23 = nvgpu.warpgroup.mma %21, %22, %arg4 {transposeB} : <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>, <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>, <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>
          %c0_22 = arith.constant 0 : index
          %24 = arith.cmpi ugt, %arg3, %c0_22 : index
          %25 = arith.andi %24, %5 : i1
          scf.if %25 {
            %c0_23 = arith.constant 0 : index
            %29 = arith.cmpi eq, %16, %c0_23 : index
            %c6_24 = arith.constant 6 : index
            %c1_25 = arith.constant 1 : index
            %30 = arith.subi %16, %c1_25 : index
            %31 = arith.select %29, %c6_24, %30 : index
            %32 = nvgpu.mbarrier.arrive %8[%31] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 7> -> !nvgpu.mbarrier.token
          }
          %c6 = arith.constant 6 : index
          %26 = arith.cmpi eq, %16, %c6 : index
          %true = arith.constant true
          %27 = arith.xori %arg5, %true : i1
          %28 = arith.select %26, %27, %arg5 : i1
          scf.yield %23, %28 : !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>, i1
        }
        nvvm.wgmma.wait.group.sync.aligned 0
        %thread_id_x_11 = gpu.thread_id  x
        %block_id_x_12 = gpu.block_id  x
        %block_id_y_13 = gpu.block_id  y
        %c128_14 = arith.constant 128 : index
        %13 = arith.muli %block_id_x_12, %c128_14 : index
        %c128_15 = arith.constant 128 : index
        %14 = arith.muli %block_id_y_13, %c128_15 : index
        %15 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
        %c0_16 = arith.constant 0 : index
        %view = memref.view %15[%c0_16][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x128xf32, #gpu.address_space<workgroup>>
        %subview = memref.subview %arg2[%13, %14] [128, 128] [1, 1] : memref<512x256xf32> to memref<128x128xf32, strided<[256, 1], offset: ?>>
        nvgpu.warpgroup.mma.store %12#0, %view : <fragmented = vector<128x128xf32>> to memref<128x128xf32, #gpu.address_space<workgroup>>
        gpu.barrier
        %c0_17 = arith.constant 0 : index
        %c128_18 = arith.constant 128 : index
        %c1_19 = arith.constant 1 : index
        scf.for %arg3 = %c0_17 to %c128_18 step %c1_19 {
          %16 = memref.load %view[%arg3, %thread_id_x_11] : memref<128x128xf32, #gpu.address_space<workgroup>>
          memref.store %16, %subview[%arg3, %thread_id_x_11] : memref<128x128xf32, strided<[256, 1], offset: ?>>
        }
      }
      gpu.return
    }
  }
}
