# AOT ID: ['4_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: inductor_cache/je/cjefzn2w2mwee23uhdkzj6ynogudr7hguofrzrrrv3uk2l6in6va.py
# Topologically Sorted Source Nodes: [gt_downsample, gt_downsample_1, gt_downsample_2, gt_downsample_3], Original ATen: [aten._to_copy, aten.arange, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
# Source node to ATen node mapping:
#   gt_downsample => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_2, add_4, add_5, add_6, clamp_max_2, clamp_max_3, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type_1, convert_element_type_2, convert_element_type_3, iota_1, mul_1, mul_2, mul_3, mul_4, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6
#   gt_downsample_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_10, add_12, add_13, add_14, clamp_max_6, clamp_max_7, clamp_min_5, clamp_min_6, clamp_min_7, convert_element_type_5, convert_element_type_6, convert_element_type_7, iota_3, mul_6, mul_7, mul_8, mul_9, sub_10, sub_11, sub_12, sub_13, sub_14, sub_9
#   gt_downsample_2 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_18, add_20, add_21, add_22, clamp_max_10, clamp_max_11, clamp_min_10, clamp_min_11, clamp_min_9, convert_element_type_10, convert_element_type_11, convert_element_type_9, iota_5, mul_11, mul_12, mul_13, mul_14, sub_17, sub_18, sub_19, sub_20, sub_21, sub_22
#   gt_downsample_3 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_26, add_28, add_29, add_30, clamp_max_14, clamp_max_15, clamp_min_13, clamp_min_14, clamp_min_15, convert_element_type_13, convert_element_type_14, convert_element_type_15, iota_7, mul_16, mul_17, mul_18, mul_19, sub_25, sub_26, sub_27, sub_28, sub_29, sub_30
# Graph fragment:
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, 1.0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0.0), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_1, torch.int64), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_3), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_4), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_4), kwargs = {})
#   %convert_element_type_5 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
#   %iota_3 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_3, torch.float32), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, 1.0), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, 0.5), kwargs = {})
#   %clamp_min_5 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_9, 0.0), kwargs = {})
#   %convert_element_type_7 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_5, torch.int64), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %clamp_max_4, %convert_element_type_7]), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_5, %convert_element_type_7), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %clamp_max_6), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_8), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %convert_element_type_5, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %convert_element_type_5, %convert_element_type_7]), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %clamp_max_6), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_7), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %add_12), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %convert_element_type_5), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_13, 0.0), kwargs = {})
#   %clamp_max_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 1.0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %clamp_max_7), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %mul_9), kwargs = {})
#   %convert_element_type_9 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
#   %iota_5 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_5, torch.float32), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_10, 0.5), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, 1.0), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_11, 0.5), kwargs = {})
#   %clamp_min_9 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_17, 0.0), kwargs = {})
#   %convert_element_type_11 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_9, torch.int64), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %clamp_max_8, %convert_element_type_11]), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_9, %convert_element_type_11), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_18, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %clamp_max_10), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_13), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %convert_element_type_9, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %convert_element_type_9, %convert_element_type_11]), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %clamp_max_10), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_12), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_21, %add_20), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %convert_element_type_9), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_21, 0.0), kwargs = {})
#   %clamp_max_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 1.0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %clamp_max_11), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %mul_14), kwargs = {})
#   %convert_element_type_13 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
#   %iota_7 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_7, torch.float32), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_14, 0.5), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, 1.0), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_16, 0.5), kwargs = {})
#   %clamp_min_13 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_25, 0.0), kwargs = {})
#   %convert_element_type_15 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_13, torch.int64), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %clamp_max_12, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_14 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %clamp_max_12, %convert_element_type_15]), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_15, %_unsafe_index_14), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_13, %convert_element_type_15), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_26, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %clamp_max_14), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_14, %mul_18), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %convert_element_type_13, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%arg1_1, [None, None, %convert_element_type_13, %convert_element_type_15]), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %clamp_max_14), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_17), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_29, %add_28), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_6, %convert_element_type_13), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_29, 0.0), kwargs = {})
#   %clamp_max_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 1.0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %clamp_max_15), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %mul_19), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 + tmp2
    tmp17 = tmp16 * tmp4
    tmp18 = tmp17 - tmp2
    tmp19 = triton_helpers.maximum(tmp18, tmp7)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tmp20 + tmp10
    tmp22 = triton_helpers.minimum(tmp21, tmp12)
    tmp23 = tl.load(in_ptr0 + (tmp22 + 4*tmp13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (tmp20 + 4*tmp13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20.to(tl.float32)
    tmp27 = tmp19 - tmp26
    tmp28 = triton_helpers.maximum(tmp27, tmp7)
    tmp29 = triton_helpers.minimum(tmp28, tmp4)
    tmp30 = tmp25 * tmp29
    tmp31 = tmp24 + tmp30
    tmp32 = tl.load(in_ptr0 + (tmp20 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (tmp22 + 4*tmp9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp34 = tmp33 - tmp32
    tmp35 = tmp34 * tmp29
    tmp36 = tmp32 + tmp35
    tmp37 = tmp31 - tmp36
    tmp38 = tmp9.to(tl.float32)
    tmp39 = tmp8 - tmp38
    tmp40 = triton_helpers.maximum(tmp39, tmp7)
    tmp41 = triton_helpers.minimum(tmp40, tmp4)
    tmp42 = tmp37 * tmp41
    tmp43 = tmp36 + tmp42
    tl.store(in_out_ptr0 + (x4), tmp43, xmask)
    tl.store(in_out_ptr1 + (x4), tmp43, xmask)
    tl.store(in_out_ptr2 + (x4), tmp43, xmask)
    tl.store(in_out_ptr3 + (x4), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sl/cslvwzhgk72kvyfpygwlweqrjv5hykmyutulhiiz7fiz5p6hdhw6.py
# Topologically Sorted Source Nodes: [sub, norm, mean], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.mean]
# Source node to ATen node mapping:
#   mean => mean
#   norm => pow_1, pow_2, sum_1
#   sub => sub_7
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select, %add_6), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_7, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_2,), kwargs = {})
triton_per_fused_linalg_vector_norm_mean_sub_1 = async_compile.triton('triton_per_fused_linalg_vector_norm_mean_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_mean_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_mean_sub_1(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp15 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/t6/ct6ctixwlvfm5vphyy52tb7dijo3pggmknb6tbw6ncue7l5d3m63.py
# Topologically Sorted Source Nodes: [sub_1, norm_1, mean_1], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.mean]
# Source node to ATen node mapping:
#   mean_1 => mean_1
#   norm_1 => pow_3, pow_4, sum_2
#   sub_1 => sub_15
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_1, %add_14), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_15, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_4,), kwargs = {})
triton_per_fused_linalg_vector_norm_mean_sub_2 = async_compile.triton('triton_per_fused_linalg_vector_norm_mean_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_mean_sub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_mean_sub_2(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    tmp0 = tl.load(in_ptr0 + (256 + r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr0 + (272 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr0 + (288 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr0 + (304 + r0 + 64*r1), None)
    tmp15 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/zd/czdi25dgaofhlbtbinpnu6ko3p4mdpyl44s7rjvnltiywd4btelt.py
# Topologically Sorted Source Nodes: [sub_2, norm_2, mean_2], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   norm_2 => pow_5, pow_6, sum_3
#   sub_2 => sub_23
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_2, %add_22), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_23, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1]), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_6,), kwargs = {})
triton_per_fused_linalg_vector_norm_mean_sub_3 = async_compile.triton('triton_per_fused_linalg_vector_norm_mean_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_mean_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_mean_sub_3(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    tmp0 = tl.load(in_ptr0 + (512 + r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr0 + (528 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr0 + (544 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr0 + (560 + r0 + 64*r1), None)
    tmp15 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/3h/c3htcyf64q5vrd7yhx3nrq677dnjducj6ws4kjib7mtaxt3ztfmx.py
# Topologically Sorted Source Nodes: [sub, norm, mean, loss_F, sub_1, norm_1, mean_1, loss_F_1, sub_2, norm_2, mean_2, loss_F_2, sub_3, norm_3, mean_3, loss_F_3], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.mean, aten.add]
# Source node to ATen node mapping:
#   loss_F => add_7
#   loss_F_1 => add_15
#   loss_F_2 => add_23
#   loss_F_3 => add_31
#   mean => mean
#   mean_1 => mean_1
#   mean_2 => mean_2
#   mean_3 => mean_3
#   norm => pow_1, pow_2, sum_1
#   norm_1 => pow_3, pow_4, sum_2
#   norm_2 => pow_5, pow_6, sum_3
#   norm_3 => pow_7, pow_8, sum_4
#   sub => sub_7
#   sub_1 => sub_15
#   sub_2 => sub_23
#   sub_3 => sub_31
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select, %add_6), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_7, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_2,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 0), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_1, %add_14), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_15, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_4,), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %mean_1), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_2, %add_22), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_23, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1]), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_6,), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %mean_2), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_3, %add_30), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_31, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1]), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_8,), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %mean_3), kwargs = {})
triton_per_fused_add_linalg_vector_norm_mean_sub_4 = async_compile.triton('triton_per_fused_add_linalg_vector_norm_mean_sub_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_linalg_vector_norm_mean_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_linalg_vector_norm_mean_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    tmp0 = tl.load(in_ptr0 + (768 + r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr0 + (784 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr0 + (800 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr0 + (816 + r0 + 64*r1), None)
    tmp15 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp23 = tl.load(in_out_ptr0 + (0))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, 1])
    tmp29 = tl.load(in_ptr2 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp33 = tl.load(in_ptr3 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tmp25 = 64.0
    tmp26 = tmp24 / tmp25
    tmp27 = 0.0
    tmp28 = tmp26 + tmp27
    tmp31 = tmp30 / tmp25
    tmp32 = tmp28 + tmp31
    tmp35 = tmp34 / tmp25
    tmp36 = tmp32 + tmp35
    tmp37 = tmp22 / tmp25
    tmp38 = tmp36 + tmp37
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp38, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4, 4), (256, 64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf12 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        buf2 = buf1; del buf1  # reuse
        buf5 = buf4; del buf4  # reuse
        buf6 = buf5; del buf5  # reuse
        buf9 = buf8; del buf8  # reuse
        buf10 = buf9; del buf9  # reuse
        buf13 = buf12; del buf12  # reuse
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [gt_downsample, gt_downsample_1, gt_downsample_2, gt_downsample_3], Original ATen: [aten._to_copy, aten.arange, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0.run(buf2, buf6, buf10, buf14, arg1_1, 256, grid=grid(256), stream=stream0)
        del arg1_1
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub, norm, mean], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_mean_sub_1.run(arg0_1, buf2, buf3, 1, 64, grid=grid(1), stream=stream0)
        del buf2
        buf7 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, norm_1, mean_1], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_mean_sub_2.run(arg0_1, buf6, buf7, 1, 64, grid=grid(1), stream=stream0)
        del buf6
        buf11 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [sub_2, norm_2, mean_2], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_mean_sub_3.run(arg0_1, buf10, buf11, 1, 64, grid=grid(1), stream=stream0)
        del buf10
        buf16 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [sub, norm, mean, loss_F, sub_1, norm_1, mean_1, loss_F_1, sub_2, norm_2, mean_2, loss_F_2, sub_3, norm_3, mean_3, loss_F_3], Original ATen: [aten.sub, aten.linalg_vector_norm, aten.mean, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_linalg_vector_norm_mean_sub_4.run(buf16, arg0_1, buf14, buf7, buf11, 1, 64, grid=grid(1), stream=stream0)
        del arg0_1
        del buf11
        del buf14
        del buf7
    return (buf16, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4, 4), (256, 64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
