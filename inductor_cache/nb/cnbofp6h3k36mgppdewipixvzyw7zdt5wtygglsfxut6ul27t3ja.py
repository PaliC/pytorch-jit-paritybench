# AOT ID: ['6_forward']
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


# kernel path: inductor_cache/ja/cjadgi7ddpjmiai4gzdhqpn4dn5cpk3n5wcnytkyvrqqp4miez6a.py
# Topologically Sorted Source Nodes: [pad, upsample, pad_1, upsample_2, pad_2, upsample_4, pad_3, upsample_6], Original ATen: [aten.constant_pad_nd, aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
# Source node to ATen node mapping:
#   pad => constant_pad_nd
#   pad_1 => constant_pad_nd_1
#   pad_2 => constant_pad_nd_2
#   pad_3 => constant_pad_nd_3
#   upsample => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add, add_4, add_5, add_6, clamp_max_2, clamp_max_3, clamp_min, clamp_min_2, clamp_min_3, convert_element_type, convert_element_type_1, convert_element_type_3, iota, mul, mul_2, mul_3, mul_4, sub, sub_2, sub_3, sub_4, sub_5, sub_6
#   upsample_2 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_20, add_21, add_22, clamp_max_10, clamp_max_11, clamp_min_10, clamp_min_11, clamp_min_8, convert_element_type_11, convert_element_type_13, mul_13, mul_15, mul_16, mul_17, sub_15, sub_17, sub_18, sub_19, sub_20, sub_21
#   upsample_4 => _unsafe_index_16, _unsafe_index_17, _unsafe_index_18, _unsafe_index_19, add_36, add_37, add_38, clamp_max_18, clamp_max_19, clamp_min_16, clamp_min_18, clamp_min_19, convert_element_type_21, convert_element_type_23, mul_26, mul_28, mul_29, mul_30, sub_30, sub_32, sub_33, sub_34, sub_35, sub_36
#   upsample_6 => _unsafe_index_24, _unsafe_index_25, _unsafe_index_26, _unsafe_index_27, add_52, add_53, add_54, clamp_max_26, clamp_max_27, clamp_min_24, clamp_min_26, clamp_min_27, convert_element_type_31, convert_element_type_33, mul_39, mul_41, mul_42, mul_43, sub_45, sub_47, sub_48, sub_49, sub_50, sub_51
# Graph fragment:
#   %constant_pad_nd : [num_users=4] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [3, 3, 3, 3], 0.0), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 10.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min, torch.int64), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_3), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_4), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_4), kwargs = {})
#   %constant_pad_nd_1 : [num_users=4] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [8, 8, 8, 8], 0.0), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 20.0), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_13, 0.5), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_15, 0.0), kwargs = {})
#   %convert_element_type_11 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
#   %convert_element_type_13 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_8, torch.int64), kwargs = {})
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_1, [None, None, %convert_element_type_11, %convert_element_type_13]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_1, [None, None, %convert_element_type_11, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_1, [None, None, %clamp_max_8, %convert_element_type_13]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_1, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_13), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_17, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %clamp_max_10), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_15), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %clamp_max_10), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_16), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %convert_element_type_11), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_20, 0.0), kwargs = {})
#   %clamp_max_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 1.0), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_21, %add_20), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %clamp_max_11), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %mul_17), kwargs = {})
#   %constant_pad_nd_2 : [num_users=4] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [13, 13, 13, 13], 0.0), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 30.0), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_26, 0.5), kwargs = {})
#   %clamp_min_16 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_30, 0.0), kwargs = {})
#   %convert_element_type_21 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.int64), kwargs = {})
#   %convert_element_type_23 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_16, torch.int64), kwargs = {})
#   %_unsafe_index_16 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_2, [None, None, %convert_element_type_21, %convert_element_type_23]), kwargs = {})
#   %_unsafe_index_17 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_2, [None, None, %convert_element_type_21, %clamp_max_17]), kwargs = {})
#   %_unsafe_index_18 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_2, [None, None, %clamp_max_16, %convert_element_type_23]), kwargs = {})
#   %_unsafe_index_19 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_2, [None, None, %clamp_max_16, %clamp_max_17]), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_16, %convert_element_type_23), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_32, 0.0), kwargs = {})
#   %clamp_max_18 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 1.0), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_17, %_unsafe_index_16), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %clamp_max_18), kwargs = {})
#   %add_36 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_16, %mul_28), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_19, %_unsafe_index_18), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %clamp_max_18), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_18, %mul_29), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_8, %convert_element_type_21), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_35, 0.0), kwargs = {})
#   %clamp_max_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 1.0), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_37, %add_36), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %clamp_max_19), kwargs = {})
#   %add_38 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %mul_30), kwargs = {})
#   %constant_pad_nd_3 : [num_users=4] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [28, 28, 28, 28], 0.0), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 60.0), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_39, 0.5), kwargs = {})
#   %clamp_min_24 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_45, 0.0), kwargs = {})
#   %convert_element_type_31 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.int64), kwargs = {})
#   %convert_element_type_33 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_24, torch.int64), kwargs = {})
#   %_unsafe_index_24 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_3, [None, None, %convert_element_type_31, %convert_element_type_33]), kwargs = {})
#   %_unsafe_index_25 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_3, [None, None, %convert_element_type_31, %clamp_max_25]), kwargs = {})
#   %_unsafe_index_26 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_3, [None, None, %clamp_max_24, %convert_element_type_33]), kwargs = {})
#   %_unsafe_index_27 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%constant_pad_nd_3, [None, None, %clamp_max_24, %clamp_max_25]), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_24, %convert_element_type_33), kwargs = {})
#   %clamp_min_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_47, 0.0), kwargs = {})
#   %clamp_max_26 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_26, 1.0), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_25, %_unsafe_index_24), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %clamp_max_26), kwargs = {})
#   %add_52 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_24, %mul_41), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_27, %_unsafe_index_26), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %clamp_max_26), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_26, %mul_42), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_12, %convert_element_type_31), kwargs = {})
#   %clamp_min_27 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_50, 0.0), kwargs = {})
#   %clamp_max_27 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_27, 1.0), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_53, %add_52), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %clamp_max_27), kwargs = {})
#   %add_54 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_52, %mul_43), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_constant_pad_nd_mul_sub_0 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_constant_pad_nd_mul_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_out_ptr7': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_constant_pad_nd_mul_sub_0', 'mutated_arg_names': ['in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_constant_pad_nd_mul_sub_0(in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 2, tl.int64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + (10 + 16*x0), tmp7 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp9 >= tmp1
    tmp11 = tmp9 < tmp3
    tmp12 = tmp5 & tmp10
    tmp13 = tmp12 & tmp11
    tmp14 = tl.load(in_ptr0 + (9 + 16*x0), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp8 - tmp14
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp10 & tmp11
    tmp20 = tmp19 & tmp2
    tmp21 = tmp20 & tmp4
    tmp22 = tl.load(in_ptr0 + (6 + 16*x0), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp19 & tmp10
    tmp24 = tmp23 & tmp11
    tmp25 = tl.load(in_ptr0 + (5 + 16*x0), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp22 - tmp25
    tmp27 = tmp26 * tmp16
    tmp28 = tmp25 + tmp27
    tmp29 = tmp18 - tmp28
    tmp30 = tmp29 * tmp16
    tmp31 = tmp28 + tmp30
    tl.store(in_out_ptr4 + (x0), tmp31, xmask)
    tl.store(in_out_ptr5 + (x0), tmp31, xmask)
    tl.store(in_out_ptr6 + (x0), tmp31, xmask)
    tl.store(in_out_ptr7 + (x0), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jm/cjmrd5yeq55rzbzvgbv66wh2zyiwphnijx4zfdi3sbqccabdwe6p.py
# Topologically Sorted Source Nodes: [batch_norm, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add_8, mul_6, mul_7, sub_7
#   relu => relu
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %unsqueeze_5), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/iy/ciygnwe334yneh7xtqkzahgqu6ex3243uynfvasty3lp73nssgb5.py
# Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   upsample_1 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_7 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4o/c4o2jxown5dgqxdc2tuexkzus2v4lwdcuthvbjycdvnixb25syqr.py
# Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   upsample_1 => add_10, clamp_max_4
# Graph fragment:
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, 1), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_10, 0), kwargs = {})
triton_poi_fused_add_clamp_3 = async_compile.triton('triton_poi_fused_add_clamp_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eh/cehrncpm7vtnlwqpbnlax63prukq2tuieym7odv235jzttw2qc5v.py
# Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   upsample_1 => add_9, clamp_max_6, clamp_min_4, clamp_min_6, convert_element_type_6, iota_2, mul_8, sub_10, sub_8
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (10,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 0.5), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, 0.1), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_8, 0.5), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_8, 0.0), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_9), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_4 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6s/c6sfjlbqtoh4uvcnuw7eq3vcc6tbhmp22k2d5f4qb4hxjjomf6by.py
# Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   upsample_1 => _unsafe_index_4, _unsafe_index_5, add_13, mul_10, sub_11
# Graph fragment:
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %convert_element_type_7, %convert_element_type_9]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %convert_element_type_7, %clamp_max_5]), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %clamp_max_6), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_10), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_5 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 10) % 10)
    x0 = (xindex % 10)
    x2 = xindex // 100
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tmp9 - tmp9
    tmp16 = tmp14 * tmp15
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ri/crivjdk2ici6iu6svtnjbmqj3btuyzqno6mrrd4qldl5eayzskgw.py
# Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   upsample_3 => convert_element_type_17
# Graph fragment:
#   %convert_element_type_17 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.05
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/is/cisjt3ef6n2vxtzfg6v2ojiubwd4wui62lh44yga7w3u6qeq2465.py
# Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   upsample_3 => add_26, clamp_max_12
# Graph fragment:
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_17, 1), kwargs = {})
#   %clamp_max_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_26, 0), kwargs = {})
triton_poi_fused_add_clamp_7 = async_compile.triton('triton_poi_fused_add_clamp_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.05
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ed/cedh3lsxocycso2sy6vve5crosnizsgbpkimmuz22osad3j2mzlt.py
# Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   upsample_3 => add_25, clamp_max_14, clamp_min_12, clamp_min_14, convert_element_type_16, iota_6, mul_21, sub_23, sub_25
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (20,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_6, torch.float32), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_16, 0.5), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_25, 0.05), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_21, 0.5), kwargs = {})
#   %clamp_min_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_23, 0.0), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_12, %convert_element_type_19), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_25, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.05
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lg/clgfiwp6ymg6nizccjewxyf7bcuev5li3w7f5i6fht5vh7w27tvh.py
# Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   upsample_3 => _unsafe_index_12, _unsafe_index_13, add_29, mul_23, sub_26
# Graph fragment:
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_1, [None, None, %convert_element_type_17, %convert_element_type_19]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_1, [None, None, %convert_element_type_17, %clamp_max_13]), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %clamp_max_14), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_23), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_9 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 20) % 20)
    x0 = (xindex % 20)
    x2 = xindex // 400
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tmp9 - tmp9
    tmp16 = tmp14 * tmp15
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ke/ckejbkz5oqaayfekbpriitmjdrdhomfkutcxp6knophwmb4g2dtq.py
# Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   upsample_5 => convert_element_type_27
# Graph fragment:
#   %convert_element_type_27 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_10, torch.int64), kwargs = {})
triton_poi_fused__to_copy_10 = async_compile.triton('triton_poi_fused__to_copy_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.03333333333333333
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci42rnhtzjjjwmsocrlt4jgqeoweovak4exz4rpdtlyiqme2b7zj.py
# Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   upsample_5 => add_42, clamp_max_20
# Graph fragment:
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_27, 1), kwargs = {})
#   %clamp_max_20 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_42, 0), kwargs = {})
triton_poi_fused_add_clamp_11 = async_compile.triton('triton_poi_fused_add_clamp_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_11(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.03333333333333333
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zp/czposy2uqpdha5iv4rv6r56gzdxrumz7g4imetsz5rhm5mtbt5g5.py
# Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   upsample_5 => add_41, clamp_max_22, clamp_min_20, clamp_min_22, convert_element_type_26, iota_10, mul_34, sub_38, sub_40
# Graph fragment:
#   %iota_10 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (30,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_26 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_10, torch.float32), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_26, 0.5), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, 0.03333333333333333), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_34, 0.5), kwargs = {})
#   %clamp_min_20 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_38, 0.0), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_20, %convert_element_type_29), kwargs = {})
#   %clamp_min_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_40, 0.0), kwargs = {})
#   %clamp_max_22 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_22, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_12 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.03333333333333333
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3a/c3a44vsez6ib3db63xkfabjpu72g3mhrszpgbavv6nbprnsdrznd.py
# Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   upsample_5 => _unsafe_index_20, _unsafe_index_21, add_45, mul_36, sub_41
# Graph fragment:
#   %_unsafe_index_20 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_2, [None, None, %convert_element_type_27, %convert_element_type_29]), kwargs = {})
#   %_unsafe_index_21 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_2, [None, None, %convert_element_type_27, %clamp_max_21]), kwargs = {})
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_21, %_unsafe_index_20), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %clamp_max_22), kwargs = {})
#   %add_45 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_20, %mul_36), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_13 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1843200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 30) % 30)
    x0 = (xindex % 30)
    x2 = xindex // 900
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tmp9 - tmp9
    tmp16 = tmp14 * tmp15
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ce/cceqe3gmcnkllzqigbok2d6ynn2ccgruuny7dzqq3w2llkjik3gu.py
# Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   upsample_7 => convert_element_type_37
# Graph fragment:
#   %convert_element_type_37 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_14, torch.int64), kwargs = {})
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 60
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.016666666666666666
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tn/ctnuwzrdzm2rted3xiqd26ilt7uiwa5i6rua37zk3x2tynjox7ns.py
# Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   upsample_7 => add_58, clamp_max_28
# Graph fragment:
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_37, 1), kwargs = {})
#   %clamp_max_28 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_58, 0), kwargs = {})
triton_poi_fused_add_clamp_15 = async_compile.triton('triton_poi_fused_add_clamp_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 60
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.016666666666666666
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4l/c4ljeqlahizy2ighhykhgdso26y6s6nh5yyxw4vzaefuzpjgtrsy.py
# Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   upsample_7 => add_57, clamp_max_30, clamp_min_28, clamp_min_30, convert_element_type_36, iota_14, mul_47, sub_53, sub_55
# Graph fragment:
#   %iota_14 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (60,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_36 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_14, torch.float32), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_36, 0.5), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, 0.016666666666666666), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_47, 0.5), kwargs = {})
#   %clamp_min_28 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_53, 0.0), kwargs = {})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_28, %convert_element_type_39), kwargs = {})
#   %clamp_min_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_55, 0.0), kwargs = {})
#   %clamp_max_30 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_30, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 60
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.016666666666666666
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7l/c7l2omnsqia77zp5lanrgw5feqgdjzcasjav47yuxl4vi7sw2xsh.py
# Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   upsample_7 => _unsafe_index_28, _unsafe_index_29, add_61, mul_49, sub_56
# Graph fragment:
#   %_unsafe_index_28 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_3, [None, None, %convert_element_type_37, %convert_element_type_39]), kwargs = {})
#   %_unsafe_index_29 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_3, [None, None, %convert_element_type_37, %clamp_max_29]), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_29, %_unsafe_index_28), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %clamp_max_30), kwargs = {})
#   %add_61 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_28, %mul_49), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_17 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7372800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 60) % 60)
    x0 = (xindex % 60)
    x2 = xindex // 3600
    x3 = (xindex % 3600)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tmp9 - tmp9
    tmp16 = tmp14 * tmp15
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x3 + 3616*x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/zj/czjfrcealjdp6mmjaiwtofifyypv6ajkwzsu3huasugasn3jbrsi.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_1, %slice_4, %slice_8, %slice_12, %slice_16], 1), kwargs = {})
triton_poi_fused_cat_18 = async_compile.triton('triton_poi_fused_cat_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*i64', 'in_ptr18': '*fp32', 'in_ptr19': '*i64', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*i64', 'in_ptr24': '*i64', 'in_ptr25': '*fp32', 'in_ptr26': '*i64', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 29, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 2052)
    x3 = xindex // 32832
    x4 = (xindex % 16)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x7 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 64*x3), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 516, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (33 + x0 + 10*x1 + 100*((-4) + x2) + 51200*x3), tmp9 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (3 + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([XBLOCK], 1, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tmp16 = tl.load(in_ptr3 + (3 + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr4 + (512*x3 + ((-4) + x2)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr5 + (3 + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp12
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tmp20 - tmp20
    tmp26 = tl.load(in_ptr6 + (3 + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 + tmp27
    tmp29 = tmp28 - tmp10
    tmp30 = tl.load(in_ptr7 + (3 + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp10 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp9, tmp32, tmp33)
    tmp35 = tmp0 >= tmp7
    tmp36 = tl.full([1], 1028, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tmp35 & tmp37
    tmp39 = tl.load(in_ptr8 + (168 + x0 + 20*x1 + 400*((-516) + x2) + 204800*x3), tmp38 & xmask, other=0.0)
    tmp40 = tl.load(in_ptr9 + (8 + x1), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full([XBLOCK], 1, tl.int32)
    tmp42 = tmp40 + tmp41
    tmp43 = tmp40 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp40)
    tmp45 = tl.load(in_ptr10 + (8 + x0), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp41
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tmp49 = tl.load(in_ptr11 + (512*x3 + ((-516) + x2)), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr12 + (8 + x0), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp41
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tmp49 - tmp49
    tmp55 = tl.load(in_ptr13 + (8 + x0), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp49 + tmp56
    tmp58 = tmp57 - tmp39
    tmp59 = tl.load(in_ptr14 + (8 + x1), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp58 * tmp59
    tmp61 = tmp39 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp38, tmp61, tmp62)
    tmp64 = tmp0 >= tmp36
    tmp65 = tl.full([1], 1540, tl.int64)
    tmp66 = tmp0 < tmp65
    tmp67 = tmp64 & tmp66
    tmp68 = tl.load(in_ptr15 + (403 + x0 + 30*x1 + 900*((-1028) + x2) + 460800*x3), tmp67 & xmask, other=0.0)
    tmp69 = tl.load(in_ptr16 + (13 + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tl.full([XBLOCK], 1, tl.int32)
    tmp71 = tmp69 + tmp70
    tmp72 = tmp69 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp69)
    tmp74 = tl.load(in_ptr17 + (13 + x0), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tmp74 + tmp70
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tmp78 = tl.load(in_ptr18 + (512*x3 + ((-1028) + x2)), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tl.load(in_ptr19 + (13 + x0), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tmp79 + tmp70
    tmp81 = tmp79 < 0
    tmp82 = tl.where(tmp81, tmp80, tmp79)
    tmp83 = tmp78 - tmp78
    tmp84 = tl.load(in_ptr20 + (13 + x0), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tmp78 + tmp85
    tmp87 = tmp86 - tmp68
    tmp88 = tl.load(in_ptr21 + (13 + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tmp68 + tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp67, tmp90, tmp91)
    tmp93 = tmp0 >= tmp65
    tmp94 = tl.full([1], 2052, tl.int64)
    tmp95 = tmp0 < tmp94
    tmp96 = tl.load(in_ptr22 + (1708 + x0 + 60*x1 + 3616*((-1540) + x2) + 1851392*x3), tmp93 & xmask, other=0.0)
    tmp97 = tl.load(in_ptr23 + (28 + x1), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp98 = tl.full([XBLOCK], 1, tl.int32)
    tmp99 = tmp97 + tmp98
    tmp100 = tmp97 < 0
    tmp101 = tl.where(tmp100, tmp99, tmp97)
    tmp102 = tl.load(in_ptr24 + (28 + x0), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp103 = tmp102 + tmp98
    tmp104 = tmp102 < 0
    tmp105 = tl.where(tmp104, tmp103, tmp102)
    tmp106 = tl.load(in_ptr25 + (512*x3 + ((-1540) + x2)), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp107 = tl.load(in_ptr26 + (28 + x0), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tmp107 + tmp98
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tmp111 = tmp106 - tmp106
    tmp112 = tl.load(in_ptr27 + (28 + x0), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tmp111 * tmp112
    tmp114 = tmp106 + tmp113
    tmp115 = tmp114 - tmp96
    tmp116 = tl.load(in_ptr28 + (28 + x1), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp117 = tmp115 * tmp116
    tmp118 = tmp96 + tmp117
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp93, tmp118, tmp119)
    tmp121 = tl.where(tmp67, tmp92, tmp120)
    tmp122 = tl.where(tmp38, tmp63, tmp121)
    tmp123 = tl.where(tmp9, tmp34, tmp122)
    tmp124 = tl.where(tmp4, tmp5, tmp123)
    tl.store(out_ptr0 + (x7), tmp124, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/74/c74cij5iappwjoahztxfi3it2bf4xfvgonar4jlfmg5hdhxzvzq6.py
# Topologically Sorted Source Nodes: [out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   out_2 => add_65, mul_53, mul_54, sub_60
#   out_3 => relu_4
# Graph fragment:
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %unsqueeze_35), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %unsqueeze_37), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (512, ), (1, ))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_6, (512, ), (1, ))
    assert_size_stride(primals_7, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_13, (512, ), (1, ))
    assert_size_stride(primals_14, (512, ), (1, ))
    assert_size_stride(primals_15, (512, ), (1, ))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, 2052, 3, 3), (18468, 9, 3, 1))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf4 = reinterpret_tensor(buf0, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf0  # reuse
        buf14 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf18 = reinterpret_tensor(buf14, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf14  # reuse
        buf28 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf32 = reinterpret_tensor(buf28, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf28  # reuse
        buf42 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf46 = reinterpret_tensor(buf42, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [pad, upsample, pad_1, upsample_2, pad_2, upsample_4, pad_3, upsample_6], Original ATen: [aten.constant_pad_nd, aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_constant_pad_nd_mul_sub_0.run(buf4, buf18, buf32, buf46, primals_1, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 512, 1, 1), (512, 1, 1, 1))
        buf6 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm, relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf5, primals_3, primals_4, primals_5, primals_6, buf6, 2048, grid=grid(2048), stream=stream0)
        buf7 = empty_strided_cuda((10, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf7, 10, grid=grid(10), stream=stream0)
        buf8 = empty_strided_cuda((10, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_3.run(buf8, 10, grid=grid(10), stream=stream0)
        buf9 = empty_strided_cuda((10, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf9, 10, grid=grid(10), stream=stream0)
        buf10 = empty_strided_cuda((10, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_3.run(buf10, 10, grid=grid(10), stream=stream0)
        buf11 = empty_strided_cuda((10, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_4.run(buf11, 10, grid=grid(10), stream=stream0)
        buf12 = empty_strided_cuda((4, 512, 10, 10), (51200, 100, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_5.run(buf7, buf9, buf6, buf10, buf11, buf12, 204800, grid=grid(204800), stream=stream0)
        buf13 = empty_strided_cuda((10, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_4.run(buf13, 10, grid=grid(10), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 512, 1, 1), (512, 1, 1, 1))
        buf20 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_1, relu_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf19, primals_8, primals_9, primals_10, primals_11, buf20, 2048, grid=grid(2048), stream=stream0)
        buf21 = empty_strided_cuda((20, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf21, 20, grid=grid(20), stream=stream0)
        buf22 = empty_strided_cuda((20, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_7.run(buf22, 20, grid=grid(20), stream=stream0)
        buf23 = empty_strided_cuda((20, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(buf23, 20, grid=grid(20), stream=stream0)
        buf24 = empty_strided_cuda((20, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_7.run(buf24, 20, grid=grid(20), stream=stream0)
        buf25 = empty_strided_cuda((20, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8.run(buf25, 20, grid=grid(20), stream=stream0)
        buf26 = empty_strided_cuda((4, 512, 20, 20), (204800, 400, 20, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_9.run(buf21, buf23, buf20, buf24, buf25, buf26, 819200, grid=grid(819200), stream=stream0)
        buf27 = empty_strided_cuda((20, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_8.run(buf27, 20, grid=grid(20), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 512, 1, 1), (512, 1, 1, 1))
        buf34 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_2, relu_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf33, primals_13, primals_14, primals_15, primals_16, buf34, 2048, grid=grid(2048), stream=stream0)
        buf35 = empty_strided_cuda((30, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(buf35, 30, grid=grid(30), stream=stream0)
        buf36 = empty_strided_cuda((30, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_11.run(buf36, 30, grid=grid(30), stream=stream0)
        buf37 = empty_strided_cuda((30, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(buf37, 30, grid=grid(30), stream=stream0)
        buf38 = empty_strided_cuda((30, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_11.run(buf38, 30, grid=grid(30), stream=stream0)
        buf39 = empty_strided_cuda((30, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_12.run(buf39, 30, grid=grid(30), stream=stream0)
        buf40 = empty_strided_cuda((4, 512, 30, 30), (460800, 900, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_13.run(buf35, buf37, buf34, buf38, buf39, buf40, 1843200, grid=grid(1843200), stream=stream0)
        buf41 = empty_strided_cuda((30, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_5], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_12.run(buf41, 30, grid=grid(30), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 512, 1, 1), (512, 1, 1, 1))
        buf48 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_3, relu_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf47, primals_18, primals_19, primals_20, primals_21, buf48, 2048, grid=grid(2048), stream=stream0)
        buf49 = empty_strided_cuda((60, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf49, 60, grid=grid(60), stream=stream0)
        buf50 = empty_strided_cuda((60, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_15.run(buf50, 60, grid=grid(60), stream=stream0)
        buf51 = empty_strided_cuda((60, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf51, 60, grid=grid(60), stream=stream0)
        buf52 = empty_strided_cuda((60, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_15.run(buf52, 60, grid=grid(60), stream=stream0)
        buf53 = empty_strided_cuda((60, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16.run(buf53, 60, grid=grid(60), stream=stream0)
        buf54 = empty_strided_cuda((4, 512, 60, 60), (1851392, 3616, 60, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_17.run(buf49, buf51, buf48, buf52, buf53, buf54, 7372800, grid=grid(7372800), stream=stream0)
        buf55 = empty_strided_cuda((60, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsample_7], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_16.run(buf55, 60, grid=grid(60), stream=stream0)
        buf56 = empty_strided_cuda((4, 2052, 4, 4), (32832, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_18.run(primals_1, buf12, buf8, buf9, buf6, buf10, buf11, buf13, buf26, buf22, buf23, buf20, buf24, buf25, buf27, buf40, buf36, buf37, buf34, buf38, buf39, buf41, buf54, buf50, buf51, buf48, buf52, buf53, buf55, buf56, 131328, grid=grid(131328), stream=stream0)
        del buf12
        del buf20
        del buf26
        del buf34
        del buf40
        del buf48
        del buf54
        del buf6
        del primals_1
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf58 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf59 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_19.run(buf57, primals_23, primals_24, primals_25, primals_26, buf58, buf59, 32768, grid=grid(32768), stream=stream0)
        del primals_26
    return (buf58, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, buf4, buf5, buf7, buf8, buf9, buf10, buf11, buf13, buf18, buf19, buf21, buf22, buf23, buf24, buf25, buf27, buf32, buf33, buf35, buf36, buf37, buf38, buf39, buf41, buf46, buf47, buf49, buf50, buf51, buf52, buf53, buf55, buf56, buf57, buf59, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, 2052, 3, 3), (18468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
