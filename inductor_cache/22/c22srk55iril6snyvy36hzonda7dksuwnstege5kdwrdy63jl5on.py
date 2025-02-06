# AOT ID: ['9_forward']
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


# kernel path: inductor_cache/mi/cmilbcz2m6d6im7pfgr22264hps745rahzp67kncfwto7ox6ngqm.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sb/csbbmwonaebsxbhlsmtjzst5o65nvibi6572w7nilp4rkrjyfpq6.py
# Topologically Sorted Source Nodes: [input_5, input_6, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => relu_1
#   x => add_4
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_1, %primals_2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), xmask)
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
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, ), (1, ))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, ), (1, ))
    assert_size_stride(primals_27, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_38, (4, ), (1, ))
    assert_size_stride(primals_39, (4, ), (1, ))
    assert_size_stride(primals_40, (4, ), (1, ))
    assert_size_stride(primals_41, (4, ), (1, ))
    assert_size_stride(primals_42, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_43, (4, ), (1, ))
    assert_size_stride(primals_44, (4, ), (1, ))
    assert_size_stride(primals_45, (4, ), (1, ))
    assert_size_stride(primals_46, (4, ), (1, ))
    assert_size_stride(primals_47, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_48, (4, ), (1, ))
    assert_size_stride(primals_49, (4, ), (1, ))
    assert_size_stride(primals_50, (4, ), (1, ))
    assert_size_stride(primals_51, (4, ), (1, ))
    assert_size_stride(primals_52, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_53, (4, ), (1, ))
    assert_size_stride(primals_54, (4, ), (1, ))
    assert_size_stride(primals_55, (4, ), (1, ))
    assert_size_stride(primals_56, (4, ), (1, ))
    assert_size_stride(primals_57, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_58, (4, ), (1, ))
    assert_size_stride(primals_59, (4, ), (1, ))
    assert_size_stride(primals_60, (4, ), (1, ))
    assert_size_stride(primals_61, (4, ), (1, ))
    assert_size_stride(primals_62, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_63, (4, ), (1, ))
    assert_size_stride(primals_64, (4, ), (1, ))
    assert_size_stride(primals_65, (4, ), (1, ))
    assert_size_stride(primals_66, (4, ), (1, ))
    assert_size_stride(primals_67, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_68, (4, ), (1, ))
    assert_size_stride(primals_69, (4, ), (1, ))
    assert_size_stride(primals_70, (4, ), (1, ))
    assert_size_stride(primals_71, (4, ), (1, ))
    assert_size_stride(primals_72, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_73, (4, ), (1, ))
    assert_size_stride(primals_74, (4, ), (1, ))
    assert_size_stride(primals_75, (4, ), (1, ))
    assert_size_stride(primals_76, (4, ), (1, ))
    assert_size_stride(primals_77, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_78, (4, ), (1, ))
    assert_size_stride(primals_79, (4, ), (1, ))
    assert_size_stride(primals_80, (4, ), (1, ))
    assert_size_stride(primals_81, (4, ), (1, ))
    assert_size_stride(primals_82, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_83, (4, ), (1, ))
    assert_size_stride(primals_84, (4, ), (1, ))
    assert_size_stride(primals_85, (4, ), (1, ))
    assert_size_stride(primals_86, (4, ), (1, ))
    assert_size_stride(primals_87, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_88, (4, ), (1, ))
    assert_size_stride(primals_89, (4, ), (1, ))
    assert_size_stride(primals_90, (4, ), (1, ))
    assert_size_stride(primals_91, (4, ), (1, ))
    assert_size_stride(primals_92, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_93, (4, ), (1, ))
    assert_size_stride(primals_94, (4, ), (1, ))
    assert_size_stride(primals_95, (4, ), (1, ))
    assert_size_stride(primals_96, (4, ), (1, ))
    assert_size_stride(primals_97, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_98, (4, ), (1, ))
    assert_size_stride(primals_99, (4, ), (1, ))
    assert_size_stride(primals_100, (4, ), (1, ))
    assert_size_stride(primals_101, (4, ), (1, ))
    assert_size_stride(primals_102, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_103, (4, ), (1, ))
    assert_size_stride(primals_104, (4, ), (1, ))
    assert_size_stride(primals_105, (4, ), (1, ))
    assert_size_stride(primals_106, (4, ), (1, ))
    assert_size_stride(primals_107, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_108, (4, ), (1, ))
    assert_size_stride(primals_109, (4, ), (1, ))
    assert_size_stride(primals_110, (4, ), (1, ))
    assert_size_stride(primals_111, (4, ), (1, ))
    assert_size_stride(primals_112, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_113, (4, ), (1, ))
    assert_size_stride(primals_114, (4, ), (1, ))
    assert_size_stride(primals_115, (4, ), (1, ))
    assert_size_stride(primals_116, (4, ), (1, ))
    assert_size_stride(primals_117, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_118, (4, ), (1, ))
    assert_size_stride(primals_119, (4, ), (1, ))
    assert_size_stride(primals_120, (4, ), (1, ))
    assert_size_stride(primals_121, (4, ), (1, ))
    assert_size_stride(primals_122, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_123, (4, ), (1, ))
    assert_size_stride(primals_124, (4, ), (1, ))
    assert_size_stride(primals_125, (4, ), (1, ))
    assert_size_stride(primals_126, (4, ), (1, ))
    assert_size_stride(primals_127, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_128, (4, ), (1, ))
    assert_size_stride(primals_129, (4, ), (1, ))
    assert_size_stride(primals_130, (4, ), (1, ))
    assert_size_stride(primals_131, (4, ), (1, ))
    assert_size_stride(primals_132, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_133, (4, ), (1, ))
    assert_size_stride(primals_134, (4, ), (1, ))
    assert_size_stride(primals_135, (4, ), (1, ))
    assert_size_stride(primals_136, (4, ), (1, ))
    assert_size_stride(primals_137, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_138, (4, ), (1, ))
    assert_size_stride(primals_139, (4, ), (1, ))
    assert_size_stride(primals_140, (4, ), (1, ))
    assert_size_stride(primals_141, (4, ), (1, ))
    assert_size_stride(primals_142, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_143, (4, ), (1, ))
    assert_size_stride(primals_144, (4, ), (1, ))
    assert_size_stride(primals_145, (4, ), (1, ))
    assert_size_stride(primals_146, (4, ), (1, ))
    assert_size_stride(primals_147, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_148, (4, ), (1, ))
    assert_size_stride(primals_149, (4, ), (1, ))
    assert_size_stride(primals_150, (4, ), (1, ))
    assert_size_stride(primals_151, (4, ), (1, ))
    assert_size_stride(primals_152, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_153, (4, ), (1, ))
    assert_size_stride(primals_154, (4, ), (1, ))
    assert_size_stride(primals_155, (4, ), (1, ))
    assert_size_stride(primals_156, (4, ), (1, ))
    assert_size_stride(primals_157, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_158, (4, ), (1, ))
    assert_size_stride(primals_159, (4, ), (1, ))
    assert_size_stride(primals_160, (4, ), (1, ))
    assert_size_stride(primals_161, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 4, 4), (64, 16, 4, 1))
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_3, primals_4, primals_5, primals_6, buf1, 256, grid=grid(256), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 4, 4, 4), (64, 16, 4, 1))
        buf3 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf2, primals_8, primals_9, primals_10, primals_11, primals_2, buf3, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 4, 4, 4), (64, 16, 4, 1))
        buf5 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf4, primals_13, primals_14, primals_15, primals_16, buf5, 256, grid=grid(256), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 4, 4, 4), (64, 16, 4, 1))
        buf7 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf6, primals_18, primals_19, primals_20, primals_21, buf3, buf7, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 4, 4, 4), (64, 16, 4, 1))
        buf9 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf8, primals_23, primals_24, primals_25, primals_26, buf9, 256, grid=grid(256), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 4, 4), (64, 16, 4, 1))
        buf11 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17, input_18, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf10, primals_28, primals_29, primals_30, primals_31, buf7, buf11, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 4, 4, 4), (64, 16, 4, 1))
        buf13 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf12, primals_33, primals_34, primals_35, primals_36, buf13, 256, grid=grid(256), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 4, 4, 4), (64, 16, 4, 1))
        buf15 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf14, primals_38, primals_39, primals_40, primals_41, buf11, buf15, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 4, 4, 4), (64, 16, 4, 1))
        buf17 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf16, primals_43, primals_44, primals_45, primals_46, buf17, 256, grid=grid(256), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 4, 4, 4), (64, 16, 4, 1))
        buf19 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29, input_30, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf18, primals_48, primals_49, primals_50, primals_51, buf15, buf19, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 4, 4, 4), (64, 16, 4, 1))
        buf21 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf20, primals_53, primals_54, primals_55, primals_56, buf21, 256, grid=grid(256), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 4, 4, 4), (64, 16, 4, 1))
        buf23 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_35, input_36, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf22, primals_58, primals_59, primals_60, primals_61, buf19, buf23, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 4, 4, 4), (64, 16, 4, 1))
        buf25 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf24, primals_63, primals_64, primals_65, primals_66, buf25, 256, grid=grid(256), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 4, 4, 4), (64, 16, 4, 1))
        buf27 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_41, input_42, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf26, primals_68, primals_69, primals_70, primals_71, buf23, buf27, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 4, 4, 4), (64, 16, 4, 1))
        buf29 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf28, primals_73, primals_74, primals_75, primals_76, buf29, 256, grid=grid(256), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_77, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 4, 4, 4), (64, 16, 4, 1))
        buf31 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47, input_48, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf30, primals_78, primals_79, primals_80, primals_81, buf27, buf31, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 4, 4, 4), (64, 16, 4, 1))
        buf33 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, input_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf32, primals_83, primals_84, primals_85, primals_86, buf33, 256, grid=grid(256), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 4, 4, 4), (64, 16, 4, 1))
        buf35 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_53, input_54, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf34, primals_88, primals_89, primals_90, primals_91, buf31, buf35, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 4, 4, 4), (64, 16, 4, 1))
        buf37 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, input_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf36, primals_93, primals_94, primals_95, primals_96, buf37, 256, grid=grid(256), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 4, 4, 4), (64, 16, 4, 1))
        buf39 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_59, input_60, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf38, primals_98, primals_99, primals_100, primals_101, buf35, buf39, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 4, 4, 4), (64, 16, 4, 1))
        buf41 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf40, primals_103, primals_104, primals_105, primals_106, buf41, 256, grid=grid(256), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 4, 4, 4), (64, 16, 4, 1))
        buf43 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_65, input_66, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf42, primals_108, primals_109, primals_110, primals_111, buf39, buf43, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 4, 4, 4), (64, 16, 4, 1))
        buf45 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf44, primals_113, primals_114, primals_115, primals_116, buf45, 256, grid=grid(256), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 4, 4, 4), (64, 16, 4, 1))
        buf47 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, input_72, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf46, primals_118, primals_119, primals_120, primals_121, buf43, buf47, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_122, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 4, 4, 4), (64, 16, 4, 1))
        buf49 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_74, input_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf48, primals_123, primals_124, primals_125, primals_126, buf49, 256, grid=grid(256), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 4, 4, 4), (64, 16, 4, 1))
        buf51 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_77, input_78, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf50, primals_128, primals_129, primals_130, primals_131, buf47, buf51, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 4, 4, 4), (64, 16, 4, 1))
        buf53 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf52, primals_133, primals_134, primals_135, primals_136, buf53, 256, grid=grid(256), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 4, 4, 4), (64, 16, 4, 1))
        buf55 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_83, input_84, x_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf54, primals_138, primals_139, primals_140, primals_141, buf51, buf55, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 4, 4, 4), (64, 16, 4, 1))
        buf57 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_86, input_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf56, primals_143, primals_144, primals_145, primals_146, buf57, 256, grid=grid(256), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 4, 4, 4), (64, 16, 4, 1))
        buf59 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_89, input_90, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf58, primals_148, primals_149, primals_150, primals_151, buf55, buf59, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 4, 4, 4), (64, 16, 4, 1))
        buf61 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf60, primals_153, primals_154, primals_155, primals_156, buf61, 256, grid=grid(256), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 4, 4, 4), (64, 16, 4, 1))
        buf63 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_95, input_96, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_1.run(buf62, primals_158, primals_159, primals_160, primals_161, buf59, buf63, 256, grid=grid(256), stream=stream0)
    return (buf63, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_157, primals_158, primals_159, primals_160, primals_161, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
