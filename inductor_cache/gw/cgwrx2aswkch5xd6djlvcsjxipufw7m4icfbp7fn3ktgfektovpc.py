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


# kernel path: inductor_cache/ih/cihkpbfqn5fs3heh6xw2ooytbo7gccchlslqgbbpfxh7tvhs7zi4.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x2 = xindex // 512
    x3 = (xindex % 512)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ev/cevorbabdjcgtd4giqli7jdzt3qvddga3f2diakictsztogo6ysh.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_2
# Graph fragment:
#   %clone_2 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 16)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yp/cypupkid6ctx3bgfk73t7p3jqxwq64es2iakrvy5atpo752222hw.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_6
# Graph fragment:
#   %view_6 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [4, 8, 4, 16]), kwargs = {})
triton_poi_fused_view_2 = async_compile.triton('triton_poi_fused_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4rwnc3glb43ncdywbp6twctcur5oudmgeugse2rwkflbxku5abn.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_7
# Graph fragment:
#   %view_7 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [4, 8, 4, 16]), kwargs = {})
triton_poi_fused_view_3 = async_compile.triton('triton_poi_fused_view_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gg/cgg7vcdigmjvz7v2mlo3ezabciekpt4avrifw545r4bzgmcbtoz3.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_8
# Graph fragment:
#   %view_8 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_6, [4, 8, 4, 16]), kwargs = {})
triton_poi_fused_view_4 = async_compile.triton('triton_poi_fused_view_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4096 + x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t2/ct2qtmt2gttpmq7w4rspm2de72vmstcwvovyzpy7dbn3eua7edp3.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 4)
    x2 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 512*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jr/cjr6buc5bhqazwfibk34zs3akqourrppbn5mq4sno3v7gktsazp4.py
# Topologically Sorted Source Nodes: [add_1, x_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_1 => add_2
#   x_4 => add_3, add_4, clone_5, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_1, %view_10), kwargs = {})
#   %clone_5 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_5, %getitem_5), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_9), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_10), kwargs = {})
#   %div_18 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 128*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r2 + 128*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r2 + 128*x3), xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr4 + (r2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 128.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0078125
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r2 + 128*x3), tmp31, xmask)
    tl.store(out_ptr2 + (r2 + 128*x3), tmp35, xmask)
    tl.store(out_ptr3 + (x3), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hf/chfnkgntwmmgbf4eqkgx6s3qzgsfvztxmj2fr7u6wpptsqba7run.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_12,), kwargs = {})
#   %le_5 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_7 = async_compile.triton('triton_poi_fused_relu_threshold_backward_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_7(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, None)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/e3/ce3wrzyduojkzsph2y7xhues4l3ows4y2u44hhjywi4rb2pnnzga.py
# Topologically Sorted Source Nodes: [add_2, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_2 => add_5
#   x_6 => add_6, add_7, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_14), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_7), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_15), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_16), kwargs = {})
#   %div_17 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0078125
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp27, xmask)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp31, xmask)
    tl.store(out_ptr3 + (x0), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tp/ctpqyqdnrosn4cmcw7wftfccpew43hoqzakki7s4hamabgds6mdv.py
# Topologically Sorted Source Nodes: [add_12, x_21, mu, sub, pow_1, sigma, add_13, sqrt, x_22, mul, output], Original ATen: [aten.add, aten.native_layer_norm, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_12 => add_35
#   add_13 => add_38
#   mu => mean
#   mul => mul_24
#   output => add_39
#   pow_1 => pow_1
#   sigma => mean_1
#   sqrt => sqrt
#   sub => sub_12
#   x_21 => add_36, add_37, mul_22, mul_23, rsqrt_11, sub_11, var_mean_11
#   x_22 => div
# Graph fragment:
#   %add_35 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %view_89), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_35, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_11 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_36,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_35, %getitem_47), kwargs = {})
#   %mul_22 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_11), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %primals_75), kwargs = {})
#   %add_37 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %primals_76), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_37, [-1], True), kwargs = {})
#   %sub_12 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_37, %mean), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_12, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-05), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_12, %sqrt), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_77, %div), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %primals_78), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_11, 128), kwargs = {})
triton_per_fused_add_div_mean_mul_native_layer_norm_native_layer_norm_backward_pow_sqrt_sub_9 = async_compile.triton('triton_per_fused_add_div_mean_mul_native_layer_norm_native_layer_norm_backward_pow_sqrt_sub_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_native_layer_norm_native_layer_norm_backward_pow_sqrt_sub_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 6, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_native_layer_norm_native_layer_norm_backward_pow_sqrt_sub_9(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr5 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp36 = tmp35 / tmp22
    tmp37 = tmp31 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp41 = tl.where(xmask, tmp39, 0)
    tmp42 = tl.sum(tmp41, 1)[:, None]
    tmp43 = tmp42 / tmp22
    tmp44 = tmp43 + tmp24
    tmp45 = libdevice.sqrt(tmp44)
    tmp46 = 0.0078125
    tmp47 = tmp26 * tmp46
    tmp49 = tmp37 / tmp45
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp27, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp36, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp45, xmask)
    tl.store(out_ptr2 + (x0), tmp47, xmask)
    tl.store(out_ptr3 + (r1 + 128*x0), tmp52, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e6/ce6nx37aju6a6tzfbxyidy2wtwrqufukko5drajw6tv23vvxjnhq.py
# Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_25 => add_40
# Graph fragment:
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_93, %primals_82), kwargs = {})
triton_poi_fused_add_10 = async_compile.triton('triton_poi_fused_add_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82 = args
    args.clear()
    assert_size_stride(primals_1, (128, 4), (4, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (5000, 1, 128), (128, 640000, 1))
    assert_size_stride(primals_5, (384, ), (1, ))
    assert_size_stride(primals_6, (384, 128), (128, 1))
    assert_size_stride(primals_7, (128, 128), (128, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (2048, 128), (128, 1))
    assert_size_stride(primals_12, (2048, ), (1, ))
    assert_size_stride(primals_13, (128, 2048), (2048, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (384, ), (1, ))
    assert_size_stride(primals_18, (384, 128), (128, 1))
    assert_size_stride(primals_19, (128, 128), (128, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (2048, 128), (128, 1))
    assert_size_stride(primals_24, (2048, ), (1, ))
    assert_size_stride(primals_25, (128, 2048), (2048, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (384, ), (1, ))
    assert_size_stride(primals_30, (384, 128), (128, 1))
    assert_size_stride(primals_31, (128, 128), (128, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (2048, 128), (128, 1))
    assert_size_stride(primals_36, (2048, ), (1, ))
    assert_size_stride(primals_37, (128, 2048), (2048, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (384, 128), (128, 1))
    assert_size_stride(primals_43, (128, 128), (128, 1))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (2048, 128), (128, 1))
    assert_size_stride(primals_48, (2048, ), (1, ))
    assert_size_stride(primals_49, (128, 2048), (2048, 1))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (384, ), (1, ))
    assert_size_stride(primals_54, (384, 128), (128, 1))
    assert_size_stride(primals_55, (128, 128), (128, 1))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (2048, 128), (128, 1))
    assert_size_stride(primals_60, (2048, ), (1, ))
    assert_size_stride(primals_61, (128, 2048), (2048, 1))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (128, ), (1, ))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_66, (384, 128), (128, 1))
    assert_size_stride(primals_67, (128, 128), (128, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (2048, 128), (128, 1))
    assert_size_stride(primals_72, (2048, ), (1, ))
    assert_size_stride(primals_73, (128, 2048), (2048, 1))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (128, ), (1, ))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (1, 128), (128, 1))
    assert_size_stride(primals_80, (1, ), (1, ))
    assert_size_stride(primals_81, (4, 4), (4, 1))
    assert_size_stride(primals_82, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (4, 128), (1, 4), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(buf0, primals_2, primals_4, buf1, 2048, grid=grid(2048), stream=stream0)
        buf2 = empty_strided_cuda((16, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (16, 128), (128, 1), 0), reinterpret_tensor(primals_6, (128, 384), (1, 128), 0), out=buf2)
        buf3 = empty_strided_cuda((3, 4, 4, 128), (2048, 512, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf2, primals_5, buf3, 6144, grid=grid(6144), stream=stream0)
        del primals_5
        buf4 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2.run(buf3, buf4, 2048, grid=grid(2048), stream=stream0)
        buf5 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf3, buf5, 2048, grid=grid(2048), stream=stream0)
        buf6 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf3, buf6, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf4, buf5, buf6, None, True)
        buf8 = buf7[0]
        buf9 = buf7[1]
        buf10 = buf7[2]
        buf11 = buf7[3]
        del buf7
        buf12 = empty_strided_cuda((4, 4, 8, 16), (512, 128, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf8, buf12, 2048, grid=grid(2048), stream=stream0)
        buf13 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf12, (16, 128), (128, 1), 0), reinterpret_tensor(primals_7, (128, 128), (1, 128), 0), out=buf13)
        buf14 = reinterpret_tensor(buf13, (4, 4, 128), (512, 128, 1), 0); del buf13  # reuse
        buf18 = buf14; del buf14  # reuse
        buf19 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf178 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_1, x_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6.run(buf18, buf0, primals_2, primals_4, primals_8, primals_9, primals_10, buf19, buf178, 16, 128, grid=grid(16), stream=stream0)
        del buf0
        del primals_10
        del primals_2
        del primals_4
        del primals_8
        buf20 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf19, (16, 128), (128, 1), 0), reinterpret_tensor(primals_11, (128, 2048), (1, 128), 0), out=buf20)
        buf21 = reinterpret_tensor(buf20, (4, 4, 2048), (8192, 2048, 1), 0); del buf20  # reuse
        buf177 = empty_strided_cuda((4, 4, 2048), (8192, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_7.run(buf21, primals_12, buf177, 32768, grid=grid(32768), stream=stream0)
        del primals_12
        buf22 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf21, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_13, (2048, 128), (1, 2048), 0), out=buf22)
        buf26 = reinterpret_tensor(buf22, (4, 4, 128), (512, 128, 1), 0); del buf22  # reuse
        buf27 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf176 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_2, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf26, buf19, primals_14, primals_15, primals_16, buf27, buf176, 16, 128, grid=grid(16), stream=stream0)
        del primals_14
        del primals_16
        buf28 = reinterpret_tensor(buf3, (16, 384), (384, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf27, (16, 128), (128, 1), 0), reinterpret_tensor(primals_18, (128, 384), (1, 128), 0), out=buf28)
        buf29 = reinterpret_tensor(buf2, (3, 4, 4, 128), (2048, 512, 128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf28, primals_17, buf29, 6144, grid=grid(6144), stream=stream0)
        del primals_17
        buf30 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2.run(buf29, buf30, 2048, grid=grid(2048), stream=stream0)
        buf31 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf29, buf31, 2048, grid=grid(2048), stream=stream0)
        buf32 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf29, buf32, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf33 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf30, buf31, buf32, None, True)
        buf34 = buf33[0]
        buf35 = buf33[1]
        buf36 = buf33[2]
        buf37 = buf33[3]
        del buf33
        buf38 = empty_strided_cuda((4, 4, 8, 16), (512, 128, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf34, buf38, 2048, grid=grid(2048), stream=stream0)
        buf39 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf38, (16, 128), (128, 1), 0), reinterpret_tensor(primals_19, (128, 128), (1, 128), 0), out=buf39)
        buf43 = reinterpret_tensor(buf39, (4, 4, 128), (512, 128, 1), 0); del buf39  # reuse
        buf44 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf175 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_3, x_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf43, buf27, primals_20, primals_21, primals_22, buf44, buf175, 16, 128, grid=grid(16), stream=stream0)
        del primals_20
        del primals_22
        buf45 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf44, (16, 128), (128, 1), 0), reinterpret_tensor(primals_23, (128, 2048), (1, 128), 0), out=buf45)
        buf46 = reinterpret_tensor(buf45, (4, 4, 2048), (8192, 2048, 1), 0); del buf45  # reuse
        buf174 = empty_strided_cuda((4, 4, 2048), (8192, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_7.run(buf46, primals_24, buf174, 32768, grid=grid(32768), stream=stream0)
        del primals_24
        buf47 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf46, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_25, (2048, 128), (1, 2048), 0), out=buf47)
        buf51 = reinterpret_tensor(buf47, (4, 4, 128), (512, 128, 1), 0); del buf47  # reuse
        buf52 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf173 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_4, x_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf51, buf44, primals_26, primals_27, primals_28, buf52, buf173, 16, 128, grid=grid(16), stream=stream0)
        del primals_26
        del primals_28
        buf53 = reinterpret_tensor(buf29, (16, 384), (384, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf52, (16, 128), (128, 1), 0), reinterpret_tensor(primals_30, (128, 384), (1, 128), 0), out=buf53)
        buf54 = reinterpret_tensor(buf28, (3, 4, 4, 128), (2048, 512, 128, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf53, primals_29, buf54, 6144, grid=grid(6144), stream=stream0)
        del primals_29
        buf55 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2.run(buf54, buf55, 2048, grid=grid(2048), stream=stream0)
        buf56 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf54, buf56, 2048, grid=grid(2048), stream=stream0)
        buf57 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf54, buf57, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf58 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf55, buf56, buf57, None, True)
        buf59 = buf58[0]
        buf60 = buf58[1]
        buf61 = buf58[2]
        buf62 = buf58[3]
        del buf58
        buf63 = empty_strided_cuda((4, 4, 8, 16), (512, 128, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf59, buf63, 2048, grid=grid(2048), stream=stream0)
        buf64 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf63, (16, 128), (128, 1), 0), reinterpret_tensor(primals_31, (128, 128), (1, 128), 0), out=buf64)
        buf68 = reinterpret_tensor(buf64, (4, 4, 128), (512, 128, 1), 0); del buf64  # reuse
        buf69 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf172 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_5, x_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf68, buf52, primals_32, primals_33, primals_34, buf69, buf172, 16, 128, grid=grid(16), stream=stream0)
        del primals_32
        del primals_34
        buf70 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf69, (16, 128), (128, 1), 0), reinterpret_tensor(primals_35, (128, 2048), (1, 128), 0), out=buf70)
        buf71 = reinterpret_tensor(buf70, (4, 4, 2048), (8192, 2048, 1), 0); del buf70  # reuse
        buf171 = empty_strided_cuda((4, 4, 2048), (8192, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_7.run(buf71, primals_36, buf171, 32768, grid=grid(32768), stream=stream0)
        del primals_36
        buf72 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf71, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_37, (2048, 128), (1, 2048), 0), out=buf72)
        buf76 = reinterpret_tensor(buf72, (4, 4, 128), (512, 128, 1), 0); del buf72  # reuse
        buf77 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf170 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_6, x_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf76, buf69, primals_38, primals_39, primals_40, buf77, buf170, 16, 128, grid=grid(16), stream=stream0)
        del primals_38
        del primals_40
        buf78 = reinterpret_tensor(buf54, (16, 384), (384, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf77, (16, 128), (128, 1), 0), reinterpret_tensor(primals_42, (128, 384), (1, 128), 0), out=buf78)
        buf79 = reinterpret_tensor(buf53, (3, 4, 4, 128), (2048, 512, 128, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf78, primals_41, buf79, 6144, grid=grid(6144), stream=stream0)
        del primals_41
        buf80 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2.run(buf79, buf80, 2048, grid=grid(2048), stream=stream0)
        buf81 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf79, buf81, 2048, grid=grid(2048), stream=stream0)
        buf82 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf79, buf82, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf83 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf80, buf81, buf82, None, True)
        buf84 = buf83[0]
        buf85 = buf83[1]
        buf86 = buf83[2]
        buf87 = buf83[3]
        del buf83
        buf88 = empty_strided_cuda((4, 4, 8, 16), (512, 128, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf84, buf88, 2048, grid=grid(2048), stream=stream0)
        buf89 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf88, (16, 128), (128, 1), 0), reinterpret_tensor(primals_43, (128, 128), (1, 128), 0), out=buf89)
        buf93 = reinterpret_tensor(buf89, (4, 4, 128), (512, 128, 1), 0); del buf89  # reuse
        buf94 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf169 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_7, x_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf93, buf77, primals_44, primals_45, primals_46, buf94, buf169, 16, 128, grid=grid(16), stream=stream0)
        del primals_44
        del primals_46
        buf95 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf94, (16, 128), (128, 1), 0), reinterpret_tensor(primals_47, (128, 2048), (1, 128), 0), out=buf95)
        buf96 = reinterpret_tensor(buf95, (4, 4, 2048), (8192, 2048, 1), 0); del buf95  # reuse
        buf168 = empty_strided_cuda((4, 4, 2048), (8192, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_3], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_7.run(buf96, primals_48, buf168, 32768, grid=grid(32768), stream=stream0)
        del primals_48
        buf97 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf96, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_49, (2048, 128), (1, 2048), 0), out=buf97)
        buf101 = reinterpret_tensor(buf97, (4, 4, 128), (512, 128, 1), 0); del buf97  # reuse
        buf102 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf167 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_8, x_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf101, buf94, primals_50, primals_51, primals_52, buf102, buf167, 16, 128, grid=grid(16), stream=stream0)
        del primals_50
        del primals_52
        buf103 = reinterpret_tensor(buf79, (16, 384), (384, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf102, (16, 128), (128, 1), 0), reinterpret_tensor(primals_54, (128, 384), (1, 128), 0), out=buf103)
        buf104 = reinterpret_tensor(buf78, (3, 4, 4, 128), (2048, 512, 128, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf103, primals_53, buf104, 6144, grid=grid(6144), stream=stream0)
        del primals_53
        buf105 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2.run(buf104, buf105, 2048, grid=grid(2048), stream=stream0)
        buf106 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf104, buf106, 2048, grid=grid(2048), stream=stream0)
        buf107 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf104, buf107, 2048, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf108 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf105, buf106, buf107, None, True)
        buf109 = buf108[0]
        buf110 = buf108[1]
        buf111 = buf108[2]
        buf112 = buf108[3]
        del buf108
        buf113 = empty_strided_cuda((4, 4, 8, 16), (512, 128, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf109, buf113, 2048, grid=grid(2048), stream=stream0)
        buf114 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf113, (16, 128), (128, 1), 0), reinterpret_tensor(primals_55, (128, 128), (1, 128), 0), out=buf114)
        buf118 = reinterpret_tensor(buf114, (4, 4, 128), (512, 128, 1), 0); del buf114  # reuse
        buf119 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf166 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_9, x_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf118, buf102, primals_56, primals_57, primals_58, buf119, buf166, 16, 128, grid=grid(16), stream=stream0)
        del primals_56
        del primals_58
        buf120 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf119, (16, 128), (128, 1), 0), reinterpret_tensor(primals_59, (128, 2048), (1, 128), 0), out=buf120)
        buf121 = reinterpret_tensor(buf120, (4, 4, 2048), (8192, 2048, 1), 0); del buf120  # reuse
        buf165 = empty_strided_cuda((4, 4, 2048), (8192, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_4], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_7.run(buf121, primals_60, buf165, 32768, grid=grid(32768), stream=stream0)
        del primals_60
        buf122 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf121, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_61, (2048, 128), (1, 2048), 0), out=buf122)
        buf126 = reinterpret_tensor(buf122, (4, 4, 128), (512, 128, 1), 0); del buf122  # reuse
        buf127 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf164 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_10, x_18], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf126, buf119, primals_62, primals_63, primals_64, buf127, buf164, 16, 128, grid=grid(16), stream=stream0)
        del primals_62
        del primals_64
        buf128 = reinterpret_tensor(buf104, (16, 384), (384, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf127, (16, 128), (128, 1), 0), reinterpret_tensor(primals_66, (128, 384), (1, 128), 0), out=buf128)
        buf129 = reinterpret_tensor(buf103, (3, 4, 4, 128), (2048, 512, 128, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf128, primals_65, buf129, 6144, grid=grid(6144), stream=stream0)
        del buf128
        del primals_65
        buf130 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_2.run(buf129, buf130, 2048, grid=grid(2048), stream=stream0)
        buf131 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf129, buf131, 2048, grid=grid(2048), stream=stream0)
        buf132 = empty_strided_cuda((4, 8, 4, 16), (128, 16, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf129, buf132, 2048, grid=grid(2048), stream=stream0)
        del buf129
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf133 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf130, buf131, buf132, None, True)
        buf134 = buf133[0]
        buf135 = buf133[1]
        buf136 = buf133[2]
        buf137 = buf133[3]
        del buf133
        buf138 = empty_strided_cuda((4, 4, 8, 16), (512, 128, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf134, buf138, 2048, grid=grid(2048), stream=stream0)
        buf139 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf138, (16, 128), (128, 1), 0), reinterpret_tensor(primals_67, (128, 128), (1, 128), 0), out=buf139)
        buf143 = reinterpret_tensor(buf139, (4, 4, 128), (512, 128, 1), 0); del buf139  # reuse
        buf144 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        buf163 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_11, x_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf143, buf127, primals_68, primals_69, primals_70, buf144, buf163, 16, 128, grid=grid(16), stream=stream0)
        del primals_68
        del primals_70
        buf145 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf144, (16, 128), (128, 1), 0), reinterpret_tensor(primals_71, (128, 2048), (1, 128), 0), out=buf145)
        buf146 = reinterpret_tensor(buf145, (4, 4, 2048), (8192, 2048, 1), 0); del buf145  # reuse
        buf162 = empty_strided_cuda((4, 4, 2048), (8192, 2048, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_5], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_7.run(buf146, primals_72, buf162, 32768, grid=grid(32768), stream=stream0)
        del primals_72
        buf147 = empty_strided_cuda((16, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf146, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_73, (2048, 128), (1, 2048), 0), out=buf147)
        buf151 = reinterpret_tensor(buf147, (4, 4, 128), (512, 128, 1), 0); del buf147  # reuse
        buf152 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf153 = reinterpret_tensor(buf152, (4, 4, 1), (4, 1, 1), 0); del buf152  # reuse
        buf154 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf155 = reinterpret_tensor(buf154, (4, 4, 1), (4, 1, 1), 0); del buf154  # reuse
        buf161 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        buf156 = empty_strided_cuda((4, 4, 128), (512, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_12, x_21, mu, sub, pow_1, sigma, add_13, sqrt, x_22, mul, output], Original ATen: [aten.add, aten.native_layer_norm, aten.mean, aten.sub, aten.pow, aten.sqrt, aten.div, aten.mul, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_native_layer_norm_native_layer_norm_backward_pow_sqrt_sub_9.run(buf151, buf153, buf155, buf144, primals_74, primals_75, primals_76, primals_77, primals_78, buf161, buf156, 16, 128, grid=grid(16), stream=stream0)
        del primals_74
        del primals_78
        buf158 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_80, reinterpret_tensor(buf156, (16, 128), (128, 1), 0), reinterpret_tensor(primals_79, (128, 1), (1, 128), 0), alpha=1, beta=1, out=buf158)
        del primals_80
        buf159 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf158, (4, 4), (1, 4), 0), reinterpret_tensor(primals_81, (4, 4), (1, 4), 0), out=buf159)
        buf160 = reinterpret_tensor(buf159, (4, 1, 4), (4, 4, 1), 0); del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_10.run(buf160, primals_82, 16, grid=grid(16), stream=stream0)
        del primals_82
    return (reinterpret_tensor(buf160, (4, 4), (4, 1), 0), primals_3, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_76, primals_77, reinterpret_tensor(buf1, (16, 128), (128, 1), 0), buf4, buf5, buf6, buf8, buf9, buf10, buf11, reinterpret_tensor(buf12, (16, 128), (128, 1), 0), buf18, reinterpret_tensor(buf19, (16, 128), (128, 1), 0), reinterpret_tensor(buf21, (16, 2048), (2048, 1), 0), buf26, reinterpret_tensor(buf27, (16, 128), (128, 1), 0), buf30, buf31, buf32, buf34, buf35, buf36, buf37, reinterpret_tensor(buf38, (16, 128), (128, 1), 0), buf43, reinterpret_tensor(buf44, (16, 128), (128, 1), 0), reinterpret_tensor(buf46, (16, 2048), (2048, 1), 0), buf51, reinterpret_tensor(buf52, (16, 128), (128, 1), 0), buf55, buf56, buf57, buf59, buf60, buf61, buf62, reinterpret_tensor(buf63, (16, 128), (128, 1), 0), buf68, reinterpret_tensor(buf69, (16, 128), (128, 1), 0), reinterpret_tensor(buf71, (16, 2048), (2048, 1), 0), buf76, reinterpret_tensor(buf77, (16, 128), (128, 1), 0), buf80, buf81, buf82, buf84, buf85, buf86, buf87, reinterpret_tensor(buf88, (16, 128), (128, 1), 0), buf93, reinterpret_tensor(buf94, (16, 128), (128, 1), 0), reinterpret_tensor(buf96, (16, 2048), (2048, 1), 0), buf101, reinterpret_tensor(buf102, (16, 128), (128, 1), 0), buf105, buf106, buf107, buf109, buf110, buf111, buf112, reinterpret_tensor(buf113, (16, 128), (128, 1), 0), buf118, reinterpret_tensor(buf119, (16, 128), (128, 1), 0), reinterpret_tensor(buf121, (16, 2048), (2048, 1), 0), buf126, reinterpret_tensor(buf127, (16, 128), (128, 1), 0), buf130, buf131, buf132, buf134, buf135, buf136, buf137, reinterpret_tensor(buf138, (16, 128), (128, 1), 0), buf143, reinterpret_tensor(buf144, (16, 128), (128, 1), 0), reinterpret_tensor(buf146, (16, 2048), (2048, 1), 0), buf151, buf153, buf155, reinterpret_tensor(buf156, (16, 128), (128, 1), 0), reinterpret_tensor(primals_81, (4, 4), (1, 4), 0), reinterpret_tensor(buf158, (4, 4), (1, 4), 0), primals_79, buf161, primals_73, buf162, primals_71, buf163, primals_67, primals_66, buf164, primals_61, buf165, primals_59, buf166, primals_55, primals_54, buf167, primals_49, buf168, primals_47, buf169, primals_43, primals_42, buf170, primals_37, buf171, primals_35, buf172, primals_31, primals_30, buf173, primals_25, buf174, primals_23, buf175, primals_19, primals_18, buf176, primals_13, buf177, primals_11, buf178, primals_7, primals_6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((5000, 1, 128), (128, 640000, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
