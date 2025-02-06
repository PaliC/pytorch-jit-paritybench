# AOT ID: ['13_forward']
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


# kernel path: inductor_cache/kj/ckjif6retnxnuvnck54sexulvfnrmjoaiqswc4kbepbrtcfu5hzx.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/yq/cyqlzdfxoe2qowddqt2mv7uxpqc5vycfb6osybb2a3slp26gfyvl.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 75*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/yx/cyxcpgov7jyoqlkfr6msld55v3yrsbm6lxtvvt7qdp7le7cquxyy.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 25
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 3200*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ep/cepg6jvsx6ahpru6lwojwbe7duyajgtjxvjofsgmiqhfakbsjp4t.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out => convolution
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [2, 2], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/jp/cjpcv2qed2fn2qz57rlhafmrq2cqgdvsa7avfxxqi35trludlmzu.py
# Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_4 = async_compile.triton('triton_per_fused_native_group_norm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 5, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_4(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*(((r3 + 128*x1) % 4096)) + 524288*x2 + ((r3 + 128*x1) // 4096)), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp8, None)
    tl.store(out_ptr1 + (x4), tmp13, None)
    tl.store(out_ptr2 + (x4), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/ab/cabyc2q3cbaianihsuagy3gki75plxphce35web7phajsj7b3xyn.py
# Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_5 = async_compile.triton('triton_per_fused_native_group_norm_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 4096*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*r2 + 4096*x1), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + 64*r2 + 4096*x1), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bd/cbduzbbpezz6ckvxgwqoh25p4epldvmktal5rk33yhmyzpujo2c7.py
# Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_per_fused_native_group_norm_6 = async_compile.triton('triton_per_fused_native_group_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 16*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uk/cukc67dgl6spgbzveih7nvz5fbheggnolcvdrv7j4uzbeftmcxy6.py
# Topologically Sorted Source Nodes: [group_norm, relu], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm => add_1, mul_1
#   relu => relu
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused_native_group_norm_relu_7 = async_compile.triton('triton_poi_fused_native_group_norm_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 524288
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (4*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (4*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/bv/cbvh7v2yax33nr637ackdkazcfqzmao6x3hyp6qoh77vqnp3rny6.py
# Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_2 => var_mean_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_group_norm_8 = async_compile.triton('triton_per_fused_native_group_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_8(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r3 = rindex
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*(((r3 + 128*x1) % 4096)) + 524288*x2 + ((r3 + 128*x1) // 4096)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (2*x0 + 128*(((r3 + 128*x1) % 4096)) + 524288*x2 + ((r3 + 128*x1) // 4096)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp10, None)
    tl.store(out_ptr1 + (x4), tmp15, None)
    tl.store(out_ptr2 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/kf/ckfewxgbrrxea4bccsq6twgx7amfu437v7kihocczz22amtv3jgl.py
# Topologically Sorted Source Nodes: [group_norm_2, relu_2], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   group_norm_2 => add_6, mul_5
#   relu_2 => relu_2
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze_17), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_14), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused_native_group_norm_relu_9 = async_compile.triton('triton_poi_fused_native_group_norm_relu_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 524288
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (4*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (4*x2 + (x0 // 32)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 131072.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/b7/cb7jma3alaeotyt3a3g66btyqt356jeoe2t7oqgyven4dqjex3mp.py
# Topologically Sorted Source Nodes: [out_1, out_2, out_3], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   out_1 => add_4
#   out_2 => add_9
#   out_3 => mean
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_4), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_9, [-1, -2], True), kwargs = {})
triton_red_fused_add_mean_10 = async_compile.triton('triton_red_fused_add_mean_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mean_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 128)
    x1 = xindex // 128
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 128*r2 + 16384*x1), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + 128*r2 + 16384*x1), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + 128*r2 + 16384*x1), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/l4/cl4c2s4ruj6sngsjigdyyl4y4ri7zev5arqdika3ikagpeflhkwd.py
# Topologically Sorted Source Nodes: [out_1, out_2, out_3, relu_4], Original ATen: [aten.add, aten.mean, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_4
#   out_2 => add_9
#   out_3 => mean
#   relu_4 => relu_4
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_4), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_9, [-1, -2], True), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_8,), kwargs = {})
triton_per_fused_add_mean_relu_11 = async_compile.triton('triton_per_fused_add_mean_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 32},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_relu_11(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r2 + 4096*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 4096.0
    tmp6 = tmp4 / tmp5
    tmp7 = tl.full([1, 1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (128, 3, 5, 5), (75, 25, 5, 1))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, 128, 5, 5), (3200, 25, 5, 1))
    assert_size_stride(primals_16, (1, 128), (128, 1))
    assert_size_stride(primals_17, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((128, 3, 5, 5), (75, 1, 15, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 384, 25, grid=grid(384, 25), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_6, buf2, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_6
        buf3 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_9, buf3, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_9
        buf4 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_12, buf4, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_12
        buf5 = empty_strided_cuda((128, 128, 5, 5), (3200, 1, 640, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_15, buf5, 16384, 25, grid=grid(16384, 25), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 128, 64, 64), (524288, 1, 8192, 128))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf7, primals_3, 2097152, grid=grid(2097152), stream=stream0)
        del primals_3
        buf8 = empty_strided_cuda((4, 4, 1, 1, 16, 64), (4096, 16, 16384, 16384, 1, 64), torch.float32)
        buf9 = empty_strided_cuda((4, 4, 1, 1, 16, 64), (4096, 16, 16384, 16384, 1, 64), torch.float32)
        buf10 = empty_strided_cuda((4, 4, 1, 1, 16, 64), (4096, 16, 16384, 16384, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf7, buf8, buf9, buf10, 16384, 128, grid=grid(16384), stream=stream0)
        buf11 = empty_strided_cuda((4, 4, 1, 1, 16), (64, 16, 256, 256, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 4, 1, 1, 16), (64, 16, 256, 256, 1), torch.float32)
        buf13 = empty_strided_cuda((4, 4, 1, 1, 16), (64, 16, 256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf8, buf9, buf10, buf11, buf12, buf13, 256, 64, grid=grid(256), stream=stream0)
        buf14 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf15 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf17 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf11, buf12, buf13, buf14, buf15, buf17, 16, 16, grid=grid(16), stream=stream0)
        buf18 = empty_strided_cuda((4, 128, 64, 64), (524288, 1, 8192, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm, relu], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_7.run(buf7, buf14, buf15, primals_4, primals_5, buf18, 2097152, grid=grid(2097152), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [dx], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, buf2, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 128, 64, 64), (524288, 1, 8192, 128))
        buf20 = buf9; del buf9  # reuse
        buf21 = buf8; del buf8  # reuse
        buf22 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf19, buf20, buf21, buf22, 16384, 128, grid=grid(16384), stream=stream0)
        buf23 = buf13; del buf13  # reuse
        buf24 = buf12; del buf12  # reuse
        buf25 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf20, buf21, buf22, buf23, buf24, buf25, 256, 64, grid=grid(256), stream=stream0)
        buf26 = buf15; del buf15  # reuse
        buf27 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf29 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf23, buf24, buf25, buf26, buf27, buf29, 16, 16, grid=grid(16), stream=stream0)
        buf30 = empty_strided_cuda((4, 128, 64, 64), (524288, 1, 8192, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_1, relu_1], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_7.run(buf19, buf26, buf27, primals_7, primals_8, buf30, 2097152, grid=grid(2097152), stream=stream0)
        del primals_8
        # Topologically Sorted Source Nodes: [dx_1], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, buf3, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 128, 64, 64), (524288, 1, 8192, 128))
        buf32 = buf22; del buf22  # reuse
        buf33 = buf21; del buf21  # reuse
        buf34 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_8.run(buf7, buf31, buf32, buf33, buf34, 16384, 128, grid=grid(16384), stream=stream0)
        buf35 = buf25; del buf25  # reuse
        buf36 = buf24; del buf24  # reuse
        buf37 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf32, buf33, buf34, buf35, buf36, buf37, 256, 64, grid=grid(256), stream=stream0)
        buf38 = buf27; del buf27  # reuse
        buf39 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf41 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf35, buf36, buf37, buf38, buf39, buf41, 16, 16, grid=grid(16), stream=stream0)
        buf42 = empty_strided_cuda((4, 128, 64, 64), (524288, 1, 8192, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_2, relu_2], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_9.run(buf7, buf31, buf38, buf39, primals_10, primals_11, buf42, 2097152, grid=grid(2097152), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [dx_2], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, buf4, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 128, 64, 64), (524288, 1, 8192, 128))
        buf44 = buf34; del buf34  # reuse
        buf45 = buf33; del buf33  # reuse
        buf46 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf43, buf44, buf45, buf46, 16384, 128, grid=grid(16384), stream=stream0)
        buf47 = buf37; del buf37  # reuse
        buf48 = buf36; del buf36  # reuse
        buf49 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf44, buf45, buf46, buf47, buf48, buf49, 256, 64, grid=grid(256), stream=stream0)
        del buf44
        del buf45
        buf50 = buf39; del buf39  # reuse
        buf51 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf53 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_3], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_6.run(buf47, buf48, buf49, buf50, buf51, buf53, 16, 16, grid=grid(16), stream=stream0)
        del buf47
        del buf48
        del buf49
        buf54 = empty_strided_cuda((4, 128, 64, 64), (524288, 1, 8192, 128), torch.float32)
        # Topologically Sorted Source Nodes: [group_norm_3, relu_3], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_7.run(buf43, buf50, buf51, primals_13, primals_14, buf54, 2097152, grid=grid(2097152), stream=stream0)
        del buf51
        del primals_14
        # Topologically Sorted Source Nodes: [dx_3], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, buf5, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 128, 64, 64), (524288, 1, 8192, 128))
        buf56 = reinterpret_tensor(buf46, (4, 128, 1, 1, 32), (4096, 1, 16384, 16384, 128), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [out_1, out_2, out_3], Original ATen: [aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mean_10.run(buf7, buf31, buf55, buf56, 16384, 128, grid=grid(16384), stream=stream0)
        del buf55
        buf57 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf58 = reinterpret_tensor(buf57, (4, 128), (128, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [out_1, out_2, out_3, relu_4], Original ATen: [aten.add, aten.mean, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_relu_11.run(buf58, buf56, 512, 32, grid=grid(512), stream=stream0)
        del buf56
        buf60 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, buf58, reinterpret_tensor(primals_16, (128, 1), (1, 128), 0), alpha=1, beta=1, out=buf60)
        del primals_17
    return (buf60, buf0, buf1, primals_4, buf2, primals_7, buf3, primals_10, buf4, primals_13, buf5, buf7, reinterpret_tensor(buf14, (4, 4), (4, 1), 0), reinterpret_tensor(buf17, (4, 4), (4, 1), 0), buf18, buf19, reinterpret_tensor(buf26, (4, 4), (4, 1), 0), reinterpret_tensor(buf29, (4, 4), (4, 1), 0), buf30, buf31, reinterpret_tensor(buf38, (4, 4), (4, 1), 0), reinterpret_tensor(buf41, (4, 4), (4, 1), 0), buf42, buf43, reinterpret_tensor(buf50, (4, 4), (4, 1), 0), reinterpret_tensor(buf53, (4, 4), (4, 1), 0), buf54, buf58, primals_16, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 3, 5, 5), (75, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, 128, 5, 5), (3200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
