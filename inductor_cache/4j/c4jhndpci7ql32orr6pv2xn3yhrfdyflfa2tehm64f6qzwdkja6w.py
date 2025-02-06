# AOT ID: ['7_forward']
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


# kernel path: inductor_cache/bq/cbqdb4r55v4dfu7yp23w475k3vxnxmyd7uikkm7k555tgnba5n6g.py
# Topologically Sorted Source Nodes: [mean, x], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean
#   x => mean_1
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%primals_1, [-1]), kwargs = {})
#   %mean_1 : [num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%mean, [-1]), kwargs = {})
triton_poi_fused_mean_0 = async_compile.triton('triton_poi_fused_mean_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 16*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 16*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + 16*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (6 + 16*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (7 + 16*x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (8 + 16*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (10 + 16*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (11 + 16*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (12 + 16*x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (14 + 16*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (15 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 / tmp7
    tmp17 = tmp8 + tmp16
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 / tmp7
    tmp26 = tmp17 + tmp25
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33 / tmp7
    tmp35 = tmp26 + tmp34
    tmp36 = tmp35 / tmp7
    tl.store(out_ptr0 + (x0 + 161*x1), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z4/cz4trrbegroqd3vpmedymrjkhdol3kaarqcend6fugyr4bijgjne.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_for_fused_1 = async_compile.triton('triton_for_fused_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'kernel_name': 'triton_for_fused_1', 'mutated_arg_names': [], 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_for_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2):
    pid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    num_xblocks_0 = tl.cdiv(576, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(40, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(12, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        xnumel = 576
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = (xindex % 144)
        x1 = xindex // 144
        tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (x0 + 161*x1), tmp0, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        xnumel = 40
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = (xindex % 10)
        x3 = xindex // 10
        tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
        tl.store(out_ptr1 + (x2 + 161*x3), tmp1, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        xnumel = 12
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = (xindex % 3)
        x5 = xindex // 3
        tmp2 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
        tl.store(out_ptr2 + (x4 + 161*x5), tmp2, xmask)
    else:
        pass
''', device_str='cuda')


# kernel path: inductor_cache/rh/crhdteesukvl5bcghqy5kbwu3tkfiztx4uwssiksrnxccie4zedb.py
# Topologically Sorted Source Nodes: [xc_5], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   xc_5 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mean_1, %add, %add_1, %add_2], 1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 644
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 161)
    x1 = xindex // 161
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (161*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 148, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (144*x1 + ((-4) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-4) + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.load(in_ptr3 + ((-4) + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp9, tmp14, tmp15)
    tmp17 = tmp0 >= tmp7
    tmp18 = tl.full([1], 158, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tl.load(in_ptr4 + (10*x1 + ((-148) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr5 + ((-148) + x0), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 + tmp22
    tmp24 = tl.load(in_ptr6 + ((-148) + x0), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp20, tmp25, tmp26)
    tmp28 = tmp0 >= tmp18
    tmp29 = tl.full([1], 161, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tl.load(in_ptr7 + (3*x1 + ((-158) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr8 + ((-158) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.load(in_ptr9 + ((-158) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp28, tmp35, tmp36)
    tmp38 = tl.where(tmp20, tmp27, tmp37)
    tmp39 = tl.where(tmp9, tmp16, tmp38)
    tmp40 = tl.where(tmp4, tmp5, tmp39)
    tl.store(out_ptr0 + (x2), tmp40, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/km/ckmfmmiiosvr2msuaftf64dzad6jmo3756elwtxsukhdq5iva4hv.py
# Topologically Sorted Source Nodes: [xc_10], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   xc_10 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mean_1, %add_3, %add_4, %add_5], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 644
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 161)
    x1 = xindex // 161
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (161*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 148, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (144*x1 + ((-4) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + ((-4) + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.load(in_ptr3 + (144*x1 + ((-4) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp13 + tmp11
    tmp15 = tl.load(in_ptr4 + ((-4) + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp12 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp9, tmp17, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tl.full([1], 158, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr5 + (10*x1 + ((-148) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr6 + ((-148) + x0), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tl.load(in_ptr7 + (10*x1 + ((-148) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp27 + tmp25
    tmp29 = tl.load(in_ptr8 + ((-148) + x0), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = tmp26 + tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp23, tmp31, tmp32)
    tmp34 = tmp0 >= tmp21
    tmp35 = tl.full([1], 161, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr9 + (3*x1 + ((-158) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr10 + ((-158) + x0), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.load(in_ptr11 + (3*x1 + ((-158) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 + tmp38
    tmp42 = tl.load(in_ptr12 + ((-158) + x0), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tmp39 + tmp43
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp34, tmp44, tmp45)
    tmp47 = tl.where(tmp23, tmp33, tmp46)
    tmp48 = tl.where(tmp9, tmp19, tmp47)
    tmp49 = tl.where(tmp4, tmp5, tmp48)
    tl.store(out_ptr0 + (x2), tmp49, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tr/ctr2cx6zxa3ejt4f5ubl43xj6eb7kk7lzmldbeucul4mertmpstm.py
# Topologically Sorted Source Nodes: [linear_3, pred_shape, linear_8, pred_shape_1, linear_13, pred_shape_2], Original ATen: [aten.addmm, aten.add]
# Source node to ATen node mapping:
#   linear_13 => add_tensor_1
#   linear_3 => add_tensor_7
#   linear_8 => add_tensor_4
#   pred_shape => add_1
#   pred_shape_1 => add_4
#   pred_shape_2 => add_7
# Graph fragment:
#   %add_tensor_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_7, %primals_12), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_7, %expand_1), kwargs = {})
#   %add_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %primals_12), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_4, %add_1), kwargs = {})
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_12), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_1, %add_4), kwargs = {})
triton_poi_fused_add_addmm_4 = async_compile.triton('triton_poi_fused_add_addmm_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 10)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 + tmp1
    tmp8 = tmp6 + tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = tmp2 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p6/cp6752ijb7oktmlcas5llhmdyi7fofi5mxeqq43r4lmacolk4lkp.py
# Topologically Sorted Source Nodes: [linear_4, pred_cam, linear_9, pred_cam_1, linear_14, pred_cam_2], Original ATen: [aten.addmm, aten.add]
# Source node to ATen node mapping:
#   linear_14 => add_tensor
#   linear_4 => add_tensor_6
#   linear_9 => add_tensor_3
#   pred_cam => add_2
#   pred_cam_1 => add_5
#   pred_cam_2 => add_8
# Graph fragment:
#   %add_tensor_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_6, %primals_14), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_6, %expand_2), kwargs = {})
#   %add_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_3, %primals_14), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_3, %add_2), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_14), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor, %add_5), kwargs = {})
triton_poi_fused_add_addmm_5 = async_compile.triton('triton_poi_fused_add_addmm_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 3)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 + tmp1
    tmp8 = tmp6 + tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = tmp2 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l3/cl3f62qidus2eqdmwvvmtnusydg6jsdz3qtllho5abubn5ag2yh4.py
# Topologically Sorted Source Nodes: [linear_2, pred_pose, linear_7, pred_pose_1, linear_12, pred_pose_2], Original ATen: [aten.addmm, aten.add]
# Source node to ATen node mapping:
#   linear_12 => add_tensor_2
#   linear_2 => add_tensor_8
#   linear_7 => add_tensor_5
#   pred_pose => add
#   pred_pose_1 => add_3
#   pred_pose_2 => add_6
# Graph fragment:
#   %add_tensor_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_8, %primals_10), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_8, %expand), kwargs = {})
#   %add_tensor_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_5, %primals_10), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_5, %add), kwargs = {})
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %primals_10), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_2, %add_3), kwargs = {})
triton_poi_fused_add_addmm_6 = async_compile.triton('triton_poi_fused_add_addmm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 + tmp1
    tmp8 = tmp6 + tmp7
    tmp9 = tmp4 + tmp8
    tmp10 = tmp2 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s4/cs4bc3jpymx5nayaa55njb73ezj5tmfmcto5l3dwbslm665iyui2.py
# Topologically Sorted Source Nodes: [b1], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   b1 => div
# Graph fragment:
#   %div : [num_users=6] = call_function[target=torch.ops.aten.div.Tensor](args = (%select, %expand_3), kwargs = {})
triton_poi_fused_div_7 = async_compile.triton('triton_poi_fused_div_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_ptr0 + (2*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (6*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 6*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (4 + 6*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = 1e-12
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = tmp0 / tmp11
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rl/crlu4wdcm6xdpmenosmyzb7lxpqxoopusbhsp55rhrxmltloacbw.py
# Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   einsum => bmm
# Graph fragment:
#   %bmm : [num_users=2] = call_function[target=torch.ops.aten.bmm.default](args = (%view_1, %view_2), kwargs = {})
triton_poi_fused_bmm_8 = async_compile.triton('triton_poi_fused_bmm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h4/ch45oxjyv5jnbq6iu6cgfnsdptwjmpxy7usro3iw3wkifrwwonku.py
# Topologically Sorted Source Nodes: [mul, sub], Original ATen: [aten.mul, aten.sub]
# Source node to ATen node mapping:
#   mul => mul
#   sub => sub
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %div), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_1, %mul), kwargs = {})
triton_poi_fused_mul_sub_9 = async_compile.triton('triton_poi_fused_mul_sub_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sub_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sub_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 3
    tmp0 = tl.load(in_ptr0 + (1 + 2*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ts/cts7dbgxpgavahd5xjhzk6xdsvi7a6oktufho6id3uaigjzbbh46.py
# Topologically Sorted Source Nodes: [b3], Original ATen: [aten.linalg_cross]
# Source node to ATen node mapping:
#   b3 => index
# Graph fragment:
#   %index : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%div, [None, %remainder]), kwargs = {})
triton_poi_fused_linalg_cross_10 = async_compile.triton('triton_poi_fused_linalg_cross_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_linalg_cross_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_linalg_cross_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x1 = xindex // 3
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x1 + (((1 + x0) % 3))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fq/cfq3thvwx2c2xndz5dozjyukr4erzugyvck52r5festrq5uejzdq.py
# Topologically Sorted Source Nodes: [b3], Original ATen: [aten.linalg_cross]
# Source node to ATen node mapping:
#   b3 => index_2
# Graph fragment:
#   %index_2 : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%div, [None, %remainder_1]), kwargs = {})
triton_poi_fused_linalg_cross_11 = async_compile.triton('triton_poi_fused_linalg_cross_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_linalg_cross_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_linalg_cross_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x1 = xindex // 3
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (3*x1 + (((2 + x0) % 3))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l2/cl27brnphirf43rwxzdlvqub4qfeov3fmqqjzhdk45o7g7zowtgg.py
# Topologically Sorted Source Nodes: [stack], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   stack => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_1, %unsqueeze_2, %unsqueeze_3], -1), kwargs = {})
triton_poi_fused_stack_12 = async_compile.triton('triton_poi_fused_stack_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x3 = xindex // 3
    x2 = xindex // 9
    x1 = ((xindex // 3) % 3)
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x3), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + (3*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 * tmp11
    tmp13 = tl.load(in_ptr1 + (1 + 3*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 + tmp14
    tmp16 = tl.load(in_ptr1 + (2 + 3*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp20 = 1e-12
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = tmp10 / tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp9, tmp22, tmp23)
    tmp25 = tmp0 >= tmp7
    tmp26 = tl.full([1], 3, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr2 + (x3), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr1 + (3*x2 + (((2 + x1) % 3))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr1 + (3*x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 * tmp30
    tmp32 = tl.load(in_ptr1 + (1 + 3*x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 + tmp33
    tmp35 = tl.load(in_ptr1 + (2 + 3*x2), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 * tmp35
    tmp37 = tmp34 + tmp36
    tmp38 = libdevice.sqrt(tmp37)
    tmp39 = 1e-12
    tmp40 = triton_helpers.maximum(tmp38, tmp39)
    tmp41 = tmp29 / tmp40
    tmp42 = tmp28 * tmp41
    tmp43 = tl.load(in_ptr3 + (x3), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr1 + (3*x2 + (((1 + x1) % 3))), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 / tmp40
    tmp46 = tmp43 * tmp45
    tmp47 = tmp42 - tmp46
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp25, tmp47, tmp48)
    tmp50 = tl.where(tmp9, tmp24, tmp49)
    tmp51 = tl.where(tmp4, tmp5, tmp50)
    tl.store(out_ptr0 + (x5), tmp51, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (1, 144), (144, 1))
    assert_size_stride(primals_3, (1, 10), (10, 1))
    assert_size_stride(primals_4, (1, 3), (3, 1))
    assert_size_stride(primals_5, (1024, 161), (161, 1))
    assert_size_stride(primals_6, (1024, ), (1, ))
    assert_size_stride(primals_7, (1024, 1024), (1024, 1))
    assert_size_stride(primals_8, (1024, ), (1, ))
    assert_size_stride(primals_9, (144, 1024), (1024, 1))
    assert_size_stride(primals_10, (144, ), (1, ))
    assert_size_stride(primals_11, (10, 1024), (1024, 1))
    assert_size_stride(primals_12, (10, ), (1, ))
    assert_size_stride(primals_13, (3, 1024), (1024, 1))
    assert_size_stride(primals_14, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((4, 161), (161, 1), torch.float32)
        buf0 = reinterpret_tensor(buf4, (4, 4), (161, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [mean, x], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_0.run(primals_1, buf0, 16, grid=grid(16), stream=stream0)
        del primals_1
        buf1 = reinterpret_tensor(buf4, (4, 144), (161, 1), 4)  # alias
        buf2 = reinterpret_tensor(buf4, (4, 10), (161, 1), 148)  # alias
        buf3 = reinterpret_tensor(buf4, (4, 3), (161, 1), 158)  # alias
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_for_fused_1.run(primals_2, primals_3, primals_4, buf1, buf2, buf3, grid=(3, 1, 1), stream=stream0)
        buf5 = empty_strided_cuda((4, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xc_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, buf4, reinterpret_tensor(primals_5, (161, 1024), (1, 161), 0), alpha=1, beta=1, out=buf5)
        buf6 = empty_strided_cuda((4, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xc_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf5, reinterpret_tensor(primals_7, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf6)
        buf7 = empty_strided_cuda((4, 144), (144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_9, (1024, 144), (1, 1024), 0), out=buf7)
        buf8 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_11, (1024, 10), (1, 1024), 0), out=buf8)
        buf9 = empty_strided_cuda((4, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf6, reinterpret_tensor(primals_13, (1024, 3), (1, 1024), 0), out=buf9)
        buf10 = empty_strided_cuda((4, 161), (161, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xc_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf0, buf7, primals_10, primals_2, buf8, primals_12, primals_3, buf9, primals_14, primals_4, buf10, 644, grid=grid(644), stream=stream0)
        buf11 = empty_strided_cuda((4, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xc_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, buf10, reinterpret_tensor(primals_5, (161, 1024), (1, 161), 0), alpha=1, beta=1, out=buf11)
        buf12 = empty_strided_cuda((4, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xc_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf11, reinterpret_tensor(primals_7, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf12)
        buf13 = empty_strided_cuda((4, 144), (144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.mm(buf12, reinterpret_tensor(primals_9, (1024, 144), (1, 1024), 0), out=buf13)
        buf14 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.mm(buf12, reinterpret_tensor(primals_11, (1024, 10), (1, 1024), 0), out=buf14)
        buf15 = empty_strided_cuda((4, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.mm(buf12, reinterpret_tensor(primals_13, (1024, 3), (1, 1024), 0), out=buf15)
        buf16 = empty_strided_cuda((4, 161), (161, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xc_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf0, buf13, primals_10, buf7, primals_2, buf14, primals_12, buf8, primals_3, buf15, primals_14, buf9, primals_4, buf16, 644, grid=grid(644), stream=stream0)
        buf17 = empty_strided_cuda((4, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xc_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, buf16, reinterpret_tensor(primals_5, (161, 1024), (1, 161), 0), alpha=1, beta=1, out=buf17)
        del primals_6
        buf18 = empty_strided_cuda((4, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [xc_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf17, reinterpret_tensor(primals_7, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf18)
        del primals_8
        buf19 = empty_strided_cuda((4, 144), (144, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(primals_9, (1024, 144), (1, 1024), 0), out=buf19)
        buf20 = empty_strided_cuda((4, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(primals_11, (1024, 10), (1, 1024), 0), out=buf20)
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [linear_3, pred_shape, linear_8, pred_shape_1, linear_13, pred_shape_2], Original ATen: [aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_4.run(buf21, primals_12, buf14, buf8, primals_3, 40, grid=grid(40), stream=stream0)
        del buf14
        del buf8
        del primals_12
        del primals_3
        buf22 = empty_strided_cuda((4, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(primals_13, (1024, 3), (1, 1024), 0), out=buf22)
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [linear_4, pred_cam, linear_9, pred_cam_1, linear_14, pred_cam_2], Original ATen: [aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_5.run(buf23, primals_14, buf15, buf9, primals_4, 12, grid=grid(12), stream=stream0)
        del buf15
        del buf9
        del primals_14
        del primals_4
        buf24 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [linear_2, pred_pose, linear_7, pred_pose_1, linear_12, pred_pose_2], Original ATen: [aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_6.run(buf24, primals_10, buf13, buf7, primals_2, 576, grid=grid(576), stream=stream0)
        del buf13
        del buf7
        del primals_10
        del primals_2
        buf25 = empty_strided_cuda((96, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [b1], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_7.run(buf24, buf25, 288, grid=grid(288), stream=stream0)
        buf26 = empty_strided_cuda((96, 3, 1), (3, 1, 288), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_8.run(buf24, buf26, 288, grid=grid(288), stream=stream0)
        buf27 = empty_strided_cuda((96, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf25, (96, 1, 3), (3, 3, 1), 0), buf26, out=buf27)
        buf28 = reinterpret_tensor(buf26, (96, 3), (3, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [mul, sub], Original ATen: [aten.mul, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sub_9.run(buf24, buf27, buf25, buf28, 288, grid=grid(288), stream=stream0)
        buf29 = empty_strided_cuda((96, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [b3], Original ATen: [aten.linalg_cross]
        stream0 = get_raw_stream(0)
        triton_poi_fused_linalg_cross_10.run(buf25, buf29, 288, grid=grid(288), stream=stream0)
        buf30 = empty_strided_cuda((96, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [b3], Original ATen: [aten.linalg_cross]
        stream0 = get_raw_stream(0)
        triton_poi_fused_linalg_cross_11.run(buf25, buf30, 288, grid=grid(288), stream=stream0)
        buf31 = empty_strided_cuda((96, 3, 3), (9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [stack], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_12.run(buf25, buf28, buf29, buf30, buf31, 864, grid=grid(864), stream=stream0)
    return (reinterpret_tensor(buf31, (4, 24, 3, 3), (216, 9, 3, 1), 0), buf21, buf23, buf4, buf5, buf6, buf10, buf11, buf12, buf16, buf17, buf18, reinterpret_tensor(buf24, (96, 3), (6, 2), 0), buf25, buf27, buf28, buf29, buf30, reinterpret_tensor(buf24, (96, 1, 3), (6, 2, 2), 1), primals_13, primals_11, primals_9, primals_7, primals_5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1024, 161), (161, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((144, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((10, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((3, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
