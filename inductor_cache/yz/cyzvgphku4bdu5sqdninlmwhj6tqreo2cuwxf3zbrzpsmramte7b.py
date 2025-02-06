# AOT ID: ['1_inference']
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


# kernel path: inductor_cache/vs/cvswek3lhq6bxpaltqqk45ch53gsssom55it5im4uypb4vaitunr.py
# Topologically Sorted Source Nodes: [sort], Original ATen: [aten.sort]
# Source node to ATen node mapping:
#   sort => sort
# Graph fragment:
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.default](args = (%arg1_1, -1, True), kwargs = {})
triton_per_fused_sort_0 = async_compile.triton('triton_per_fused_sort_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i16', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sort_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sort_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), xmask, other=0.0)
    tmp1 = r1
    tmp2 = tmp1.to(tl.int16)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5, tmp6, = triton_helpers.sort_with_index(tmp3, tmp4, None, 1, stable=False, descending=True)
    tl.store(out_ptr0 + (r1 + 4*x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sb/csb57fvrilenjlzyvvw4f2yuhhpca3ylwison6ssgodroiafbsau.py
# Topologically Sorted Source Nodes: [gamma], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   gamma => max_1
# Graph fragment:
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.default](args = (%view_1,), kwargs = {})
triton_red_fused_max_1 = async_compile.triton('triton_red_fused_max_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_max_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_max_1(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (128*x0 + (r1 // 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 4")
        tmp7 = tl.load(in_ptr1 + (64*tmp5 + ((r1 % 64))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wk/cwkmpbk773uzbltmrbjznf2xw5qye4gdwutctey7lcytvnpobw2z.py
# Topologically Sorted Source Nodes: [gamma], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   gamma => max_1
# Graph fragment:
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.default](args = (%view_1,), kwargs = {})
triton_per_fused_max_2 = async_compile.triton('triton_per_fused_max_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_max_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_max_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = triton_helpers.max2(tmp1, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/u3/cu3oarkjqpfu4sb5iqoujm7un6vfw6w7vbpj2u7blwc2zedwhl5g.py
# Topologically Sorted Source Nodes: [sub, exp, cumsum], Original ATen: [aten.sub, aten.exp, aten.cumsum]
# Source node to ATen node mapping:
#   cumsum => cumsum
#   exp => exp
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %max_1), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %cumsum : [num_users=1] = call_function[target=torch.ops.aten.cumsum.default](args = (%exp, 0), kwargs = {})
triton_spl_fused_cumsum_exp_sub_3 = async_compile.triton('triton_spl_fused_cumsum_exp_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.split_scan(
    size_hints={'x': 1, 'r': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ws_ptr': '*u8', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_spl_fused_cumsum_exp_sub_3', 'mutated_arg_names': ['ws_ptr'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_spl_fused_cumsum_exp_sub_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ws_ptr, xnumel, rnumel, RBLOCK : tl.constexpr):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 16384
    xoffset = tl.program_id(1) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    roffset = tl.program_id(0) * RBLOCK
    rindex = roffset + tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 // 64), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp12 = tl.num_programs(0)
    tmp13 = ws_ptr.to(tl.pointer_type(tl.uint64)) + xoffset * 1 * tmp12
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([RBLOCK], 4, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(rmask), "index out of bounds: 0 <= tmp5 < 4")
    tmp7 = tl.load(in_ptr1 + (64*tmp5 + ((r0 % 64))), rmask, other=0.0)
    tmp10 = tmp7 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp14 = tmp11.to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp16 = tl.reduce(tmp15, 0, _triton_helper_fn_add0)
    tmp17 = triton_helpers.exclusive_scan_decoupled_lookback(
        tmp13,
        tmp16,
        tl.program_id(0),
        _triton_helper_fn_add0,
        DTYPE_VALUE_AS_UINT=tl.uint32,
        DTYPE_PACK=tl.uint64,
    )
    tmp18 = tl.associative_scan(tmp15, 0, _triton_helper_fn_add0)
    tmp19 = _triton_helper_fn_add0(tmp17, tmp18)
    tmp20 = tl.where(roffset == 0, tmp18, tmp19)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp20, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/fw/cfwg2jz343gu7bntrhj7jyhnglszeokryb4ls4hxmuff3ghouij7.py
# Topologically Sorted Source Nodes: [add, log, log_cumsum_h, sub_1, mul, sum_1, sum_2], Original ATen: [aten.add, aten.log, aten.sub, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add => add
#   log => log
#   log_cumsum_h => add_1
#   mul => mul
#   sub_1 => sub_1
#   sum_1 => sum_1
#   sum_2 => sum_2
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cumsum, 1e-07), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %max_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %add_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %view), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%view,), kwargs = {})
triton_red_fused_add_log_mul_sub_sum_4 = async_compile.triton('triton_red_fused_add_log_mul_sub_sum_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_log_mul_sub_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_log_mul_sub_sum_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp12 = tl.load(in_ptr3 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (128*x0 + (r1 // 64)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp5 < 4")
        tmp7 = tl.load(in_ptr1 + (64*tmp5 + ((r1 % 64))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = 1e-07
        tmp10 = tmp8 + tmp9
        tmp11 = tl_math.log(tmp10)
        tmp14 = tmp11 + tmp13
        tmp15 = tmp7 - tmp14
        tmp16 = tl.load(in_ptr4 + (64*tmp5 + ((r1 % 64))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp21 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tl.store(out_ptr1 + (x0), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rh/crhsijltbdskjevw444s7j53oaurmvqm2yti3d4swhxnjhk3d2ii.py
# Topologically Sorted Source Nodes: [add, log, log_cumsum_h, sub_1, mul, sum_1, sum_2, div, neg], Original ATen: [aten.add, aten.log, aten.sub, aten.mul, aten.sum, aten.div, aten.neg]
# Source node to ATen node mapping:
#   add => add
#   div => div
#   log => log
#   log_cumsum_h => add_1
#   mul => mul
#   neg => neg
#   sub_1 => sub_1
#   sum_1 => sum_1
#   sum_2 => sum_2
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cumsum, 1e-07), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %max_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %add_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %view), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%view,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_1, %sum_2), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
triton_per_fused_add_div_log_mul_neg_sub_sum_5 = async_compile.triton('triton_per_fused_add_div_log_mul_neg_sub_sum_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_log_mul_neg_sub_sum_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_log_mul_neg_sub_sum_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp4 = tl.load(in_ptr1 + (r0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp3 / tmp7
    tmp9 = -tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.int16)
        # Topologically Sorted Source Nodes: [sort], Original ATen: [aten.sort]
        stream0 = get_raw_stream(0)
        triton_per_fused_sort_0.run(arg1_1, buf1, 64, 4, grid=grid(64), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [gamma], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_red_fused_max_1.run(buf1, arg0_1, buf2, 2, 8192, grid=grid(2), stream=stream0)
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [gamma], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_per_fused_max_2.run(buf2, buf3, 1, 2, grid=grid(1), stream=stream0)
        buf4 = empty_strided_cuda((16384, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sub, exp, cumsum], Original ATen: [aten.sub, aten.exp, aten.cumsum]
        workspace_0 = empty_strided_cuda((512, ), (1, ), torch.uint8)
        workspace_0.zero_()
        stream0 = get_raw_stream(0)
        triton_spl_fused_cumsum_exp_sub_3.run(buf1, arg0_1, buf3, buf4, workspace_0, 1, 16384, grid=split_scan_grid(1, 16384), stream=stream0)
        del workspace_0
        buf5 = buf2; del buf2  # reuse
        buf7 = empty_strided_cuda((2, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [add, log, log_cumsum_h, sub_1, mul, sum_1, sum_2], Original ATen: [aten.add, aten.log, aten.sub, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_log_mul_sub_sum_4.run(buf1, arg0_1, buf4, buf3, arg2_1, buf5, buf7, 2, 8192, grid=grid(2), stream=stream0)
        del arg0_1
        del arg2_1
        del buf1
        del buf4
        buf6 = buf3; del buf3  # reuse
        buf9 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [add, log, log_cumsum_h, sub_1, mul, sum_1, sum_2, div, neg], Original ATen: [aten.add, aten.log, aten.sub, aten.mul, aten.sum, aten.div, aten.neg]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_log_mul_neg_sub_sum_5.run(buf9, buf5, buf7, 1, 2, grid=grid(1), stream=stream0)
        del buf5
        del buf7
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
