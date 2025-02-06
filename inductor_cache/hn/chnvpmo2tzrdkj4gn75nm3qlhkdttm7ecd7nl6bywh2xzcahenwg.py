# AOT ID: ['44_inference']
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


# kernel path: inductor_cache/l3/cl3fvuofurtxyqa75w7j37htx2nlazsrw5byrfjutlrmshpwrfl6.py
# Topologically Sorted Source Nodes: [pos, dim, floordiv, mul, truediv, mul_1, div, pos_1, sin, setitem], Original ATen: [aten.repeat, aten.floor_divide, aten.mul, aten.div, aten.exp, aten.sin, aten.copy]
# Source node to ATen node mapping:
#   dim => repeat_1
#   div => exp
#   floordiv => div
#   mul => mul_2
#   mul_1 => mul_3
#   pos => repeat
#   pos_1 => mul_4
#   setitem => copy
#   sin => sin
#   truediv => div_1
# Graph fragment:
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze, [1, 4]), kwargs = {})
#   %repeat_1 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze_1, [4, 1]), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%repeat_1, 2), kwargs = {rounding_mode: floor})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, 4), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, -9.210340371976184), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_3,), kwargs = {})
#   %mul_4 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%repeat, %exp), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%slice_4,), kwargs = {})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_8, %sin), kwargs = {})
#   %slice_scatter_default : [num_users=3] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%mul_4, %copy, 1, 0, 9223372036854775807, 2), kwargs = {})
triton_poi_fused_copy_div_exp_floor_divide_mul_repeat_sin_0 = async_compile.triton('triton_poi_fused_copy_div_exp_floor_divide_mul_repeat_sin_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_div_exp_floor_divide_mul_repeat_sin_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_div_exp_floor_divide_mul_repeat_sin_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = (x2 % 2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = 2*(x0 // 2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.floor(tmp6)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.25
    tmp11 = tmp9 * tmp10
    tmp12 = -9.210340371976184
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = x1
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp14
    tmp18 = tl_math.sin(tmp17)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp2, tmp18, tmp19)
    tmp21 = x0
    tmp22 = tmp21.to(tl.float32)
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = libdevice.floor(tmp24)
    tmp26 = 2.0
    tmp27 = tmp25 * tmp26
    tmp28 = 0.25
    tmp29 = tmp27 * tmp28
    tmp30 = -9.210340371976184
    tmp31 = tmp29 * tmp30
    tmp32 = tl_math.exp(tmp31)
    tmp33 = x1
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp32
    tmp36 = tl.where(tmp2, tmp20, tmp35)
    tl.store(out_ptr0 + (x2), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ns/cnsiereytptbftffypjkbc7o65nkn2ivtnzye62vilv5wy24ilup.py
# Topologically Sorted Source Nodes: [cos, setitem_1], Original ATen: [aten.cos, aten.copy]
# Source node to ATen node mapping:
#   cos => cos
#   setitem_1 => copy_1
# Graph fragment:
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%slice_15,), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_19, %cos), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%slice_scatter_default, %copy_1, 1, 1, 9223372036854775807, 2), kwargs = {})
triton_poi_fused_copy_cos_1 = async_compile.triton('triton_poi_fused_copy_cos_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_cos_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_cos_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (((-1) + x0) % 2)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (1 + 2*(triton_helpers.div_floor_integer((-1) + x0,  2)) + 4*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl_math.cos(tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = (x2 % 2)
    tmp12 = tmp11 == tmp4
    tmp13 = 2*(x0 // 2)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = libdevice.floor(tmp16)
    tmp18 = 2.0
    tmp19 = tmp17 * tmp18
    tmp20 = 0.25
    tmp21 = tmp19 * tmp20
    tmp22 = -9.210340371976184
    tmp23 = tmp21 * tmp22
    tmp24 = tl_math.exp(tmp23)
    tmp25 = x1
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp24
    tmp28 = tl_math.sin(tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp12, tmp28, tmp29)
    tmp31 = tmp0.to(tl.float32)
    tmp32 = 0.5
    tmp33 = tmp31 * tmp32
    tmp34 = libdevice.floor(tmp33)
    tmp35 = 2.0
    tmp36 = tmp34 * tmp35
    tmp37 = 0.25
    tmp38 = tmp36 * tmp37
    tmp39 = -9.210340371976184
    tmp40 = tmp38 * tmp39
    tmp41 = tl_math.exp(tmp40)
    tmp42 = x1
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp43 * tmp41
    tmp45 = tl.where(tmp12, tmp30, tmp44)
    tmp46 = tl.where(tmp6, tmp10, tmp45)
    tl.store(out_ptr0 + (x2), tmp46, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ng/cngcpnzfium2e67i5lx4i4rwwff3w6tgkenn6tf5wn37ny3zamag.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add_2
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %unsqueeze_3), kwargs = {})
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pos, dim, floordiv, mul, truediv, mul_1, div, pos_1, sin, setitem], Original ATen: [aten.repeat, aten.floor_divide, aten.mul, aten.div, aten.exp, aten.sin, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_div_exp_floor_divide_mul_repeat_sin_0.run(buf0, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cos, setitem_1], Original ATen: [aten.cos, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_cos_1.run(buf0, buf1, 16, grid=grid(16), stream=stream0)
        del buf0
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(arg0_1, buf1, buf2, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del buf1
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
