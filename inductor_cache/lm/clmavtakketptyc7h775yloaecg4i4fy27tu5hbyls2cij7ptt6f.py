# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/cr/ccr6tczfw5pafttfhklfpd4tgfzfikg5dp2bify4nv5n4qehqqkt.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.tanh]
# Source node to ATen node mapping:
#   mean => tanh
# Graph fragment:
#   %tanh : [num_users=3] = call_function[target=torch.ops.aten.tanh.default](args = (%view_1,), kwargs = {})
triton_poi_fused_tanh_0 = async_compile.triton('triton_poi_fused_tanh_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.tanh(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zu/czuijx2bnoc4uwpcj2xpbhz5eq2p2kbklpwt22263slvlvh5zc4c.py
# Topologically Sorted Source Nodes: [action, var, mul], Original ATen: [aten.normal, aten.pow, aten.mul]
# Source node to ATen node mapping:
#   action => normal
#   mul => mul
#   var => pow_1
# Graph fragment:
#   %normal : [num_users=2] = call_function[target=torch.ops.aten.normal.Tensor_Tensor](args = (%expand_1, %expand_2), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%expand, 2), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 2), kwargs = {})
triton_poi_fused_mul_normal_pow_1 = async_compile.triton('triton_poi_fused_mul_normal_pow_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_normal_pow_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_normal_pow_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 * tmp5
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp5, xmask)
    tl.store(out_ptr1 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tx/ctxb4bkdl55oflyukdechu4bmy3kccgc5jl57cx357eftnd2sark.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   sub => sub
# Graph fragment:
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%normal, %tanh), kwargs = {})
triton_poi_fused_sub_2 = async_compile.triton('triton_poi_fused_sub_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sub_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 - tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/it/citlret6hj7tduih7lz7pgx6hrirp6w5scaucp7n4immw4aqrp2j.py
# Topologically Sorted Source Nodes: [log_scale, pow_2, neg, truediv, sub_1, sub_2, sum_1, add, sum_2], Original ATen: [aten.log, aten.pow, aten.neg, aten.div, aten.sub, aten.sum, aten.add]
# Source node to ATen node mapping:
#   add => add
#   log_scale => log
#   neg => neg
#   pow_2 => pow_2
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sum_1 => sum_1
#   sum_2 => sum_2
#   truediv => div
# Graph fragment:
#   %log : [num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%expand,), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%pow_2,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg, %mul), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %log), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, 0.9189385332046727), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sub_2, [-1]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, 1.4189385332046727), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add, [-1]), kwargs = {})
triton_poi_fused_add_div_log_neg_pow_sub_sum_3 = async_compile.triton('triton_poi_fused_add_div_log_neg_pow_sub_sum_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_log_neg_pow_sub_sum_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_log_neg_pow_sub_sum_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (1))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (2))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp46 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr2 + (3))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp1 = tmp0 * tmp0
    tmp2 = -tmp1
    tmp4 = tmp2 / tmp3
    tmp7 = 20.0
    tmp8 = tmp6 > tmp7
    tmp9 = tl_math.exp(tmp6)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tl.where(tmp8, tmp6, tmp10)
    tmp12 = tl_math.log(tmp11)
    tmp13 = tmp4 - tmp12
    tmp14 = 0.9189385332046727
    tmp15 = tmp13 - tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = -tmp17
    tmp20 = tmp18 / tmp19
    tmp23 = tmp22 > tmp7
    tmp24 = tl_math.exp(tmp22)
    tmp25 = libdevice.log1p(tmp24)
    tmp26 = tl.where(tmp23, tmp22, tmp25)
    tmp27 = tl_math.log(tmp26)
    tmp28 = tmp20 - tmp27
    tmp29 = tmp28 - tmp14
    tmp30 = tmp15 + tmp29
    tmp32 = tmp31 * tmp31
    tmp33 = -tmp32
    tmp35 = tmp33 / tmp34
    tmp38 = tmp37 > tmp7
    tmp39 = tl_math.exp(tmp37)
    tmp40 = libdevice.log1p(tmp39)
    tmp41 = tl.where(tmp38, tmp37, tmp40)
    tmp42 = tl_math.log(tmp41)
    tmp43 = tmp35 - tmp42
    tmp44 = tmp43 - tmp14
    tmp45 = tmp30 + tmp44
    tmp47 = tmp46 * tmp46
    tmp48 = -tmp47
    tmp50 = tmp48 / tmp49
    tmp53 = tmp52 > tmp7
    tmp54 = tl_math.exp(tmp52)
    tmp55 = libdevice.log1p(tmp54)
    tmp56 = tl.where(tmp53, tmp52, tmp55)
    tmp57 = tl_math.log(tmp56)
    tmp58 = tmp50 - tmp57
    tmp59 = tmp58 - tmp14
    tmp60 = tmp45 + tmp59
    tmp61 = 1.4189385332046727
    tmp62 = tmp12 + tmp61
    tmp63 = tmp27 + tmp61
    tmp64 = tmp62 + tmp63
    tmp65 = tmp42 + tmp61
    tmp66 = tmp64 + tmp65
    tmp67 = tmp57 + tmp61
    tmp68 = tmp66 + tmp67
    tl.store(out_ptr0 + (x0), tmp60, xmask)
    tl.store(out_ptr1 + (x0), tmp68, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (1, 4), (4, 1))
    assert_size_stride(primals_5, (1, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf0)
        del primals_2
        buf1 = reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_0.run(buf1, primals_3, 256, grid=grid(256), stream=stream0)
        del primals_3
        buf3 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 1), (1, 4), 0), alpha=1, beta=1, out=buf3)
        del primals_4
        del primals_5
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [action, var, mul], Original ATen: [aten.normal, aten.pow, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_normal_pow_1.run(primals_6, buf4, buf8, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [action], Original ATen: [aten.normal]
        buf5 = torch.ops.aten.normal.Tensor_Tensor(buf1, buf4)
        buf6 = buf5
        del buf5
        buf7 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sub_2.run(buf6, buf1, buf7, 256, grid=grid(256), stream=stream0)
        buf9 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf10 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [log_scale, pow_2, neg, truediv, sub_1, sub_2, sum_1, add, sum_2], Original ATen: [aten.log, aten.pow, aten.neg, aten.div, aten.sub, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_log_neg_pow_sub_sum_3.run(buf7, buf8, primals_6, buf9, buf10, 64, grid=grid(64), stream=stream0)
    return (buf6, reinterpret_tensor(buf9, (4, 4, 4, 1), (16, 4, 1, 1), 0), reinterpret_tensor(buf10, (4, 4, 4, 1), (16, 4, 1, 1), 0), buf1, reinterpret_tensor(buf3, (4, 4, 4, 1), (16, 4, 1, 1), 0), primals_6, reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), buf1, buf7, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
