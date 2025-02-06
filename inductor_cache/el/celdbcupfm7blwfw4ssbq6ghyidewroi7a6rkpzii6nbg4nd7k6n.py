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


# kernel path: inductor_cache/hk/chky3kvupnutlibt3fn5qkuriskz35kgbiukcfkzdwg5n7zjqvf4.py
# Topologically Sorted Source Nodes: [add, weights, pow_1, sum_1, add_1, d, weights_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.sum, aten.rsqrt]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   d => rsqrt
#   pow_1 => pow_1
#   sum_1 => sum_1
#   weights => mul
#   weights_1 => mul_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_4, 1), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %add), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [2, 3, 4], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-08), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %rsqrt), kwargs = {})
triton_red_fused_add_mul_pow_rsqrt_sum_0 = async_compile.triton('triton_red_fused_add_mul_pow_rsqrt_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_pow_rsqrt_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_pow_rsqrt_sum_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 96)
    x1 = xindex // 96
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 9
        tmp0 = tl.load(in_ptr0 + (r5 + 864*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + 96*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 1.0
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp9 = 1e-08
    tmp10 = tmp7 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp11, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 9
        tmp12 = tl.load(in_ptr0 + (r5 + 864*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r3 + 96*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = 1.0
        tmp15 = tmp13 + tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tmp16 * tmp11
        tl.store(out_ptr0 + (r5 + 864*x4), tmp17, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l2/cl2pnc26yqipa3jf3dip7msj2oeirw2ewwiqft7xjjjq2tahop75.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_1 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution, %primals_3, %primals_4, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 96)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/do/cdo7s3bbo4zlyurebr5lp6u6vqx7vjy2xrh5mmjcfjiqczfawemm.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
# Source node to ATen node mapping:
#   x_5 => gt, mul_2, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_2, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.2), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %view_2, %mul_2), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%view_8, 0), kwargs = {})
triton_poi_fused_leaky_relu_leaky_relu_backward_2 = async_compile.triton('triton_poi_fused_leaky_relu_leaky_relu_backward_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_leaky_relu_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_leaky_relu_backward_2(in_out_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tmp5 > tmp1
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qk/cqkhdcqa5pgitrl5bisrzpq2qs75mfnwcbffp27xkh7y4g4oiwic.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   x_7 => view_9
# Graph fragment:
#   %view_9 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, -1, 4, 4]), kwargs = {})
triton_poi_fused_view_3 = async_compile.triton('triton_poi_fused_view_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1 + 1536*(((x1 % 96)) // 96)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 96, 4, 4), (1536, 16, 4, 1))
    assert_size_stride(primals_3, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_4, (96, ), (1, ))
    assert_size_stride(primals_5, (96, 4), (4, 1))
    assert_size_stride(primals_6, (96, ), (1, ))
    assert_size_stride(primals_7, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_8, (96, 4), (4, 1))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (96, 96, 3, 3), (864, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_1, (4, 4, 1, 1), (4, 1, 1, 1), 0), primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 96, 4, 4), (1536, 16, 4, 1))
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 96, 4, 4), (1536, 16, 4, 1))
        buf2 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [style1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, primals_1, reinterpret_tensor(primals_5, (4, 96), (1, 4), 0), alpha=1, beta=1, out=buf2)
        del primals_5
        del primals_6
        buf3 = empty_strided_cuda((4, 96, 1, 1, 1), (96, 1, 384, 384, 384), torch.float32)
        buf4 = reinterpret_tensor(buf3, (4, 96, 1, 1, 1), (96, 1, 1, 1, 1), 0); del buf3  # reuse
        buf6 = empty_strided_cuda((4, 96, 96, 3, 3), (82944, 864, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, weights, pow_1, sum_1, add_1, d, weights_1], Original ATen: [aten.add, aten.mul, aten.pow, aten.sum, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_pow_rsqrt_sum_0.run(buf4, primals_7, buf2, buf6, 384, 864, grid=grid(384), stream=stream0)
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf5, primals_4, 6144, grid=grid(6144), stream=stream0)
        del primals_4
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(buf5, (1, 384, 4, 4), (0, 16, 4, 1), 0), reinterpret_tensor(buf6, (384, 96, 3, 3), (864, 9, 3, 1), 0), stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf7, (1, 384, 4, 4), (6144, 16, 4, 1))
        buf8 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [style2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, primals_1, reinterpret_tensor(primals_8, (4, 96), (1, 4), 0), alpha=1, beta=1, out=buf8)
        del primals_8
        del primals_9
        buf9 = empty_strided_cuda((4, 96, 1, 1, 1), (96, 1, 384, 384, 384), torch.float32)
        buf10 = reinterpret_tensor(buf9, (4, 96, 1, 1, 1), (96, 1, 1, 1, 1), 0); del buf9  # reuse
        buf11 = empty_strided_cuda((4, 96, 96, 3, 3), (82944, 864, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_2, weights_3, pow_2, sum_2, add_3, d_1, weights_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.sum, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_pow_rsqrt_sum_0.run(buf10, primals_10, buf8, buf11, 384, 864, grid=grid(384), stream=stream0)
        buf12 = reinterpret_tensor(buf7, (4, 96, 4, 4), (1536, 16, 4, 1), 0); del buf7  # reuse
        buf17 = empty_strided_cuda((4, 96, 4, 4), (1536, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_2.run(buf12, buf17, 6144, grid=grid(6144), stream=stream0)
        buf13 = empty_strided_cuda((1, 384, 4, 4), (6144, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf12, buf13, 6144, grid=grid(6144), stream=stream0)
        del buf12
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, reinterpret_tensor(buf11, (384, 96, 3, 3), (864, 9, 3, 1), 0), stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf14, (1, 384, 4, 4), (6144, 16, 4, 1))
        buf15 = reinterpret_tensor(buf14, (4, 96, 4, 4), (1536, 16, 4, 1), 0); del buf14  # reuse
        buf16 = empty_strided_cuda((4, 96, 4, 4), (1536, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.leaky_relu, aten.leaky_relu_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_leaky_relu_backward_2.run(buf15, buf16, 6144, grid=grid(6144), stream=stream0)
    return (buf15, primals_1, primals_2, primals_3, primals_7, primals_10, buf0, buf2, buf4, reinterpret_tensor(buf5, (1, 384, 4, 4), (6144, 16, 4, 1), 0), reinterpret_tensor(buf6, (384, 96, 3, 3), (864, 9, 3, 1), 0), buf8, buf10, reinterpret_tensor(buf11, (384, 96, 3, 3), (864, 9, 3, 1), 0), buf13, buf16, buf17, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 96, 4, 4), (1536, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((96, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((96, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
