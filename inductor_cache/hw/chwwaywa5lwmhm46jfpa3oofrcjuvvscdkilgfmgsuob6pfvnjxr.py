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


# kernel path: inductor_cache/ko/ckoktn35sxrbsng7ocpywdskplqn6aaeyfjlxvah2bs4io3cbg3j.py
# Topologically Sorted Source Nodes: [truediv, K_3], Original ATen: [aten.div, aten.exp]
# Source node to ATen node mapping:
#   K_3 => exp
#   truediv => div
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_4, 0.1), kwargs = {})
#   %exp : [num_users=202] = call_function[target=torch.ops.aten.exp.default](args = (%div,), kwargs = {})
triton_poi_fused_div_exp_0 = async_compile.triton('triton_poi_fused_div_exp_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_exp_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_exp_0(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 10.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl_math.exp(tmp2)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4i/c4isbw5hq4zsztkh6wwxbqp7vpfo5tv4px7iiaplcncvyxp7awxv.py
# Topologically Sorted Source Nodes: [v], Original ATen: [aten.new_ones]
# Source node to ATen node mapping:
#   v => full_1
# Graph fragment:
#   %full_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_ones_1 = async_compile.triton('triton_poi_fused_new_ones_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_ones_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_ones_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3z/c3zadin23ieawfnvgej7rb5xg6njeitt37apspgwfzay5iootrtq.py
# Topologically Sorted Source Nodes: [u_1], Original ATen: [aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   u_1 => mul, reciprocal
# Graph fragment:
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_6,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 0.25), kwargs = {})
triton_poi_fused_mul_reciprocal_2 = async_compile.triton('triton_poi_fused_mul_reciprocal_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_reciprocal_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_reciprocal_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp1 / tmp0
    tmp3 = 0.25
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xd/cxd4otirawph5r4ro4lpljs33eujaplr7abi62eruup6e7gv7ifb.py
# Topologically Sorted Source Nodes: [v_1], Original ATen: [aten.reciprocal, aten.mul]
# Source node to ATen node mapping:
#   v_1 => mul_1, reciprocal_1
# Graph fragment:
#   %reciprocal_1 : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%view_8,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1.0), kwargs = {})
triton_poi_fused_mul_reciprocal_3 = async_compile.triton('triton_poi_fused_mul_reciprocal_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_reciprocal_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_reciprocal_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp1 / tmp0
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6a/c6aapt5dnzlpyy4npmbjchnenh5sqvbaq4po7v2f3xt5eerodur5.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul_200
# Graph fragment:
#   %mul_200 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp, %view_406), kwargs = {})
triton_poi_fused_mul_4 = async_compile.triton('triton_poi_fused_mul_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 1, tl.int32)
    tmp3 = tmp2 / tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 * tmp5
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gr/cgrw6phh4ntzyqvuvwt7e4okomphky4tkqwpucz3vxbzjaam5nxo.py
# Topologically Sorted Source Nodes: [K_6], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   K_6 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex // 4
    y0 = (yindex % 4)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 4*x2 + 64*y1), xmask & ymask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + 16*y3), tmp2, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (1, 4, 4), (16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [K], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf0)
        del primals_2
        buf1 = reinterpret_tensor(buf0, (4, 16, 4), (64, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [truediv, K_3], Original ATen: [aten.div, aten.exp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_exp_0.run(buf1, 256, grid=grid(256), stream=stream0)
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten.new_ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_ones_1.run(buf2, 16, grid=grid(16), stream=stream0)
        buf3 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf2, (4, 4, 1), (4, 1, 0), 0), out=buf3)
        buf4 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_1], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf3, buf4, 64, grid=grid(64), stream=stream0)
        buf5 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf5)
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_1], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf5, buf6, 16, grid=grid(16), stream=stream0)
        buf7 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf6, (4, 4, 1), (4, 1, 0), 0), out=buf7)
        buf8 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_2], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf7, buf8, 64, grid=grid(64), stream=stream0)
        buf9 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf9)
        buf10 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_2], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf9, buf10, 16, grid=grid(16), stream=stream0)
        buf11 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf10, (4, 4, 1), (4, 1, 0), 0), out=buf11)
        buf12 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_3], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf11, buf12, 64, grid=grid(64), stream=stream0)
        buf13 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf13)
        buf14 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_3], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf13, buf14, 16, grid=grid(16), stream=stream0)
        buf15 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf14, (4, 4, 1), (4, 1, 0), 0), out=buf15)
        buf16 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_4], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf15, buf16, 64, grid=grid(64), stream=stream0)
        buf17 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf17)
        buf18 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_4], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf17, buf18, 16, grid=grid(16), stream=stream0)
        buf19 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf18, (4, 4, 1), (4, 1, 0), 0), out=buf19)
        buf20 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_5], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf19, buf20, 64, grid=grid(64), stream=stream0)
        buf21 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf21)
        buf22 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_5], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf21, buf22, 16, grid=grid(16), stream=stream0)
        buf23 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf22, (4, 4, 1), (4, 1, 0), 0), out=buf23)
        buf24 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_6], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf23, buf24, 64, grid=grid(64), stream=stream0)
        buf25 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf25)
        buf26 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_6], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf25, buf26, 16, grid=grid(16), stream=stream0)
        buf27 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf26, (4, 4, 1), (4, 1, 0), 0), out=buf27)
        buf28 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_7], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf27, buf28, 64, grid=grid(64), stream=stream0)
        buf29 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf29)
        buf30 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_7], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf29, buf30, 16, grid=grid(16), stream=stream0)
        buf31 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf30, (4, 4, 1), (4, 1, 0), 0), out=buf31)
        buf32 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_8], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf31, buf32, 64, grid=grid(64), stream=stream0)
        buf33 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf33)
        buf34 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_8], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf33, buf34, 16, grid=grid(16), stream=stream0)
        buf35 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf34, (4, 4, 1), (4, 1, 0), 0), out=buf35)
        buf36 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_9], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf35, buf36, 64, grid=grid(64), stream=stream0)
        buf37 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf37)
        buf38 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_9], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf37, buf38, 16, grid=grid(16), stream=stream0)
        buf39 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf38, (4, 4, 1), (4, 1, 0), 0), out=buf39)
        buf40 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_10], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf39, buf40, 64, grid=grid(64), stream=stream0)
        buf41 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf40, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf41)
        buf42 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_10], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf41, buf42, 16, grid=grid(16), stream=stream0)
        buf43 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf42, (4, 4, 1), (4, 1, 0), 0), out=buf43)
        buf44 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_11], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf43, buf44, 64, grid=grid(64), stream=stream0)
        buf45 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf44, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf45)
        buf46 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_11], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf45, buf46, 16, grid=grid(16), stream=stream0)
        buf47 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf46, (4, 4, 1), (4, 1, 0), 0), out=buf47)
        buf48 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_12], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf47, buf48, 64, grid=grid(64), stream=stream0)
        buf49 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf49)
        buf50 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_12], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf49, buf50, 16, grid=grid(16), stream=stream0)
        buf51 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf50, (4, 4, 1), (4, 1, 0), 0), out=buf51)
        buf52 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_13], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf51, buf52, 64, grid=grid(64), stream=stream0)
        buf53 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf52, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf53)
        buf54 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_13], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf53, buf54, 16, grid=grid(16), stream=stream0)
        buf55 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf54, (4, 4, 1), (4, 1, 0), 0), out=buf55)
        buf56 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_14], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf55, buf56, 64, grid=grid(64), stream=stream0)
        buf57 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf56, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf57)
        buf58 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_14], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf57, buf58, 16, grid=grid(16), stream=stream0)
        buf59 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf58, (4, 4, 1), (4, 1, 0), 0), out=buf59)
        buf60 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_15], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf59, buf60, 64, grid=grid(64), stream=stream0)
        buf61 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf61)
        buf62 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_15], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf61, buf62, 16, grid=grid(16), stream=stream0)
        buf63 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf62, (4, 4, 1), (4, 1, 0), 0), out=buf63)
        buf64 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_16], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf63, buf64, 64, grid=grid(64), stream=stream0)
        buf65 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf65)
        buf66 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_16], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf65, buf66, 16, grid=grid(16), stream=stream0)
        buf67 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf66, (4, 4, 1), (4, 1, 0), 0), out=buf67)
        buf68 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_17], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf67, buf68, 64, grid=grid(64), stream=stream0)
        buf69 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf69)
        buf70 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_17], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf69, buf70, 16, grid=grid(16), stream=stream0)
        buf71 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf70, (4, 4, 1), (4, 1, 0), 0), out=buf71)
        buf72 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_18], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf71, buf72, 64, grid=grid(64), stream=stream0)
        buf73 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf73)
        buf74 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_18], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf73, buf74, 16, grid=grid(16), stream=stream0)
        buf75 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf74, (4, 4, 1), (4, 1, 0), 0), out=buf75)
        buf76 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_19], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf75, buf76, 64, grid=grid(64), stream=stream0)
        buf77 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf76, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf77)
        buf78 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_19], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf77, buf78, 16, grid=grid(16), stream=stream0)
        buf79 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf78, (4, 4, 1), (4, 1, 0), 0), out=buf79)
        buf80 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_20], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf79, buf80, 64, grid=grid(64), stream=stream0)
        buf81 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf80, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf81)
        buf82 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_20], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf81, buf82, 16, grid=grid(16), stream=stream0)
        buf83 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf82, (4, 4, 1), (4, 1, 0), 0), out=buf83)
        buf84 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_21], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf83, buf84, 64, grid=grid(64), stream=stream0)
        buf85 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf85)
        buf86 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_21], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf85, buf86, 16, grid=grid(16), stream=stream0)
        buf87 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf86, (4, 4, 1), (4, 1, 0), 0), out=buf87)
        buf88 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_22], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf87, buf88, 64, grid=grid(64), stream=stream0)
        buf89 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf89)
        buf90 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_22], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf89, buf90, 16, grid=grid(16), stream=stream0)
        buf91 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf90, (4, 4, 1), (4, 1, 0), 0), out=buf91)
        buf92 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_23], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf91, buf92, 64, grid=grid(64), stream=stream0)
        buf93 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf92, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf93)
        buf94 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_23], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf93, buf94, 16, grid=grid(16), stream=stream0)
        buf95 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf94, (4, 4, 1), (4, 1, 0), 0), out=buf95)
        buf96 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_24], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf95, buf96, 64, grid=grid(64), stream=stream0)
        buf97 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf97)
        buf98 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_24], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf97, buf98, 16, grid=grid(16), stream=stream0)
        buf99 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf98, (4, 4, 1), (4, 1, 0), 0), out=buf99)
        buf100 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_25], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf99, buf100, 64, grid=grid(64), stream=stream0)
        buf101 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf100, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf101)
        buf102 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_25], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf101, buf102, 16, grid=grid(16), stream=stream0)
        buf103 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf102, (4, 4, 1), (4, 1, 0), 0), out=buf103)
        buf104 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_26], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf103, buf104, 64, grid=grid(64), stream=stream0)
        buf105 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf105)
        buf106 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_26], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf105, buf106, 16, grid=grid(16), stream=stream0)
        buf107 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf106, (4, 4, 1), (4, 1, 0), 0), out=buf107)
        buf108 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_27], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf107, buf108, 64, grid=grid(64), stream=stream0)
        buf109 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_53], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf108, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf109)
        buf110 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_27], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf109, buf110, 16, grid=grid(16), stream=stream0)
        buf111 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf110, (4, 4, 1), (4, 1, 0), 0), out=buf111)
        buf112 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_28], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf111, buf112, 64, grid=grid(64), stream=stream0)
        buf113 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf112, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf113)
        buf114 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_28], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf113, buf114, 16, grid=grid(16), stream=stream0)
        buf115 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf114, (4, 4, 1), (4, 1, 0), 0), out=buf115)
        buf116 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_29], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf115, buf116, 64, grid=grid(64), stream=stream0)
        buf117 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_57], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf117)
        buf118 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_29], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf117, buf118, 16, grid=grid(16), stream=stream0)
        buf119 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_58], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf118, (4, 4, 1), (4, 1, 0), 0), out=buf119)
        buf120 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_30], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf119, buf120, 64, grid=grid(64), stream=stream0)
        buf121 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_59], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf121)
        buf122 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_30], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf121, buf122, 16, grid=grid(16), stream=stream0)
        buf123 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf122, (4, 4, 1), (4, 1, 0), 0), out=buf123)
        buf124 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_31], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf123, buf124, 64, grid=grid(64), stream=stream0)
        buf125 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_61], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf124, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf125)
        buf126 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_31], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf125, buf126, 16, grid=grid(16), stream=stream0)
        buf127 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_62], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf126, (4, 4, 1), (4, 1, 0), 0), out=buf127)
        buf128 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_32], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf127, buf128, 64, grid=grid(64), stream=stream0)
        buf129 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf128, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf129)
        buf130 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_32], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf129, buf130, 16, grid=grid(16), stream=stream0)
        buf131 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_64], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf130, (4, 4, 1), (4, 1, 0), 0), out=buf131)
        buf132 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_33], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf131, buf132, 64, grid=grid(64), stream=stream0)
        buf133 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_65], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf133)
        buf134 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_33], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf133, buf134, 16, grid=grid(16), stream=stream0)
        buf135 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_66], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf134, (4, 4, 1), (4, 1, 0), 0), out=buf135)
        buf136 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_34], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf135, buf136, 64, grid=grid(64), stream=stream0)
        buf137 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_67], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf136, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf137)
        buf138 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_34], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf137, buf138, 16, grid=grid(16), stream=stream0)
        buf139 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_68], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf138, (4, 4, 1), (4, 1, 0), 0), out=buf139)
        buf140 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_35], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf139, buf140, 64, grid=grid(64), stream=stream0)
        buf141 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_69], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf140, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf141)
        buf142 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_35], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf141, buf142, 16, grid=grid(16), stream=stream0)
        buf143 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_70], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf142, (4, 4, 1), (4, 1, 0), 0), out=buf143)
        buf144 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_36], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf143, buf144, 64, grid=grid(64), stream=stream0)
        buf145 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_71], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf144, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf145)
        buf146 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_36], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf145, buf146, 16, grid=grid(16), stream=stream0)
        buf147 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_72], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf146, (4, 4, 1), (4, 1, 0), 0), out=buf147)
        buf148 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_37], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf147, buf148, 64, grid=grid(64), stream=stream0)
        buf149 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_73], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf149)
        buf150 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_37], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf149, buf150, 16, grid=grid(16), stream=stream0)
        buf151 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_74], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf150, (4, 4, 1), (4, 1, 0), 0), out=buf151)
        buf152 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_38], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf151, buf152, 64, grid=grid(64), stream=stream0)
        buf153 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_75], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf152, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf153)
        buf154 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_38], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf153, buf154, 16, grid=grid(16), stream=stream0)
        buf155 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_76], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf154, (4, 4, 1), (4, 1, 0), 0), out=buf155)
        buf156 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_39], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf155, buf156, 64, grid=grid(64), stream=stream0)
        buf157 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_77], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf156, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf157)
        buf158 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_39], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf157, buf158, 16, grid=grid(16), stream=stream0)
        buf159 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_78], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf158, (4, 4, 1), (4, 1, 0), 0), out=buf159)
        buf160 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_40], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf159, buf160, 64, grid=grid(64), stream=stream0)
        buf161 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_79], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf161)
        buf162 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_40], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf161, buf162, 16, grid=grid(16), stream=stream0)
        buf163 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_80], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf162, (4, 4, 1), (4, 1, 0), 0), out=buf163)
        buf164 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_41], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf163, buf164, 64, grid=grid(64), stream=stream0)
        buf165 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_81], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf164, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf165)
        buf166 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_41], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf165, buf166, 16, grid=grid(16), stream=stream0)
        buf167 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_82], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf166, (4, 4, 1), (4, 1, 0), 0), out=buf167)
        buf168 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_42], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf167, buf168, 64, grid=grid(64), stream=stream0)
        buf169 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_83], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf169)
        buf170 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_42], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf169, buf170, 16, grid=grid(16), stream=stream0)
        buf171 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_84], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf170, (4, 4, 1), (4, 1, 0), 0), out=buf171)
        buf172 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_43], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf171, buf172, 64, grid=grid(64), stream=stream0)
        buf173 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_85], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf172, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf173)
        buf174 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_43], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf173, buf174, 16, grid=grid(16), stream=stream0)
        buf175 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_86], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf174, (4, 4, 1), (4, 1, 0), 0), out=buf175)
        buf176 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_44], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf175, buf176, 64, grid=grid(64), stream=stream0)
        buf177 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_87], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf177)
        buf178 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_44], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf177, buf178, 16, grid=grid(16), stream=stream0)
        buf179 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_88], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf178, (4, 4, 1), (4, 1, 0), 0), out=buf179)
        buf180 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_45], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf179, buf180, 64, grid=grid(64), stream=stream0)
        buf181 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_89], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf181)
        buf182 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_45], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf181, buf182, 16, grid=grid(16), stream=stream0)
        buf183 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_90], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf182, (4, 4, 1), (4, 1, 0), 0), out=buf183)
        buf184 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_46], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf183, buf184, 64, grid=grid(64), stream=stream0)
        buf185 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_91], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf185)
        buf186 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_46], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf185, buf186, 16, grid=grid(16), stream=stream0)
        buf187 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_92], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf186, (4, 4, 1), (4, 1, 0), 0), out=buf187)
        buf188 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_47], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf187, buf188, 64, grid=grid(64), stream=stream0)
        buf189 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_93], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf188, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf189)
        buf190 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_47], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf189, buf190, 16, grid=grid(16), stream=stream0)
        buf191 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_94], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf190, (4, 4, 1), (4, 1, 0), 0), out=buf191)
        buf192 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_48], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf191, buf192, 64, grid=grid(64), stream=stream0)
        buf193 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_95], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf192, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf193)
        buf194 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_48], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf193, buf194, 16, grid=grid(16), stream=stream0)
        buf195 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_96], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf194, (4, 4, 1), (4, 1, 0), 0), out=buf195)
        buf196 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_49], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf195, buf196, 64, grid=grid(64), stream=stream0)
        buf197 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_97], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf196, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf197)
        buf198 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_49], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf197, buf198, 16, grid=grid(16), stream=stream0)
        buf199 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_98], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf198, (4, 4, 1), (4, 1, 0), 0), out=buf199)
        buf200 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_50], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf199, buf200, 64, grid=grid(64), stream=stream0)
        buf201 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_99], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf200, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf201)
        buf202 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_50], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf201, buf202, 16, grid=grid(16), stream=stream0)
        buf203 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_100], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf202, (4, 4, 1), (4, 1, 0), 0), out=buf203)
        buf204 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_51], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf203, buf204, 64, grid=grid(64), stream=stream0)
        buf205 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_101], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf204, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf205)
        buf206 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_51], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf205, buf206, 16, grid=grid(16), stream=stream0)
        buf207 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_102], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf206, (4, 4, 1), (4, 1, 0), 0), out=buf207)
        buf208 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_52], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf207, buf208, 64, grid=grid(64), stream=stream0)
        buf209 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_103], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf208, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf209)
        buf210 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_52], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf209, buf210, 16, grid=grid(16), stream=stream0)
        buf211 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_104], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf210, (4, 4, 1), (4, 1, 0), 0), out=buf211)
        buf212 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_53], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf211, buf212, 64, grid=grid(64), stream=stream0)
        buf213 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_105], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf212, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf213)
        buf214 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_53], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf213, buf214, 16, grid=grid(16), stream=stream0)
        buf215 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_106], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf214, (4, 4, 1), (4, 1, 0), 0), out=buf215)
        buf216 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_54], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf215, buf216, 64, grid=grid(64), stream=stream0)
        buf217 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_107], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf216, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf217)
        buf218 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_54], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf217, buf218, 16, grid=grid(16), stream=stream0)
        buf219 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_108], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf218, (4, 4, 1), (4, 1, 0), 0), out=buf219)
        buf220 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_55], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf219, buf220, 64, grid=grid(64), stream=stream0)
        buf221 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_109], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf221)
        buf222 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_55], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf221, buf222, 16, grid=grid(16), stream=stream0)
        buf223 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_110], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf222, (4, 4, 1), (4, 1, 0), 0), out=buf223)
        buf224 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_56], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf223, buf224, 64, grid=grid(64), stream=stream0)
        buf225 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_111], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf225)
        buf226 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_56], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf225, buf226, 16, grid=grid(16), stream=stream0)
        buf227 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_112], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf226, (4, 4, 1), (4, 1, 0), 0), out=buf227)
        buf228 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_57], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf227, buf228, 64, grid=grid(64), stream=stream0)
        buf229 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_113], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf228, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf229)
        buf230 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_57], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf229, buf230, 16, grid=grid(16), stream=stream0)
        buf231 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_114], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf230, (4, 4, 1), (4, 1, 0), 0), out=buf231)
        buf232 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_58], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf231, buf232, 64, grid=grid(64), stream=stream0)
        buf233 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_115], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf232, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf233)
        buf234 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_58], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf233, buf234, 16, grid=grid(16), stream=stream0)
        buf235 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_116], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf234, (4, 4, 1), (4, 1, 0), 0), out=buf235)
        buf236 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_59], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf235, buf236, 64, grid=grid(64), stream=stream0)
        buf237 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_117], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf236, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf237)
        buf238 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_59], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf237, buf238, 16, grid=grid(16), stream=stream0)
        buf239 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_118], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf238, (4, 4, 1), (4, 1, 0), 0), out=buf239)
        buf240 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_60], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf239, buf240, 64, grid=grid(64), stream=stream0)
        buf241 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_119], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf240, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf241)
        buf242 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_60], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf241, buf242, 16, grid=grid(16), stream=stream0)
        buf243 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_120], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf242, (4, 4, 1), (4, 1, 0), 0), out=buf243)
        buf244 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_61], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf243, buf244, 64, grid=grid(64), stream=stream0)
        buf245 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_121], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf244, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf245)
        buf246 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_61], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf245, buf246, 16, grid=grid(16), stream=stream0)
        buf247 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_122], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf246, (4, 4, 1), (4, 1, 0), 0), out=buf247)
        buf248 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_62], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf247, buf248, 64, grid=grid(64), stream=stream0)
        buf249 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_123], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf248, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf249)
        buf250 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_62], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf249, buf250, 16, grid=grid(16), stream=stream0)
        buf251 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_124], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf250, (4, 4, 1), (4, 1, 0), 0), out=buf251)
        buf252 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_63], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf251, buf252, 64, grid=grid(64), stream=stream0)
        buf253 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_125], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf252, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf253)
        buf254 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_63], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf253, buf254, 16, grid=grid(16), stream=stream0)
        buf255 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_126], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf254, (4, 4, 1), (4, 1, 0), 0), out=buf255)
        buf256 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_64], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf255, buf256, 64, grid=grid(64), stream=stream0)
        buf257 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_127], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf256, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf257)
        buf258 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_64], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf257, buf258, 16, grid=grid(16), stream=stream0)
        buf259 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_128], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf258, (4, 4, 1), (4, 1, 0), 0), out=buf259)
        buf260 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_65], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf259, buf260, 64, grid=grid(64), stream=stream0)
        buf261 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_129], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf260, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf261)
        buf262 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_65], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf261, buf262, 16, grid=grid(16), stream=stream0)
        buf263 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_130], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf262, (4, 4, 1), (4, 1, 0), 0), out=buf263)
        buf264 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_66], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf263, buf264, 64, grid=grid(64), stream=stream0)
        buf265 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_131], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf264, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf265)
        buf266 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_66], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf265, buf266, 16, grid=grid(16), stream=stream0)
        buf267 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_132], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf266, (4, 4, 1), (4, 1, 0), 0), out=buf267)
        buf268 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_67], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf267, buf268, 64, grid=grid(64), stream=stream0)
        buf269 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_133], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf268, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf269)
        buf270 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_67], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf269, buf270, 16, grid=grid(16), stream=stream0)
        buf271 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_134], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf270, (4, 4, 1), (4, 1, 0), 0), out=buf271)
        buf272 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_68], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf271, buf272, 64, grid=grid(64), stream=stream0)
        buf273 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_135], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf272, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf273)
        buf274 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_68], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf273, buf274, 16, grid=grid(16), stream=stream0)
        buf275 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_136], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf274, (4, 4, 1), (4, 1, 0), 0), out=buf275)
        buf276 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_69], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf275, buf276, 64, grid=grid(64), stream=stream0)
        buf277 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_137], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf276, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf277)
        buf278 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_69], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf277, buf278, 16, grid=grid(16), stream=stream0)
        buf279 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_138], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf278, (4, 4, 1), (4, 1, 0), 0), out=buf279)
        buf280 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_70], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf279, buf280, 64, grid=grid(64), stream=stream0)
        buf281 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_139], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf280, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf281)
        buf282 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_70], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf281, buf282, 16, grid=grid(16), stream=stream0)
        buf283 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_140], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf282, (4, 4, 1), (4, 1, 0), 0), out=buf283)
        buf284 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_71], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf283, buf284, 64, grid=grid(64), stream=stream0)
        buf285 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_141], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf284, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf285)
        buf286 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_71], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf285, buf286, 16, grid=grid(16), stream=stream0)
        buf287 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_142], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf286, (4, 4, 1), (4, 1, 0), 0), out=buf287)
        buf288 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_72], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf287, buf288, 64, grid=grid(64), stream=stream0)
        buf289 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_143], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf288, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf289)
        buf290 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_72], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf289, buf290, 16, grid=grid(16), stream=stream0)
        buf291 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_144], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf290, (4, 4, 1), (4, 1, 0), 0), out=buf291)
        buf292 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_73], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf291, buf292, 64, grid=grid(64), stream=stream0)
        buf293 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_145], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf293)
        buf294 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_73], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf293, buf294, 16, grid=grid(16), stream=stream0)
        buf295 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_146], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf294, (4, 4, 1), (4, 1, 0), 0), out=buf295)
        buf296 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_74], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf295, buf296, 64, grid=grid(64), stream=stream0)
        buf297 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_147], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf297)
        buf298 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_74], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf297, buf298, 16, grid=grid(16), stream=stream0)
        buf299 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_148], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf298, (4, 4, 1), (4, 1, 0), 0), out=buf299)
        buf300 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_75], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf299, buf300, 64, grid=grid(64), stream=stream0)
        buf301 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_149], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf300, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf301)
        buf302 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_75], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf301, buf302, 16, grid=grid(16), stream=stream0)
        buf303 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_150], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf302, (4, 4, 1), (4, 1, 0), 0), out=buf303)
        buf304 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_76], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf303, buf304, 64, grid=grid(64), stream=stream0)
        buf305 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_151], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf305)
        buf306 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_76], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf305, buf306, 16, grid=grid(16), stream=stream0)
        buf307 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_152], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf306, (4, 4, 1), (4, 1, 0), 0), out=buf307)
        buf308 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_77], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf307, buf308, 64, grid=grid(64), stream=stream0)
        buf309 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_153], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf308, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf309)
        buf310 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_77], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf309, buf310, 16, grid=grid(16), stream=stream0)
        buf311 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_154], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf310, (4, 4, 1), (4, 1, 0), 0), out=buf311)
        buf312 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_78], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf311, buf312, 64, grid=grid(64), stream=stream0)
        buf313 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_155], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf312, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf313)
        buf314 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_78], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf313, buf314, 16, grid=grid(16), stream=stream0)
        buf315 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_156], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf314, (4, 4, 1), (4, 1, 0), 0), out=buf315)
        buf316 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_79], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf315, buf316, 64, grid=grid(64), stream=stream0)
        buf317 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_157], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf316, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf317)
        buf318 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_79], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf317, buf318, 16, grid=grid(16), stream=stream0)
        buf319 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_158], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf318, (4, 4, 1), (4, 1, 0), 0), out=buf319)
        buf320 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_80], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf319, buf320, 64, grid=grid(64), stream=stream0)
        buf321 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_159], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf321)
        buf322 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_80], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf321, buf322, 16, grid=grid(16), stream=stream0)
        buf323 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_160], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf322, (4, 4, 1), (4, 1, 0), 0), out=buf323)
        buf324 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_81], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf323, buf324, 64, grid=grid(64), stream=stream0)
        buf325 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_161], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf325)
        buf326 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_81], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf325, buf326, 16, grid=grid(16), stream=stream0)
        buf327 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_162], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf326, (4, 4, 1), (4, 1, 0), 0), out=buf327)
        buf328 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_82], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf327, buf328, 64, grid=grid(64), stream=stream0)
        buf329 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_163], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf328, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf329)
        buf330 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_82], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf329, buf330, 16, grid=grid(16), stream=stream0)
        buf331 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_164], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf330, (4, 4, 1), (4, 1, 0), 0), out=buf331)
        buf332 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_83], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf331, buf332, 64, grid=grid(64), stream=stream0)
        buf333 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_165], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf332, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf333)
        buf334 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_83], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf333, buf334, 16, grid=grid(16), stream=stream0)
        buf335 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_166], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf334, (4, 4, 1), (4, 1, 0), 0), out=buf335)
        buf336 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_84], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf335, buf336, 64, grid=grid(64), stream=stream0)
        buf337 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_167], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf336, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf337)
        buf338 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_84], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf337, buf338, 16, grid=grid(16), stream=stream0)
        buf339 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_168], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf338, (4, 4, 1), (4, 1, 0), 0), out=buf339)
        buf340 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_85], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf339, buf340, 64, grid=grid(64), stream=stream0)
        buf341 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_169], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf341)
        buf342 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_85], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf341, buf342, 16, grid=grid(16), stream=stream0)
        buf343 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_170], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf342, (4, 4, 1), (4, 1, 0), 0), out=buf343)
        buf344 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_86], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf343, buf344, 64, grid=grid(64), stream=stream0)
        buf345 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_171], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf344, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf345)
        buf346 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_86], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf345, buf346, 16, grid=grid(16), stream=stream0)
        buf347 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_172], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf346, (4, 4, 1), (4, 1, 0), 0), out=buf347)
        buf348 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_87], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf347, buf348, 64, grid=grid(64), stream=stream0)
        buf349 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_173], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf348, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf349)
        buf350 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_87], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf349, buf350, 16, grid=grid(16), stream=stream0)
        buf351 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_174], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf350, (4, 4, 1), (4, 1, 0), 0), out=buf351)
        buf352 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_88], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf351, buf352, 64, grid=grid(64), stream=stream0)
        buf353 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_175], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf352, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf353)
        buf354 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_88], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf353, buf354, 16, grid=grid(16), stream=stream0)
        buf355 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_176], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf354, (4, 4, 1), (4, 1, 0), 0), out=buf355)
        buf356 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_89], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf355, buf356, 64, grid=grid(64), stream=stream0)
        buf357 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_177], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf356, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf357)
        buf358 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_89], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf357, buf358, 16, grid=grid(16), stream=stream0)
        buf359 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_178], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf358, (4, 4, 1), (4, 1, 0), 0), out=buf359)
        buf360 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_90], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf359, buf360, 64, grid=grid(64), stream=stream0)
        buf361 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_179], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf361)
        buf362 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_90], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf361, buf362, 16, grid=grid(16), stream=stream0)
        buf363 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_180], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf362, (4, 4, 1), (4, 1, 0), 0), out=buf363)
        buf364 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_91], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf363, buf364, 64, grid=grid(64), stream=stream0)
        buf365 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_181], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf364, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf365)
        buf366 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_91], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf365, buf366, 16, grid=grid(16), stream=stream0)
        buf367 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_182], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf366, (4, 4, 1), (4, 1, 0), 0), out=buf367)
        buf368 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_92], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf367, buf368, 64, grid=grid(64), stream=stream0)
        buf369 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_183], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf368, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf369)
        buf370 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_92], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf369, buf370, 16, grid=grid(16), stream=stream0)
        buf371 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_184], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf370, (4, 4, 1), (4, 1, 0), 0), out=buf371)
        buf372 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_93], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf371, buf372, 64, grid=grid(64), stream=stream0)
        buf373 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_185], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf373)
        buf374 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_93], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf373, buf374, 16, grid=grid(16), stream=stream0)
        buf375 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_186], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf374, (4, 4, 1), (4, 1, 0), 0), out=buf375)
        buf376 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_94], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf375, buf376, 64, grid=grid(64), stream=stream0)
        buf377 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_187], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf376, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf377)
        buf378 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_94], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf377, buf378, 16, grid=grid(16), stream=stream0)
        buf379 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_188], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf378, (4, 4, 1), (4, 1, 0), 0), out=buf379)
        buf380 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_95], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf379, buf380, 64, grid=grid(64), stream=stream0)
        buf381 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_189], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf380, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf381)
        buf382 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_95], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf381, buf382, 16, grid=grid(16), stream=stream0)
        buf383 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_190], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf382, (4, 4, 1), (4, 1, 0), 0), out=buf383)
        buf384 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_96], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf383, buf384, 64, grid=grid(64), stream=stream0)
        buf385 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_191], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf384, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf385)
        buf386 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_96], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf385, buf386, 16, grid=grid(16), stream=stream0)
        buf387 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_192], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf386, (4, 4, 1), (4, 1, 0), 0), out=buf387)
        buf388 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_97], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf387, buf388, 64, grid=grid(64), stream=stream0)
        buf389 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_193], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf388, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf389)
        buf390 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_97], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf389, buf390, 16, grid=grid(16), stream=stream0)
        buf391 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_194], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf390, (4, 4, 1), (4, 1, 0), 0), out=buf391)
        buf392 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_98], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf391, buf392, 64, grid=grid(64), stream=stream0)
        buf393 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_195], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf392, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf393)
        buf394 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_98], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf393, buf394, 16, grid=grid(16), stream=stream0)
        buf395 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_196], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf394, (4, 4, 1), (4, 1, 0), 0), out=buf395)
        buf396 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_99], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf395, buf396, 64, grid=grid(64), stream=stream0)
        buf397 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_197], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf396, (4, 1, 16), (16, 0, 1), 0), buf1, out=buf397)
        buf398 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_99], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_3.run(buf397, buf398, 16, grid=grid(16), stream=stream0)
        buf399 = empty_strided_cuda((4, 16, 1), (16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_198], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, reinterpret_tensor(buf398, (4, 4, 1), (4, 1, 0), 0), out=buf399)
        buf400 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [u_100], Original ATen: [aten.reciprocal, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_reciprocal_2.run(buf399, buf400, 64, grid=grid(64), stream=stream0)
        buf401 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_199], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf400, (4, 1, 16), (16, 16, 1), 0), buf1, out=buf401)
        buf402 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_4.run(buf1, buf401, buf402, 256, grid=grid(256), stream=stream0)
        buf403 = empty_strided_cuda((4, 4, 1, 16), (64, 16, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [K_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf400, buf402, buf403, 16, 16, grid=grid(16, 16), stream=stream0)
        buf404 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf403, (4, 4, 16), (64, 16, 1), 0), reinterpret_tensor(primals_1, (4, 16, 4), (64, 4, 1), 0), out=buf404)
        del buf403
    return (reinterpret_tensor(buf404, (4, 2, 2, 4), (16, 8, 4, 1), 0), reinterpret_tensor(primals_1, (4, 16, 4), (64, 4, 1), 0), buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf51, buf53, buf55, buf57, buf59, buf61, buf63, buf65, buf67, buf69, buf71, buf73, buf75, buf77, buf79, buf81, buf83, buf85, buf87, buf89, buf91, buf93, buf95, buf97, buf99, buf101, buf103, buf105, buf107, buf109, buf111, buf113, buf115, buf117, buf119, buf121, buf123, buf125, buf127, buf129, buf131, buf133, buf135, buf137, buf139, buf141, buf143, buf145, buf147, buf149, buf151, buf153, buf155, buf157, buf159, buf161, buf163, buf165, buf167, buf169, buf171, buf173, buf175, buf177, buf179, buf181, buf183, buf185, buf187, buf189, buf191, buf193, buf195, buf197, buf199, buf201, buf203, buf205, buf207, buf209, buf211, buf213, buf215, buf217, buf219, buf221, buf223, buf225, buf227, buf229, buf231, buf233, buf235, buf237, buf239, buf241, buf243, buf245, buf247, buf249, buf251, buf253, buf255, buf257, buf259, buf261, buf263, buf265, buf267, buf269, buf271, buf273, buf275, buf277, buf279, buf281, buf283, buf285, buf287, buf289, buf291, buf293, buf295, buf297, buf299, buf301, buf303, buf305, buf307, buf309, buf311, buf313, buf315, buf317, buf319, buf321, buf323, buf325, buf327, buf329, buf331, buf333, buf335, buf337, buf339, buf341, buf343, buf345, buf347, buf349, buf351, buf353, buf355, buf357, buf359, buf361, buf363, buf365, buf367, buf369, buf371, buf373, buf375, buf377, buf379, buf381, buf383, buf385, buf387, buf389, buf391, buf393, buf395, buf397, buf399, buf400, buf401, buf402, reinterpret_tensor(buf398, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf396, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf394, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf392, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf390, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf388, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf386, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf384, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf382, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf380, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf378, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf376, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf374, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf372, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf370, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf368, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf366, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf364, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf362, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf360, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf358, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf356, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf354, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf352, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf350, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf348, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf346, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf344, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf342, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf340, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf338, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf336, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf334, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf332, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf330, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf328, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf326, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf324, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf322, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf320, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf318, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf316, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf314, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf312, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf310, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf308, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf306, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf304, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf302, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf300, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf298, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf296, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf294, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf292, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf290, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf288, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf286, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf284, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf282, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf280, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf278, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf276, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf274, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf272, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf270, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf268, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf266, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf264, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf262, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf260, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf258, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf256, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf254, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf252, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf250, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf248, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf246, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf244, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf242, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf240, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf238, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf236, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf234, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf232, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf230, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf228, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf226, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf224, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf222, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf220, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf218, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf216, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf214, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf212, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf210, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf208, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf206, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf204, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf202, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf200, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf198, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf196, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf194, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf192, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf190, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf188, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf186, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf184, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf182, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf180, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf178, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf176, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf174, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf172, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf170, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf168, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf166, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf164, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf162, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf160, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf158, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf156, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf154, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf152, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf150, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf148, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf146, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf144, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf142, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf140, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf138, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf136, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf134, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf132, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf130, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf128, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf126, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf124, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf122, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf120, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf118, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf116, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf114, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf112, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf110, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf108, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf106, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf104, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf102, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf100, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf98, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf96, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf94, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf92, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf90, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf88, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf86, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf84, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf82, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf80, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf78, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf76, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf74, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf72, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf70, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf68, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf66, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf64, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf62, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf60, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf58, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf56, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf54, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf52, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf50, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf48, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf46, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf44, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf42, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf40, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf38, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf36, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf34, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf32, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf30, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf28, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf26, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf24, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf22, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf20, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf18, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf16, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf14, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf12, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf10, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf8, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf6, (4, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf4, (4, 16, 1), (16, 1, 16), 0), reinterpret_tensor(buf2, (4, 1, 4), (4, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
