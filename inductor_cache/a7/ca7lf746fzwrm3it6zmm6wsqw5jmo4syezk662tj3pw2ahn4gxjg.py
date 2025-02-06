# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/cy/ccyyi3yzwd3lp3ghjey3t3ldkvlcg6pnyhzuaerh3kjag6lh7oyx.py
# Topologically Sorted Source Nodes: [x1], Original ATen: [aten.elu]
# Source node to ATen node mapping:
#   x1 => expm1, gt, mul, mul_2, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mm, 0), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul, %mul_2), kwargs = {})
triton_poi_fused_elu_0 = async_compile.triton('triton_poi_fused_elu_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zf/czfjrciwflddbylqck7tkxhjdhd75hr7k4qeopidaz6tpgcu4z2s.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%_trilinear, %_trilinear_1, %_trilinear_2, %_trilinear_3, %_trilinear_4, %_trilinear_5, %_trilinear_6, %_trilinear_7, %_trilinear_8, %_trilinear_9, %_trilinear_10, %_trilinear_11, %_trilinear_12, %_trilinear_13, %_trilinear_14, %_trilinear_15, %_trilinear_16, %_trilinear_17, %_trilinear_18, %_trilinear_19, %_trilinear_20, %_trilinear_21, %_trilinear_22, %_trilinear_23, %_trilinear_24, %_trilinear_25, %_trilinear_26, %_trilinear_27, %_trilinear_28, %_trilinear_29, %_trilinear_30, %_trilinear_31], 1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 128*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4w/c4wedljbeocuk4a6uttelxjhb2qwb7tu7ehouoju677dlfb2lxx4.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%_trilinear, %_trilinear_1, %_trilinear_2, %_trilinear_3, %_trilinear_4, %_trilinear_5, %_trilinear_6, %_trilinear_7, %_trilinear_8, %_trilinear_9, %_trilinear_10, %_trilinear_11, %_trilinear_12, %_trilinear_13, %_trilinear_14, %_trilinear_15, %_trilinear_16, %_trilinear_17, %_trilinear_18, %_trilinear_19, %_trilinear_20, %_trilinear_21, %_trilinear_22, %_trilinear_23, %_trilinear_24, %_trilinear_25, %_trilinear_26, %_trilinear_27, %_trilinear_28, %_trilinear_29, %_trilinear_30, %_trilinear_31], 1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 128*x1), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37 = args
    args.clear()
    assert_size_stride(primals_1, (128, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (128, 4), (4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_6, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_7, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_8, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_9, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_10, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_11, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_12, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_13, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_14, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_15, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_16, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_17, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_18, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_19, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_20, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_21, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_22, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_23, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_24, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_25, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_26, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_27, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_28, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_29, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_30, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_31, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_32, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_33, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_34, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_35, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_36, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_37, (4, 128), (128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(primals_2, reinterpret_tensor(primals_1, (4, 128), (1, 4), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(primals_4, reinterpret_tensor(primals_3, (4, 128), (1, 4), 0), out=buf1)
        del primals_3
        buf2 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x1], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf0, buf2, 512, grid=grid(512), stream=stream0)
        buf3 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x2], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf1, buf3, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [bilinear], Original ATen: [aten._trilinear]
        buf4 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 0), primals_5, reinterpret_tensor(buf3, (4, 4), (128, 1), 0), [1, 3], [0], [1, 2], [2, 3])
        buf5 = buf4
        del buf4
        # Topologically Sorted Source Nodes: [bilinear_1], Original ATen: [aten._trilinear]
        buf6 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 4), primals_6, reinterpret_tensor(buf3, (4, 4), (128, 1), 4), [1, 3], [0], [1, 2], [2, 3])
        buf7 = buf6
        del buf6
        # Topologically Sorted Source Nodes: [bilinear_2], Original ATen: [aten._trilinear]
        buf8 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 8), primals_7, reinterpret_tensor(buf3, (4, 4), (128, 1), 8), [1, 3], [0], [1, 2], [2, 3])
        buf9 = buf8
        del buf8
        # Topologically Sorted Source Nodes: [bilinear_3], Original ATen: [aten._trilinear]
        buf10 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 12), primals_8, reinterpret_tensor(buf3, (4, 4), (128, 1), 12), [1, 3], [0], [1, 2], [2, 3])
        buf11 = buf10
        del buf10
        # Topologically Sorted Source Nodes: [bilinear_4], Original ATen: [aten._trilinear]
        buf12 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 16), primals_9, reinterpret_tensor(buf3, (4, 4), (128, 1), 16), [1, 3], [0], [1, 2], [2, 3])
        buf13 = buf12
        del buf12
        # Topologically Sorted Source Nodes: [bilinear_5], Original ATen: [aten._trilinear]
        buf14 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 20), primals_10, reinterpret_tensor(buf3, (4, 4), (128, 1), 20), [1, 3], [0], [1, 2], [2, 3])
        buf15 = buf14
        del buf14
        # Topologically Sorted Source Nodes: [bilinear_6], Original ATen: [aten._trilinear]
        buf16 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 24), primals_11, reinterpret_tensor(buf3, (4, 4), (128, 1), 24), [1, 3], [0], [1, 2], [2, 3])
        buf17 = buf16
        del buf16
        # Topologically Sorted Source Nodes: [bilinear_7], Original ATen: [aten._trilinear]
        buf18 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 28), primals_12, reinterpret_tensor(buf3, (4, 4), (128, 1), 28), [1, 3], [0], [1, 2], [2, 3])
        buf19 = buf18
        del buf18
        # Topologically Sorted Source Nodes: [bilinear_8], Original ATen: [aten._trilinear]
        buf20 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 32), primals_13, reinterpret_tensor(buf3, (4, 4), (128, 1), 32), [1, 3], [0], [1, 2], [2, 3])
        buf21 = buf20
        del buf20
        # Topologically Sorted Source Nodes: [bilinear_9], Original ATen: [aten._trilinear]
        buf22 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 36), primals_14, reinterpret_tensor(buf3, (4, 4), (128, 1), 36), [1, 3], [0], [1, 2], [2, 3])
        buf23 = buf22
        del buf22
        # Topologically Sorted Source Nodes: [bilinear_10], Original ATen: [aten._trilinear]
        buf24 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 40), primals_15, reinterpret_tensor(buf3, (4, 4), (128, 1), 40), [1, 3], [0], [1, 2], [2, 3])
        buf25 = buf24
        del buf24
        # Topologically Sorted Source Nodes: [bilinear_11], Original ATen: [aten._trilinear]
        buf26 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 44), primals_16, reinterpret_tensor(buf3, (4, 4), (128, 1), 44), [1, 3], [0], [1, 2], [2, 3])
        buf27 = buf26
        del buf26
        # Topologically Sorted Source Nodes: [bilinear_12], Original ATen: [aten._trilinear]
        buf28 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 48), primals_17, reinterpret_tensor(buf3, (4, 4), (128, 1), 48), [1, 3], [0], [1, 2], [2, 3])
        buf29 = buf28
        del buf28
        # Topologically Sorted Source Nodes: [bilinear_13], Original ATen: [aten._trilinear]
        buf30 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 52), primals_18, reinterpret_tensor(buf3, (4, 4), (128, 1), 52), [1, 3], [0], [1, 2], [2, 3])
        buf31 = buf30
        del buf30
        # Topologically Sorted Source Nodes: [bilinear_14], Original ATen: [aten._trilinear]
        buf32 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 56), primals_19, reinterpret_tensor(buf3, (4, 4), (128, 1), 56), [1, 3], [0], [1, 2], [2, 3])
        buf33 = buf32
        del buf32
        # Topologically Sorted Source Nodes: [bilinear_15], Original ATen: [aten._trilinear]
        buf34 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 60), primals_20, reinterpret_tensor(buf3, (4, 4), (128, 1), 60), [1, 3], [0], [1, 2], [2, 3])
        buf35 = buf34
        del buf34
        # Topologically Sorted Source Nodes: [bilinear_16], Original ATen: [aten._trilinear]
        buf36 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 64), primals_21, reinterpret_tensor(buf3, (4, 4), (128, 1), 64), [1, 3], [0], [1, 2], [2, 3])
        buf37 = buf36
        del buf36
        # Topologically Sorted Source Nodes: [bilinear_17], Original ATen: [aten._trilinear]
        buf38 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 68), primals_22, reinterpret_tensor(buf3, (4, 4), (128, 1), 68), [1, 3], [0], [1, 2], [2, 3])
        buf39 = buf38
        del buf38
        # Topologically Sorted Source Nodes: [bilinear_18], Original ATen: [aten._trilinear]
        buf40 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 72), primals_23, reinterpret_tensor(buf3, (4, 4), (128, 1), 72), [1, 3], [0], [1, 2], [2, 3])
        buf41 = buf40
        del buf40
        # Topologically Sorted Source Nodes: [bilinear_19], Original ATen: [aten._trilinear]
        buf42 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 76), primals_24, reinterpret_tensor(buf3, (4, 4), (128, 1), 76), [1, 3], [0], [1, 2], [2, 3])
        buf43 = buf42
        del buf42
        # Topologically Sorted Source Nodes: [bilinear_20], Original ATen: [aten._trilinear]
        buf44 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 80), primals_25, reinterpret_tensor(buf3, (4, 4), (128, 1), 80), [1, 3], [0], [1, 2], [2, 3])
        buf45 = buf44
        del buf44
        # Topologically Sorted Source Nodes: [bilinear_21], Original ATen: [aten._trilinear]
        buf46 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 84), primals_26, reinterpret_tensor(buf3, (4, 4), (128, 1), 84), [1, 3], [0], [1, 2], [2, 3])
        buf47 = buf46
        del buf46
        # Topologically Sorted Source Nodes: [bilinear_22], Original ATen: [aten._trilinear]
        buf48 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 88), primals_27, reinterpret_tensor(buf3, (4, 4), (128, 1), 88), [1, 3], [0], [1, 2], [2, 3])
        buf49 = buf48
        del buf48
        # Topologically Sorted Source Nodes: [bilinear_23], Original ATen: [aten._trilinear]
        buf50 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 92), primals_28, reinterpret_tensor(buf3, (4, 4), (128, 1), 92), [1, 3], [0], [1, 2], [2, 3])
        buf51 = buf50
        del buf50
        # Topologically Sorted Source Nodes: [bilinear_24], Original ATen: [aten._trilinear]
        buf52 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 96), primals_29, reinterpret_tensor(buf3, (4, 4), (128, 1), 96), [1, 3], [0], [1, 2], [2, 3])
        buf53 = buf52
        del buf52
        # Topologically Sorted Source Nodes: [bilinear_25], Original ATen: [aten._trilinear]
        buf54 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 100), primals_30, reinterpret_tensor(buf3, (4, 4), (128, 1), 100), [1, 3], [0], [1, 2], [2, 3])
        buf55 = buf54
        del buf54
        # Topologically Sorted Source Nodes: [bilinear_26], Original ATen: [aten._trilinear]
        buf56 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 104), primals_31, reinterpret_tensor(buf3, (4, 4), (128, 1), 104), [1, 3], [0], [1, 2], [2, 3])
        buf57 = buf56
        del buf56
        # Topologically Sorted Source Nodes: [bilinear_27], Original ATen: [aten._trilinear]
        buf58 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 108), primals_32, reinterpret_tensor(buf3, (4, 4), (128, 1), 108), [1, 3], [0], [1, 2], [2, 3])
        buf59 = buf58
        del buf58
        # Topologically Sorted Source Nodes: [bilinear_28], Original ATen: [aten._trilinear]
        buf60 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 112), primals_33, reinterpret_tensor(buf3, (4, 4), (128, 1), 112), [1, 3], [0], [1, 2], [2, 3])
        buf61 = buf60
        del buf60
        # Topologically Sorted Source Nodes: [bilinear_29], Original ATen: [aten._trilinear]
        buf62 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 116), primals_34, reinterpret_tensor(buf3, (4, 4), (128, 1), 116), [1, 3], [0], [1, 2], [2, 3])
        buf63 = buf62
        del buf62
        # Topologically Sorted Source Nodes: [bilinear_30], Original ATen: [aten._trilinear]
        buf64 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 120), primals_35, reinterpret_tensor(buf3, (4, 4), (128, 1), 120), [1, 3], [0], [1, 2], [2, 3])
        buf65 = buf64
        del buf64
        # Topologically Sorted Source Nodes: [bilinear_31], Original ATen: [aten._trilinear]
        buf66 = torch.ops.aten._trilinear.default(reinterpret_tensor(buf2, (4, 4), (128, 1), 124), primals_36, reinterpret_tensor(buf3, (4, 4), (128, 1), 124), [1, 3], [0], [1, 2], [2, 3])
        buf67 = buf66
        del buf66
        buf100 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        buf68 = reinterpret_tensor(buf100, (4, 4), (128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf5, buf68, 16, grid=grid(16), stream=stream0)
        del buf5
        buf69 = reinterpret_tensor(buf100, (4, 4), (128, 1), 4)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf7, buf69, 16, grid=grid(16), stream=stream0)
        del buf7
        buf70 = reinterpret_tensor(buf100, (4, 4), (128, 1), 8)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf9, buf70, 16, grid=grid(16), stream=stream0)
        del buf9
        buf71 = reinterpret_tensor(buf100, (4, 4), (128, 1), 12)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf11, buf71, 16, grid=grid(16), stream=stream0)
        del buf11
        buf72 = reinterpret_tensor(buf100, (4, 4), (128, 1), 16)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf13, buf72, 16, grid=grid(16), stream=stream0)
        del buf13
        buf73 = reinterpret_tensor(buf100, (4, 4), (128, 1), 20)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf15, buf73, 16, grid=grid(16), stream=stream0)
        del buf15
        buf74 = reinterpret_tensor(buf100, (4, 4), (128, 1), 24)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf17, buf74, 16, grid=grid(16), stream=stream0)
        del buf17
        buf75 = reinterpret_tensor(buf100, (4, 4), (128, 1), 28)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf19, buf75, 16, grid=grid(16), stream=stream0)
        del buf19
        buf76 = reinterpret_tensor(buf100, (4, 4), (128, 1), 32)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf21, buf76, 16, grid=grid(16), stream=stream0)
        del buf21
        buf77 = reinterpret_tensor(buf100, (4, 4), (128, 1), 36)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf23, buf77, 16, grid=grid(16), stream=stream0)
        del buf23
        buf78 = reinterpret_tensor(buf100, (4, 4), (128, 1), 40)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf25, buf78, 16, grid=grid(16), stream=stream0)
        del buf25
        buf79 = reinterpret_tensor(buf100, (4, 4), (128, 1), 44)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf27, buf79, 16, grid=grid(16), stream=stream0)
        del buf27
        buf80 = reinterpret_tensor(buf100, (4, 4), (128, 1), 48)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf29, buf80, 16, grid=grid(16), stream=stream0)
        del buf29
        buf81 = reinterpret_tensor(buf100, (4, 4), (128, 1), 52)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf31, buf81, 16, grid=grid(16), stream=stream0)
        del buf31
        buf82 = reinterpret_tensor(buf100, (4, 4), (128, 1), 56)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf33, buf82, 16, grid=grid(16), stream=stream0)
        del buf33
        buf83 = reinterpret_tensor(buf100, (4, 4), (128, 1), 60)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf35, buf83, 16, grid=grid(16), stream=stream0)
        del buf35
        buf84 = reinterpret_tensor(buf100, (4, 4), (128, 1), 64)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf37, buf84, 16, grid=grid(16), stream=stream0)
        del buf37
        buf85 = reinterpret_tensor(buf100, (4, 4), (128, 1), 68)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf39, buf85, 16, grid=grid(16), stream=stream0)
        del buf39
        buf86 = reinterpret_tensor(buf100, (4, 4), (128, 1), 72)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf41, buf86, 16, grid=grid(16), stream=stream0)
        del buf41
        buf87 = reinterpret_tensor(buf100, (4, 4), (128, 1), 76)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf43, buf87, 16, grid=grid(16), stream=stream0)
        del buf43
        buf88 = reinterpret_tensor(buf100, (4, 4), (128, 1), 80)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf45, buf88, 16, grid=grid(16), stream=stream0)
        del buf45
        buf89 = reinterpret_tensor(buf100, (4, 4), (128, 1), 84)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf47, buf89, 16, grid=grid(16), stream=stream0)
        del buf47
        buf90 = reinterpret_tensor(buf100, (4, 4), (128, 1), 88)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf49, buf90, 16, grid=grid(16), stream=stream0)
        del buf49
        buf91 = reinterpret_tensor(buf100, (4, 4), (128, 1), 92)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf51, buf91, 16, grid=grid(16), stream=stream0)
        del buf51
        buf92 = reinterpret_tensor(buf100, (4, 4), (128, 1), 96)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf53, buf92, 16, grid=grid(16), stream=stream0)
        del buf53
        buf93 = reinterpret_tensor(buf100, (4, 4), (128, 1), 100)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf55, buf93, 16, grid=grid(16), stream=stream0)
        del buf55
        buf94 = reinterpret_tensor(buf100, (4, 4), (128, 1), 104)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf57, buf94, 16, grid=grid(16), stream=stream0)
        del buf57
        buf95 = reinterpret_tensor(buf100, (4, 4), (128, 1), 108)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf59, buf95, 16, grid=grid(16), stream=stream0)
        del buf59
        buf96 = reinterpret_tensor(buf100, (4, 4), (128, 1), 112)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf61, buf96, 16, grid=grid(16), stream=stream0)
        del buf61
        buf97 = reinterpret_tensor(buf100, (4, 4), (128, 1), 116)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf63, buf97, 16, grid=grid(16), stream=stream0)
        del buf63
        buf98 = reinterpret_tensor(buf100, (4, 4), (128, 1), 120)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf65, buf98, 16, grid=grid(16), stream=stream0)
        del buf65
        buf99 = reinterpret_tensor(buf100, (4, 4), (128, 1), 124)  # alias
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf67, buf99, 16, grid=grid(16), stream=stream0)
        buf101 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf100, buf101, 512, grid=grid(512), stream=stream0)
        buf102 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf101, reinterpret_tensor(primals_37, (128, 4), (1, 128), 0), out=buf102)
    return (buf102, primals_2, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, buf0, buf1, reinterpret_tensor(buf2, (4, 4), (128, 1), 0), reinterpret_tensor(buf2, (4, 4), (128, 1), 4), reinterpret_tensor(buf2, (4, 4), (128, 1), 8), reinterpret_tensor(buf2, (4, 4), (128, 1), 12), reinterpret_tensor(buf2, (4, 4), (128, 1), 16), reinterpret_tensor(buf2, (4, 4), (128, 1), 20), reinterpret_tensor(buf2, (4, 4), (128, 1), 24), reinterpret_tensor(buf2, (4, 4), (128, 1), 28), reinterpret_tensor(buf2, (4, 4), (128, 1), 32), reinterpret_tensor(buf2, (4, 4), (128, 1), 36), reinterpret_tensor(buf2, (4, 4), (128, 1), 40), reinterpret_tensor(buf2, (4, 4), (128, 1), 44), reinterpret_tensor(buf2, (4, 4), (128, 1), 48), reinterpret_tensor(buf2, (4, 4), (128, 1), 52), reinterpret_tensor(buf2, (4, 4), (128, 1), 56), reinterpret_tensor(buf2, (4, 4), (128, 1), 60), reinterpret_tensor(buf2, (4, 4), (128, 1), 64), reinterpret_tensor(buf2, (4, 4), (128, 1), 68), reinterpret_tensor(buf2, (4, 4), (128, 1), 72), reinterpret_tensor(buf2, (4, 4), (128, 1), 76), reinterpret_tensor(buf2, (4, 4), (128, 1), 80), reinterpret_tensor(buf2, (4, 4), (128, 1), 84), reinterpret_tensor(buf2, (4, 4), (128, 1), 88), reinterpret_tensor(buf2, (4, 4), (128, 1), 92), reinterpret_tensor(buf2, (4, 4), (128, 1), 96), reinterpret_tensor(buf2, (4, 4), (128, 1), 100), reinterpret_tensor(buf2, (4, 4), (128, 1), 104), reinterpret_tensor(buf2, (4, 4), (128, 1), 108), reinterpret_tensor(buf2, (4, 4), (128, 1), 112), reinterpret_tensor(buf2, (4, 4), (128, 1), 116), reinterpret_tensor(buf2, (4, 4), (128, 1), 120), reinterpret_tensor(buf2, (4, 4), (128, 1), 124), reinterpret_tensor(buf3, (4, 4), (128, 1), 0), reinterpret_tensor(buf3, (4, 4), (128, 1), 4), reinterpret_tensor(buf3, (4, 4), (128, 1), 8), reinterpret_tensor(buf3, (4, 4), (128, 1), 12), reinterpret_tensor(buf3, (4, 4), (128, 1), 16), reinterpret_tensor(buf3, (4, 4), (128, 1), 20), reinterpret_tensor(buf3, (4, 4), (128, 1), 24), reinterpret_tensor(buf3, (4, 4), (128, 1), 28), reinterpret_tensor(buf3, (4, 4), (128, 1), 32), reinterpret_tensor(buf3, (4, 4), (128, 1), 36), reinterpret_tensor(buf3, (4, 4), (128, 1), 40), reinterpret_tensor(buf3, (4, 4), (128, 1), 44), reinterpret_tensor(buf3, (4, 4), (128, 1), 48), reinterpret_tensor(buf3, (4, 4), (128, 1), 52), reinterpret_tensor(buf3, (4, 4), (128, 1), 56), reinterpret_tensor(buf3, (4, 4), (128, 1), 60), reinterpret_tensor(buf3, (4, 4), (128, 1), 64), reinterpret_tensor(buf3, (4, 4), (128, 1), 68), reinterpret_tensor(buf3, (4, 4), (128, 1), 72), reinterpret_tensor(buf3, (4, 4), (128, 1), 76), reinterpret_tensor(buf3, (4, 4), (128, 1), 80), reinterpret_tensor(buf3, (4, 4), (128, 1), 84), reinterpret_tensor(buf3, (4, 4), (128, 1), 88), reinterpret_tensor(buf3, (4, 4), (128, 1), 92), reinterpret_tensor(buf3, (4, 4), (128, 1), 96), reinterpret_tensor(buf3, (4, 4), (128, 1), 100), reinterpret_tensor(buf3, (4, 4), (128, 1), 104), reinterpret_tensor(buf3, (4, 4), (128, 1), 108), reinterpret_tensor(buf3, (4, 4), (128, 1), 112), reinterpret_tensor(buf3, (4, 4), (128, 1), 116), reinterpret_tensor(buf3, (4, 4), (128, 1), 120), reinterpret_tensor(buf3, (4, 4), (128, 1), 124), buf100, buf101, primals_37, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
