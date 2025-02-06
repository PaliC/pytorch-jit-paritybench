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


# kernel path: inductor_cache/ln/clnxtofo4e7d7gqpaohssg5u2e34k72wivczwn36eozvzohqeysx.py
# Topologically Sorted Source Nodes: [D], Original ATen: [aten.diag_embed]
# Source node to ATen node mapping:
#   D => eq, full_default, iota, where
# Graph fragment:
#   %iota : [num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%iota, %unsqueeze_1), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %permute, %full_default), kwargs = {})
triton_poi_fused_diag_embed_0 = async_compile.triton('triton_poi_fused_diag_embed_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_diag_embed_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_diag_embed_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 == tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = -0.5
    tmp11 = libdevice.pow(tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fl/cflfytx523md3tr6e65zwe75xere2bbkm5t2iyscvbrx3pjkqdcj.py
# Topologically Sorted Source Nodes: [D, eye, L], Original ATen: [aten.diag_embed, aten.eye, aten.sub]
# Source node to ATen node mapping:
#   D => full_default, iota
#   L => sub
#   eye => eq_1, full_default_1, where_1
# Graph fragment:
#   %iota : [num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_1, %iota), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_1, %full_default), kwargs = {})
#   %sub : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %mm_1), kwargs = {})
triton_poi_fused_diag_embed_eye_sub_1 = async_compile.triton('triton_poi_fused_diag_embed_eye_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_diag_embed_eye_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_diag_embed_eye_sub_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp6 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp0 = x1
    tmp1 = x0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 - tmp6
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6f/c6fott5pxqyarc56fxjtoi4uwcxnuhq67wpagcgvtncfz3vyziza.py
# Topologically Sorted Source Nodes: [D, eye, multi_order_laplacian], Original ATen: [aten.diag_embed, aten.eye, aten.zeros]
# Source node to ATen node mapping:
#   D => full_default, iota
#   eye => eq_1, full_default_1, where_1
#   multi_order_laplacian => full_default_3
# Graph fragment:
#   %iota : [num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_1, %iota), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_1, %full_default), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([5, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_3, %where_1, 0, 0), kwargs = {})
#   %select_scatter_default_1 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %sub, 0, 1), kwargs = {})
triton_poi_fused_diag_embed_eye_zeros_2 = async_compile.triton('triton_poi_fused_diag_embed_eye_zeros_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_diag_embed_eye_zeros_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_diag_embed_eye_zeros_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16
    x3 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = x1
    tmp7 = x0
    tmp8 = tmp6 == tmp7
    tmp9 = 1.0
    tmp10 = 0.0
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp5, tmp11, tmp10)
    tmp13 = tl.where(tmp2, tmp3, tmp12)
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ya/cyawpjbnx44rvy2naljxuhgckj3pxldinmamryu776kkmlpn7zsy.py
# Topologically Sorted Source Nodes: [mul, sub_1], Original ATen: [aten.mul, aten.sub]
# Source node to ATen node mapping:
#   mul => mul
#   sub_1 => sub_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_2, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %select_8), kwargs = {})
#   %select_scatter_default_2 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %sub_1, 0, 2), kwargs = {})
triton_poi_fused_mul_sub_3 = async_compile.triton('triton_poi_fused_mul_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sub_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x0 = (xindex % 16)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7w/c7wslyijb7lm2oxxonw4erjd5w2tfmfhdjd2ldrmey7k2yjs2l22.py
# Topologically Sorted Source Nodes: [mul_1, sub_2], Original ATen: [aten.mul, aten.sub]
# Source node to ATen node mapping:
#   mul_1 => mul_1
#   sub_2 => sub_2
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_3, 2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, %select_15), kwargs = {})
#   %select_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %sub_2, 0, 3), kwargs = {})
triton_poi_fused_mul_sub_4 = async_compile.triton('triton_poi_fused_mul_sub_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sub_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sub_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x0 = (xindex % 16)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (16 + x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u4/cu4axqwvunn4sv3e2q7ugjdeyuturkx3ayjycji7p63b2eyvtc5i.py
# Topologically Sorted Source Nodes: [mul_2, sub_3], Original ATen: [aten.mul, aten.sub]
# Source node to ATen node mapping:
#   mul_2 => mul_2
#   sub_3 => sub_3
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_4, 2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %select_22), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_3, %sub_3, 0, 4), kwargs = {})
triton_poi_fused_mul_sub_5 = async_compile.triton('triton_poi_fused_mul_sub_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sub_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sub_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16
    x0 = (xindex % 16)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (32 + x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 4, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 - tmp6
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4q/c4qdxgykhvqtrkmn5fh5ufcu6eatuxku6kzgyspf6getpme2pt2n.py
# Topologically Sorted Source Nodes: [sum_2, result_2], Original ATen: [aten.sum, aten.add]
# Source node to ATen node mapping:
#   result_2 => add
#   sum_2 => sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_6, [0]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, %primals_4), kwargs = {})
triton_poi_fused_add_sum_6 = async_compile.triton('triton_poi_fused_add_sum_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sum_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sum_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x2), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x2), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x2), xmask)
    tmp7 = tl.load(in_ptr0 + (64 + x2), xmask)
    tmp9 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (5, 1, 4, 4), (16, 16, 4, 1))
    assert_size_stride(primals_4, (1, 1, 4), (4, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [D], Original ATen: [aten.diag_embed]
        stream0 = get_raw_stream(0)
        triton_poi_fused_diag_embed_0.run(primals_1, buf0, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, primals_1, out=buf1)
        del primals_1
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mm_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1, buf0, out=buf2)
        del buf0
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [D, eye, L], Original ATen: [aten.diag_embed, aten.eye, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_diag_embed_eye_sub_1.run(buf3, 16, grid=grid(16), stream=stream0)
        buf4 = empty_strided_cuda((5, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [D, eye, multi_order_laplacian], Original ATen: [aten.diag_embed, aten.eye, aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_diag_embed_eye_zeros_2.run(buf3, buf4, 80, grid=grid(80), stream=stream0)
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [mm_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(buf4, (4, 4), (4, 1), 16), out=buf5)
        buf6 = empty_strided_cuda((5, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, sub_1], Original ATen: [aten.mul, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sub_3.run(buf5, buf4, buf6, 80, grid=grid(80), stream=stream0)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [mm_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(buf6, (4, 4), (4, 1), 32), out=buf7)
        buf8 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [mul_1, sub_2], Original ATen: [aten.mul, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sub_4.run(buf7, buf6, buf8, 80, grid=grid(80), stream=stream0)
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [mm_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(buf8, (4, 4), (4, 1), 48), out=buf9)
        del buf3
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [mul_2, sub_3], Original ATen: [aten.mul, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sub_5.run(buf9, buf8, buf10, 80, grid=grid(80), stream=stream0)
        buf11 = reinterpret_tensor(buf8, (20, 4), (4, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [result], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (20, 4), (4, 1), 0), primals_2, out=buf11)
        del primals_2
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [result_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (5, 4, 4), (16, 4, 1), 0), reinterpret_tensor(primals_3, (5, 4, 4), (16, 4, 1), 0), out=buf12)
        del primals_3
        buf13 = reinterpret_tensor(buf9, (1, 4, 4), (16, 4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [sum_2, result_2], Original ATen: [aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sum_6.run(buf12, primals_4, buf13, 16, grid=grid(16), stream=stream0)
        del buf12
        del primals_4
    return (buf13, reinterpret_tensor(buf11, (5, 4, 4), (16, 1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((5, 1, 4, 4), (16, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 1, 4), (4, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
