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


# kernel path: inductor_cache/we/cwevyt7nnbodtob5nscmwb4kgyn26jtnbl7ue7hzebf75wcrvjpg.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %primals_1, %repeat_1], 1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 28)
    x0 = (xindex % 4)
    x2 = xindex // 112
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 16, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 4*((-12) + x1) + 16*x2), tmp9 & xmask, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 28, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (12 + x0 + 16*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cv/ccvw4kt2kc6ilhttn7kdkz3vbvweu2sdss6g4o46skvyoos5tfax.py
# Topologically Sorted Source Nodes: [x_1, res], Original ATen: [aten.avg_pool2d, aten.sub]
# Source node to ATen node mapping:
#   res => sub
#   x_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%unsqueeze, [1, 25], [1, 1]), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %permute_1), kwargs = {})
triton_poi_fused_avg_pool2d_sub_1 = async_compile.triton('triton_poi_fused_avg_pool2d_sub_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_sub_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (4 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (12 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (16 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (20 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (24 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (28 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (32 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (36 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (40 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (44 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (48 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (52 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (56 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (60 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (64 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (68 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (72 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (76 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (80 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (84 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (88 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (92 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr0 + (96 + x2 + 4*y0 + 112*y1), xmask & ymask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp32 = tmp31 + tmp30
    tmp34 = tmp33 + tmp32
    tmp36 = tmp35 + tmp34
    tmp38 = tmp37 + tmp36
    tmp40 = tmp39 + tmp38
    tmp42 = tmp41 + tmp40
    tmp44 = tmp43 + tmp42
    tmp46 = tmp45 + tmp44
    tmp48 = tmp47 + tmp46
    tmp49 = 0.04
    tmp50 = tmp48 * tmp49
    tmp52 = tmp51 - tmp50
    tl.store(out_ptr0 + (y0 + 4*x2 + 16*y1), tmp50, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 4*y3), tmp52, xmask & ymask)
''', device_str='cuda')


cpp_fused_zeros_2 = async_compile.cpp_pybinding(['float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = static_cast<float>(0.0);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp0;
                }
            }
        }
    }
}
''')


# kernel path: inductor_cache/fa/cfa4ezcpipdba2zgit3hx2sm3jvukgbcc4xdooxhrf54neua5iyk.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear => addmm
# Graph fragment:
#   %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_3, %select, %permute_4), kwargs = {})
triton_poi_fused_addmm_3 = async_compile.triton('triton_poi_fused_addmm_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l2/cl2nre6ftlpiws2uou66j6dkic7q7xziugd2r3k2lublejeu3hoq.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear_2 => addmm_2
# Graph fragment:
#   %addmm_2 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_7, %select_10, %permute_6), kwargs = {})
triton_poi_fused_addmm_4 = async_compile.triton('triton_poi_fused_addmm_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fx/cfxqehf5d7nsrss2on37w4b77l7cobmhd35mf2swlhe76jub74u3.py
# Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear_4 => addmm_4
# Graph fragment:
#   %addmm_4 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_11, %select_22, %permute_8), kwargs = {})
triton_poi_fused_addmm_5 = async_compile.triton('triton_poi_fused_addmm_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h4/ch45gqrnjg4mtzuqfu2vb4t4qgslwdouuh27g4eflpw56poclrym.py
# Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   linear_6 => addmm_6
# Graph fragment:
#   %addmm_6 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_15, %select_34, %permute_10), kwargs = {})
triton_poi_fused_addmm_6 = async_compile.triton('triton_poi_fused_addmm_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


cpp_fused_add_copy_7 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                #pragma GCC ivdep
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(4L); x2+=static_cast<int64_t>(1L))
                {
                    {
                        {
                            auto tmp4 = in_ptr0[static_cast<int64_t>(x2 + 4L*x0)];
                            auto tmp7 = in_ptr1[static_cast<int64_t>(x2 + 4L*x0)];
                            auto tmp10 = in_ptr2[static_cast<int64_t>(x2 + 4L*x0)];
                            auto tmp13 = in_ptr3[static_cast<int64_t>(x2 + 4L*x0)];
                            auto tmp19 = in_ptr4[static_cast<int64_t>(x2 + 4L*x0)];
                            auto tmp20 = in_ptr5[static_cast<int64_t>(x2 + 4L*x0)];
                            auto tmp21 = in_ptr6[static_cast<int64_t>(x2 + 4L*x0)];
                            auto tmp22 = in_ptr7[static_cast<int64_t>(x2 + 4L*x0)];
                            auto tmp0 = x1;
                            auto tmp1 = c10::convert<int32_t>(tmp0);
                            auto tmp2 = static_cast<int32_t>(3);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<int32_t>(2);
                            auto tmp6 = tmp1 == tmp5;
                            auto tmp8 = static_cast<int32_t>(1);
                            auto tmp9 = tmp1 == tmp8;
                            auto tmp11 = static_cast<int32_t>(0);
                            auto tmp12 = tmp1 == tmp11;
                            auto tmp14 = static_cast<float>(0.0);
                            auto tmp15 = tmp12 ? tmp13 : tmp14;
                            auto tmp16 = tmp9 ? tmp10 : tmp15;
                            auto tmp17 = tmp6 ? tmp7 : tmp16;
                            auto tmp18 = tmp3 ? tmp4 : tmp17;
                            auto tmp23 = tmp12 ? tmp22 : tmp14;
                            auto tmp24 = tmp9 ? tmp21 : tmp23;
                            auto tmp25 = tmp6 ? tmp20 : tmp24;
                            auto tmp26 = tmp3 ? tmp19 : tmp25;
                            auto tmp27 = decltype(tmp18)(tmp18 + tmp26);
                            out_ptr0[static_cast<int64_t>(x2 + 4L*x1 + 16L*x0)] = tmp27;
                        }
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4), (4, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4), (4, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4), (4, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4), (4, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, 4), (4, 1))
    assert_size_stride(primals_17, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 28, 4), (112, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_1, buf0, 448, grid=grid(448), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 1, 4), (16, 4, 4, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, res], Original ATen: [aten.avg_pool2d, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_sub_1.run(buf0, primals_1, buf1, buf3, 16, 4, grid=grid(16, 4), stream=stream0)
        del buf0
        del primals_1
    buf2 = empty_strided_cpu((4, 4, 4), (16, 4, 1), torch.float32)
    cpp_fused_zeros_2(buf2)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_3.run(buf3, buf4, 16, grid=grid(16), stream=stream0)
        buf5 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_3, buf4, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf5)
        del primals_2
        del primals_3
    buf6 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    buf6.copy_(buf5, False)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf7 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(buf1, (4, 4), (16, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf7)
        del primals_4
        del primals_5
    buf8 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    buf8.copy_(buf7, False)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_4.run(buf3, buf9, 16, grid=grid(16), stream=stream0)
        buf10 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_7, buf9, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf10)
        del primals_6
        del primals_7
    buf11 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    buf11.copy_(buf10, False)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf1, (4, 4), (16, 1), 4), reinterpret_tensor(primals_8, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf12)
        del primals_8
        del primals_9
    buf13 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    buf13.copy_(buf12, False)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf14 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_5.run(buf3, buf14, 16, grid=grid(16), stream=stream0)
        buf15 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf14, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf15)
        del primals_10
        del primals_11
    buf16 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    buf16.copy_(buf15, False)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf17 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf1, (4, 4), (16, 1), 8), reinterpret_tensor(primals_12, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf17)
        del primals_12
        del primals_13
    buf18 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    buf18.copy_(buf17, False)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_6.run(buf3, buf19, 16, grid=grid(16), stream=stream0)
        buf20 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_15, buf19, reinterpret_tensor(primals_14, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf20)
        del buf19
        del primals_14
        del primals_15
    buf21 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    buf21.copy_(buf20, False)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf22 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_17, reinterpret_tensor(buf1, (4, 4), (16, 1), 12), reinterpret_tensor(primals_16, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf22)
        del primals_16
        del primals_17
    buf23 = empty_strided_cpu((4, 4), (4, 1), torch.float32)
    buf23.copy_(buf22, False)
    del buf22
    buf24 = empty_strided_cpu((4, 4, 4), (16, 4, 1), torch.float32)
    cpp_fused_add_copy_7(buf21, buf16, buf11, buf6, buf23, buf18, buf13, buf8, buf24)
    return (reinterpret_tensor(buf24, (4, 4), (16, 1), 12), buf2, reinterpret_tensor(buf3, (4, 4), (16, 4), 0), reinterpret_tensor(buf1, (4, 4), (16, 1), 0), reinterpret_tensor(buf3, (4, 4), (16, 4), 1), reinterpret_tensor(buf1, (4, 4), (16, 1), 4), reinterpret_tensor(buf3, (4, 4), (16, 4), 2), reinterpret_tensor(buf1, (4, 4), (16, 1), 8), reinterpret_tensor(buf3, (4, 4), (16, 4), 3), reinterpret_tensor(buf1, (4, 4), (16, 1), 12), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
