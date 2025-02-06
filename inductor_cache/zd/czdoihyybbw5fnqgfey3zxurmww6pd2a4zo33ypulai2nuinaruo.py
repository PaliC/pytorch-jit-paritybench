# AOT ID: ['4_forward']
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


# kernel path: inductor_cache/vo/cvobweae4ddnryggjgxveax7yxifstbptgaslfpdwbfsn4xlnpub.py
# Topologically Sorted Source Nodes: [linear, x_1], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear => add_tensor_2
#   x_1 => relu
# Graph fragment:
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %primals_3), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_2,), kwargs = {})
triton_poi_fused_addmm_relu_0 = async_compile.triton('triton_poi_fused_addmm_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 500)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cg/ccg45iokbmwt4brnajixig4fdjo7qenmngjh3vf4tvdp6nzrxoim.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   hx => full_default
# Graph fragment:
#   %full_default : [num_users=5] = call_function[target=torch.ops.aten.full.default](args = ([4, 1000], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_1 = async_compile.triton('triton_poi_fused_zeros_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y2/cy2plxl4ld3ex6dz77rccs5xkbtvkjrpane2rldunqjpvuqp77gp.py
# Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   rnn_out => cat
# Graph fragment:
#   %cat : [num_users=16] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3, %unsqueeze_4, %unsqueeze_5, %unsqueeze_6, %unsqueeze_7, %unsqueeze_8, %unsqueeze_9, %unsqueeze_10, %unsqueeze_11, %unsqueeze_12, %unsqueeze_13, %unsqueeze_14, %unsqueeze_15],), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lf/clf7fvin5hjb3n4bsbnryof5o6tuitbz6vtxaawjqnvxu2lm2zow.py
# Topologically Sorted Source Nodes: [linear_3, x_6], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear_3 => add_tensor
#   x_6 => relu_2
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_17), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
triton_poi_fused_addmm_relu_3 = async_compile.triton('triton_poi_fused_addmm_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 100)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (500, 4), (4, 1))
    assert_size_stride(primals_3, (500, ), (1, ))
    assert_size_stride(primals_4, (500, 500), (500, 1))
    assert_size_stride(primals_5, (500, ), (1, ))
    assert_size_stride(primals_6, (4000, 500), (500, 1))
    assert_size_stride(primals_7, (4000, 1000), (1000, 1))
    assert_size_stride(primals_8, (4000, ), (1, ))
    assert_size_stride(primals_9, (4000, ), (1, ))
    assert_size_stride(primals_10, (4000, 1000), (1000, 1))
    assert_size_stride(primals_11, (4000, 1000), (1000, 1))
    assert_size_stride(primals_12, (4000, ), (1, ))
    assert_size_stride(primals_13, (4000, ), (1, ))
    assert_size_stride(primals_14, (500, 1000), (1000, 1))
    assert_size_stride(primals_15, (500, ), (1, ))
    assert_size_stride(primals_16, (100, 500), (500, 1))
    assert_size_stride(primals_17, (100, ), (1, ))
    assert_size_stride(primals_18, (4, 100), (100, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 500), (500, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 500), (1, 4), 0), out=buf0)
        del primals_2
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [linear, x_1], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf1, primals_3, 32000, grid=grid(32000), stream=stream0)
        del primals_3
        buf2 = empty_strided_cuda((64, 500), (500, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf1, reinterpret_tensor(primals_4, (500, 500), (1, 500), 0), alpha=1, beta=1, out=buf2)
        del primals_5
        buf3 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_1.run(buf3, 4000, grid=grid(4000), stream=stream0)
        buf4 = empty_strided_cuda((4, 4000), (4000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 0), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf4)
        buf5 = empty_strided_cuda((4, 4000), (4000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf5)
        # Topologically Sorted Source Nodes: [lstm_cell], Original ATen: [aten._thnn_fused_lstm_cell]
        buf6 = torch.ops.aten._thnn_fused_lstm_cell.default(buf4, buf5, buf3, primals_8, primals_9)
        buf7 = buf6[0]
        buf8 = buf6[1]
        buf9 = buf6[2]
        del buf6
        buf10 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 2000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf10)
        buf11 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf7, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf11)
        # Topologically Sorted Source Nodes: [lstm_cell_1], Original ATen: [aten._thnn_fused_lstm_cell]
        buf12 = torch.ops.aten._thnn_fused_lstm_cell.default(buf10, buf11, buf8, primals_8, primals_9)
        buf13 = buf12[0]
        buf14 = buf12[1]
        buf15 = buf12[2]
        del buf12
        buf16 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 4000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf16)
        buf17 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf13, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf17)
        # Topologically Sorted Source Nodes: [lstm_cell_2], Original ATen: [aten._thnn_fused_lstm_cell]
        buf18 = torch.ops.aten._thnn_fused_lstm_cell.default(buf16, buf17, buf14, primals_8, primals_9)
        buf19 = buf18[0]
        buf20 = buf18[1]
        buf21 = buf18[2]
        del buf18
        buf22 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 6000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf22)
        buf23 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf23)
        # Topologically Sorted Source Nodes: [lstm_cell_3], Original ATen: [aten._thnn_fused_lstm_cell]
        buf24 = torch.ops.aten._thnn_fused_lstm_cell.default(buf22, buf23, buf20, primals_8, primals_9)
        buf25 = buf24[0]
        buf26 = buf24[1]
        buf27 = buf24[2]
        del buf24
        buf28 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 8000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf28)
        buf29 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf29)
        # Topologically Sorted Source Nodes: [lstm_cell_4], Original ATen: [aten._thnn_fused_lstm_cell]
        buf30 = torch.ops.aten._thnn_fused_lstm_cell.default(buf28, buf29, buf26, primals_8, primals_9)
        buf31 = buf30[0]
        buf32 = buf30[1]
        buf33 = buf30[2]
        del buf30
        buf34 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 10000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf34)
        buf35 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf35)
        # Topologically Sorted Source Nodes: [lstm_cell_5], Original ATen: [aten._thnn_fused_lstm_cell]
        buf36 = torch.ops.aten._thnn_fused_lstm_cell.default(buf34, buf35, buf32, primals_8, primals_9)
        buf37 = buf36[0]
        buf38 = buf36[1]
        buf39 = buf36[2]
        del buf36
        buf40 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 12000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf40)
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf37, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf41)
        # Topologically Sorted Source Nodes: [lstm_cell_6], Original ATen: [aten._thnn_fused_lstm_cell]
        buf42 = torch.ops.aten._thnn_fused_lstm_cell.default(buf40, buf41, buf38, primals_8, primals_9)
        buf43 = buf42[0]
        buf44 = buf42[1]
        buf45 = buf42[2]
        del buf42
        buf46 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 14000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf46)
        buf47 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf43, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf47)
        # Topologically Sorted Source Nodes: [lstm_cell_7], Original ATen: [aten._thnn_fused_lstm_cell]
        buf48 = torch.ops.aten._thnn_fused_lstm_cell.default(buf46, buf47, buf44, primals_8, primals_9)
        buf49 = buf48[0]
        buf50 = buf48[1]
        buf51 = buf48[2]
        del buf48
        buf52 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 16000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf52)
        buf53 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf49, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf53)
        # Topologically Sorted Source Nodes: [lstm_cell_8], Original ATen: [aten._thnn_fused_lstm_cell]
        buf54 = torch.ops.aten._thnn_fused_lstm_cell.default(buf52, buf53, buf50, primals_8, primals_9)
        buf55 = buf54[0]
        buf56 = buf54[1]
        buf57 = buf54[2]
        del buf54
        buf58 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 18000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf58)
        buf59 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf55, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf59)
        # Topologically Sorted Source Nodes: [lstm_cell_9], Original ATen: [aten._thnn_fused_lstm_cell]
        buf60 = torch.ops.aten._thnn_fused_lstm_cell.default(buf58, buf59, buf56, primals_8, primals_9)
        buf61 = buf60[0]
        buf62 = buf60[1]
        buf63 = buf60[2]
        del buf60
        buf64 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 20000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf64)
        buf65 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf61, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf65)
        # Topologically Sorted Source Nodes: [lstm_cell_10], Original ATen: [aten._thnn_fused_lstm_cell]
        buf66 = torch.ops.aten._thnn_fused_lstm_cell.default(buf64, buf65, buf62, primals_8, primals_9)
        buf67 = buf66[0]
        buf68 = buf66[1]
        buf69 = buf66[2]
        del buf66
        buf70 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 22000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf70)
        buf71 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf67, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf71)
        # Topologically Sorted Source Nodes: [lstm_cell_11], Original ATen: [aten._thnn_fused_lstm_cell]
        buf72 = torch.ops.aten._thnn_fused_lstm_cell.default(buf70, buf71, buf68, primals_8, primals_9)
        buf73 = buf72[0]
        buf74 = buf72[1]
        buf75 = buf72[2]
        del buf72
        buf76 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 24000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf76)
        buf77 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf73, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf77)
        # Topologically Sorted Source Nodes: [lstm_cell_12], Original ATen: [aten._thnn_fused_lstm_cell]
        buf78 = torch.ops.aten._thnn_fused_lstm_cell.default(buf76, buf77, buf74, primals_8, primals_9)
        buf79 = buf78[0]
        buf80 = buf78[1]
        buf81 = buf78[2]
        del buf78
        buf82 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 26000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf82)
        buf83 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf79, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf83)
        # Topologically Sorted Source Nodes: [lstm_cell_13], Original ATen: [aten._thnn_fused_lstm_cell]
        buf84 = torch.ops.aten._thnn_fused_lstm_cell.default(buf82, buf83, buf80, primals_8, primals_9)
        buf85 = buf84[0]
        buf86 = buf84[1]
        buf87 = buf84[2]
        del buf84
        buf88 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 28000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf88)
        buf89 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf85, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf89)
        # Topologically Sorted Source Nodes: [lstm_cell_14], Original ATen: [aten._thnn_fused_lstm_cell]
        buf90 = torch.ops.aten._thnn_fused_lstm_cell.default(buf88, buf89, buf86, primals_8, primals_9)
        buf91 = buf90[0]
        buf92 = buf90[1]
        buf93 = buf90[2]
        del buf90
        buf94 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 500), (500, 1), 30000), reinterpret_tensor(primals_6, (500, 4000), (1, 500), 0), out=buf94)
        buf95 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf91, reinterpret_tensor(primals_7, (1000, 4000), (1, 1000), 0), out=buf95)
        # Topologically Sorted Source Nodes: [lstm_cell_15], Original ATen: [aten._thnn_fused_lstm_cell]
        buf96 = torch.ops.aten._thnn_fused_lstm_cell.default(buf94, buf95, buf92, primals_8, primals_9)
        del primals_8
        del primals_9
        buf97 = buf96[0]
        buf98 = buf96[1]
        buf99 = buf96[2]
        del buf96
        buf116 = empty_strided_cuda((16, 4, 1000), (4000, 1000, 1), torch.float32)
        buf100 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf7, buf100, 4000, grid=grid(4000), stream=stream0)
        buf101 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 4000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf13, buf101, 4000, grid=grid(4000), stream=stream0)
        buf102 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 8000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf19, buf102, 4000, grid=grid(4000), stream=stream0)
        buf103 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 12000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf25, buf103, 4000, grid=grid(4000), stream=stream0)
        buf104 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 16000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf31, buf104, 4000, grid=grid(4000), stream=stream0)
        buf105 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 20000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf37, buf105, 4000, grid=grid(4000), stream=stream0)
        buf106 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 24000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf43, buf106, 4000, grid=grid(4000), stream=stream0)
        buf107 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 28000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf49, buf107, 4000, grid=grid(4000), stream=stream0)
        buf108 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 32000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf55, buf108, 4000, grid=grid(4000), stream=stream0)
        buf109 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 36000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf61, buf109, 4000, grid=grid(4000), stream=stream0)
        buf110 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 40000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf67, buf110, 4000, grid=grid(4000), stream=stream0)
        buf111 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 44000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf73, buf111, 4000, grid=grid(4000), stream=stream0)
        buf112 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 48000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf79, buf112, 4000, grid=grid(4000), stream=stream0)
        buf113 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 52000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf85, buf113, 4000, grid=grid(4000), stream=stream0)
        buf114 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 56000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf91, buf114, 4000, grid=grid(4000), stream=stream0)
        buf115 = reinterpret_tensor(buf116, (1, 4, 1000), (4000, 1000, 1), 60000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf97, buf115, 4000, grid=grid(4000), stream=stream0)
        del buf97
        buf117 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 0), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf117)
        buf118 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf118)
        # Topologically Sorted Source Nodes: [lstm_cell_16], Original ATen: [aten._thnn_fused_lstm_cell]
        buf119 = torch.ops.aten._thnn_fused_lstm_cell.default(buf117, buf118, buf3, primals_12, primals_13)
        buf120 = buf119[0]
        buf121 = buf119[1]
        buf122 = buf119[2]
        del buf119
        buf123 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 4000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf123)
        buf124 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf120, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf124)
        # Topologically Sorted Source Nodes: [lstm_cell_17], Original ATen: [aten._thnn_fused_lstm_cell]
        buf125 = torch.ops.aten._thnn_fused_lstm_cell.default(buf123, buf124, buf121, primals_12, primals_13)
        buf126 = buf125[0]
        buf127 = buf125[1]
        buf128 = buf125[2]
        del buf125
        buf129 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 8000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf129)
        buf130 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf126, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf130)
        # Topologically Sorted Source Nodes: [lstm_cell_18], Original ATen: [aten._thnn_fused_lstm_cell]
        buf131 = torch.ops.aten._thnn_fused_lstm_cell.default(buf129, buf130, buf127, primals_12, primals_13)
        buf132 = buf131[0]
        buf133 = buf131[1]
        buf134 = buf131[2]
        del buf131
        buf135 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 12000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf135)
        buf136 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf132, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf136)
        # Topologically Sorted Source Nodes: [lstm_cell_19], Original ATen: [aten._thnn_fused_lstm_cell]
        buf137 = torch.ops.aten._thnn_fused_lstm_cell.default(buf135, buf136, buf133, primals_12, primals_13)
        buf138 = buf137[0]
        buf139 = buf137[1]
        buf140 = buf137[2]
        del buf137
        buf141 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 16000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf141)
        buf142 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf138, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf142)
        # Topologically Sorted Source Nodes: [lstm_cell_20], Original ATen: [aten._thnn_fused_lstm_cell]
        buf143 = torch.ops.aten._thnn_fused_lstm_cell.default(buf141, buf142, buf139, primals_12, primals_13)
        buf144 = buf143[0]
        buf145 = buf143[1]
        buf146 = buf143[2]
        del buf143
        buf147 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 20000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf147)
        buf148 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf144, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf148)
        # Topologically Sorted Source Nodes: [lstm_cell_21], Original ATen: [aten._thnn_fused_lstm_cell]
        buf149 = torch.ops.aten._thnn_fused_lstm_cell.default(buf147, buf148, buf145, primals_12, primals_13)
        buf150 = buf149[0]
        buf151 = buf149[1]
        buf152 = buf149[2]
        del buf149
        buf153 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 24000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf153)
        buf154 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_22], Original ATen: [aten.mm]
        extern_kernels.mm(buf150, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf154)
        # Topologically Sorted Source Nodes: [lstm_cell_22], Original ATen: [aten._thnn_fused_lstm_cell]
        buf155 = torch.ops.aten._thnn_fused_lstm_cell.default(buf153, buf154, buf151, primals_12, primals_13)
        buf156 = buf155[0]
        buf157 = buf155[1]
        buf158 = buf155[2]
        del buf155
        buf159 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 28000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf159)
        buf160 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf156, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf160)
        # Topologically Sorted Source Nodes: [lstm_cell_23], Original ATen: [aten._thnn_fused_lstm_cell]
        buf161 = torch.ops.aten._thnn_fused_lstm_cell.default(buf159, buf160, buf157, primals_12, primals_13)
        buf162 = buf161[0]
        buf163 = buf161[1]
        buf164 = buf161[2]
        del buf161
        buf165 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 32000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf165)
        buf166 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf162, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf166)
        # Topologically Sorted Source Nodes: [lstm_cell_24], Original ATen: [aten._thnn_fused_lstm_cell]
        buf167 = torch.ops.aten._thnn_fused_lstm_cell.default(buf165, buf166, buf163, primals_12, primals_13)
        buf168 = buf167[0]
        buf169 = buf167[1]
        buf170 = buf167[2]
        del buf167
        buf171 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 36000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf171)
        buf172 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_25], Original ATen: [aten.mm]
        extern_kernels.mm(buf168, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf172)
        # Topologically Sorted Source Nodes: [lstm_cell_25], Original ATen: [aten._thnn_fused_lstm_cell]
        buf173 = torch.ops.aten._thnn_fused_lstm_cell.default(buf171, buf172, buf169, primals_12, primals_13)
        buf174 = buf173[0]
        buf175 = buf173[1]
        buf176 = buf173[2]
        del buf173
        buf177 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 40000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf177)
        buf178 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_26], Original ATen: [aten.mm]
        extern_kernels.mm(buf174, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf178)
        # Topologically Sorted Source Nodes: [lstm_cell_26], Original ATen: [aten._thnn_fused_lstm_cell]
        buf179 = torch.ops.aten._thnn_fused_lstm_cell.default(buf177, buf178, buf175, primals_12, primals_13)
        buf180 = buf179[0]
        buf181 = buf179[1]
        buf182 = buf179[2]
        del buf179
        buf183 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 44000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf183)
        buf184 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf180, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf184)
        # Topologically Sorted Source Nodes: [lstm_cell_27], Original ATen: [aten._thnn_fused_lstm_cell]
        buf185 = torch.ops.aten._thnn_fused_lstm_cell.default(buf183, buf184, buf181, primals_12, primals_13)
        buf186 = buf185[0]
        buf187 = buf185[1]
        buf188 = buf185[2]
        del buf185
        buf189 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 48000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf189)
        buf190 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_28], Original ATen: [aten.mm]
        extern_kernels.mm(buf186, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf190)
        # Topologically Sorted Source Nodes: [lstm_cell_28], Original ATen: [aten._thnn_fused_lstm_cell]
        buf191 = torch.ops.aten._thnn_fused_lstm_cell.default(buf189, buf190, buf187, primals_12, primals_13)
        buf192 = buf191[0]
        buf193 = buf191[1]
        buf194 = buf191[2]
        del buf191
        buf195 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 52000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf195)
        buf196 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf192, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf196)
        # Topologically Sorted Source Nodes: [lstm_cell_29], Original ATen: [aten._thnn_fused_lstm_cell]
        buf197 = torch.ops.aten._thnn_fused_lstm_cell.default(buf195, buf196, buf193, primals_12, primals_13)
        buf198 = buf197[0]
        buf199 = buf197[1]
        buf200 = buf197[2]
        del buf197
        buf201 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 56000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf201)
        buf202 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_30], Original ATen: [aten.mm]
        extern_kernels.mm(buf198, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf202)
        # Topologically Sorted Source Nodes: [lstm_cell_30], Original ATen: [aten._thnn_fused_lstm_cell]
        buf203 = torch.ops.aten._thnn_fused_lstm_cell.default(buf201, buf202, buf199, primals_12, primals_13)
        buf204 = buf203[0]
        buf205 = buf203[1]
        buf206 = buf203[2]
        del buf203
        buf207 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (4, 1000), (1000, 1), 60000), reinterpret_tensor(primals_10, (1000, 4000), (1, 1000), 0), out=buf207)
        buf208 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [lstm_cell_31], Original ATen: [aten.mm]
        extern_kernels.mm(buf204, reinterpret_tensor(primals_11, (1000, 4000), (1, 1000), 0), out=buf208)
        # Topologically Sorted Source Nodes: [lstm_cell_31], Original ATen: [aten._thnn_fused_lstm_cell]
        buf209 = torch.ops.aten._thnn_fused_lstm_cell.default(buf207, buf208, buf205, primals_12, primals_13)
        del buf207
        del buf208
        del primals_12
        del primals_13
        buf210 = buf209[0]
        buf211 = buf209[1]
        buf212 = buf209[2]
        del buf209
        buf229 = empty_strided_cuda((16, 4, 1000), (4000, 1000, 1), torch.float32)
        buf213 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf120, buf213, 4000, grid=grid(4000), stream=stream0)
        buf214 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 4000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf126, buf214, 4000, grid=grid(4000), stream=stream0)
        buf215 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 8000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf132, buf215, 4000, grid=grid(4000), stream=stream0)
        buf216 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 12000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf138, buf216, 4000, grid=grid(4000), stream=stream0)
        buf217 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 16000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf144, buf217, 4000, grid=grid(4000), stream=stream0)
        buf218 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 20000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf150, buf218, 4000, grid=grid(4000), stream=stream0)
        buf219 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 24000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf156, buf219, 4000, grid=grid(4000), stream=stream0)
        buf220 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 28000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf162, buf220, 4000, grid=grid(4000), stream=stream0)
        buf221 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 32000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf168, buf221, 4000, grid=grid(4000), stream=stream0)
        buf222 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 36000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf174, buf222, 4000, grid=grid(4000), stream=stream0)
        buf223 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 40000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf180, buf223, 4000, grid=grid(4000), stream=stream0)
        buf224 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 44000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf186, buf224, 4000, grid=grid(4000), stream=stream0)
        buf225 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 48000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf192, buf225, 4000, grid=grid(4000), stream=stream0)
        buf226 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 52000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf198, buf226, 4000, grid=grid(4000), stream=stream0)
        buf227 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 56000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf204, buf227, 4000, grid=grid(4000), stream=stream0)
        buf228 = reinterpret_tensor(buf229, (1, 4, 1000), (4000, 1000, 1), 60000)  # alias
        # Topologically Sorted Source Nodes: [rnn_out_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf210, buf228, 4000, grid=grid(4000), stream=stream0)
        del buf210
        buf230 = empty_strided_cuda((64, 500), (500, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf229, (64, 1000), (1000, 1), 0), reinterpret_tensor(primals_14, (1000, 500), (1, 1000), 0), out=buf230)
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [linear_2, x_5], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf231, primals_15, 32000, grid=grid(32000), stream=stream0)
        del primals_15
        buf232 = empty_strided_cuda((64, 100), (100, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf231, reinterpret_tensor(primals_16, (500, 100), (1, 500), 0), out=buf232)
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [linear_3, x_6], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_3.run(buf233, primals_17, 6400, grid=grid(6400), stream=stream0)
        del primals_17
        buf234 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, buf233, reinterpret_tensor(primals_18, (100, 4), (1, 100), 0), alpha=1, beta=1, out=buf234)
        del primals_19
    return (buf234, reinterpret_tensor(primals_1, (64, 4), (4, 1), 0), buf1, buf3, reinterpret_tensor(buf2, (4, 500), (500, 1), 0), buf7, buf8, buf9, reinterpret_tensor(buf2, (4, 500), (500, 1), 2000), buf13, buf14, buf15, reinterpret_tensor(buf2, (4, 500), (500, 1), 4000), buf19, buf20, buf21, reinterpret_tensor(buf2, (4, 500), (500, 1), 6000), buf25, buf26, buf27, reinterpret_tensor(buf2, (4, 500), (500, 1), 8000), buf31, buf32, buf33, reinterpret_tensor(buf2, (4, 500), (500, 1), 10000), buf37, buf38, buf39, reinterpret_tensor(buf2, (4, 500), (500, 1), 12000), buf43, buf44, buf45, reinterpret_tensor(buf2, (4, 500), (500, 1), 14000), buf49, buf50, buf51, reinterpret_tensor(buf2, (4, 500), (500, 1), 16000), buf55, buf56, buf57, reinterpret_tensor(buf2, (4, 500), (500, 1), 18000), buf61, buf62, buf63, reinterpret_tensor(buf2, (4, 500), (500, 1), 20000), buf67, buf68, buf69, reinterpret_tensor(buf2, (4, 500), (500, 1), 22000), buf73, buf74, buf75, reinterpret_tensor(buf2, (4, 500), (500, 1), 24000), buf79, buf80, buf81, reinterpret_tensor(buf2, (4, 500), (500, 1), 26000), buf85, buf86, buf87, reinterpret_tensor(buf2, (4, 500), (500, 1), 28000), buf91, buf92, buf93, reinterpret_tensor(buf2, (4, 500), (500, 1), 30000), buf98, buf99, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 0), buf120, buf121, buf122, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 4000), buf126, buf127, buf128, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 8000), buf132, buf133, buf134, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 12000), buf138, buf139, buf140, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 16000), buf144, buf145, buf146, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 20000), buf150, buf151, buf152, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 24000), buf156, buf157, buf158, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 28000), buf162, buf163, buf164, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 32000), buf168, buf169, buf170, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 36000), buf174, buf175, buf176, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 40000), buf180, buf181, buf182, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 44000), buf186, buf187, buf188, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 48000), buf192, buf193, buf194, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 52000), buf198, buf199, buf200, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 56000), buf204, buf205, buf206, reinterpret_tensor(buf116, (4, 1000), (1000, 1), 60000), buf211, buf212, reinterpret_tensor(buf229, (64, 1000), (1000, 1), 0), buf231, buf233, primals_18, primals_16, primals_14, primals_11, primals_10, primals_7, primals_6, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((500, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((500, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((500, 500), (500, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((500, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4000, 500), (500, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4000, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4000, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4000, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((500, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((500, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((100, 500), (500, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 100), (100, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
