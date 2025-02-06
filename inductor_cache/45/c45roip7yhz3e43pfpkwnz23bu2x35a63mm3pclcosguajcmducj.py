# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/64/c64izzbgmessc4ppnvumn2vm2yn3rxmmiwz6xbibwyl46mvse5r6.py
# Topologically Sorted Source Nodes: [locs], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   locs => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %view_1, %view_2, %view_3, %view_4, %view_5], 1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1966080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 122880)
    x0 = (xindex % 4)
    x2 = xindex // 491520
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4096*(((x0 + 4*(x1)) % 16)) + 65536*((((x0 + 4*(x1) + 65536*x2) // 65536) % 4)) + ((((x0 + 4*(x1)) // 16) % 4096))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (((x0 + 4*(x1)) % 16)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 40960, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (4096*(((x0 + 4*((-16384) + x1)) % 24)) + 98304*((((x0 + 4*((-16384) + x1) + 98304*x2) // 98304) % 4)) + ((((x0 + 4*((-16384) + x1)) // 24) % 4096))), tmp13, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (((x0 + 4*((-16384) + x1)) % 24)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 65536, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr4 + (4096*(((x0 + 4*((-40960) + x1)) % 24)) + 98304*((((x0 + 4*((-40960) + x1) + 98304*x2) // 98304) % 4)) + ((((x0 + 4*((-40960) + x1)) // 24) % 4096))), tmp22, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr5 + (((x0 + 4*((-40960) + x1)) % 24)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 90112, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tl.load(in_ptr6 + (4096*(((x0 + 4*((-65536) + x1)) % 24)) + 98304*((((x0 + 4*((-65536) + x1) + 98304*x2) // 98304) % 4)) + ((((x0 + 4*((-65536) + x1)) // 24) % 4096))), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr7 + (((x0 + 4*((-65536) + x1)) % 24)), tmp31, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tmp0 >= tmp29
    tmp38 = tl.full([1], 106496, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = tl.load(in_ptr8 + (4096*(((x0 + 4*((-90112) + x1)) % 16)) + 65536*((((x0 + 4*((-90112) + x1) + 65536*x2) // 65536) % 4)) + ((((x0 + 4*((-90112) + x1)) // 16) % 4096))), tmp40, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr9 + (((x0 + 4*((-90112) + x1)) % 16)), tmp40, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp40, tmp43, tmp44)
    tmp46 = tmp0 >= tmp38
    tmp47 = tl.full([1], 122880, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tl.load(in_ptr10 + (4096*(((x0 + 4*((-106496) + x1)) % 16)) + 65536*((((x0 + 4*((-106496) + x1) + 65536*x2) // 65536) % 4)) + ((((x0 + 4*((-106496) + x1)) // 16) % 4096))), tmp46, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr11 + (((x0 + 4*((-106496) + x1)) % 16)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp46, tmp51, tmp52)
    tmp54 = tl.where(tmp40, tmp45, tmp53)
    tmp55 = tl.where(tmp31, tmp36, tmp54)
    tmp56 = tl.where(tmp22, tmp27, tmp55)
    tmp57 = tl.where(tmp13, tmp18, tmp56)
    tmp58 = tl.where(tmp4, tmp9, tmp57)
    tl.store(out_ptr0 + (x3), tmp58, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30 = args
    args.clear()
    assert_size_stride(primals_1, (4, 512, 64, 64), (2097152, 4096, 64, 1))
    assert_size_stride(primals_2, (16, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (24, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (4, 1024, 64, 64), (4194304, 4096, 64, 1))
    assert_size_stride(primals_7, (24, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_8, (24, ), (1, ))
    assert_size_stride(primals_9, (4, 512, 64, 64), (2097152, 4096, 64, 1))
    assert_size_stride(primals_10, (24, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (4, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(primals_13, (16, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_14, (16, ), (1, ))
    assert_size_stride(primals_15, (4, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(primals_16, (16, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_17, (16, ), (1, ))
    assert_size_stride(primals_18, (4, 256, 64, 64), (1048576, 4096, 64, 1))
    assert_size_stride(primals_19, (16, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (24, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_22, (24, ), (1, ))
    assert_size_stride(primals_23, (24, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_25, (24, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_27, (16, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_28, (16, ), (1, ))
    assert_size_stride(primals_29, (16, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_30, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [l_conv4_3], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 16, 64, 64), (65536, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [l_conv7], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_6, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 24, 64, 64), (98304, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [l_conv8_2], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(primals_9, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 24, 64, 64), (98304, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [l_conv9_2], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_12, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 24, 64, 64), (98304, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [l_conv10_2], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(primals_15, primals_13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 16, 64, 64), (65536, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [l_conv11_2], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(primals_18, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 16, 64, 64), (65536, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [c_conv4_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(primals_1, primals_19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 16, 64, 64), (65536, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [c_conv7], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(primals_6, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 24, 64, 64), (98304, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [c_conv8_2], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(primals_9, primals_23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 24, 64, 64), (98304, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [c_conv9_2], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(primals_12, primals_25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 24, 64, 64), (98304, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [c_conv10_2], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(primals_15, primals_27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 16, 64, 64), (65536, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [c_conv11_2], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(primals_18, primals_29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 16, 64, 64), (65536, 4096, 64, 1))
        buf12 = empty_strided_cuda((4, 122880, 4), (491520, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [locs], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(buf0, primals_3, buf1, primals_5, buf2, primals_8, buf3, primals_11, buf4, primals_14, buf5, primals_17, buf12, 1966080, grid=grid(1966080), stream=stream0)
        del buf0
        del buf1
        del buf2
        del buf3
        del buf4
        del buf5
        del primals_11
        del primals_14
        del primals_17
        del primals_3
        del primals_5
        del primals_8
        buf13 = empty_strided_cuda((4, 122880, 4), (491520, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [classes_scores], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(buf6, primals_20, buf7, primals_22, buf8, primals_24, buf9, primals_26, buf10, primals_28, buf11, primals_30, buf13, 1966080, grid=grid(1966080), stream=stream0)
        del buf10
        del buf11
        del buf6
        del buf7
        del buf8
        del buf9
        del primals_20
        del primals_22
        del primals_24
        del primals_26
        del primals_28
        del primals_30
    return (buf12, buf13, primals_1, primals_2, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 512, 64, 64), (2097152, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 1024, 64, 64), (4194304, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 512, 64, 64), (2097152, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((24, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((16, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 256, 64, 64), (1048576, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((24, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((24, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((16, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((16, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
