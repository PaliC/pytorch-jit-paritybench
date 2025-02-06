
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_52(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y5 = yindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    x4 = xindex // 8
    x3 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y5), xmask, eviction_policy='evict_last')
    tmp1 = (-1) + x4
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 8, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (-1) + x3
    tmp8 = tmp7 >= tmp2
    tmp9 = tmp7 < tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp6 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-9) + x2 + 64*y5), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = x3
    tmp14 = tmp13 >= tmp2
    tmp15 = tmp13 < tmp4
    tmp16 = tmp14 & tmp15
    tmp17 = tmp6 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-8) + x2 + 64*y5), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 + tmp12
    tmp20 = 1 + x3
    tmp21 = tmp20 >= tmp2
    tmp22 = tmp20 < tmp4
    tmp23 = tmp21 & tmp22
    tmp24 = tmp6 & tmp23
    tmp25 = tl.load(in_ptr0 + ((-7) + x2 + 64*y5), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 + tmp19
    tmp27 = x4
    tmp28 = tmp27 >= tmp2
    tmp29 = tmp27 < tmp4
    tmp30 = tmp28 & tmp29
    tmp31 = tmp30 & tmp10
    tmp32 = tl.load(in_ptr0 + ((-1) + x2 + 64*y5), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp26
    tmp34 = tmp30 & tmp16
    tmp35 = tl.load(in_ptr0 + (x2 + 64*y5), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = tmp30 & tmp23
    tmp38 = tl.load(in_ptr0 + (1 + x2 + 64*y5), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38 + tmp36
    tmp40 = 1 + x4
    tmp41 = tmp40 >= tmp2
    tmp42 = tmp40 < tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp10
    tmp45 = tl.load(in_ptr0 + (7 + x2 + 64*y5), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp39
    tmp47 = tmp43 & tmp16
    tmp48 = tl.load(in_ptr0 + (8 + x2 + 64*y5), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 + tmp46
    tmp50 = tmp43 & tmp23
    tmp51 = tl.load(in_ptr0 + (9 + x2 + 64*y5), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 + tmp49
    tmp53 = 1 + ((-1)*x3) + ((-1)*x4) + x3*x4 + ((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9)))*((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9))) + ((-1)*x3*((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9)))) + ((-1)*x4*((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9)))) + ((9) * ((9) <= (2 + x3)) + (2 + x3) * ((2 + x3) < (9))) + ((9) * ((9) <= (2 + x4)) + (2 + x4) * ((2 + x4) < (9)))
    tmp54 = tmp52 / tmp53
    tl.store(out_ptr0 + (y0 + 2048*x2 + 131072*y1), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + 2048*x2 + 131072*y1), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + 2048*x2 + 131072*y1), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + 2048*x2 + 131072*y1), tmp54, xmask)
