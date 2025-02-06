
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1056
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 264)
    x2 = xindex
    y1 = yindex // 264
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 44, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (44*x2 + 2816*y1 + (y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 88, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (44*x2 + 2816*y1 + ((-44) + y0)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 132, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (44*x2 + 2816*y1 + ((-88) + y0)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1, 1], 176, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (44*x2 + 2816*y1 + ((-132) + y0)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr0 + (44*x2 + 2816*y1 + ((-132) + y0)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tmp0 >= tmp17
    tmp26 = tl.full([1, 1], 220, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr4 + (44*x2 + 2816*y1 + ((-176) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp28, tmp30, tmp31)
    tmp33 = tmp0 >= tmp26
    tmp34 = tl.full([1, 1], 264, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tl.load(in_ptr5 + (44*x2 + 2816*y1 + ((-220) + y0)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.load(in_ptr6 + (tl.broadcast_to((-220) + y0, [XBLOCK, YBLOCK])), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 - tmp37
    tmp39 = tl.load(in_ptr7 + (tl.broadcast_to((-220) + y0, [XBLOCK, YBLOCK])), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp40 = 0.001
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.sqrt(tmp41)
    tmp43 = tl.full([1, 1], 1, tl.int32)
    tmp44 = tmp43 / tmp42
    tmp45 = 1.0
    tmp46 = tmp44 * tmp45
    tmp47 = tmp38 * tmp46
    tmp48 = tl.load(in_ptr8 + (tl.broadcast_to((-220) + y0, [XBLOCK, YBLOCK])), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp47 * tmp48
    tmp50 = tl.load(in_ptr9 + (tl.broadcast_to((-220) + y0, [XBLOCK, YBLOCK])), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 + tmp50
    tmp52 = tl.load(in_ptr10 + (44*x2 + 2816*y1 + ((-220) + y0)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp33, tmp53, tmp54)
    tmp56 = tl.where(tmp28, tmp32, tmp55)
    tmp57 = tl.where(tmp19, tmp24, tmp56)
    tmp58 = tl.where(tmp14, tmp15, tmp57)
    tmp59 = tl.where(tmp9, tmp10, tmp58)
    tmp60 = tl.where(tmp4, tmp5, tmp59)
    tmp61 = tl.full([1, 1], 0, tl.int32)
    tmp62 = triton_helpers.maximum(tmp61, tmp60)
    tl.store(out_ptr1 + (y0 + 264*x2 + 16896*y1), tmp62, xmask & ymask)
