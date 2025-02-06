
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1408
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = (yindex % 352)
    x4 = xindex
    y1 = yindex // 352
    x2 = (xindex % 4)
    x3 = xindex // 4
    y5 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 88, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (88*x4 + 1408*y1 + (y0)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 176, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (528 + 88*x2 + 440*x3 + 2200*y1 + ((-88) + y0)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (88*x4 + 1408*y1 + ((-88) + y0)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr3 + (tl.broadcast_to((-88) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr4 + (tl.broadcast_to((-88) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = 0.001
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tl.full([1, 1], 1, tl.int32)
    tmp19 = tmp18 / tmp17
    tmp20 = 1.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp13 * tmp21
    tmp23 = tl.load(in_ptr5 + (tl.broadcast_to((-88) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp22 * tmp23
    tmp25 = tl.load(in_ptr6 + (tl.broadcast_to((-88) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp10 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp9, tmp27, tmp28)
    tmp30 = tmp0 >= tmp7
    tmp31 = tl.full([1, 1], 264, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr7 + (88*x4 + 1408*y1 + ((-176) + y0)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr0 + (88*x4 + 1408*y1 + ((-176) + y0)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 + tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp33, tmp36, tmp37)
    tmp39 = tmp0 >= tmp31
    tmp40 = tl.full([1, 1], 352, tl.int64)
    tmp41 = tmp0 < tmp40
    tmp42 = tl.load(in_ptr8 + (88*x4 + 1408*y1 + ((-264) + y0)), tmp39 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr9 + (tl.broadcast_to((-264) + y0, [XBLOCK, YBLOCK])), tmp39 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 - tmp43
    tmp45 = tl.load(in_ptr10 + (tl.broadcast_to((-264) + y0, [XBLOCK, YBLOCK])), tmp39 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = 0.001
    tmp47 = tmp45 + tmp46
    tmp48 = libdevice.sqrt(tmp47)
    tmp49 = tl.full([1, 1], 1, tl.int32)
    tmp50 = tmp49 / tmp48
    tmp51 = 1.0
    tmp52 = tmp50 * tmp51
    tmp53 = tmp44 * tmp52
    tmp54 = tl.load(in_ptr11 + (tl.broadcast_to((-264) + y0, [XBLOCK, YBLOCK])), tmp39 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 * tmp54
    tmp56 = tl.load(in_ptr12 + (tl.broadcast_to((-264) + y0, [XBLOCK, YBLOCK])), tmp39 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 + tmp56
    tmp58 = tl.load(in_ptr13 + (528 + 88*x2 + 440*x3 + 2200*y1 + ((-264) + y0)), tmp39 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp39, tmp59, tmp60)
    tmp62 = tl.where(tmp33, tmp38, tmp61)
    tmp63 = tl.where(tmp9, tmp29, tmp62)
    tmp64 = tl.where(tmp4, tmp5, tmp63)
    tmp65 = tl.full([1, 1], 0, tl.int32)
    tmp66 = triton_helpers.maximum(tmp65, tmp64)
    tl.store(out_ptr1 + (y0 + 352*x4 + 5632*y1), tmp66, xmask & ymask)
