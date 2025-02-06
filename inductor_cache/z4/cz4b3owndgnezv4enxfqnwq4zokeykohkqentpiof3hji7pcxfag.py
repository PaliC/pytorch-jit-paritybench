
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 20)
    x0 = (xindex % 16)
    x2 = xindex // 320
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 64*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 8, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr5 + (x0 + 16*((-4) + x1) + 64*x2), tmp26 & xmask, other=0.0)
    tmp28 = tl.load(in_ptr6 + ((-4) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.load(in_ptr7 + ((-4) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp39 = tl.load(in_ptr8 + ((-4) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.load(in_ptr9 + ((-4) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp26, tmp42, tmp43)
    tmp45 = tmp0 >= tmp24
    tmp46 = tl.full([1], 12, tl.int64)
    tmp47 = tmp0 < tmp46
    tmp48 = tmp45 & tmp47
    tmp49 = tl.load(in_ptr10 + (x0 + 16*((-8) + x1) + 64*x2), tmp48 & xmask, other=0.0)
    tmp50 = tl.load(in_ptr11 + ((-8) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 - tmp50
    tmp52 = tl.load(in_ptr12 + ((-8) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = 1e-05
    tmp54 = tmp52 + tmp53
    tmp55 = libdevice.sqrt(tmp54)
    tmp56 = tl.full([1], 1, tl.int32)
    tmp57 = tmp56 / tmp55
    tmp58 = 1.0
    tmp59 = tmp57 * tmp58
    tmp60 = tmp51 * tmp59
    tmp61 = tl.load(in_ptr13 + ((-8) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp60 * tmp61
    tmp63 = tl.load(in_ptr14 + ((-8) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp62 + tmp63
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp48, tmp64, tmp65)
    tmp67 = tmp0 >= tmp46
    tmp68 = tl.full([1], 16, tl.int64)
    tmp69 = tmp0 < tmp68
    tmp70 = tmp67 & tmp69
    tmp71 = tl.load(in_ptr15 + (x0 + 16*((-12) + x1) + 64*x2), tmp70 & xmask, other=0.0)
    tmp72 = tl.load(in_ptr16 + ((-12) + x1), tmp70 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tmp71 - tmp72
    tmp74 = tl.load(in_ptr17 + ((-12) + x1), tmp70 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = 1e-05
    tmp76 = tmp74 + tmp75
    tmp77 = libdevice.sqrt(tmp76)
    tmp78 = tl.full([1], 1, tl.int32)
    tmp79 = tmp78 / tmp77
    tmp80 = 1.0
    tmp81 = tmp79 * tmp80
    tmp82 = tmp73 * tmp81
    tmp83 = tl.load(in_ptr18 + ((-12) + x1), tmp70 & xmask, eviction_policy='evict_last', other=0.0)
    tmp84 = tmp82 * tmp83
    tmp85 = tl.load(in_ptr19 + ((-12) + x1), tmp70 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp84 + tmp85
    tmp87 = tl.full(tmp86.shape, 0.0, tmp86.dtype)
    tmp88 = tl.where(tmp70, tmp86, tmp87)
    tmp89 = tmp0 >= tmp68
    tmp90 = tl.full([1], 20, tl.int64)
    tmp91 = tmp0 < tmp90
    tmp92 = tl.load(in_ptr20 + (4*x2 + ((-16) + x1)), tmp89 & xmask, eviction_policy='evict_last', other=0.0)
    tmp93 = tl.where(tmp70, tmp88, tmp92)
    tmp94 = tl.where(tmp48, tmp66, tmp93)
    tmp95 = tl.where(tmp26, tmp44, tmp94)
    tmp96 = tl.where(tmp4, tmp22, tmp95)
    tl.store(out_ptr0 + (x3), tmp96, xmask)
