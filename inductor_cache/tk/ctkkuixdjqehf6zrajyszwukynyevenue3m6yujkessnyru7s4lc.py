
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 16)
    x0 = (xindex % 4)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 16*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp14 / tmp13
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 * tmp17
    tmp19 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 8, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 5*((-4) + x1) + 20*x2), tmp28 & xmask, other=0.0)
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = tl.load(in_ptr6 + ((-4) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-4) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-4) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-4) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 12, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (x0 + 4*((-8) + x1) + 16*x2), tmp52 & xmask, other=0.0)
    tmp54 = tl.full([1], 0, tl.int32)
    tmp55 = triton_helpers.maximum(tmp54, tmp53)
    tmp56 = tl.load(in_ptr11 + ((-8) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr12 + ((-8) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = 1e-05
    tmp60 = tmp58 + tmp59
    tmp61 = libdevice.sqrt(tmp60)
    tmp62 = tl.full([1], 1, tl.int32)
    tmp63 = tmp62 / tmp61
    tmp64 = 1.0
    tmp65 = tmp63 * tmp64
    tmp66 = tmp57 * tmp65
    tmp67 = tl.load(in_ptr13 + ((-8) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 * tmp67
    tmp69 = tl.load(in_ptr14 + ((-8) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 + tmp69
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 16, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (x0 + 5*((-12) + x1) + 20*x2), tmp73 & xmask, other=0.0)
    tmp77 = tl.full([1], 0, tl.int32)
    tmp78 = triton_helpers.maximum(tmp77, tmp76)
    tmp79 = tl.load(in_ptr16 + ((-12) + x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tmp78 - tmp79
    tmp81 = tl.load(in_ptr17 + ((-12) + x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp82 = 1e-05
    tmp83 = tmp81 + tmp82
    tmp84 = libdevice.sqrt(tmp83)
    tmp85 = tl.full([1], 1, tl.int32)
    tmp86 = tmp85 / tmp84
    tmp87 = 1.0
    tmp88 = tmp86 * tmp87
    tmp89 = tmp80 * tmp88
    tmp90 = tl.load(in_ptr18 + ((-12) + x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 * tmp90
    tmp92 = tl.load(in_ptr19 + ((-12) + x1), tmp73 & xmask, eviction_policy='evict_last', other=0.0)
    tmp93 = tmp91 + tmp92
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x3), tmp98, xmask)
