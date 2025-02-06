
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 128)
    x0 = (xindex % 4)
    x2 = xindex // 512
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 128*x2), tmp4 & xmask, other=0.0)
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
    tmp24 = tl.full([1], 64, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr5 + (x0 + 4*((-32) + x1) + 128*x2), tmp26 & xmask, other=0.0)
    tmp28 = tl.load(in_ptr6 + ((-32) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.load(in_ptr7 + ((-32) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.sqrt(tmp32)
    tmp34 = tl.full([1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = 1.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 * tmp37
    tmp39 = tl.load(in_ptr8 + ((-32) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.load(in_ptr9 + ((-32) + x1), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp26, tmp42, tmp43)
    tmp45 = tmp0 >= tmp24
    tmp46 = tl.full([1], 96, tl.int64)
    tmp47 = tmp0 < tmp46
    tmp48 = tmp45 & tmp47
    tmp49 = tl.load(in_ptr10 + (x0 + 4*((-64) + x1) + 128*x2), tmp48 & xmask, other=0.0)
    tmp50 = tl.load(in_ptr11 + ((-64) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 - tmp50
    tmp52 = tl.load(in_ptr12 + ((-64) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = 1e-05
    tmp54 = tmp52 + tmp53
    tmp55 = libdevice.sqrt(tmp54)
    tmp56 = tl.full([1], 1, tl.int32)
    tmp57 = tmp56 / tmp55
    tmp58 = 1.0
    tmp59 = tmp57 * tmp58
    tmp60 = tmp51 * tmp59
    tmp61 = tl.load(in_ptr13 + ((-64) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp60 * tmp61
    tmp63 = tl.load(in_ptr14 + ((-64) + x1), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp62 + tmp63
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp48, tmp64, tmp65)
    tmp67 = tmp0 >= tmp46
    tmp68 = tl.full([1], 128, tl.int64)
    tmp69 = tmp0 < tmp68
    tmp70 = tl.load(in_ptr15 + (x0 + 4*((-96) + x1) + 128*x2), tmp67 & xmask, other=0.0)
    tmp71 = tl.load(in_ptr16 + ((-96) + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 - tmp71
    tmp73 = tl.load(in_ptr17 + ((-96) + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = 1e-05
    tmp75 = tmp73 + tmp74
    tmp76 = libdevice.sqrt(tmp75)
    tmp77 = tl.full([1], 1, tl.int32)
    tmp78 = tmp77 / tmp76
    tmp79 = 1.0
    tmp80 = tmp78 * tmp79
    tmp81 = tmp72 * tmp80
    tmp82 = tl.load(in_ptr18 + ((-96) + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp81 * tmp82
    tmp84 = tl.load(in_ptr19 + ((-96) + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 + tmp84
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp67, tmp85, tmp86)
    tmp88 = tl.where(tmp48, tmp66, tmp87)
    tmp89 = tl.where(tmp26, tmp44, tmp88)
    tmp90 = tl.where(tmp4, tmp22, tmp89)
    tl.store(out_ptr0 + (x3), tmp90, xmask)
