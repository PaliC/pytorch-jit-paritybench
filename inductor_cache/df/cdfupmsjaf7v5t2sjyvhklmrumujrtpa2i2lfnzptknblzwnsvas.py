
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2240)
    x1 = xindex // 2240
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 900, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (900*x1 + (((x0) % 900))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (((x0) % 4)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1800, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr2 + (900*x1 + ((((-900) + x0) % 900))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + ((((-900) + x0) % 4)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1], 1996, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.load(in_ptr4 + (196*x1 + ((((-1800) + x0) % 196))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr5 + ((((-1800) + x0) % 4)), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tmp0 >= tmp20
    tmp29 = tl.full([1], 2192, tl.int64)
    tmp30 = tmp0 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tl.load(in_ptr6 + (196*x1 + ((((-1996) + x0) % 196))), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr7 + ((((-1996) + x0) % 4)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tmp0 >= tmp29
    tmp38 = tl.full([1], 2228, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = tl.load(in_ptr8 + (36*x1 + ((((-2192) + x0) % 36))), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr9 + ((((-2192) + x0) % 4)), tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp40, tmp43, tmp44)
    tmp46 = tmp0 >= tmp38
    tmp47 = tl.full([1], 2232, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr10 + (4*x1 + ((-2228) + x0)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr11 + ((-2228) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp50 + tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp49, tmp52, tmp53)
    tmp55 = tmp0 >= tmp47
    tmp56 = tl.full([1], 2236, tl.int64)
    tmp57 = tmp0 < tmp56
    tmp58 = tmp55 & tmp57
    tmp59 = tl.load(in_ptr12 + (4*x1 + ((-2232) + x0)), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.load(in_ptr13 + ((-2232) + x0), tmp58 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp58, tmp61, tmp62)
    tmp64 = tmp0 >= tmp56
    tmp65 = tl.full([1], 2240, tl.int64)
    tmp66 = tmp0 < tmp65
    tmp67 = tl.load(in_ptr14 + (4*x1 + ((-2236) + x0)), tmp64 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.load(in_ptr15 + ((-2236) + x0), tmp64 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp64, tmp69, tmp70)
    tmp72 = tl.where(tmp58, tmp63, tmp71)
    tmp73 = tl.where(tmp49, tmp54, tmp72)
    tmp74 = tl.where(tmp40, tmp45, tmp73)
    tmp75 = tl.where(tmp31, tmp36, tmp74)
    tmp76 = tl.where(tmp22, tmp27, tmp75)
    tmp77 = tl.where(tmp13, tmp18, tmp76)
    tmp78 = tl.where(tmp4, tmp9, tmp77)
    tl.store(out_ptr0 + (x2), tmp78, xmask)
