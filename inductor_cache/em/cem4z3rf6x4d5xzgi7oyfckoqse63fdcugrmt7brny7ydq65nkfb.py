
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 384)
    x3 = xindex // 1572864
    x4 = (xindex % 4096)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 4096*(x2) + 262144*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 4096*((-64) + x2) + 1048576*x3), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 192, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x4 + 4096*((-128) + x2) + 1310720*x3), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 256, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (x4 + 4096*((-192) + x2) + 262144*x3), tmp19, other=0.0)
    tmp21 = tl.load(in_ptr4 + ((-192) + x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 - tmp21
    tmp23 = tl.load(in_ptr5 + ((-192) + x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.sqrt(tmp25)
    tmp27 = tl.full([1], 1, tl.int32)
    tmp28 = tmp27 / tmp26
    tmp29 = 1.0
    tmp30 = tmp28 * tmp29
    tmp31 = tmp22 * tmp30
    tmp32 = tl.load(in_ptr6 + ((-192) + x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 * tmp32
    tmp34 = tl.load(in_ptr7 + ((-192) + x2), tmp19, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full([1], 0, tl.int32)
    tmp37 = triton_helpers.maximum(tmp36, tmp35)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp19, tmp37, tmp38)
    tmp40 = tmp0 >= tmp17
    tmp41 = tl.full([1], 384, tl.int64)
    tmp42 = tmp0 < tmp41
    tmp43 = tl.load(in_ptr8 + (x4 + 4096*((-256) + x2) + 524288*x3), tmp40, other=0.0)
    tmp44 = tl.load(in_ptr9 + (x1), tmp40, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full([XBLOCK], 32, tl.int32)
    tmp46 = tmp44 + tmp45
    tmp47 = tmp44 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp44)
    tmp49 = tl.load(in_ptr10 + (x0), tmp40, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 + tmp45
    tmp51 = tmp49 < 0
    tmp52 = tl.where(tmp51, tmp50, tmp49)
    tmp53 = tl.load(in_ptr11 + (tmp52 + 32*tmp48 + 1024*((-256) + x2) + 131072*x3), tmp40, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr12 + (x0), tmp40, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 + tmp45
    tmp56 = tmp54 < 0
    tmp57 = tl.where(tmp56, tmp55, tmp54)
    tmp58 = tl.load(in_ptr11 + (tmp57 + 32*tmp48 + 1024*((-256) + x2) + 131072*x3), tmp40, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 - tmp53
    tmp60 = tl.load(in_ptr13 + (x0), tmp40, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 * tmp60
    tmp62 = tmp53 + tmp61
    tmp63 = tmp62 - tmp43
    tmp64 = tl.load(in_ptr14 + (x1), tmp40, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tmp43 + tmp65
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp40, tmp66, tmp67)
    tmp69 = tl.where(tmp19, tmp39, tmp68)
    tmp70 = tl.where(tmp14, tmp15, tmp69)
    tmp71 = tl.where(tmp9, tmp10, tmp70)
    tmp72 = tl.where(tmp4, tmp5, tmp71)
    tl.store(out_ptr0 + (x5), tmp72, None)
