
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1905152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 64, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (32*x1 + ((-32) + x0)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-32) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-32) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-32) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-32) + x0), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 128, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tl.load(in_ptr10 + (64*x1 + ((-64) + x0)), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = tl.load(in_ptr11 + ((-64) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tmp52 - tmp53
    tmp55 = tl.load(in_ptr12 + ((-64) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = 0.001
    tmp57 = tmp55 + tmp56
    tmp58 = libdevice.sqrt(tmp57)
    tmp59 = tl.full([1], 1, tl.int32)
    tmp60 = tmp59 / tmp58
    tmp61 = 1.0
    tmp62 = tmp60 * tmp61
    tmp63 = tmp54 * tmp62
    tmp64 = tl.load(in_ptr13 + ((-64) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr14 + ((-64) + x0), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.full([1], 0, tl.int32)
    tmp69 = triton_helpers.maximum(tmp68, tmp67)
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp49, tmp69, tmp70)
    tmp72 = tl.where(tmp28, tmp48, tmp71)
    tmp73 = tl.where(tmp4, tmp24, tmp72)
    tl.store(out_ptr0 + (x2), tmp73, xmask)
