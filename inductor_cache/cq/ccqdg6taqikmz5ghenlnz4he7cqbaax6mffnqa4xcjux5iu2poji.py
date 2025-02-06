
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 320)
    x3 = xindex // 81920
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 65536*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x4 + 256*(x2) + 65536*x3), tmp4, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 - tmp7
    tmp9 = tl.load(in_ptr3 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp18 = tl.load(in_ptr4 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr5 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tmp5 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 320, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr6 + (x4 + 256*((-256) + x2) + 16384*x3), tmp25, other=0.0)
    tmp29 = tl.load(in_ptr7 + (x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.full([XBLOCK], 32, tl.int32)
    tmp31 = tmp29 + tmp30
    tmp32 = tmp29 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp29)
    tmp34 = tl.load(in_ptr8 + (x0), tmp25, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp30
    tmp36 = tmp34 < 0
    tmp37 = tl.where(tmp36, tmp35, tmp34)
    tmp38 = tl.load(in_ptr9 + (tmp37 + 32*tmp33 + 1024*((-256) + x2) + 65536*x3), tmp25, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_ptr10 + (x0), tmp25, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp39 + tmp30
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.load(in_ptr9 + (tmp42 + 32*tmp33 + 1024*((-256) + x2) + 65536*x3), tmp25, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp43 - tmp38
    tmp45 = tl.load(in_ptr11 + (x0), tmp25, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 * tmp45
    tmp47 = tmp38 + tmp46
    tmp48 = tmp47 - tmp28
    tmp49 = tl.load(in_ptr12 + (x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp48 * tmp49
    tmp51 = tmp28 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp25, tmp51, tmp52)
    tmp54 = tl.where(tmp4, tmp24, tmp53)
    tl.store(out_ptr0 + (x5), tmp54, None)
