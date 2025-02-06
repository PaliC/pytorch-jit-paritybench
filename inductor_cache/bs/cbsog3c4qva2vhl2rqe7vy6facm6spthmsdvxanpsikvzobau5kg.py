
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 4, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (tmp14 + 4*tmp10 + 16*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (tmp19 + 4*tmp10 + 16*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 4, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 16*((-1) + x2) + 48*x3), tmp31 & xmask, other=0.0)
    tmp35 = tl.load(in_ptr1 + (x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full([XBLOCK], 4, tl.int32)
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tmp40 = tl.load(in_ptr2 + (x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 + tmp36
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp44 = tl.load(in_ptr8 + (tmp43 + 4*tmp39 + 16*((-1) + x2) + 48*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr4 + (x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp36
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tmp49 = tl.load(in_ptr8 + (tmp48 + 4*tmp39 + 16*((-1) + x2) + 48*x3), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 - tmp44
    tmp51 = tl.load(in_ptr5 + (x0), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp50 * tmp51
    tmp53 = tmp44 + tmp52
    tmp54 = tmp53 - tmp34
    tmp55 = tl.load(in_ptr6 + (x1), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp34 + tmp56
    tmp58 = 1.0
    tmp59 = tmp57 * tmp58
    tmp60 = tmp59 * tmp58
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp31, tmp60, tmp61)
    tmp63 = tl.where(tmp4, tmp30, tmp62)
    tl.store(out_ptr0 + (x6), tmp63, xmask)
