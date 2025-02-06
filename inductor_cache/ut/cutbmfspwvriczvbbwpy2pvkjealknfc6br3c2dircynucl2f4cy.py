
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 8)
    x3 = xindex // 128
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp21 = tl.load(in_ptr5 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 64*x3), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 5, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 16*x3), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([XBLOCK], 1, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tmp16 = tl.load(in_ptr3 + (x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr4 + (x3), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp20 + tmp22
    tmp24 = tl.load(in_ptr6 + (x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp12
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tmp28 = tmp23 - tmp23
    tmp29 = tl.load(in_ptr7 + (x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp23 + tmp30
    tmp32 = tmp31 - tmp10
    tmp33 = tl.load(in_ptr8 + (x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp10 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp9, tmp35, tmp36)
    tmp38 = tmp0 >= tmp7
    tmp39 = tl.full([1], 6, tl.int64)
    tmp40 = tmp0 < tmp39
    tmp41 = tmp38 & tmp40
    tmp42 = tl.load(in_ptr9 + (x4 + 16*x3), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr10 + (x4 + 16*x3), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tmp0 >= tmp39
    tmp48 = tl.full([1], 7, tl.int64)
    tmp49 = tmp0 < tmp48
    tmp50 = tmp47 & tmp49
    tmp51 = tl.load(in_ptr11 + (x4 + 16*x3), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr12 + (x4 + 16*x3), tmp50 & xmask, eviction_policy='evict_last', other=0.0)
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp50, tmp53, tmp54)
    tmp56 = tmp0 >= tmp48
    tmp57 = tl.full([1], 8, tl.int64)
    tmp58 = tmp0 < tmp57
    tmp59 = tl.load(in_ptr13 + (x4 + 16*x3), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.load(in_ptr14 + (x4 + 16*x3), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp56, tmp61, tmp62)
    tmp64 = tl.where(tmp50, tmp55, tmp63)
    tmp65 = tl.where(tmp41, tmp46, tmp64)
    tmp66 = tl.where(tmp9, tmp37, tmp65)
    tmp67 = tl.where(tmp4, tmp5, tmp66)
    tl.store(out_ptr0 + (x5), tmp67, xmask)
