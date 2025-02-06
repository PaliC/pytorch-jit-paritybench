
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_copy_index_mean_minimum_mul_remainder_rsub_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_copy_index_mean_minimum_mul_remainder_rsub_sub_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x0 = (xindex % 3)
    r2 = rindex
    x1 = xindex // 3
    x3 = xindex
    tmp13 = tl.load(in_ptr0 + (32 + r2 + 48*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (r2 + 48*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr0 + (16 + r2 + 48*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1, 1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 3.0
    tmp6 = 1.0
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = 5.0
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp10 = tl.full([1, 1], 0, tl.int32)
    tmp11 = tl.full([1, 1], 2, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp15 = tmp13 * tmp14
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = triton_helpers.minimum(tmp19, tmp6)
    tmp21 = 6.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp9 + tmp22
    tmp24 = tmp23 % tmp21
    tmp25 = tmp24 != tmp10
    tmp26 = (libdevice.signbit(tmp24) != 0) if (tmp24).dtype is tl.float32 else tmp24 < 0
    tmp27 = (libdevice.signbit(tmp21) != 0) if (tmp21).dtype is tl.float32 else tmp21 < 0
    tmp28 = tmp26 != tmp27
    tmp29 = tmp25 & tmp28
    tmp30 = tmp24 + tmp21
    tmp31 = tl.where(tmp29, tmp30, tmp24)
    tmp32 = 4.0
    tmp33 = tmp32 - tmp31
    tmp34 = triton_helpers.minimum(tmp31, tmp33)
    tmp35 = triton_helpers.maximum(tmp34, tmp18)
    tmp36 = tmp11 == tmp11
    tmp37 = tl.where(tmp36, tmp15, tmp13)
    tmp38 = triton_helpers.maximum(tmp37, tmp18)
    tmp39 = triton_helpers.minimum(tmp38, tmp6)
    tmp40 = tl.full([1, 1], 1, tl.int32)
    tmp41 = tmp40 == tmp11
    tmp43 = tl.where(tmp41, tmp15, tmp42)
    tmp44 = triton_helpers.maximum(tmp43, tmp18)
    tmp45 = triton_helpers.minimum(tmp44, tmp6)
    tmp46 = tmp39 * tmp45
    tmp47 = triton_helpers.minimum(tmp35, tmp6)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp39 - tmp48
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
    tmp52 = tl.where(xmask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tmp54 = 16.0
    tmp55 = tmp53 / tmp54
    tmp56 = tmp49 - tmp55
    tmp58 = tmp56 * tmp57
    tmp59 = tmp58 + tmp55
    tmp60 = triton_helpers.maximum(tmp59, tmp18)
    tmp61 = triton_helpers.minimum(tmp60, tmp6)
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp61, xmask)
