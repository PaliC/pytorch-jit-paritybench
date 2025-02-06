
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_index_abs_add_arange_clamp_exp_mul_pow_sub_sum_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_index_abs_add_arange_clamp_exp_mul_pow_sub_sum_3(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x7 = xindex
    r6 = rindex
    x3 = (xindex % 16)
    x5 = xindex // 64
    tmp39 = tl.load(in_ptr1 + (r6 + 16*x3 + 256*x5), xmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1, 1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1, 1], 3, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = x0
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = triton_helpers.maximum(tmp13, tmp4)
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + 4*tmp10 + 16*x2), xmask, eviction_policy='evict_last')
    tmp17 = tmp15 + tmp7
    tmp18 = triton_helpers.minimum(tmp17, tmp9)
    tmp19 = tl.load(in_ptr0 + (tmp18 + 4*tmp10 + 16*x2), xmask, eviction_policy='evict_last')
    tmp20 = tmp19 - tmp16
    tmp21 = tmp15.to(tl.float32)
    tmp22 = tmp14 - tmp21
    tmp23 = triton_helpers.maximum(tmp22, tmp4)
    tmp24 = triton_helpers.minimum(tmp23, tmp2)
    tmp25 = tmp20 * tmp24
    tmp26 = tmp16 + tmp25
    tmp27 = tl.load(in_ptr0 + (tmp15 + 4*tmp6 + 16*x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (tmp18 + 4*tmp6 + 16*x2), xmask, eviction_policy='evict_last')
    tmp29 = tmp28 - tmp27
    tmp30 = tmp29 * tmp24
    tmp31 = tmp27 + tmp30
    tmp32 = tmp26 - tmp31
    tmp33 = tmp6.to(tl.float32)
    tmp34 = tmp5 - tmp33
    tmp35 = triton_helpers.maximum(tmp34, tmp4)
    tmp36 = triton_helpers.minimum(tmp35, tmp2)
    tmp37 = tmp32 * tmp36
    tmp38 = tmp31 + tmp37
    tmp40 = tmp39 * tmp2
    tmp41 = 20.0
    tmp42 = tmp40 > tmp41
    tmp43 = tl_math.exp(tmp40)
    tmp44 = libdevice.log1p(tmp43)
    tmp45 = tmp44 * tmp2
    tmp46 = tl.where(tmp42, tmp39, tmp45)
    tmp47 = tmp46 - tmp38
    tmp48 = tl_math.abs(tmp47)
    tmp49 = tmp48 * tmp48
    tmp50 = -300.0
    tmp51 = tmp49 * tmp50
    tmp52 = tl_math.exp(tmp51)
    tmp53 = tmp52 * tmp47
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp56 = tl.where(xmask, tmp54, 0)
    tmp57 = tl.sum(tmp56, 1)[:, None]
    tmp58 = tmp38 + tmp57
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x7), tmp38, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x7), tmp58, xmask)
