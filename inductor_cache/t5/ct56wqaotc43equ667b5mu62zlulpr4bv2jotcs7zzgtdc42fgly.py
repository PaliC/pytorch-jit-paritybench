
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_log10_mean_mul_neg_pow_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_log10_mean_mul_neg_pow_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tmp1 * tmp1
    tmp16 = tmp4 * tmp4
    tmp17 = tmp15 + tmp16
    tmp18 = tmp8 * tmp8
    tmp19 = tmp17 + tmp18
    tmp20 = tmp12 * tmp12
    tmp21 = tmp19 + tmp20
    tmp22 = 1e-08
    tmp23 = tmp21 + tmp22
    tmp24 = tmp14 / tmp23
    tmp25 = tmp1 * tmp24
    tmp26 = tmp0 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tmp4 * tmp24
    tmp29 = tmp3 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tmp27 + tmp30
    tmp32 = tmp8 * tmp24
    tmp33 = tmp7 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tmp12 * tmp24
    tmp37 = tmp11 - tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tmp35 + tmp38
    tmp40 = tmp25 * tmp25
    tmp41 = tmp28 * tmp28
    tmp42 = tmp40 + tmp41
    tmp43 = tmp32 * tmp32
    tmp44 = tmp42 + tmp43
    tmp45 = tmp36 * tmp36
    tmp46 = tmp44 + tmp45
    tmp47 = tmp39 + tmp22
    tmp48 = tmp46 / tmp47
    tmp49 = tmp48 + tmp22
    tmp50 = libdevice.log10(tmp49)
    tmp51 = 10.0
    tmp52 = tmp50 * tmp51
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
    tmp55 = tl.sum(tmp53, 1)[:, None]
    tmp56 = 64.0
    tmp57 = tmp55 / tmp56
    tmp58 = -tmp57
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp58, None)
