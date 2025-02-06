
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_max_mean_mul_neg_pow_relu_sub_sum_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_max_mean_mul_neg_pow_relu_sub_sum_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_out_ptr0 + (r0), None)
    tmp18 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = -3.0
    tmp5 = tmp3 * tmp4
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1, 1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = -0.3333333333333333
    tmp11 = libdevice.pow(tmp9, tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = -0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tmp0 + tmp16
    tmp20 = tmp19 - tmp2
    tmp21 = tmp20 * tmp4
    tmp22 = tmp21 + tmp6
    tmp23 = triton_helpers.maximum(tmp8, tmp22)
    tmp24 = libdevice.pow(tmp23, tmp10)
    tmp25 = tmp12 / tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tmp26 * tmp15
    tmp28 = tmp18 + tmp27
    tmp29 = tmp17 + tmp28
    tmp32 = tmp31 - tmp2
    tmp33 = tmp32 * tmp4
    tmp34 = tmp33 + tmp6
    tmp35 = triton_helpers.maximum(tmp8, tmp34)
    tmp36 = libdevice.pow(tmp35, tmp10)
    tmp37 = tmp12 / tmp36
    tmp38 = tmp37 * tmp37
    tmp39 = tmp38 * tmp15
    tmp40 = tmp30 + tmp39
    tmp41 = tmp29 + tmp40
    tmp44 = tmp43 - tmp2
    tmp45 = tmp44 * tmp4
    tmp46 = tmp45 + tmp6
    tmp47 = triton_helpers.maximum(tmp8, tmp46)
    tmp48 = libdevice.pow(tmp47, tmp10)
    tmp49 = tmp12 / tmp48
    tmp50 = tmp49 * tmp49
    tmp51 = tmp50 * tmp15
    tmp52 = tmp42 + tmp51
    tmp53 = tmp41 + tmp52
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp56 = tl.sum(tmp54, 1)[:, None]
    tmp57 = 64.0
    tmp58 = tmp56 / tmp57
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp58, None)
