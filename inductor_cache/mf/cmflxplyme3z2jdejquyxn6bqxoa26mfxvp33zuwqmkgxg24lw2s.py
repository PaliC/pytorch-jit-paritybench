
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax__softmax_add_div_mean_mul_neg_reciprocal_sub_sum_0', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax__softmax_add_div_mean_mul_neg_reciprocal_sub_sum_0(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp8 = tl.load(in_ptr2 + (4*r0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr2 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr3 + (4*r0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr3 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr3 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([1, 1], 1, tl.int32)
    tmp5 = tmp4 / tmp3
    tmp6 = -3.0
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = 4.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp0 / tmp11
    tmp15 = tmp14 - tmp2
    tmp16 = tmp4 / tmp15
    tmp17 = tmp16 * tmp6
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19 + tmp10
    tmp21 = tmp13 / tmp20
    tmp22 = triton_helpers.maximum(tmp12, tmp21)
    tmp25 = tmp24 - tmp2
    tmp26 = tmp4 / tmp25
    tmp27 = tmp26 * tmp6
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29 + tmp10
    tmp31 = tmp23 / tmp30
    tmp32 = triton_helpers.maximum(tmp22, tmp31)
    tmp35 = tmp34 - tmp2
    tmp36 = tmp4 / tmp35
    tmp37 = tmp36 * tmp6
    tmp39 = tmp37 * tmp38
    tmp40 = tmp39 + tmp10
    tmp41 = tmp33 / tmp40
    tmp42 = triton_helpers.maximum(tmp32, tmp41)
    tmp44 = tmp43 / tmp11
    tmp46 = tmp45 / tmp20
    tmp47 = triton_helpers.maximum(tmp44, tmp46)
    tmp49 = tmp48 / tmp30
    tmp50 = triton_helpers.maximum(tmp47, tmp49)
    tmp52 = tmp51 / tmp40
    tmp53 = triton_helpers.maximum(tmp50, tmp52)
    tmp54 = tmp44 - tmp53
    tmp55 = tl_math.exp(tmp54)
    tmp56 = tmp46 - tmp53
    tmp57 = tl_math.exp(tmp56)
    tmp58 = tmp55 + tmp57
    tmp59 = tmp49 - tmp53
    tmp60 = tl_math.exp(tmp59)
    tmp61 = tmp58 + tmp60
    tmp62 = tmp52 - tmp53
    tmp63 = tl_math.exp(tmp62)
    tmp64 = tmp61 + tmp63
    tmp65 = tmp12 - tmp42
    tmp66 = tl_math.exp(tmp65)
    tmp67 = tmp21 - tmp42
    tmp68 = tl_math.exp(tmp67)
    tmp69 = tmp66 + tmp68
    tmp70 = tmp31 - tmp42
    tmp71 = tl_math.exp(tmp70)
    tmp72 = tmp69 + tmp71
    tmp73 = tmp41 - tmp42
    tmp74 = tl_math.exp(tmp73)
    tmp75 = tmp72 + tmp74
    tmp76 = tmp55 / tmp64
    tmp77 = tl_math.log(tmp75)
    tmp78 = tmp65 - tmp77
    tmp79 = tmp76 * tmp78
    tmp80 = tmp57 / tmp64
    tmp81 = tmp67 - tmp77
    tmp82 = tmp80 * tmp81
    tmp83 = tmp79 + tmp82
    tmp84 = tmp60 / tmp64
    tmp85 = tmp70 - tmp77
    tmp86 = tmp84 * tmp85
    tmp87 = tmp83 + tmp86
    tmp88 = tmp63 / tmp64
    tmp89 = tmp73 - tmp77
    tmp90 = tmp88 * tmp89
    tmp91 = tmp87 + tmp90
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK, RBLOCK])
    tmp94 = tl.sum(tmp92, 1)[:, None]
    tmp95 = 64.0
    tmp96 = tmp94 / tmp95
    tmp97 = -tmp96
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp97, None)
