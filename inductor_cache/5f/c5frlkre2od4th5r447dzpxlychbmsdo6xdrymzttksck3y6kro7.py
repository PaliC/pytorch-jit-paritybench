
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_exp_mean_mul_stack_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_exp_mean_mul_stack_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp16 = tl.load(in_ptr0 + (4))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp41 = tl.load(in_ptr0 + (11))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, RBLOCK])
    tmp71 = tl.load(in_ptr1 + (4))
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK, RBLOCK])
    tmp78 = tl.load(in_ptr1 + (11))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK, RBLOCK])
    tmp104 = tl.load(in_ptr2 + (0))
    tmp105 = tl.broadcast_to(tmp104, [XBLOCK, 1])
    tmp0 = r0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 3, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(1 + (r0), [XBLOCK, RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 6, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.broadcast_to((-3) + r0, [XBLOCK, RBLOCK])
    tmp11 = tl.full([1, 1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1, 1], 1, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tmp14 & tmp9
    tmp18 = tmp10 >= tmp13
    tmp19 = tl.full([1, 1], 3, tl.int64)
    tmp20 = tmp10 < tmp19
    tmp21 = tmp18 & tmp9
    tmp22 = tl.load(in_ptr0 + (tl.broadcast_to(6 + ((-1) + ((-3) + r0)), [XBLOCK, RBLOCK])), rmask & tmp21, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp14, tmp17, tmp22)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp9, tmp23, tmp24)
    tmp26 = tmp0 >= tmp7
    tmp27 = tl.full([1, 1], 9, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.broadcast_to((-6) + r0, [XBLOCK, RBLOCK])
    tmp31 = tl.full([1, 1], 0, tl.int64)
    tmp32 = tmp30 >= tmp31
    tmp33 = tl.full([1, 1], 2, tl.int64)
    tmp34 = tmp30 < tmp33
    tmp35 = tmp34 & tmp29
    tmp36 = tl.load(in_ptr0 + (tl.broadcast_to(8 + ((-6) + r0), [XBLOCK, RBLOCK])), rmask & tmp35, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp30 >= tmp33
    tmp38 = tl.full([1, 1], 3, tl.int64)
    tmp39 = tmp30 < tmp38
    tmp40 = tmp37 & tmp29
    tmp43 = tl.where(tmp34, tmp36, tmp42)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp29, tmp43, tmp44)
    tmp46 = tmp0 >= tmp27
    tmp47 = tl.full([1, 1], 12, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tl.load(in_ptr0 + (tl.broadcast_to(12 + ((-9) + r0), [XBLOCK, RBLOCK])), rmask & tmp46, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.where(tmp29, tmp45, tmp49)
    tmp51 = tl.where(tmp9, tmp25, tmp50)
    tmp52 = tl.where(tmp4, tmp5, tmp51)
    tmp53 = -0.5
    tmp54 = tmp52 * tmp53
    tmp55 = tl_math.exp(tmp54)
    tmp56 = 0.0
    tmp57 = tmp55 + tmp56
    tmp58 = -0.02
    tmp59 = tmp52 * tmp58
    tmp60 = tl_math.exp(tmp59)
    tmp61 = tmp57 + tmp60
    tmp62 = -0.005
    tmp63 = tmp52 * tmp62
    tmp64 = tl_math.exp(tmp63)
    tmp65 = tmp61 + tmp64
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, RBLOCK])
    tmp68 = tl.where(rmask, tmp66, 0)
    tmp69 = tl.sum(tmp68, 1)[:, None]
    tmp70 = tl.load(in_ptr1 + (tl.broadcast_to(1 + (r0), [XBLOCK, RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.load(in_ptr1 + (tl.broadcast_to(6 + ((-1) + ((-3) + r0)), [XBLOCK, RBLOCK])), rmask & tmp21, eviction_policy='evict_last', other=0.0)
    tmp74 = tl.where(tmp14, tmp72, tmp73)
    tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
    tmp76 = tl.where(tmp9, tmp74, tmp75)
    tmp77 = tl.load(in_ptr1 + (tl.broadcast_to(8 + ((-6) + r0), [XBLOCK, RBLOCK])), rmask & tmp35, eviction_policy='evict_last', other=0.0)
    tmp80 = tl.where(tmp34, tmp77, tmp79)
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp29, tmp80, tmp81)
    tmp83 = tl.load(in_ptr1 + (tl.broadcast_to(12 + ((-9) + r0), [XBLOCK, RBLOCK])), rmask & tmp46, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.where(tmp29, tmp82, tmp83)
    tmp85 = tl.where(tmp9, tmp76, tmp84)
    tmp86 = tl.where(tmp4, tmp70, tmp85)
    tmp87 = tmp86 * tmp53
    tmp88 = tl_math.exp(tmp87)
    tmp89 = tmp88 + tmp56
    tmp90 = tmp86 * tmp58
    tmp91 = tl_math.exp(tmp90)
    tmp92 = tmp89 + tmp91
    tmp93 = tmp86 * tmp62
    tmp94 = tl_math.exp(tmp93)
    tmp95 = tmp92 + tmp94
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK, RBLOCK])
    tmp98 = tl.where(rmask, tmp96, 0)
    tmp99 = tl.sum(tmp98, 1)[:, None]
    tmp100 = 12.0
    tmp101 = tmp69 / tmp100
    tmp102 = tmp99 / tmp100
    tmp103 = tmp101 + tmp102
    tmp106 = 16.0
    tmp107 = tmp105 / tmp106
    tmp108 = 2.0
    tmp109 = tmp107 * tmp108
    tmp110 = tmp103 - tmp109
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp110, None)
