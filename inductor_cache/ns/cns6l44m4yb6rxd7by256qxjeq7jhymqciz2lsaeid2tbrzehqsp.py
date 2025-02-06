
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_exp_log_mean_mul_neg_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 22, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_exp_log_mean_mul_neg_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 4)
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (4*r2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (1 + 4*r2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (2 + 4*r2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (3 + 4*r2), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (r0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr1 + (r0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (r0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (4 + r0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (4 + r0), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (8 + r0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr1 + (8 + r0), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (12 + r0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr1 + (12 + r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp3 = 0.25
    tmp4 = tmp2 * tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp7 = tmp5 / tmp6
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 / tmp11
    tmp13 = tmp12 * tmp3
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp14 / tmp6
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp21 = tmp19 / tmp20
    tmp22 = tmp21 * tmp3
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp23 / tmp6
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 + tmp26
    tmp30 = tmp28 / tmp29
    tmp31 = tmp30 * tmp3
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp32 / tmp6
    tmp35 = tmp33 * tmp34
    tmp36 = tmp27 + tmp35
    tmp39 = tmp37 / tmp38
    tmp40 = tmp39 * tmp3
    tmp41 = tl_math.exp(tmp40)
    tmp43 = tmp41 / tmp42
    tmp44 = tmp43 * tmp8
    tmp47 = tmp45 / tmp46
    tmp48 = tmp47 * tmp3
    tmp49 = tl_math.exp(tmp48)
    tmp50 = tmp49 / tmp42
    tmp51 = tmp50 * tmp16
    tmp52 = tmp44 + tmp51
    tmp55 = tmp53 / tmp54
    tmp56 = tmp55 * tmp3
    tmp57 = tl_math.exp(tmp56)
    tmp58 = tmp57 / tmp42
    tmp59 = tmp58 * tmp25
    tmp60 = tmp52 + tmp59
    tmp63 = tmp61 / tmp62
    tmp64 = tmp63 * tmp3
    tmp65 = tl_math.exp(tmp64)
    tmp66 = tmp65 / tmp42
    tmp67 = tmp66 * tmp34
    tmp68 = tmp60 + tmp67
    tmp69 = tl_math.log(tmp36)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, RBLOCK])
    tmp72 = tl.sum(tmp70, 1)[:, None]
    tmp73 = tl_math.log(tmp68)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, RBLOCK])
    tmp76 = tl.sum(tmp74, 1)[:, None]
    tmp77 = 64.0
    tmp78 = tmp72 / tmp77
    tmp79 = -tmp78
    tmp80 = 4.0
    tmp81 = tmp79 * tmp80
    tmp82 = tmp76 / tmp77
    tmp83 = -tmp82
    tmp84 = -3.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp81 + tmp85
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp86, None)
