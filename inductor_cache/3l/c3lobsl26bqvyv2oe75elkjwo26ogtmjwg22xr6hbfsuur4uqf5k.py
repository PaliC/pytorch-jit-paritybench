
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_log_sigmoid_forward_mean_mul_neg_rsub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_log_sigmoid_forward_mean_mul_neg_rsub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp22 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = triton_helpers.minimum(tmp2, tmp1)
    tmp4 = tl_math.abs(tmp1)
    tmp5 = -tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = libdevice.log1p(tmp6)
    tmp8 = tmp3 - tmp7
    tmp9 = tmp0 * tmp8
    tmp10 = 1.0
    tmp11 = tmp10 - tmp0
    tmp12 = -tmp1
    tmp13 = triton_helpers.minimum(tmp2, tmp12)
    tmp14 = tl_math.abs(tmp12)
    tmp15 = -tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = libdevice.log1p(tmp16)
    tmp18 = tmp13 - tmp17
    tmp19 = tmp11 * tmp18
    tmp20 = tmp9 + tmp19
    tmp21 = -tmp20
    tmp24 = triton_helpers.minimum(tmp2, tmp23)
    tmp25 = tl_math.abs(tmp23)
    tmp26 = -tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp28 = libdevice.log1p(tmp27)
    tmp29 = tmp24 - tmp28
    tmp30 = tmp22 * tmp29
    tmp31 = tmp10 - tmp22
    tmp32 = -tmp23
    tmp33 = triton_helpers.minimum(tmp2, tmp32)
    tmp34 = tl_math.abs(tmp32)
    tmp35 = -tmp34
    tmp36 = tl_math.exp(tmp35)
    tmp37 = libdevice.log1p(tmp36)
    tmp38 = tmp33 - tmp37
    tmp39 = tmp31 * tmp38
    tmp40 = tmp30 + tmp39
    tmp41 = -tmp40
    tmp42 = tmp21 + tmp41
    tmp45 = triton_helpers.minimum(tmp2, tmp44)
    tmp46 = tl_math.abs(tmp44)
    tmp47 = -tmp46
    tmp48 = tl_math.exp(tmp47)
    tmp49 = libdevice.log1p(tmp48)
    tmp50 = tmp45 - tmp49
    tmp51 = tmp43 * tmp50
    tmp52 = tmp10 - tmp43
    tmp53 = -tmp44
    tmp54 = triton_helpers.minimum(tmp2, tmp53)
    tmp55 = tl_math.abs(tmp53)
    tmp56 = -tmp55
    tmp57 = tl_math.exp(tmp56)
    tmp58 = libdevice.log1p(tmp57)
    tmp59 = tmp54 - tmp58
    tmp60 = tmp52 * tmp59
    tmp61 = tmp51 + tmp60
    tmp62 = -tmp61
    tmp63 = tmp42 + tmp62
    tmp66 = triton_helpers.minimum(tmp2, tmp65)
    tmp67 = tl_math.abs(tmp65)
    tmp68 = -tmp67
    tmp69 = tl_math.exp(tmp68)
    tmp70 = libdevice.log1p(tmp69)
    tmp71 = tmp66 - tmp70
    tmp72 = tmp64 * tmp71
    tmp73 = tmp10 - tmp64
    tmp74 = -tmp65
    tmp75 = triton_helpers.minimum(tmp2, tmp74)
    tmp76 = tl_math.abs(tmp74)
    tmp77 = -tmp76
    tmp78 = tl_math.exp(tmp77)
    tmp79 = libdevice.log1p(tmp78)
    tmp80 = tmp75 - tmp79
    tmp81 = tmp73 * tmp80
    tmp82 = tmp72 + tmp81
    tmp83 = -tmp82
    tmp84 = tmp63 + tmp83
    tmp85 = 0.25
    tmp86 = tmp84 * tmp85
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK, RBLOCK])
    tmp89 = tl.sum(tmp87, 1)[:, None]
    tmp90 = 64.0
    tmp91 = tmp89 / tmp90
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp91, None)
