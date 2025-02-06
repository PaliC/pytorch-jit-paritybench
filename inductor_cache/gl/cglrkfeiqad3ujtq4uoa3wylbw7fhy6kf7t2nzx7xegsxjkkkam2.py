
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_exp_mean_neg_pow_sub_sum_tanh_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_exp_mean_neg_pow_sub_sum_tanh_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr2 + (4*r0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr2 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (4*r0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr3 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr3 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr3 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = -tmp3
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp4 / tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = -tmp12
    tmp15 = libdevice.tanh(tmp14)
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp13 / tmp16
    tmp18 = tmp8 + tmp17
    tmp21 = tmp19 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = -tmp22
    tmp25 = libdevice.tanh(tmp24)
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp23 / tmp26
    tmp28 = tmp18 + tmp27
    tmp31 = tmp29 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = -tmp32
    tmp35 = libdevice.tanh(tmp34)
    tmp36 = tl_math.exp(tmp35)
    tmp37 = tmp33 / tmp36
    tmp38 = tmp28 + tmp37
    tmp40 = tmp0 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = -tmp41
    tmp43 = tmp42 / tmp7
    tmp45 = tmp9 - tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = -tmp46
    tmp48 = tmp47 / tmp16
    tmp49 = tmp43 + tmp48
    tmp51 = tmp19 - tmp50
    tmp52 = tmp51 * tmp51
    tmp53 = -tmp52
    tmp54 = tmp53 / tmp26
    tmp55 = tmp49 + tmp54
    tmp57 = tmp29 - tmp56
    tmp58 = tmp57 * tmp57
    tmp59 = -tmp58
    tmp60 = tmp59 / tmp36
    tmp61 = tmp55 + tmp60
    tmp62 = tmp38 - tmp61
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK, RBLOCK])
    tmp65 = tl.sum(tmp63, 1)[:, None]
    tmp66 = 64.0
    tmp67 = tmp65 / tmp66
    tmp68 = 0.5
    tmp69 = tmp67 * tmp68
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp69, None)
