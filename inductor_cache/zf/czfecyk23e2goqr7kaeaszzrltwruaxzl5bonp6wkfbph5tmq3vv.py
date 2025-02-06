
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_mean_mul_sub_sum_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_mean_mul_sub_sum_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (4*r0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr2 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr2 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp3 = tl_math.exp(tmp2)
    tmp4 = tmp1 + tmp3
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tmp4 + tmp6
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tmp7 + tmp9
    tmp11 = tl_math.log(tmp10)
    tmp12 = tmp0 - tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tmp2 - tmp11
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp5 - tmp11
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp8 - tmp11
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
    tmp29 = tl.sum(tmp27, 1)[:, None]
    tmp31 = tl_math.exp(tmp30)
    tmp33 = tl_math.exp(tmp32)
    tmp34 = tmp31 + tmp33
    tmp36 = tl_math.exp(tmp35)
    tmp37 = tmp34 + tmp36
    tmp39 = tl_math.exp(tmp38)
    tmp40 = tmp37 + tmp39
    tmp41 = tl_math.log(tmp40)
    tmp42 = tmp30 - tmp41
    tmp43 = tmp42 * tmp13
    tmp44 = tmp32 - tmp41
    tmp45 = tmp44 * tmp16
    tmp46 = tmp43 + tmp45
    tmp47 = tmp35 - tmp41
    tmp48 = tmp47 * tmp20
    tmp49 = tmp46 + tmp48
    tmp50 = tmp38 - tmp41
    tmp51 = tmp50 * tmp24
    tmp52 = tmp49 + tmp51
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK, RBLOCK])
    tmp55 = tl.sum(tmp53, 1)[:, None]
    tmp56 = 64.0
    tmp57 = tmp29 / tmp56
    tmp58 = tmp55 / tmp56
    tmp59 = tmp57 - tmp58
    tmp60 = 0.1
    tmp61 = tmp59 * tmp60
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp61, None)
