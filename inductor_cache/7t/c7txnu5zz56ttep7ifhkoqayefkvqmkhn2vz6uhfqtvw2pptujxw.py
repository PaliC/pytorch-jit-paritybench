
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_log_mean_mul_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_log_mean_mul_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    r1 = rindex // 4
    tmp0 = tl.load(in_ptr0 + (r2), None)
    tmp5 = tl.load(in_ptr0 + (4*r1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (r2), None)
    tmp26 = tl.load(in_ptr1 + (4*r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr1 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp6 = tmp5 + tmp1
    tmp7 = tmp6 * tmp3
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 * tmp3
    tmp11 = tmp7 + tmp10
    tmp13 = tmp12 + tmp1
    tmp14 = tmp13 * tmp3
    tmp15 = tmp11 + tmp14
    tmp17 = tmp16 + tmp1
    tmp18 = tmp17 * tmp3
    tmp19 = tmp15 + tmp18
    tmp20 = tmp4 / tmp19
    tmp21 = 1e-07
    tmp22 = tmp20 + tmp21
    tmp24 = tmp23 + tmp1
    tmp25 = tmp24 * tmp3
    tmp27 = tmp26 + tmp1
    tmp28 = tmp27 * tmp3
    tmp30 = tmp29 + tmp1
    tmp31 = tmp30 * tmp3
    tmp32 = tmp28 + tmp31
    tmp34 = tmp33 + tmp1
    tmp35 = tmp34 * tmp3
    tmp36 = tmp32 + tmp35
    tmp38 = tmp37 + tmp1
    tmp39 = tmp38 * tmp3
    tmp40 = tmp36 + tmp39
    tmp41 = tmp25 / tmp40
    tmp42 = tmp41 + tmp21
    tmp43 = tmp22 / tmp42
    tmp44 = tl_math.log(tmp43)
    tmp45 = tmp20 * tmp44
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp48 = tl.sum(tmp46, 1)[:, None]
    tmp49 = 16.0
    tmp50 = tmp48 / tmp49
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp50, None)
