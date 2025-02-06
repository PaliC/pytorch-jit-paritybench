
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_min_mul_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_min_mul_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex // 4
    r0 = (rindex % 4)
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (16*r1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (5*r0 + 16*r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r0 + 16*r1), None)
    tmp7 = tl.load(in_ptr0 + (5 + 16*r1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (4 + r0 + 16*r1), None)
    tmp13 = tl.load(in_ptr0 + (10 + 16*r1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (8 + r0 + 16*r1), None)
    tmp19 = tl.load(in_ptr0 + (15 + 16*r1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (12 + r0 + 16*r1), None)
    tmp28 = tl.load(in_ptr0 + (5*r0 + 16*r1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (16*r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (4*r2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (5 + 16*r1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (1 + 4*r2), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr1 + (10 + 16*r1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr2 + (2 + 4*r2), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (15 + 16*r1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr2 + (3 + 4*r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 - tmp5
    tmp8 = tmp7 + tmp1
    tmp10 = tmp9 * tmp4
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.minimum(tmp6, tmp11)
    tmp14 = tmp13 + tmp1
    tmp16 = tmp15 * tmp4
    tmp17 = tmp14 - tmp16
    tmp18 = triton_helpers.minimum(tmp12, tmp17)
    tmp20 = tmp19 + tmp1
    tmp22 = tmp21 * tmp4
    tmp23 = tmp20 - tmp22
    tmp24 = triton_helpers.minimum(tmp18, tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp30 = tmp28 + tmp29
    tmp32 = tmp31 * tmp4
    tmp33 = tmp30 - tmp32
    tmp35 = tmp28 + tmp34
    tmp37 = tmp36 * tmp4
    tmp38 = tmp35 - tmp37
    tmp39 = triton_helpers.minimum(tmp33, tmp38)
    tmp41 = tmp28 + tmp40
    tmp43 = tmp42 * tmp4
    tmp44 = tmp41 - tmp43
    tmp45 = triton_helpers.minimum(tmp39, tmp44)
    tmp47 = tmp28 + tmp46
    tmp49 = tmp48 * tmp4
    tmp50 = tmp47 - tmp49
    tmp51 = triton_helpers.minimum(tmp45, tmp50)
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, RBLOCK])
    tmp54 = tl.sum(tmp52, 1)[:, None]
    tmp55 = tmp27 + tmp54
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp55, None)
