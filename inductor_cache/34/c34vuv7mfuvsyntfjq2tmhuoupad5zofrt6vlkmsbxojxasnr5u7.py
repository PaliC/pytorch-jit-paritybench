
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bitwise_and_bitwise_not_bitwise_or_div_eq_gt_mul_relu_sub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_bitwise_and_bitwise_not_bitwise_or_div_eq_gt_mul_relu_sub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = ((rindex // 16) % 4)
    r1 = ((rindex // 4) % 4)
    r0 = (rindex % 4)
    r3 = rindex // 64
    r6 = (rindex % 16)
    r7 = rindex
    r4 = ((rindex // 4) % 16)
    tmp19 = tl.load(in_ptr0 + (r0 + 4*r2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (r0 + 4*r3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (r6), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr1 + (r4), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (r0 + 4*r2), None, eviction_policy='evict_last')
    tmp0 = r2
    tmp1 = r1
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp7 = tmp6 == 0
    tmp8 = r0
    tmp9 = tmp0 == tmp8
    tmp10 = tl.where(tmp9, tmp3, tmp4)
    tmp11 = (tmp10 != 0)
    tmp12 = tmp11 == 0
    tmp13 = tmp7 & tmp12
    tmp14 = tmp1 == tmp8
    tmp15 = tl.where(tmp14, tmp3, tmp4)
    tmp16 = (tmp15 != 0)
    tmp17 = tmp16 == 0
    tmp18 = tmp13 & tmp17
    tmp21 = tmp19 == tmp20
    tmp23 = tmp22 == tmp20
    tmp24 = tmp23 == 0
    tmp25 = tmp21 & tmp24
    tmp26 = tmp18 & tmp25
    tmp27 = -1.0
    tmp28 = tmp20 >= tmp27
    tmp29 = tmp28 == 0
    tmp30 = tmp19 >= tmp27
    tmp31 = tmp30 == 0
    tmp32 = tmp29 & tmp31
    tmp33 = tmp20 + tmp27
    tmp34 = tmp33 == tmp22
    tmp35 = tmp22 >= tmp27
    tmp36 = tmp35 == 0
    tmp37 = tmp34 | tmp36
    tmp38 = tmp32 & tmp37
    tmp39 = tmp26 & tmp38
    tmp41 = tmp3 - tmp40
    tmp43 = tmp3 - tmp42
    tmp44 = tmp41 - tmp43
    tmp45 = 0.3
    tmp46 = tmp44 + tmp45
    tmp47 = tl.full([1], 0, tl.int32)
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tmp49 = tmp39.to(tl.float32)
    tmp50 = tmp48 * tmp49
    tmp51 = tl.broadcast_to(tmp50, [RBLOCK])
    tmp53 = triton_helpers.promote_to_tensor(tl.sum(tmp51, 0))
    tmp54 = 1e-08
    tmp55 = tmp50 > tmp54
    tmp56 = tmp55.to(tl.int64)
    tmp57 = tl.broadcast_to(tmp56, [RBLOCK])
    tmp59 = triton_helpers.promote_to_tensor(tl.sum(tmp57, 0))
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 + tmp54
    tmp62 = tmp53 / tmp61
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp62, None)
