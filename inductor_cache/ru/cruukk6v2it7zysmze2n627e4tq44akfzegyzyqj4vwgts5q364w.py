
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr8': '*i64', 'out_ptr9': '*i64', 'out_ptr10': '*i64', 'out_ptr11': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_argmax_eq_stack_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_argmax_eq_stack_sum_0(in_ptr0, out_ptr8, out_ptr9, out_ptr10, out_ptr11, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 == tmp1
    tmp4 = tmp0 != tmp0
    tmp5 = tmp1 != tmp1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp2 | tmp6
    tmp8 = tmp4 & tmp5
    tmp9 = tmp3 | tmp8
    tmp10 = tl.full([1, 1], 0, tl.int64)
    tmp11 = tl.full([1, 1], 1, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tmp7 | tmp13
    tmp15 = tl.where(tmp14, tmp0, tmp1)
    tmp16 = tl.where(tmp14, tmp10, tmp11)
    tmp18 = tmp15 > tmp17
    tmp19 = tmp15 == tmp17
    tmp20 = tmp15 != tmp15
    tmp21 = tmp17 != tmp17
    tmp22 = tmp20 > tmp21
    tmp23 = tmp18 | tmp22
    tmp24 = tmp20 & tmp21
    tmp25 = tmp19 | tmp24
    tmp26 = tl.full([1, 1], 2, tl.int64)
    tmp27 = tmp16 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tmp23 | tmp28
    tmp30 = tl.where(tmp29, tmp15, tmp17)
    tmp31 = tl.where(tmp29, tmp16, tmp26)
    tmp33 = tmp30 > tmp32
    tmp34 = tmp30 == tmp32
    tmp35 = tmp30 != tmp30
    tmp36 = tmp32 != tmp32
    tmp37 = tmp35 > tmp36
    tmp38 = tmp33 | tmp37
    tmp39 = tmp35 & tmp36
    tmp40 = tmp34 | tmp39
    tmp41 = tl.full([1, 1], 3, tl.int64)
    tmp42 = tmp31 < tmp41
    tmp43 = tmp40 & tmp42
    tmp44 = tmp38 | tmp43
    tmp45 = tl.where(tmp44, tmp30, tmp32)
    tmp46 = tl.where(tmp44, tmp31, tmp41)
    tmp47 = tmp46 == tmp10
    tmp48 = tmp47.to(tl.int64)
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.sum(tmp49, 1)[:, None]
    tmp52 = tmp46 == tmp11
    tmp53 = tmp52.to(tl.int64)
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp56 = tl.sum(tmp54, 1)[:, None]
    tmp57 = tmp46 == tmp26
    tmp58 = tmp57.to(tl.int64)
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
    tmp61 = tl.sum(tmp59, 1)[:, None]
    tmp62 = tmp46 == tmp41
    tmp63 = tmp62.to(tl.int64)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp66 = tl.sum(tmp64, 1)[:, None]
    tl.store(out_ptr8 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp66, None)
    tl.store(out_ptr9 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp61, None)
    tl.store(out_ptr10 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp56, None)
    tl.store(out_ptr11 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp51, None)
