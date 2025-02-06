
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mul_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    r0 = (rindex % 16)
    tmp0 = tl.load(in_ptr0 + (r2), None)
    tmp2 = tl.load(in_ptr0 + (r0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (16 + r0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (32 + r0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (48 + r0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (r2), None)
    tmp20 = tl.load(in_ptr2 + (r2), None)
    tmp22 = tl.load(in_ptr2 + (r0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (16 + r0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (32 + r0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (48 + r0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr3 + (r2), None)
    tmp40 = tl.load(in_ptr4 + (r2), None)
    tmp42 = tl.load(in_ptr4 + (r0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (16 + r0), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (32 + r0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr4 + (48 + r0), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (r2), None)
    tmp60 = tl.load(in_ptr6 + (r2), None)
    tmp62 = tl.load(in_ptr6 + (r0), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr6 + (16 + r0), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr6 + (32 + r0), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr6 + (48 + r0), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr7 + (r2), None)
    tmp1 = tmp0 / tmp0
    tmp3 = tmp2 / tmp2
    tmp5 = tmp4 / tmp4
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7 / tmp7
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10 / tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = tmp1 / tmp12
    tmp14 = tmp13 / tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp21 = tmp20 / tmp20
    tmp23 = tmp22 / tmp22
    tmp25 = tmp24 / tmp24
    tmp26 = tmp23 + tmp25
    tmp28 = tmp27 / tmp27
    tmp29 = tmp26 + tmp28
    tmp31 = tmp30 / tmp30
    tmp32 = tmp29 + tmp31
    tmp33 = tmp21 / tmp32
    tmp34 = tmp33 / tmp33
    tmp36 = tmp34 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
    tmp39 = tl.sum(tmp37, 1)[:, None]
    tmp41 = tmp40 / tmp40
    tmp43 = tmp42 / tmp42
    tmp45 = tmp44 / tmp44
    tmp46 = tmp43 + tmp45
    tmp48 = tmp47 / tmp47
    tmp49 = tmp46 + tmp48
    tmp51 = tmp50 / tmp50
    tmp52 = tmp49 + tmp51
    tmp53 = tmp41 / tmp52
    tmp54 = tmp53 / tmp53
    tmp56 = tmp54 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.sum(tmp57, 1)[:, None]
    tmp61 = tmp60 / tmp60
    tmp63 = tmp62 / tmp62
    tmp65 = tmp64 / tmp64
    tmp66 = tmp63 + tmp65
    tmp68 = tmp67 / tmp67
    tmp69 = tmp66 + tmp68
    tmp71 = tmp70 / tmp70
    tmp72 = tmp69 + tmp71
    tmp73 = tmp61 / tmp72
    tmp74 = tmp73 / tmp73
    tmp76 = tmp74 * tmp75
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp79 = tl.sum(tmp77, 1)[:, None]
    tmp80 = 0.0
    tmp81 = tmp19 + tmp80
    tmp82 = tmp81 + tmp39
    tmp83 = tmp82 + tmp59
    tmp84 = tmp83 + tmp79
    tmp85 = 0.001
    tmp86 = tmp84 * tmp85
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp86, None)
