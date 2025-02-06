
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_exp_mean_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addmm_exp_mean_mul_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (4*r1), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (4*r1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (1 + 4*r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (2 + 4*r1), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (3 + 4*r1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_out_ptr0 + (r2), None)
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 + tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp10 + tmp21
    tmp24 = tmp23 * tmp23
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 + tmp29
    tmp32 = tmp31 * tmp31
    tmp33 = tmp30 + tmp32
    tmp35 = tmp34 * tmp34
    tmp37 = tmp36 * tmp36
    tmp38 = tmp35 + tmp37
    tmp40 = tmp39 * tmp39
    tmp41 = tmp38 + tmp40
    tmp43 = tmp42 * tmp42
    tmp44 = tmp41 + tmp43
    tmp45 = tmp33 + tmp44
    tmp47 = tmp10 + tmp44
    tmp48 = tmp46 + tmp47
    tmp49 = -0.5
    tmp50 = tmp48 * tmp49
    tmp51 = tl_math.exp(tmp50)
    tmp52 = 0.0
    tmp53 = tmp51 + tmp52
    tmp54 = -0.02
    tmp55 = tmp48 * tmp54
    tmp56 = tl_math.exp(tmp55)
    tmp57 = tmp53 + tmp56
    tmp58 = -0.005
    tmp59 = tmp48 * tmp58
    tmp60 = tl_math.exp(tmp59)
    tmp61 = tmp57 + tmp60
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK, RBLOCK])
    tmp64 = tl.sum(tmp62, 1)[:, None]
    tl.store(out_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp22, None)
    tl.store(out_ptr1 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), tmp45, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp64, None)
