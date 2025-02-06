
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_index_put_lift_fresh_linalg_vector_norm_mean_pow_rsub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_index_put_lift_fresh_linalg_vector_norm_mean_pow_rsub_0(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 4)
    r1 = rindex // 4
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr0 + (4 + r0 + 64*r1), None)
    tmp11 = tl.load(in_ptr0 + (20 + r0 + 64*r1), None)
    tmp13 = tl.load(in_ptr0 + (36 + r0 + 64*r1), None)
    tmp15 = tl.load(in_ptr0 + (52 + r0 + 64*r1), None)
    tmp20 = tl.load(in_ptr0 + (8 + r0 + 64*r1), None)
    tmp21 = tl.load(in_ptr0 + (24 + r0 + 64*r1), None)
    tmp23 = tl.load(in_ptr0 + (40 + r0 + 64*r1), None)
    tmp25 = tl.load(in_ptr0 + (56 + r0 + 64*r1), None)
    tmp30 = tl.load(in_ptr0 + (12 + r0 + 64*r1), None)
    tmp31 = tl.load(in_ptr0 + (28 + r0 + 64*r1), None)
    tmp33 = tl.load(in_ptr0 + (44 + r0 + 64*r1), None)
    tmp35 = tl.load(in_ptr0 + (60 + r0 + 64*r1), None)
    tmp45 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp46 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp48 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp50 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp54 = tl.load(in_ptr1 + (4 + r0 + 64*r1), None)
    tmp55 = tl.load(in_ptr1 + (20 + r0 + 64*r1), None)
    tmp57 = tl.load(in_ptr1 + (36 + r0 + 64*r1), None)
    tmp59 = tl.load(in_ptr1 + (52 + r0 + 64*r1), None)
    tmp64 = tl.load(in_ptr1 + (8 + r0 + 64*r1), None)
    tmp65 = tl.load(in_ptr1 + (24 + r0 + 64*r1), None)
    tmp67 = tl.load(in_ptr1 + (40 + r0 + 64*r1), None)
    tmp69 = tl.load(in_ptr1 + (56 + r0 + 64*r1), None)
    tmp74 = tl.load(in_ptr1 + (12 + r0 + 64*r1), None)
    tmp75 = tl.load(in_ptr1 + (28 + r0 + 64*r1), None)
    tmp77 = tl.load(in_ptr1 + (44 + r0 + 64*r1), None)
    tmp79 = tl.load(in_ptr1 + (60 + r0 + 64*r1), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 * tmp8
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 / tmp7
    tmp18 = tmp17 * tmp17
    tmp19 = tmp9 + tmp18
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 / tmp7
    tmp28 = tmp27 * tmp27
    tmp29 = tmp19 + tmp28
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36 / tmp7
    tmp38 = tmp37 * tmp37
    tmp39 = tmp29 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tmp7 - tmp40
    tmp42 = 0.0
    tmp43 = tmp41 < tmp42
    tmp44 = tl.where(tmp43, tmp42, tmp41)
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 + tmp50
    tmp52 = tmp51 / tmp7
    tmp53 = tmp52 * tmp52
    tmp56 = tmp54 + tmp55
    tmp58 = tmp56 + tmp57
    tmp60 = tmp58 + tmp59
    tmp61 = tmp60 / tmp7
    tmp62 = tmp61 * tmp61
    tmp63 = tmp53 + tmp62
    tmp66 = tmp64 + tmp65
    tmp68 = tmp66 + tmp67
    tmp70 = tmp68 + tmp69
    tmp71 = tmp70 / tmp7
    tmp72 = tmp71 * tmp71
    tmp73 = tmp63 + tmp72
    tmp76 = tmp74 + tmp75
    tmp78 = tmp76 + tmp77
    tmp80 = tmp78 + tmp79
    tmp81 = tmp80 / tmp7
    tmp82 = tmp81 * tmp81
    tmp83 = tmp73 + tmp82
    tmp84 = libdevice.sqrt(tmp83)
    tmp85 = tmp44 + tmp84
    tmp86 = tmp85 * tmp85
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK, RBLOCK])
    tmp89 = tl.sum(tmp87, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp89, None)
