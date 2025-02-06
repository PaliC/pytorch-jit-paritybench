
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_atan_div_mean_mul_pow_rsub_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_atan_div_mean_mul_pow_rsub_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp11 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp13 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = 0.25
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 + tmp11
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp17 * tmp8
    tmp19 = tmp9 + tmp18
    tmp20 = triton_helpers.maximum(tmp4, tmp1)
    tmp21 = triton_helpers.minimum(tmp3, tmp0)
    tmp22 = tmp20 - tmp21
    tmp23 = 0.0
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp25 = tmp24 * tmp24
    tmp26 = triton_helpers.maximum(tmp14, tmp11)
    tmp27 = triton_helpers.minimum(tmp13, tmp10)
    tmp28 = tmp26 - tmp27
    tmp29 = triton_helpers.maximum(tmp28, tmp23)
    tmp30 = tmp29 * tmp29
    tmp31 = tmp25 + tmp30
    tmp32 = 1e-06
    tmp33 = tmp31 + tmp32
    tmp34 = tmp19 / tmp33
    tmp35 = tmp1 - tmp0
    tmp36 = tmp11 - tmp10
    tmp37 = tmp36 + tmp32
    tmp38 = tmp35 / tmp37
    tmp39 = libdevice.atan(tmp38)
    tmp40 = tmp4 - tmp3
    tmp41 = tmp14 - tmp13
    tmp42 = tmp41 + tmp32
    tmp43 = tmp40 / tmp42
    tmp44 = libdevice.atan(tmp43)
    tmp45 = tmp39 - tmp44
    tmp46 = tmp45 * tmp45
    tmp47 = 0.4052847345693511
    tmp48 = tmp46 * tmp47
    tmp49 = triton_helpers.minimum(tmp4, tmp1)
    tmp50 = triton_helpers.maximum(tmp3, tmp0)
    tmp51 = tmp49 - tmp50
    tmp52 = triton_helpers.maximum(tmp51, tmp23)
    tmp53 = triton_helpers.minimum(tmp14, tmp11)
    tmp54 = triton_helpers.maximum(tmp13, tmp10)
    tmp55 = tmp53 - tmp54
    tmp56 = triton_helpers.maximum(tmp55, tmp23)
    tmp57 = tmp52 * tmp56
    tmp58 = tmp40 * tmp41
    tmp59 = tmp35 * tmp36
    tmp60 = tmp58 + tmp59
    tmp61 = tmp60 - tmp57
    tmp62 = tmp61 + tmp32
    tmp63 = tmp57 / tmp62
    tmp64 = tmp48 * tmp48
    tmp65 = 1.0
    tmp66 = tmp65 - tmp63
    tmp67 = tmp66 + tmp48
    tmp68 = tmp64 / tmp67
    tmp69 = tmp34 + tmp68
    tmp70 = tmp63 - tmp69
    tmp71 = tmp65 - tmp70
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK, RBLOCK])
    tmp74 = tl.sum(tmp72, 1)[:, None]
    tmp75 = 64.0
    tmp76 = tmp74 / tmp75
    tmp77 = tmp76 * tmp65
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp77, None)
