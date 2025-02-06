
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_rsub_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_pow_rsub_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp12 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp13 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp2 = triton_helpers.minimum(tmp0, tmp1)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp2 - tmp5
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp11 - tmp14
    tmp16 = triton_helpers.maximum(tmp15, tmp7)
    tmp17 = tmp8 * tmp16
    tmp18 = tmp0 - tmp3
    tmp19 = tmp9 - tmp12
    tmp20 = tmp18 * tmp19
    tmp21 = tmp1 - tmp4
    tmp22 = tmp10 - tmp13
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = tmp24 - tmp17
    tmp26 = tmp4 + tmp1
    tmp27 = tmp3 + tmp0
    tmp28 = tmp26 - tmp27
    tmp29 = tmp28 * tmp28
    tmp30 = 0.25
    tmp31 = tmp29 * tmp30
    tmp32 = tmp13 + tmp10
    tmp33 = tmp12 + tmp9
    tmp34 = tmp32 - tmp33
    tmp35 = tmp34 * tmp34
    tmp36 = tmp35 * tmp30
    tmp37 = tmp31 + tmp36
    tmp38 = triton_helpers.maximum(tmp0, tmp1)
    tmp39 = triton_helpers.minimum(tmp3, tmp4)
    tmp40 = tmp38 - tmp39
    tmp41 = triton_helpers.maximum(tmp40, tmp7)
    tmp42 = tmp41 * tmp41
    tmp43 = triton_helpers.maximum(tmp9, tmp10)
    tmp44 = triton_helpers.minimum(tmp12, tmp13)
    tmp45 = tmp43 - tmp44
    tmp46 = triton_helpers.maximum(tmp45, tmp7)
    tmp47 = tmp46 * tmp46
    tmp48 = tmp42 + tmp47
    tmp49 = 1e-06
    tmp50 = tmp48 + tmp49
    tmp51 = tmp37 / tmp50
    tmp52 = tmp25 + tmp49
    tmp53 = tmp17 / tmp52
    tmp54 = tmp53 - tmp51
    tmp55 = 1.0
    tmp56 = tmp55 - tmp54
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.sum(tmp57, 1)[:, None]
    tmp60 = 64.0
    tmp61 = tmp59 / tmp60
    tmp62 = tmp61 * tmp55
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp62, None)
