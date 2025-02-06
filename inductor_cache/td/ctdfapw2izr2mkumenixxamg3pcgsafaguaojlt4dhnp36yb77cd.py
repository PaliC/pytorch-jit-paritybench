
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_mean_mul_sub_xlogy_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_mean_mul_sub_xlogy_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp9 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp11 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp17 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp24 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp35 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp46 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp1 = libdevice.isnan(tmp0).to(tl.int1)
    tmp2 = 0.0
    tmp3 = tmp0 == tmp2
    tmp4 = tl_math.log(tmp0)
    tmp5 = tmp0 * tmp4
    tmp6 = tl.where(tmp3, tmp2, tmp5)
    tmp7 = float("nan")
    tmp8 = tl.where(tmp1, tmp7, tmp6)
    tmp10 = tl_math.exp(tmp9)
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tmp10 + tmp12
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp13 + tmp15
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp20 = tl_math.log(tmp19)
    tmp21 = tmp9 - tmp20
    tmp22 = tmp0 * tmp21
    tmp23 = tmp8 - tmp22
    tmp25 = libdevice.isnan(tmp24).to(tl.int1)
    tmp26 = tmp24 == tmp2
    tmp27 = tl_math.log(tmp24)
    tmp28 = tmp24 * tmp27
    tmp29 = tl.where(tmp26, tmp2, tmp28)
    tmp30 = tl.where(tmp25, tmp7, tmp29)
    tmp31 = tmp11 - tmp20
    tmp32 = tmp24 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tmp23 + tmp33
    tmp36 = libdevice.isnan(tmp35).to(tl.int1)
    tmp37 = tmp35 == tmp2
    tmp38 = tl_math.log(tmp35)
    tmp39 = tmp35 * tmp38
    tmp40 = tl.where(tmp37, tmp2, tmp39)
    tmp41 = tl.where(tmp36, tmp7, tmp40)
    tmp42 = tmp14 - tmp20
    tmp43 = tmp35 * tmp42
    tmp44 = tmp41 - tmp43
    tmp45 = tmp34 + tmp44
    tmp47 = libdevice.isnan(tmp46).to(tl.int1)
    tmp48 = tmp46 == tmp2
    tmp49 = tl_math.log(tmp46)
    tmp50 = tmp46 * tmp49
    tmp51 = tl.where(tmp48, tmp2, tmp50)
    tmp52 = tl.where(tmp47, tmp7, tmp51)
    tmp53 = tmp17 - tmp20
    tmp54 = tmp46 * tmp53
    tmp55 = tmp52 - tmp54
    tmp56 = tmp45 + tmp55
    tmp57 = 4.0
    tmp58 = tmp56 / tmp57
    tmp59 = 100.0
    tmp60 = tmp58 * tmp59
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
    tmp63 = tl.sum(tmp61, 1)[:, None]
    tmp64 = 64.0
    tmp65 = tmp63 / tmp64
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp67, None)
