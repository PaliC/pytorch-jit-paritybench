
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_index_put_lift_fresh_mean_mul_pow_rsub_sqrt_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 7, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_div_index_put_lift_fresh_mean_mul_pow_rsub_sqrt_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (16 + r1 + 64*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (32 + r1 + 64*x0), xmask, other=0.0)
    tmp5 = tl.load(in_ptr0 + (48 + r1 + 64*x0), xmask, other=0.0)
    tmp16 = tl.load(in_ptr1 + (r1 + 64*x0), xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (16 + r1 + 64*x0), xmask, other=0.0)
    tmp19 = tl.load(in_ptr1 + (32 + r1 + 64*x0), xmask, other=0.0)
    tmp21 = tl.load(in_ptr1 + (48 + r1 + 64*x0), xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = 16.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp8 - tmp14
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tmp22 / tmp7
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tmp27 / tmp13
    tmp29 = tmp23 - tmp28
    tmp30 = tmp15 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(xmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tmp15 * tmp15
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    tmp38 = tl.where(xmask, tmp36, 0)
    tmp39 = tl.sum(tmp38, 1)[:, None]
    tmp40 = tmp29 * tmp29
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
    tmp43 = tl.where(xmask, tmp41, 0)
    tmp44 = tl.sum(tmp43, 1)[:, None]
    tmp45 = 0.01
    tmp46 = tmp39 < tmp45
    tmp47 = tmp44 < tmp45
    tmp48 = tmp46 | tmp47
    tmp49 = tmp39 * tmp44
    tmp50 = 1e-06
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.sqrt(tmp51)
    tmp53 = tmp52 + tmp50
    tmp54 = tmp34 / tmp53
    tmp55 = 1.0
    tmp56 = tl.where(tmp48, tmp55, tmp54)
    tmp57 = -1.0
    tmp58 = triton_helpers.maximum(tmp56, tmp57)
    tmp59 = triton_helpers.minimum(tmp58, tmp55)
    tmp60 = tmp55 - tmp59
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp60, xmask)
