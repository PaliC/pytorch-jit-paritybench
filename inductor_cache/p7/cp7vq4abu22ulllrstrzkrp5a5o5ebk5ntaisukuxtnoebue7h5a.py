
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_eq_log_sigmoid_forward_mean_mul_neg_pow_rsub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_eq_log_sigmoid_forward_mean_mul_neg_pow_rsub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp20 = tl.load(in_ptr1 + (r0), None)
    tmp1 = 1.0
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = 0.0
    tmp8 = tmp0 == tmp7
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp6 + tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = 1e-07
    tmp17 = tmp15 + tmp16
    tmp18 = tmp13 / tmp17
    tmp19 = tmp18 * tmp0
    tmp21 = triton_helpers.minimum(tmp7, tmp20)
    tmp22 = tl_math.abs(tmp20)
    tmp23 = -tmp22
    tmp24 = tl_math.exp(tmp23)
    tmp25 = libdevice.log1p(tmp24)
    tmp26 = tmp21 - tmp25
    tmp27 = tmp19 * tmp26
    tmp28 = tmp1 - tmp18
    tmp29 = tmp1 - tmp0
    tmp30 = tmp28 * tmp29
    tmp31 = -tmp20
    tmp32 = triton_helpers.minimum(tmp7, tmp31)
    tmp33 = tl_math.abs(tmp31)
    tmp34 = -tmp33
    tmp35 = tl_math.exp(tmp34)
    tmp36 = libdevice.log1p(tmp35)
    tmp37 = tmp32 - tmp36
    tmp38 = tmp30 * tmp37
    tmp39 = tmp27 + tmp38
    tmp40 = -tmp39
    tmp41 = tl.broadcast_to(tmp40, [RBLOCK])
    tmp43 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp44 = 256.0
    tmp45 = tmp43 / tmp44
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp45, None)
