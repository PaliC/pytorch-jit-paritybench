
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_exp_log_mean_neg_pow_rsub_sub_sum_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_exp_log_mean_neg_pow_rsub_sub_sum_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
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
    tmp9 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp16 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp23 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp32 = tl.load(in_ptr1 + (r0 + 64*((r1 % 4))), None)
    tmp33 = tl.load(in_ptr1 + (16 + r0 + 64*((r1 % 4))), None)
    tmp35 = tl.load(in_ptr1 + (32 + r0 + 64*((r1 % 4))), None)
    tmp37 = tl.load(in_ptr1 + (48 + r0 + 64*((r1 % 4))), None)
    tmp1 = tmp0 * tmp0
    tmp2 = 2.0
    tmp3 = tmp2 - tmp1
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = 0.25
    tmp7 = tmp5 * tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp10 = tmp9 * tmp9
    tmp11 = tmp2 - tmp10
    tmp12 = tmp11 * tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp8 + tmp14
    tmp17 = tmp16 * tmp16
    tmp18 = tmp2 - tmp17
    tmp19 = tmp18 * tmp4
    tmp20 = tmp19 * tmp6
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp15 + tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp2 - tmp24
    tmp26 = tmp25 * tmp4
    tmp27 = tmp26 * tmp6
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp22 + tmp28
    tmp30 = 1.2840254166877414
    tmp31 = tmp29 - tmp30
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38 * tmp6
    tmp40 = -tmp39
    tmp41 = tl_math.log(tmp31)
    tmp42 = tmp40 + tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.sum(tmp43, 1)[:, None]
    tmp46 = 128.0
    tmp47 = tmp45 / tmp46
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp47, None)
