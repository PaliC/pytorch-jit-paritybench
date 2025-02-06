
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_div_linalg_vector_norm_mse_loss_pow_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_div_linalg_vector_norm_mse_loss_pow_sum_0(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp4 = tl.load(in_ptr0 + (16 + r1 + 64*x0), xmask, other=0.0)
    tmp9 = tl.load(in_ptr0 + (32 + r1 + 64*x0), xmask, other=0.0)
    tmp14 = tl.load(in_ptr0 + (48 + r1 + 64*x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1 + 64*x0), xmask, other=0.0)
    tmp28 = tl.load(in_ptr1 + (16 + r1 + 64*x0), xmask, other=0.0)
    tmp33 = tl.load(in_ptr1 + (32 + r1 + 64*x0), xmask, other=0.0)
    tmp38 = tl.load(in_ptr1 + (48 + r1 + 64*x0), xmask, other=0.0)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = tmp1 * tmp1
    tmp3 = tmp2 * tmp2
    tmp5 = tl_math.abs(tmp4)
    tmp6 = tmp5 * tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp10 = tl_math.abs(tmp9)
    tmp11 = tmp10 * tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp15 = tl_math.abs(tmp14)
    tmp16 = tmp15 * tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp25 = tl_math.abs(tmp24)
    tmp26 = tmp25 * tmp25
    tmp27 = tmp26 * tmp26
    tmp29 = tl_math.abs(tmp28)
    tmp30 = tmp29 * tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tmp27 + tmp31
    tmp34 = tl_math.abs(tmp33)
    tmp35 = tmp34 * tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tmp32 + tmp36
    tmp39 = tl_math.abs(tmp38)
    tmp40 = tmp39 * tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp37 + tmp41
    tmp43 = tmp42 * tmp42
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK, RBLOCK])
    tmp46 = tl.where(xmask, tmp44, 0)
    tmp47 = tl.sum(tmp46, 1)[:, None]
    tmp48 = libdevice.sqrt(tmp23)
    tmp49 = 1e-06
    tmp50 = tmp48 + tmp49
    tmp51 = tmp18 / tmp50
    tmp52 = libdevice.sqrt(tmp47)
    tmp53 = tmp52 + tmp49
    tmp54 = tmp42 / tmp53
    tmp55 = tmp51 - tmp54
    tl.store(out_ptr2 + (r1 + 16*x0), tmp55, xmask)
