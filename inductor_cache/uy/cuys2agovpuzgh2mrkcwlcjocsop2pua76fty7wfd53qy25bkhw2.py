
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_neg_softplus_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_mul_neg_softplus_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp7 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp8 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp11 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp12 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp16 = tl.load(in_ptr2 + (r0 + 64*r1), None)
    tmp18 = tl.load(in_ptr2 + (16 + r0 + 64*r1), None)
    tmp21 = tl.load(in_ptr2 + (32 + r0 + 64*r1), None)
    tmp24 = tl.load(in_ptr2 + (48 + r0 + 64*r1), None)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = -tmp14
    tmp17 = tmp0 * tmp16
    tmp19 = tmp3 * tmp18
    tmp20 = tmp17 + tmp19
    tmp22 = tmp7 * tmp21
    tmp23 = tmp20 + tmp22
    tmp25 = tmp11 * tmp24
    tmp26 = tmp23 + tmp25
    tmp27 = 20.0
    tmp28 = tmp15 > tmp27
    tmp29 = tl_math.exp(tmp15)
    tmp30 = libdevice.log1p(tmp29)
    tmp31 = tl.where(tmp28, tmp15, tmp30)
    tmp32 = tmp26 > tmp27
    tmp33 = tl_math.exp(tmp26)
    tmp34 = libdevice.log1p(tmp33)
    tmp35 = tl.where(tmp32, tmp26, tmp34)
    tmp36 = tmp31 + tmp35
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
    tmp39 = tl.sum(tmp37, 1)[:, None]
    tmp40 = 64.0
    tmp41 = tmp39 / tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp41, None)
