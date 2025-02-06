
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_abs_add_binary_cross_entropy_with_logits_gt_le_mean_mul_pow_sigmoid_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_abs_add_binary_cross_entropy_with_logits_gt_le_mean_mul_pow_sigmoid_sub_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr1 + (r0), None)
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.minimum(tmp5, tmp3)
    tmp7 = tl_math.abs(tmp3)
    tmp8 = -tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = libdevice.log1p(tmp9)
    tmp11 = tmp6 - tmp10
    tmp12 = tmp4 - tmp11
    tmp13 = tmp0 > tmp5
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp0 * tmp14
    tmp16 = tl.sigmoid(tmp3)
    tmp17 = tmp16 - tmp0
    tmp18 = tl_math.abs(tmp17)
    tmp19 = tmp18 * tmp18
    tmp20 = 0.75
    tmp21 = tmp19 * tmp20
    tmp22 = tmp0 <= tmp5
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 * tmp23
    tmp25 = tmp15 + tmp24
    tmp26 = tmp12 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = 256.0
    tmp31 = tmp29 / tmp30
    tmp32 = tmp31 * tmp1
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp32, None)
