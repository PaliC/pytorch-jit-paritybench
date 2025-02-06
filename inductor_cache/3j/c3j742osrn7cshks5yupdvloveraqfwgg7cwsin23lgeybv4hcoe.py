
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 8), 'tt.equal_to': (7,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_binary_cross_entropy_div_linalg_vector_norm_mean_mul_ones_like_pow_sum_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_binary_cross_entropy_div_linalg_vector_norm_mean_mul_ones_like_pow_sum_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
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
    r1 = (rindex % 16)
    r3 = rindex // 64
    tmp4 = tl.load(in_ptr0 + (r0), None)
    tmp17 = tl.load(in_ptr1 + (r0), None)
    tmp18 = tl.load(in_ptr1 + (r1 + 64*r3), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (16 + r1 + 64*r3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (32 + r1 + 64*r3), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (48 + r1 + 64*r3), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (r0), None)
    tmp41 = tl.load(in_out_ptr1 + (0))
    tmp42 = tl.broadcast_to(tmp41, [1])
    tmp0 = 1.0
    tmp1 = 4.0
    tmp2 = tmp0 / tmp1
    tmp3 = tmp2 - tmp0
    tmp5 = -tmp4
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = -100.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp3 * tmp8
    tmp10 = tl_math.log(tmp4)
    tmp11 = triton_helpers.maximum(tmp10, tmp7)
    tmp12 = tmp2 * tmp11
    tmp13 = tmp9 - tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp17 / tmp24
    tmp26 = tmp25 - tmp0
    tmp28 = -tmp27
    tmp29 = libdevice.log1p(tmp28)
    tmp30 = triton_helpers.maximum(tmp29, tmp7)
    tmp31 = tmp26 * tmp30
    tmp32 = tl_math.log(tmp27)
    tmp33 = triton_helpers.maximum(tmp32, tmp7)
    tmp34 = tmp25 * tmp33
    tmp35 = tmp31 - tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp39 = 256.0
    tmp40 = tmp38 / tmp39
    tmp43 = 16.0
    tmp44 = tmp42 / tmp43
    tmp45 = tmp16 / tmp39
    tmp46 = tmp44 * tmp1
    tmp47 = tmp40 + tmp46
    tmp48 = tmp45 * tmp1
    tmp49 = tmp47 + tmp48
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp40, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([1], 0, tl.int32)), tmp44, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (tl.full([1], 0, tl.int32)), tmp45, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp49, None)
