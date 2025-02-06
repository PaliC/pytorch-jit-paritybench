
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_div_mse_loss_mul_repeat_rsub_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_add_div_mse_loss_mul_repeat_rsub_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    r4 = rindex
    r1 = ((rindex // 4) % 4)
    r3 = rindex // 64
    r0 = (rindex % 4)
    r6 = rindex // 4
    tmp0 = tl.load(in_ptr0 + (r4), None)
    tmp3 = tl.load(in_ptr1 + (r1 + 4*r3), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (r0 + 4*r3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (r6), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (r6), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (r4), None)
    tmp19 = tl.load(in_ptr1 + (4*(r4 // 64) + (((r4 // 4) % 4))), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (4*(r4 // 64) + ((r4 % 4))), None)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tmp1 - tmp5
    tmp8 = -10000.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tmp10 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp15 = tmp13 / tmp14
    tmp16 = tmp15 * tmp5
    tmp18 = tmp17 * tmp1
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 * tmp21
    tmp23 = tmp2 * tmp21
    tmp24 = tmp22 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp29, 0))
    tl.store(out_ptr0 + (tl.broadcast_to(r4, [RBLOCK])), tmp16, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp28, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp31, None)
