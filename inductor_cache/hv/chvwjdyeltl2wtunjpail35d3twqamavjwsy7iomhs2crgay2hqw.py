
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mse_loss_mul_pow_relu_rsub_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mse_loss_mul_pow_relu_rsub_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.load(in_ptr1 + (r0), None)
    tmp21 = tl.load(in_ptr2 + (r0), None)
    tmp22 = tl.load(in_ptr3 + (r0), None)
    tmp2 = 0.9
    tmp3 = tmp2 - tmp1
    tmp4 = tl.full([1, 1], 0, tl.int32)
    tmp5 = triton_helpers.maximum(tmp4, tmp3)
    tmp6 = tmp5 * tmp5
    tmp7 = tmp0 * tmp6
    tmp8 = 1.0
    tmp9 = tmp8 - tmp0
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp12 = 0.1
    tmp13 = tmp1 - tmp12
    tmp14 = triton_helpers.maximum(tmp4, tmp13)
    tmp15 = tmp14 * tmp14
    tmp16 = tmp11 * tmp15
    tmp17 = tmp7 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tmp23 = tmp21 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp28 = 0.0005
    tmp29 = tmp27 * tmp28
    tmp30 = tmp20 + tmp29
    tmp31 = 0.25
    tmp32 = tmp30 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp32, None)
