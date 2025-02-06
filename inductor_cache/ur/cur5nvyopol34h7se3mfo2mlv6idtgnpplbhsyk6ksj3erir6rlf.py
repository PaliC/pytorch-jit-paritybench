
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mse_loss_mul_pow_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mse_loss_mul_pow_1(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0), None)
    tmp17 = tl.load(in_ptr0 + (r0), None)
    tmp1 = 0.002683701023220095
    tmp2 = tmp0 * tmp1
    tmp3 = 0.00033546262790251185
    tmp4 = tmp3 + tmp2
    tmp5 = tmp0 * tmp0
    tmp6 = 0.01073480409288038
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tmp5 * tmp0
    tmp10 = 0.02862614424768101
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = 0.05725228849536202
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp18 = tmp17 * tmp1
    tmp19 = tmp3 + tmp18
    tmp20 = tmp17 * tmp17
    tmp21 = tmp20 * tmp6
    tmp22 = tmp19 + tmp21
    tmp23 = tmp20 * tmp17
    tmp24 = tmp23 * tmp10
    tmp25 = tmp22 + tmp24
    tmp26 = tmp20 * tmp20
    tmp27 = tmp26 * tmp14
    tmp28 = tmp25 + tmp27
    tmp29 = tmp16 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.sum(tmp31, 1)[:, None]
    tmp34 = 16.0
    tmp35 = tmp33 / tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp35, None)
