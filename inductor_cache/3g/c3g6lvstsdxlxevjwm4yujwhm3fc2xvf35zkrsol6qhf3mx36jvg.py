
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_smooth_l1_loss_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_smooth_l1_loss_0(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex // 16
    r0 = (rindex % 4)
    r1 = ((rindex // 4) % 4)
    r3 = rindex
    tmp0 = r2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + 16*r1 + 64*(r2), [XBLOCK, RBLOCK])), rmask & tmp4, other=0.0)
    tmp6 = tl.load(in_ptr0 + (tl.broadcast_to(4 + r0 + 16*r1 + 64*(r2), [XBLOCK, RBLOCK])), rmask & tmp4, other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 8, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr0 + (tl.broadcast_to(4 + r0 + 16*r1 + 64*((-4) + r2), [XBLOCK, RBLOCK])), rmask & tmp13, other=0.0)
    tmp15 = tl.load(in_ptr0 + (tl.broadcast_to(8 + r0 + 16*r1 + 64*((-4) + r2), [XBLOCK, RBLOCK])), rmask & tmp13, other=0.0)
    tmp16 = tmp14 - tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tmp0 >= tmp11
    tmp20 = tl.full([1, 1], 12, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (tl.broadcast_to(8 + r0 + 16*r1 + 64*((-8) + r2), [XBLOCK, RBLOCK])), rmask & tmp19, other=0.0)
    tmp23 = tl.load(in_ptr0 + (tl.broadcast_to(12 + r0 + 16*r1 + 64*((-8) + r2), [XBLOCK, RBLOCK])), rmask & tmp19, other=0.0)
    tmp24 = tmp22 - tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp19, tmp24, tmp25)
    tmp27 = tl.where(tmp13, tmp18, tmp26)
    tmp28 = tl.where(tmp4, tmp9, tmp27)
    tmp29 = tl_math.abs(tmp28)
    tmp30 = 1.0
    tmp31 = tmp29 < tmp30
    tmp32 = tmp29 * tmp29
    tmp33 = 0.5
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34 * tmp30
    tmp36 = tmp29 - tmp33
    tmp37 = tl.where(tmp31, tmp35, tmp36)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp40 = tl.where(rmask, tmp38, 0)
    tmp41 = tl.sum(tmp40, 1)[:, None]
    tmp42 = 192.0
    tmp43 = tmp41 / tmp42
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp43, None)
