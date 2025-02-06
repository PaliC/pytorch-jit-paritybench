
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_add_mul_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_add_mul_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 12
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = (rindex % 3)
    r3 = rindex // 3
    x0 = (xindex % 4)
    x1 = xindex // 4
    x5 = xindex
    r4 = rindex
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (4 + x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (4 + x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (x0 + 4*r2 + 16*r3 + 64*x1), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr0 + (16 + x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr0 + (x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr1 + (16 + x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr2 + (16 + x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr1 + (x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (x0 + 4*r4 + 64*x1), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp0 * tmp5
    tmp9 = tmp7 - tmp8
    tmp10 = tmp1 * tmp9
    tmp11 = tmp6 - tmp10
    tmp12 = tl_math.abs(tmp11)
    tmp13 = tmp2 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp20 = tmp18 * tmp19
    tmp23 = tmp21 - tmp22
    tmp24 = tmp18 * tmp23
    tmp27 = tmp25 - tmp26
    tmp28 = tmp19 * tmp27
    tmp29 = tmp24 - tmp28
    tmp30 = tl_math.abs(tmp29)
    tmp31 = tmp20 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp36 = tmp17 + tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5), tmp36, xmask)
