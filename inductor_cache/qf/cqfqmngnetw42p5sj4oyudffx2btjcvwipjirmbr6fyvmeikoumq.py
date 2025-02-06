
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_nll_loss2d_forward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_nll_loss2d_forward_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp12 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp17 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp20 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1, 1], 255, tl.int64)
    tmp3 = tmp1 != tmp2
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp6 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert((0 <= tmp9) & (tmp9 < 4), "index out of bounds: 0 <= tmp9 < 4")
    tmp11 = tl.load(in_ptr1 + (r0 + 16*tmp9 + 64*r1), None)
    tmp13 = tl_math.exp(tmp12)
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp13 + tmp15
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp19 + tmp21
    tmp23 = tl_math.log(tmp22)
    tmp24 = tmp11 - tmp23
    tmp25 = -tmp24
    tmp26 = 0.0
    tmp27 = tl.where(tmp3, tmp25, tmp26)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.sum(tmp28, 1)[:, None]
    tmp31 = tmp3.to(tl.int64)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.sum(tmp32, 1)[:, None]
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp30 / tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp36, None)
