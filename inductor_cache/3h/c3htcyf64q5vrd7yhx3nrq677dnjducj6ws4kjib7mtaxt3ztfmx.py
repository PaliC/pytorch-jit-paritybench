
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_linalg_vector_norm_mean_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_linalg_vector_norm_mean_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (768 + r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr1 + (r0 + 64*r1), None)
    tmp4 = tl.load(in_ptr0 + (784 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr1 + (16 + r0 + 64*r1), None)
    tmp9 = tl.load(in_ptr0 + (800 + r0 + 64*r1), None)
    tmp10 = tl.load(in_ptr1 + (32 + r0 + 64*r1), None)
    tmp14 = tl.load(in_ptr0 + (816 + r0 + 64*r1), None)
    tmp15 = tl.load(in_ptr1 + (48 + r0 + 64*r1), None)
    tmp23 = tl.load(in_out_ptr0 + (0))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, 1])
    tmp29 = tl.load(in_ptr2 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp33 = tl.load(in_ptr3 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp6 = tmp4 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp3 + tmp7
    tmp11 = tmp9 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp8 + tmp12
    tmp16 = tmp14 - tmp15
    tmp17 = tmp16 * tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = libdevice.sqrt(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tmp25 = 64.0
    tmp26 = tmp24 / tmp25
    tmp27 = 0.0
    tmp28 = tmp26 + tmp27
    tmp31 = tmp30 / tmp25
    tmp32 = tmp28 + tmp31
    tmp35 = tmp34 / tmp25
    tmp36 = tmp32 + tmp35
    tmp37 = tmp22 / tmp25
    tmp38 = tmp36 + tmp37
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp38, None)
