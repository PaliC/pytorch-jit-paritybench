
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': (16,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_pow_sub_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mean_pow_sub_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp4 = tl.load(in_ptr1 + (r0), None)
    tmp8 = tl.load(in_ptr2 + (r0), None)
    tmp12 = tl.load(in_ptr3 + (r0), None)
    tmp16 = tl.load(in_out_ptr0 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, 1])
    tmp20 = tl.load(in_ptr4 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, 1])
    tmp24 = tl.load(in_ptr5 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, 1])
    tmp29 = tl.load(in_ptr6 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 1])
    tmp33 = tl.load(in_ptr7 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 1])
    tmp38 = tl.load(in_ptr8 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, 1])
    tmp42 = tl.load(in_ptr9 + (0))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, 1])
    tmp46 = tl.load(in_ptr10 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, 1])
    tmp50 = tl.load(in_ptr11 + (0))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, 1])
    tmp55 = tl.load(in_ptr12 + (0))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, 1])
    tmp59 = tl.load(in_ptr13 + (0))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, 1])
    tmp63 = tl.load(in_ptr14 + (0))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp18 = 524288.0
    tmp19 = tmp17 / tmp18
    tmp22 = tmp21 / tmp18
    tmp23 = tmp19 + tmp22
    tmp26 = 262144.0
    tmp27 = tmp25 / tmp26
    tmp28 = tmp23 + tmp27
    tmp31 = tmp30 / tmp26
    tmp32 = tmp28 + tmp31
    tmp35 = 131072.0
    tmp36 = tmp34 / tmp35
    tmp37 = tmp32 + tmp36
    tmp40 = tmp39 / tmp35
    tmp41 = tmp37 + tmp40
    tmp44 = tmp43 / tmp35
    tmp45 = tmp41 + tmp44
    tmp48 = tmp47 / tmp35
    tmp49 = tmp45 + tmp48
    tmp52 = 65536.0
    tmp53 = tmp51 / tmp52
    tmp54 = tmp49 + tmp53
    tmp57 = tmp56 / tmp52
    tmp58 = tmp54 + tmp57
    tmp61 = tmp60 / tmp52
    tmp62 = tmp58 + tmp61
    tmp65 = tmp64 / tmp52
    tmp66 = tmp62 + tmp65
    tmp67 = 16384.0
    tmp68 = tmp3 / tmp67
    tmp69 = tmp66 + tmp68
    tmp70 = tmp7 / tmp67
    tmp71 = tmp69 + tmp70
    tmp72 = tmp11 / tmp67
    tmp73 = tmp71 + tmp72
    tmp74 = tmp15 / tmp67
    tmp75 = tmp73 + tmp74
    tmp76 = 0.0625
    tmp77 = tmp75 * tmp76
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp77, None)
