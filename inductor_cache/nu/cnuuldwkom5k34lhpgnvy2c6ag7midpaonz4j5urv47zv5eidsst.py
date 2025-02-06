
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21), 'tt.equal_to': (20,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_sqrt_sub_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 8, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_pow_sqrt_sub_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 32
    RBLOCK: tl.constexpr = 32
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
    tmp16 = tl.load(in_ptr4 + (r0), None)
    tmp20 = tl.load(in_ptr5 + (r0), None)
    tmp24 = tl.load(in_ptr6 + (r0), None)
    tmp28 = tl.load(in_ptr7 + (r0), None)
    tmp32 = tl.load(in_out_ptr0 + (0))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, 1])
    tmp41 = tl.load(in_ptr8 + (0))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, 1])
    tmp47 = tl.load(in_ptr9 + (0))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, 1])
    tmp53 = tl.load(in_ptr10 + (0))
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, 1])
    tmp59 = tl.load(in_ptr11 + (0))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, 1])
    tmp67 = tl.load(in_ptr12 + (0))
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK, 1])
    tmp73 = tl.load(in_ptr13 + (0))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, 1])
    tmp79 = tl.load(in_ptr14 + (0))
    tmp80 = tl.broadcast_to(tmp79, [XBLOCK, 1])
    tmp85 = tl.load(in_ptr15 + (0))
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK, 1])
    tmp93 = tl.load(in_ptr16 + (0))
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK, 1])
    tmp99 = tl.load(in_ptr17 + (0))
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK, 1])
    tmp105 = tl.load(in_ptr18 + (0))
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK, 1])
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None]
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.sum(tmp25, 1)[:, None]
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp34 = 4096.0
    tmp35 = tmp33 / tmp34
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = 0.03125
    tmp38 = tmp36 * tmp37
    tmp39 = 0.0
    tmp40 = tmp38 + tmp39
    tmp43 = tmp42 / tmp34
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tmp44 * tmp37
    tmp46 = tmp40 + tmp45
    tmp49 = tmp48 / tmp34
    tmp50 = libdevice.sqrt(tmp49)
    tmp51 = tmp50 * tmp37
    tmp52 = tmp46 + tmp51
    tmp55 = tmp54 / tmp34
    tmp56 = libdevice.sqrt(tmp55)
    tmp57 = tmp56 * tmp37
    tmp58 = tmp52 + tmp57
    tmp61 = 16384.0
    tmp62 = tmp60 / tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = 0.0625
    tmp65 = tmp63 * tmp64
    tmp66 = tmp58 + tmp65
    tmp69 = tmp68 / tmp61
    tmp70 = libdevice.sqrt(tmp69)
    tmp71 = tmp70 * tmp64
    tmp72 = tmp66 + tmp71
    tmp75 = tmp74 / tmp61
    tmp76 = libdevice.sqrt(tmp75)
    tmp77 = tmp76 * tmp64
    tmp78 = tmp72 + tmp77
    tmp81 = tmp80 / tmp61
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tmp82 * tmp64
    tmp84 = tmp78 + tmp83
    tmp87 = 65536.0
    tmp88 = tmp86 / tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = 0.125
    tmp91 = tmp89 * tmp90
    tmp92 = tmp84 + tmp91
    tmp95 = tmp94 / tmp87
    tmp96 = libdevice.sqrt(tmp95)
    tmp97 = tmp96 * tmp90
    tmp98 = tmp92 + tmp97
    tmp101 = tmp100 / tmp87
    tmp102 = libdevice.sqrt(tmp101)
    tmp103 = tmp102 * tmp90
    tmp104 = tmp98 + tmp103
    tmp107 = tmp106 / tmp87
    tmp108 = libdevice.sqrt(tmp107)
    tmp109 = tmp108 * tmp90
    tmp110 = tmp104 + tmp109
    tmp111 = 262144.0
    tmp112 = tmp3 / tmp111
    tmp113 = libdevice.sqrt(tmp112)
    tmp114 = 0.25
    tmp115 = tmp113 * tmp114
    tmp116 = tmp110 + tmp115
    tmp117 = tmp7 / tmp111
    tmp118 = libdevice.sqrt(tmp117)
    tmp119 = tmp118 * tmp114
    tmp120 = tmp116 + tmp119
    tmp121 = tmp11 / tmp111
    tmp122 = libdevice.sqrt(tmp121)
    tmp123 = tmp122 * tmp114
    tmp124 = tmp120 + tmp123
    tmp125 = tmp15 / tmp111
    tmp126 = libdevice.sqrt(tmp125)
    tmp127 = tmp126 * tmp114
    tmp128 = tmp124 + tmp127
    tmp129 = tmp19 / tmp111
    tmp130 = libdevice.sqrt(tmp129)
    tmp131 = 1.0
    tmp132 = tmp130 * tmp131
    tmp133 = tmp128 + tmp132
    tmp134 = tmp23 / tmp111
    tmp135 = libdevice.sqrt(tmp134)
    tmp136 = tmp135 * tmp131
    tmp137 = tmp133 + tmp136
    tmp138 = tmp31 / tmp111
    tmp139 = libdevice.sqrt(tmp138)
    tmp140 = tmp139 * tmp131
    tmp141 = tmp137 + tmp140
    tmp142 = tmp27 / tmp111
    tmp143 = libdevice.sqrt(tmp142)
    tmp144 = tmp143 * tmp131
    tmp145 = tmp141 + tmp144
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp145, None)
