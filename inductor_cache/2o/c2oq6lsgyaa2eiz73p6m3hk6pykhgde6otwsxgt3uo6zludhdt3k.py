
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mse_loss_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 28, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mse_loss_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + (4))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.load(in_ptr1 + (4))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr2 + (4))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (8))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp23 = tl.load(in_ptr1 + (8))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp26 = tl.load(in_ptr2 + (8))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp32 = tl.load(in_ptr0 + (12))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp34 = tl.load(in_ptr1 + (12))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp37 = tl.load(in_ptr2 + (12))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp49 = tl.load(in_ptr0 + (1))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp52 = tl.load(in_ptr2 + (1))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp57 = tl.load(in_ptr0 + (5))
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK])
    tmp60 = tl.load(in_ptr2 + (5))
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK])
    tmp66 = tl.load(in_ptr0 + (9))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp69 = tl.load(in_ptr2 + (9))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp75 = tl.load(in_ptr0 + (13))
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK])
    tmp78 = tl.load(in_ptr2 + (13))
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK])
    tmp87 = tl.load(in_ptr0 + (2))
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK])
    tmp90 = tl.load(in_ptr2 + (2))
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK])
    tmp95 = tl.load(in_ptr0 + (6))
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK])
    tmp98 = tl.load(in_ptr2 + (6))
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK])
    tmp104 = tl.load(in_ptr0 + (10))
    tmp105 = tl.broadcast_to(tmp104, [XBLOCK])
    tmp107 = tl.load(in_ptr2 + (10))
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK])
    tmp113 = tl.load(in_ptr0 + (14))
    tmp114 = tl.broadcast_to(tmp113, [XBLOCK])
    tmp116 = tl.load(in_ptr2 + (14))
    tmp117 = tl.broadcast_to(tmp116, [XBLOCK])
    tmp4 = tmp1 * tmp3
    tmp7 = tmp6 * tmp3
    tmp8 = tmp4 - tmp7
    tmp9 = tmp8 * tmp8
    tmp14 = tmp11 * tmp13
    tmp17 = tmp16 * tmp13
    tmp18 = tmp14 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tmp9 + tmp19
    tmp25 = tmp22 * tmp24
    tmp28 = tmp27 * tmp24
    tmp29 = tmp25 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tmp20 + tmp30
    tmp36 = tmp33 * tmp35
    tmp39 = tmp38 * tmp35
    tmp40 = tmp36 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp31 + tmp41
    tmp43 = 4.0
    tmp44 = tmp42 / tmp43
    tmp45 = 0.5
    tmp46 = tmp44 * tmp45
    tmp47 = 0.0
    tmp48 = tmp46 + tmp47
    tmp51 = tmp7 * tmp50
    tmp54 = tmp7 * tmp53
    tmp55 = tmp51 - tmp54
    tmp56 = tmp55 * tmp55
    tmp59 = tmp17 * tmp58
    tmp62 = tmp17 * tmp61
    tmp63 = tmp59 - tmp62
    tmp64 = tmp63 * tmp63
    tmp65 = tmp56 + tmp64
    tmp68 = tmp28 * tmp67
    tmp71 = tmp28 * tmp70
    tmp72 = tmp68 - tmp71
    tmp73 = tmp72 * tmp72
    tmp74 = tmp65 + tmp73
    tmp77 = tmp39 * tmp76
    tmp80 = tmp39 * tmp79
    tmp81 = tmp77 - tmp80
    tmp82 = tmp81 * tmp81
    tmp83 = tmp74 + tmp82
    tmp84 = tmp83 / tmp43
    tmp85 = tmp84 * tmp45
    tmp86 = tmp48 + tmp85
    tmp89 = tmp7 * tmp88
    tmp92 = tmp7 * tmp91
    tmp93 = tmp89 - tmp92
    tmp94 = tmp93 * tmp93
    tmp97 = tmp17 * tmp96
    tmp100 = tmp17 * tmp99
    tmp101 = tmp97 - tmp100
    tmp102 = tmp101 * tmp101
    tmp103 = tmp94 + tmp102
    tmp106 = tmp28 * tmp105
    tmp109 = tmp28 * tmp108
    tmp110 = tmp106 - tmp109
    tmp111 = tmp110 * tmp110
    tmp112 = tmp103 + tmp111
    tmp115 = tmp39 * tmp114
    tmp118 = tmp39 * tmp117
    tmp119 = tmp115 - tmp118
    tmp120 = tmp119 * tmp119
    tmp121 = tmp112 + tmp120
    tmp122 = tmp121 / tmp43
    tmp123 = tmp122 * tmp45
    tmp124 = tmp86 + tmp123
    tmp125 = 1.0
    tmp126 = tmp124 * tmp125
    tmp127 = tmp126 * tmp125
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp127, None)
