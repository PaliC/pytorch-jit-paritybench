
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_argmin_linalg_vector_norm_sub_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_argmin_linalg_vector_norm_sub_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp6 = tl.load(in_ptr1 + (1))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (8 + x0 + 16*x1), xmask)
    tmp12 = tl.load(in_ptr1 + (2))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (12 + x0 + 16*x1), xmask)
    tmp18 = tl.load(in_ptr1 + (3))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (4))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp28 = tl.load(in_ptr1 + (5))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp33 = tl.load(in_ptr1 + (6))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp38 = tl.load(in_ptr1 + (7))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp59 = tl.load(in_ptr1 + (8))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp63 = tl.load(in_ptr1 + (9))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp68 = tl.load(in_ptr1 + (10))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
    tmp73 = tl.load(in_ptr1 + (11))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp93 = tl.load(in_ptr1 + (12))
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK])
    tmp97 = tl.load(in_ptr1 + (13))
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK])
    tmp102 = tl.load(in_ptr1 + (14))
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK])
    tmp107 = tl.load(in_ptr1 + (15))
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp4 = tmp3 * tmp3
    tmp8 = tmp5 - tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tmp4 + tmp9
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tmp10 + tmp15
    tmp20 = tmp17 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tmp16 + tmp21
    tmp23 = libdevice.sqrt(tmp22)
    tmp26 = tmp0 - tmp25
    tmp27 = tmp26 * tmp26
    tmp30 = tmp5 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tmp27 + tmp31
    tmp35 = tmp11 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tmp32 + tmp36
    tmp40 = tmp17 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp37 + tmp41
    tmp43 = libdevice.sqrt(tmp42)
    tmp44 = tmp23 < tmp43
    tmp45 = tmp23 == tmp43
    tmp46 = tmp23 != tmp23
    tmp47 = tmp43 != tmp43
    tmp48 = tmp46 > tmp47
    tmp49 = tmp44 | tmp48
    tmp50 = tmp46 & tmp47
    tmp51 = tmp45 | tmp50
    tmp52 = tl.full([1], 0, tl.int64)
    tmp53 = tl.full([1], 1, tl.int64)
    tmp54 = tmp52 < tmp53
    tmp55 = tmp51 & tmp54
    tmp56 = tmp49 | tmp55
    tmp57 = tl.where(tmp56, tmp23, tmp43)
    tmp58 = tl.where(tmp56, tmp52, tmp53)
    tmp61 = tmp0 - tmp60
    tmp62 = tmp61 * tmp61
    tmp65 = tmp5 - tmp64
    tmp66 = tmp65 * tmp65
    tmp67 = tmp62 + tmp66
    tmp70 = tmp11 - tmp69
    tmp71 = tmp70 * tmp70
    tmp72 = tmp67 + tmp71
    tmp75 = tmp17 - tmp74
    tmp76 = tmp75 * tmp75
    tmp77 = tmp72 + tmp76
    tmp78 = libdevice.sqrt(tmp77)
    tmp79 = tmp57 < tmp78
    tmp80 = tmp57 == tmp78
    tmp81 = tmp57 != tmp57
    tmp82 = tmp78 != tmp78
    tmp83 = tmp81 > tmp82
    tmp84 = tmp79 | tmp83
    tmp85 = tmp81 & tmp82
    tmp86 = tmp80 | tmp85
    tmp87 = tl.full([1], 2, tl.int64)
    tmp88 = tmp58 < tmp87
    tmp89 = tmp86 & tmp88
    tmp90 = tmp84 | tmp89
    tmp91 = tl.where(tmp90, tmp57, tmp78)
    tmp92 = tl.where(tmp90, tmp58, tmp87)
    tmp95 = tmp0 - tmp94
    tmp96 = tmp95 * tmp95
    tmp99 = tmp5 - tmp98
    tmp100 = tmp99 * tmp99
    tmp101 = tmp96 + tmp100
    tmp104 = tmp11 - tmp103
    tmp105 = tmp104 * tmp104
    tmp106 = tmp101 + tmp105
    tmp109 = tmp17 - tmp108
    tmp110 = tmp109 * tmp109
    tmp111 = tmp106 + tmp110
    tmp112 = libdevice.sqrt(tmp111)
    tmp113 = tmp91 < tmp112
    tmp114 = tmp91 == tmp112
    tmp115 = tmp91 != tmp91
    tmp116 = tmp112 != tmp112
    tmp117 = tmp115 > tmp116
    tmp118 = tmp113 | tmp117
    tmp119 = tmp115 & tmp116
    tmp120 = tmp114 | tmp119
    tmp121 = tl.full([1], 3, tl.int64)
    tmp122 = tmp92 < tmp121
    tmp123 = tmp120 & tmp122
    tmp124 = tmp118 | tmp123
    tmp125 = tl.where(tmp124, tmp91, tmp112)
    tmp126 = tl.where(tmp124, tmp92, tmp121)
    tl.store(out_ptr0 + (x2), tmp126, xmask)
