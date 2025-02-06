
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_div_nll_loss_forward_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_div_nll_loss_forward_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr0 + (2))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (3))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp29 = tl.load(in_ptr0 + (4))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp32 = tl.load(in_ptr0 + (5))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp36 = tl.load(in_ptr0 + (6))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp40 = tl.load(in_ptr0 + (7))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp53 = tl.load(in_ptr0 + (8))
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK])
    tmp56 = tl.load(in_ptr0 + (9))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp60 = tl.load(in_ptr0 + (10))
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK])
    tmp64 = tl.load(in_ptr0 + (11))
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
    tmp77 = tl.load(in_ptr0 + (12))
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK])
    tmp80 = tl.load(in_ptr0 + (13))
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK])
    tmp84 = tl.load(in_ptr0 + (14))
    tmp85 = tl.broadcast_to(tmp84, [XBLOCK])
    tmp88 = tl.load(in_ptr0 + (15))
    tmp89 = tl.broadcast_to(tmp88, [XBLOCK])
    tmp98 = tl.load(in_ptr1 + (0))
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK])
    tmp101 = tl.load(in_ptr1 + (1))
    tmp102 = tl.broadcast_to(tmp101, [XBLOCK])
    tmp105 = tl.load(in_ptr1 + (2))
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK])
    tmp109 = tl.load(in_ptr1 + (3))
    tmp110 = tl.broadcast_to(tmp109, [XBLOCK])
    tmp118 = tl.load(in_ptr1 + (4))
    tmp119 = tl.broadcast_to(tmp118, [XBLOCK])
    tmp121 = tl.load(in_ptr1 + (5))
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK])
    tmp125 = tl.load(in_ptr1 + (6))
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK])
    tmp129 = tl.load(in_ptr1 + (7))
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK])
    tmp139 = tl.load(in_ptr1 + (8))
    tmp140 = tl.broadcast_to(tmp139, [XBLOCK])
    tmp142 = tl.load(in_ptr1 + (9))
    tmp143 = tl.broadcast_to(tmp142, [XBLOCK])
    tmp146 = tl.load(in_ptr1 + (10))
    tmp147 = tl.broadcast_to(tmp146, [XBLOCK])
    tmp150 = tl.load(in_ptr1 + (11))
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK])
    tmp160 = tl.load(in_ptr1 + (12))
    tmp161 = tl.broadcast_to(tmp160, [XBLOCK])
    tmp163 = tl.load(in_ptr1 + (13))
    tmp164 = tl.broadcast_to(tmp163, [XBLOCK])
    tmp167 = tl.load(in_ptr1 + (14))
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK])
    tmp171 = tl.load(in_ptr1 + (15))
    tmp172 = tl.broadcast_to(tmp171, [XBLOCK])
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.where(tmp2, tmp0, tmp0)
    tmp4 = tl.load(in_ptr0 + (tmp3), None, eviction_policy='evict_last')
    tmp7 = tl_math.exp(tmp6)
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp7 + tmp10
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp11 + tmp14
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp15 + tmp18
    tmp20 = tl_math.log(tmp19)
    tmp21 = tmp4 - tmp20
    tmp22 = -tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.full([1], 1, tl.int64)
    tmp26 = tmp25 != tmp1
    tmp27 = tl.where(tmp26, tmp25, tmp0)
    tmp28 = tl.load(in_ptr0 + (4 + tmp27), None, eviction_policy='evict_last')
    tmp31 = tl_math.exp(tmp30)
    tmp34 = tl_math.exp(tmp33)
    tmp35 = tmp31 + tmp34
    tmp38 = tl_math.exp(tmp37)
    tmp39 = tmp35 + tmp38
    tmp42 = tl_math.exp(tmp41)
    tmp43 = tmp39 + tmp42
    tmp44 = tl_math.log(tmp43)
    tmp45 = tmp28 - tmp44
    tmp46 = -tmp45
    tmp47 = tl.where(tmp26, tmp46, tmp23)
    tmp48 = tmp24 + tmp47
    tmp49 = tl.full([1], 2, tl.int64)
    tmp50 = tmp49 != tmp1
    tmp51 = tl.where(tmp50, tmp49, tmp0)
    tmp52 = tl.load(in_ptr0 + (8 + tmp51), None, eviction_policy='evict_last')
    tmp55 = tl_math.exp(tmp54)
    tmp58 = tl_math.exp(tmp57)
    tmp59 = tmp55 + tmp58
    tmp62 = tl_math.exp(tmp61)
    tmp63 = tmp59 + tmp62
    tmp66 = tl_math.exp(tmp65)
    tmp67 = tmp63 + tmp66
    tmp68 = tl_math.log(tmp67)
    tmp69 = tmp52 - tmp68
    tmp70 = -tmp69
    tmp71 = tl.where(tmp50, tmp70, tmp23)
    tmp72 = tmp48 + tmp71
    tmp73 = tl.full([1], 3, tl.int64)
    tmp74 = tmp73 != tmp1
    tmp75 = tl.where(tmp74, tmp73, tmp0)
    tmp76 = tl.load(in_ptr0 + (12 + tmp75), None, eviction_policy='evict_last')
    tmp79 = tl_math.exp(tmp78)
    tmp82 = tl_math.exp(tmp81)
    tmp83 = tmp79 + tmp82
    tmp86 = tl_math.exp(tmp85)
    tmp87 = tmp83 + tmp86
    tmp90 = tl_math.exp(tmp89)
    tmp91 = tmp87 + tmp90
    tmp92 = tl_math.log(tmp91)
    tmp93 = tmp76 - tmp92
    tmp94 = -tmp93
    tmp95 = tl.where(tmp74, tmp94, tmp23)
    tmp96 = tmp72 + tmp95
    tmp97 = tl.load(in_ptr1 + (tmp3), None, eviction_policy='evict_last')
    tmp100 = tl_math.exp(tmp99)
    tmp103 = tl_math.exp(tmp102)
    tmp104 = tmp100 + tmp103
    tmp107 = tl_math.exp(tmp106)
    tmp108 = tmp104 + tmp107
    tmp111 = tl_math.exp(tmp110)
    tmp112 = tmp108 + tmp111
    tmp113 = tl_math.log(tmp112)
    tmp114 = tmp97 - tmp113
    tmp115 = -tmp114
    tmp116 = tl.where(tmp2, tmp115, tmp23)
    tmp117 = tl.load(in_ptr1 + (4 + tmp27), None, eviction_policy='evict_last')
    tmp120 = tl_math.exp(tmp119)
    tmp123 = tl_math.exp(tmp122)
    tmp124 = tmp120 + tmp123
    tmp127 = tl_math.exp(tmp126)
    tmp128 = tmp124 + tmp127
    tmp131 = tl_math.exp(tmp130)
    tmp132 = tmp128 + tmp131
    tmp133 = tl_math.log(tmp132)
    tmp134 = tmp117 - tmp133
    tmp135 = -tmp134
    tmp136 = tl.where(tmp26, tmp135, tmp23)
    tmp137 = tmp116 + tmp136
    tmp138 = tl.load(in_ptr1 + (8 + tmp51), None, eviction_policy='evict_last')
    tmp141 = tl_math.exp(tmp140)
    tmp144 = tl_math.exp(tmp143)
    tmp145 = tmp141 + tmp144
    tmp148 = tl_math.exp(tmp147)
    tmp149 = tmp145 + tmp148
    tmp152 = tl_math.exp(tmp151)
    tmp153 = tmp149 + tmp152
    tmp154 = tl_math.log(tmp153)
    tmp155 = tmp138 - tmp154
    tmp156 = -tmp155
    tmp157 = tl.where(tmp50, tmp156, tmp23)
    tmp158 = tmp137 + tmp157
    tmp159 = tl.load(in_ptr1 + (12 + tmp75), None, eviction_policy='evict_last')
    tmp162 = tl_math.exp(tmp161)
    tmp165 = tl_math.exp(tmp164)
    tmp166 = tmp162 + tmp165
    tmp169 = tl_math.exp(tmp168)
    tmp170 = tmp166 + tmp169
    tmp173 = tl_math.exp(tmp172)
    tmp174 = tmp170 + tmp173
    tmp175 = tl_math.log(tmp174)
    tmp176 = tmp159 - tmp175
    tmp177 = -tmp176
    tmp178 = tl.where(tmp74, tmp177, tmp23)
    tmp179 = tmp158 + tmp178
    tmp180 = tmp2.to(tl.int32)
    tmp181 = tmp26.to(tl.int32)
    tmp182 = tmp180 + tmp181
    tmp183 = tmp50.to(tl.int32)
    tmp184 = tmp182 + tmp183
    tmp185 = tmp74.to(tl.int32)
    tmp186 = tmp184 + tmp185
    tmp187 = tmp186.to(tl.float32)
    tmp188 = tmp96 / tmp187
    tmp189 = tmp179 / tmp187
    tmp190 = tmp188 + tmp189
    tmp191 = 0.5
    tmp192 = tmp190 * tmp191
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp192, None)
