
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_rsub_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 27, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_rsub_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 56)
    x1 = xindex // 56
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), None)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 64*x1), None)
    tmp3 = tl.load(in_ptr0 + (2 + x0 + 64*x1), None)
    tmp5 = tl.load(in_ptr0 + (3 + x0 + 64*x1), None)
    tmp7 = tl.load(in_ptr0 + (4 + x0 + 64*x1), None)
    tmp9 = tl.load(in_ptr0 + (5 + x0 + 64*x1), None)
    tmp11 = tl.load(in_ptr0 + (6 + x0 + 64*x1), None)
    tmp13 = tl.load(in_ptr0 + (7 + x0 + 64*x1), None)
    tmp15 = tl.load(in_ptr0 + (8 + x0 + 64*x1), None)
    tmp19 = tl.load(in_ptr1 + (x0 + 64*x1), None)
    tmp20 = tl.load(in_ptr1 + (1 + x0 + 64*x1), None)
    tmp22 = tl.load(in_ptr1 + (2 + x0 + 64*x1), None)
    tmp24 = tl.load(in_ptr1 + (3 + x0 + 64*x1), None)
    tmp26 = tl.load(in_ptr1 + (4 + x0 + 64*x1), None)
    tmp28 = tl.load(in_ptr1 + (5 + x0 + 64*x1), None)
    tmp30 = tl.load(in_ptr1 + (6 + x0 + 64*x1), None)
    tmp32 = tl.load(in_ptr1 + (7 + x0 + 64*x1), None)
    tmp34 = tl.load(in_ptr1 + (8 + x0 + 64*x1), None)
    tmp55 = tl.load(in_ptr2 + (x0 + 64*x1), None)
    tmp56 = tl.load(in_ptr2 + (1 + x0 + 64*x1), None)
    tmp58 = tl.load(in_ptr2 + (2 + x0 + 64*x1), None)
    tmp60 = tl.load(in_ptr2 + (3 + x0 + 64*x1), None)
    tmp62 = tl.load(in_ptr2 + (4 + x0 + 64*x1), None)
    tmp64 = tl.load(in_ptr2 + (5 + x0 + 64*x1), None)
    tmp66 = tl.load(in_ptr2 + (6 + x0 + 64*x1), None)
    tmp68 = tl.load(in_ptr2 + (7 + x0 + 64*x1), None)
    tmp70 = tl.load(in_ptr2 + (8 + x0 + 64*x1), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp21 = tmp20 + tmp19
    tmp23 = tmp22 + tmp21
    tmp25 = tmp24 + tmp23
    tmp27 = tmp26 + tmp25
    tmp29 = tmp28 + tmp27
    tmp31 = tmp30 + tmp29
    tmp33 = tmp32 + tmp31
    tmp35 = tmp34 + tmp33
    tmp36 = tmp35 * tmp17
    tmp37 = tmp19 * tmp19
    tmp38 = tmp20 * tmp20
    tmp39 = tmp38 + tmp37
    tmp40 = tmp22 * tmp22
    tmp41 = tmp40 + tmp39
    tmp42 = tmp24 * tmp24
    tmp43 = tmp42 + tmp41
    tmp44 = tmp26 * tmp26
    tmp45 = tmp44 + tmp43
    tmp46 = tmp28 * tmp28
    tmp47 = tmp46 + tmp45
    tmp48 = tmp30 * tmp30
    tmp49 = tmp48 + tmp47
    tmp50 = tmp32 * tmp32
    tmp51 = tmp50 + tmp49
    tmp52 = tmp34 * tmp34
    tmp53 = tmp52 + tmp51
    tmp54 = tmp53 * tmp17
    tmp57 = tmp56 + tmp55
    tmp59 = tmp58 + tmp57
    tmp61 = tmp60 + tmp59
    tmp63 = tmp62 + tmp61
    tmp65 = tmp64 + tmp63
    tmp67 = tmp66 + tmp65
    tmp69 = tmp68 + tmp67
    tmp71 = tmp70 + tmp69
    tmp72 = tmp71 * tmp17
    tmp73 = tmp55 * tmp55
    tmp74 = tmp56 * tmp56
    tmp75 = tmp74 + tmp73
    tmp76 = tmp58 * tmp58
    tmp77 = tmp76 + tmp75
    tmp78 = tmp60 * tmp60
    tmp79 = tmp78 + tmp77
    tmp80 = tmp62 * tmp62
    tmp81 = tmp80 + tmp79
    tmp82 = tmp64 * tmp64
    tmp83 = tmp82 + tmp81
    tmp84 = tmp66 * tmp66
    tmp85 = tmp84 + tmp83
    tmp86 = tmp68 * tmp68
    tmp87 = tmp86 + tmp85
    tmp88 = tmp70 * tmp70
    tmp89 = tmp88 + tmp87
    tmp90 = tmp89 * tmp17
    tmp91 = 2.0
    tmp92 = tmp36 * tmp91
    tmp93 = tmp92 * tmp72
    tmp94 = 0.0001
    tmp95 = tmp93 + tmp94
    tmp96 = tmp36 * tmp72
    tmp97 = tmp18 - tmp96
    tmp98 = tmp97 * tmp91
    tmp99 = 0.0009
    tmp100 = tmp98 + tmp99
    tmp101 = tmp95 * tmp100
    tmp102 = tmp36 * tmp36
    tmp103 = tmp72 * tmp72
    tmp104 = tmp102 + tmp103
    tmp105 = tmp104 + tmp94
    tmp106 = tmp54 - tmp102
    tmp107 = tmp90 - tmp103
    tmp108 = tmp106 + tmp107
    tmp109 = tmp108 + tmp99
    tmp110 = tmp105 * tmp109
    tmp111 = tmp101 / tmp110
    tmp112 = 1.0
    tmp113 = tmp112 - tmp111
    tmp114 = 0.5
    tmp115 = tmp113 * tmp114
    tmp116 = 0.0
    tmp117 = triton_helpers.maximum(tmp115, tmp116)
    tmp118 = triton_helpers.minimum(tmp117, tmp112)
    tl.store(in_out_ptr0 + (x2), tmp118, None)
