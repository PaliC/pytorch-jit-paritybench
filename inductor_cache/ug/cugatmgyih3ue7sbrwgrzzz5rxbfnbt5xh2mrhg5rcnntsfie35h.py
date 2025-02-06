
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_div_log_neg_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_div_log_neg_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (7*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr1 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp1 = 10.0
    tmp2 = tmp0 * tmp1
    tmp4 = tl_math.log(tmp3)
    tmp5 = -tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tl_math.log(tmp11)
    tmp13 = -tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tmp14 * tmp7
    tmp16 = triton_helpers.maximum(tmp8, tmp15)
    tmp18 = tmp17 * tmp1
    tmp20 = tl_math.log(tmp19)
    tmp21 = -tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp22 * tmp7
    tmp24 = triton_helpers.maximum(tmp16, tmp23)
    tmp26 = tmp25 * tmp1
    tmp28 = tl_math.log(tmp27)
    tmp29 = -tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tmp30 * tmp7
    tmp32 = triton_helpers.maximum(tmp24, tmp31)
    tmp34 = tmp33 * tmp1
    tmp36 = tl_math.log(tmp35)
    tmp37 = -tmp36
    tmp38 = tmp34 + tmp37
    tmp39 = tmp38 * tmp7
    tmp40 = triton_helpers.maximum(tmp32, tmp39)
    tmp42 = tmp41 * tmp1
    tmp44 = tl_math.log(tmp43)
    tmp45 = -tmp44
    tmp46 = tmp42 + tmp45
    tmp47 = tmp46 * tmp7
    tmp48 = triton_helpers.maximum(tmp40, tmp47)
    tmp50 = tmp49 * tmp1
    tmp52 = tl_math.log(tmp51)
    tmp53 = -tmp52
    tmp54 = tmp50 + tmp53
    tmp55 = tmp54 * tmp7
    tmp56 = triton_helpers.maximum(tmp48, tmp55)
    tmp57 = tmp8 - tmp56
    tmp58 = tmp57 * tmp7
    tmp59 = tl_math.exp(tmp58)
    tmp60 = tmp15 - tmp56
    tmp61 = tmp60 * tmp7
    tmp62 = tl_math.exp(tmp61)
    tmp63 = tmp59 + tmp62
    tmp64 = tmp23 - tmp56
    tmp65 = tmp64 * tmp7
    tmp66 = tl_math.exp(tmp65)
    tmp67 = tmp63 + tmp66
    tmp68 = tmp31 - tmp56
    tmp69 = tmp68 * tmp7
    tmp70 = tl_math.exp(tmp69)
    tmp71 = tmp67 + tmp70
    tmp72 = tmp39 - tmp56
    tmp73 = tmp72 * tmp7
    tmp74 = tl_math.exp(tmp73)
    tmp75 = tmp71 + tmp74
    tmp76 = tmp47 - tmp56
    tmp77 = tmp76 * tmp7
    tmp78 = tl_math.exp(tmp77)
    tmp79 = tmp75 + tmp78
    tmp80 = tmp55 - tmp56
    tmp81 = tmp80 * tmp7
    tmp82 = tl_math.exp(tmp81)
    tmp83 = tmp79 + tmp82
    tmp84 = tmp0 * tmp7
    tmp85 = tmp9 * tmp7
    tmp86 = triton_helpers.maximum(tmp84, tmp85)
    tmp87 = tmp17 * tmp7
    tmp88 = triton_helpers.maximum(tmp86, tmp87)
    tmp89 = tmp25 * tmp7
    tmp90 = triton_helpers.maximum(tmp88, tmp89)
    tmp91 = tmp33 * tmp7
    tmp92 = triton_helpers.maximum(tmp90, tmp91)
    tmp93 = tmp41 * tmp7
    tmp94 = triton_helpers.maximum(tmp92, tmp93)
    tmp95 = tmp49 * tmp7
    tmp96 = triton_helpers.maximum(tmp94, tmp95)
    tmp97 = tmp84 - tmp96
    tmp98 = tmp97 * tmp1
    tmp99 = tl_math.exp(tmp98)
    tmp100 = tmp85 - tmp96
    tmp101 = tmp100 * tmp1
    tmp102 = tl_math.exp(tmp101)
    tmp103 = tmp99 + tmp102
    tmp104 = tmp87 - tmp96
    tmp105 = tmp104 * tmp1
    tmp106 = tl_math.exp(tmp105)
    tmp107 = tmp103 + tmp106
    tmp108 = tmp89 - tmp96
    tmp109 = tmp108 * tmp1
    tmp110 = tl_math.exp(tmp109)
    tmp111 = tmp107 + tmp110
    tmp112 = tmp91 - tmp96
    tmp113 = tmp112 * tmp1
    tmp114 = tl_math.exp(tmp113)
    tmp115 = tmp111 + tmp114
    tmp116 = tmp93 - tmp96
    tmp117 = tmp116 * tmp1
    tmp118 = tl_math.exp(tmp117)
    tmp119 = tmp115 + tmp118
    tmp120 = tmp95 - tmp96
    tmp121 = tmp120 * tmp1
    tmp122 = tl_math.exp(tmp121)
    tmp123 = tmp119 + tmp122
    tl.store(out_ptr0 + (x0), tmp56, xmask)
    tl.store(out_ptr1 + (x0), tmp83, xmask)
    tl.store(out_ptr2 + (x0), tmp96, xmask)
    tl.store(out_ptr3 + (x0), tmp123, xmask)
