
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_mul_pow_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_mul_pow_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (4*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr3 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr4 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp76 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr3 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr4 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0001
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7 - tmp2
    tmp9 = tmp8 * tmp3
    tmp10 = 0.0009
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp0 * tmp0
    tmp14 = tmp1 * tmp1
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 + tmp5
    tmp18 = tmp17 - tmp13
    tmp20 = tmp19 - tmp14
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21 + tmp10
    tmp23 = tmp16 * tmp22
    tmp24 = tmp12 / tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27 * tmp3
    tmp29 = tmp28 + tmp5
    tmp31 = tmp30 - tmp27
    tmp32 = tmp31 * tmp3
    tmp33 = tmp32 + tmp10
    tmp34 = tmp29 * tmp33
    tmp35 = tmp25 * tmp25
    tmp36 = tmp26 * tmp26
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37 + tmp5
    tmp40 = tmp39 - tmp35
    tmp42 = tmp41 - tmp36
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43 + tmp10
    tmp45 = tmp38 * tmp44
    tmp46 = tmp34 / tmp45
    tmp47 = tmp24 + tmp46
    tmp50 = tmp48 * tmp49
    tmp51 = tmp50 * tmp3
    tmp52 = tmp51 + tmp5
    tmp54 = tmp53 - tmp50
    tmp55 = tmp54 * tmp3
    tmp56 = tmp55 + tmp10
    tmp57 = tmp52 * tmp56
    tmp58 = tmp48 * tmp48
    tmp59 = tmp49 * tmp49
    tmp60 = tmp58 + tmp59
    tmp61 = tmp60 + tmp5
    tmp63 = tmp62 - tmp58
    tmp65 = tmp64 - tmp59
    tmp66 = tmp63 + tmp65
    tmp67 = tmp66 + tmp10
    tmp68 = tmp61 * tmp67
    tmp69 = tmp57 / tmp68
    tmp70 = tmp47 + tmp69
    tmp73 = tmp71 * tmp72
    tmp74 = tmp73 * tmp3
    tmp75 = tmp74 + tmp5
    tmp77 = tmp76 - tmp73
    tmp78 = tmp77 * tmp3
    tmp79 = tmp78 + tmp10
    tmp80 = tmp75 * tmp79
    tmp81 = tmp71 * tmp71
    tmp82 = tmp72 * tmp72
    tmp83 = tmp81 + tmp82
    tmp84 = tmp83 + tmp5
    tmp86 = tmp85 - tmp81
    tmp88 = tmp87 - tmp82
    tmp89 = tmp86 + tmp88
    tmp90 = tmp89 + tmp10
    tmp91 = tmp84 * tmp90
    tmp92 = tmp80 / tmp91
    tmp93 = tmp70 + tmp92
    tmp94 = 4.0
    tmp95 = tmp93 / tmp94
    tl.store(out_ptr0 + (x0), tmp95, xmask)
