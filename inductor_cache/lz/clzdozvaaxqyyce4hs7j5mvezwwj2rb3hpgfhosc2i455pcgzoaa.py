
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_div_mul_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_div_mul_sub_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x5 = xindex
    tmp97 = tl.load(in_ptr4 + (x5), xmask)
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 3, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 + tmp2
    tmp17 = tmp16 * tmp4
    tmp18 = tmp17 - tmp2
    tmp19 = triton_helpers.maximum(tmp18, tmp7)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tmp20 + tmp10
    tmp22 = triton_helpers.minimum(tmp21, tmp12)
    tmp23 = tl.load(in_ptr0 + (x2 + 4*tmp22 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (x2 + 4*tmp22 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (x2 + 4*tmp22 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 - tmp26
    tmp28 = tl.load(in_ptr3 + (x2 + 4*tmp22 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp29 = tmp24 * tmp24
    tmp30 = tmp28 - tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = tmp27 / tmp32
    tmp34 = tl.load(in_ptr0 + (x2 + 4*tmp20 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (x2 + 4*tmp20 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (x2 + 4*tmp20 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp37 = tmp35 * tmp36
    tmp38 = tmp34 - tmp37
    tmp39 = tl.load(in_ptr3 + (x2 + 4*tmp20 + 16*tmp13 + 64*x3), xmask, eviction_policy='evict_last')
    tmp40 = tmp35 * tmp35
    tmp41 = tmp39 - tmp40
    tmp42 = tmp41 + tmp31
    tmp43 = tmp38 / tmp42
    tmp44 = tl.load(in_ptr0 + (x2 + 4*tmp22 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr1 + (x2 + 4*tmp22 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr2 + (x2 + 4*tmp22 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp47 = tmp45 * tmp46
    tmp48 = tmp44 - tmp47
    tmp49 = tl.load(in_ptr3 + (x2 + 4*tmp22 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp50 = tmp45 * tmp45
    tmp51 = tmp49 - tmp50
    tmp52 = tmp51 + tmp31
    tmp53 = tmp48 / tmp52
    tmp54 = tl.load(in_ptr0 + (x2 + 4*tmp20 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr1 + (x2 + 4*tmp20 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr2 + (x2 + 4*tmp20 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp57 = tmp55 * tmp56
    tmp58 = tmp54 - tmp57
    tmp59 = tl.load(in_ptr3 + (x2 + 4*tmp20 + 16*tmp9 + 64*x3), xmask, eviction_policy='evict_last')
    tmp60 = tmp55 * tmp55
    tmp61 = tmp59 - tmp60
    tmp62 = tmp61 + tmp31
    tmp63 = tmp58 / tmp62
    tmp64 = tmp53 - tmp63
    tmp65 = tmp20.to(tl.float32)
    tmp66 = tmp19 - tmp65
    tmp67 = triton_helpers.maximum(tmp66, tmp7)
    tmp68 = triton_helpers.minimum(tmp67, tmp4)
    tmp69 = tmp64 * tmp68
    tmp70 = tmp63 + tmp69
    tmp71 = tmp33 * tmp24
    tmp72 = tmp25 - tmp71
    tmp73 = tmp43 * tmp35
    tmp74 = tmp36 - tmp73
    tmp75 = tmp53 * tmp45
    tmp76 = tmp46 - tmp75
    tmp77 = tmp63 * tmp55
    tmp78 = tmp56 - tmp77
    tmp79 = tmp76 - tmp78
    tmp80 = tmp79 * tmp68
    tmp81 = tmp78 + tmp80
    tmp82 = tmp33 - tmp43
    tmp83 = tmp82 * tmp68
    tmp84 = tmp43 + tmp83
    tmp85 = tmp84 - tmp70
    tmp86 = tmp9.to(tl.float32)
    tmp87 = tmp8 - tmp86
    tmp88 = triton_helpers.maximum(tmp87, tmp7)
    tmp89 = triton_helpers.minimum(tmp88, tmp4)
    tmp90 = tmp85 * tmp89
    tmp91 = tmp72 - tmp74
    tmp92 = tmp91 * tmp68
    tmp93 = tmp74 + tmp92
    tmp94 = tmp93 - tmp81
    tmp95 = tmp94 * tmp89
    tmp96 = tmp70 + tmp90
    tmp98 = tmp96 * tmp97
    tmp99 = tmp81 + tmp95
    tmp100 = tmp98 + tmp99
    tl.store(in_out_ptr0 + (x5), tmp100, xmask)
