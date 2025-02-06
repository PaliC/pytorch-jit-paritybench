
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_grid_sampler_2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_grid_sampler_2d_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex // 32
    x4 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr2 + (x3), xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.5
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 3.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = libdevice.floor(tmp8)
    tmp10 = tmp9 >= tmp5
    tmp11 = 4.0
    tmp12 = tmp9 < tmp11
    tmp14 = tmp13 * tmp1
    tmp15 = tmp14 + tmp3
    tmp16 = triton_helpers.maximum(tmp15, tmp5)
    tmp17 = triton_helpers.minimum(tmp16, tmp7)
    tmp18 = libdevice.floor(tmp17)
    tmp19 = tmp18 >= tmp5
    tmp20 = tmp18 < tmp11
    tmp21 = tmp19 & tmp20
    tmp22 = tmp12 & tmp21
    tmp23 = tmp10 & tmp22
    tmp24 = tmp18.to(tl.int64)
    tmp25 = tl.full([1], 0, tl.int64)
    tmp26 = tl.where(tmp23, tmp24, tmp25)
    tmp27 = tl.full([XBLOCK], 4, tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tmp30) & (tmp30 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp30 < 4")
    tmp32 = tmp9.to(tl.int64)
    tmp33 = tl.where(tmp23, tmp32, tmp25)
    tmp34 = tmp33 + tmp27
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tl.device_assert(((0 <= tmp36) & (tmp36 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp36 < 4")
    tmp38 = tl.load(in_ptr1 + (tmp36 + 4*tmp30 + 16*x4), xmask, eviction_policy='evict_last')
    tmp39 = 1.0
    tmp40 = tmp9 + tmp39
    tmp41 = tmp40 >= tmp5
    tmp42 = tmp40 < tmp11
    tmp43 = tmp42 & tmp21
    tmp44 = tmp41 & tmp43
    tmp45 = tl.where(tmp44, tmp24, tmp25)
    tmp46 = tmp45 + tmp27
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tl.device_assert(((0 <= tmp48) & (tmp48 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp48 < 4")
    tmp50 = tmp40.to(tl.int64)
    tmp51 = tl.where(tmp44, tmp50, tmp25)
    tmp52 = tmp51 + tmp27
    tmp53 = tmp51 < 0
    tmp54 = tl.where(tmp53, tmp52, tmp51)
    tl.device_assert(((0 <= tmp54) & (tmp54 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp54 < 4")
    tmp56 = tl.load(in_ptr1 + (tmp54 + 4*tmp48 + 16*x4), xmask, eviction_policy='evict_last')
    tmp57 = tmp18 + tmp39
    tmp58 = tmp57 >= tmp5
    tmp59 = tmp57 < tmp11
    tmp60 = tmp58 & tmp59
    tmp61 = tmp12 & tmp60
    tmp62 = tmp10 & tmp61
    tmp63 = tmp57.to(tl.int64)
    tmp64 = tl.where(tmp62, tmp63, tmp25)
    tmp65 = tmp64 + tmp27
    tmp66 = tmp64 < 0
    tmp67 = tl.where(tmp66, tmp65, tmp64)
    tl.device_assert(((0 <= tmp67) & (tmp67 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp67 < 4")
    tmp69 = tl.where(tmp62, tmp32, tmp25)
    tmp70 = tmp69 + tmp27
    tmp71 = tmp69 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp69)
    tl.device_assert(((0 <= tmp72) & (tmp72 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp72 < 4")
    tmp74 = tl.load(in_ptr1 + (tmp72 + 4*tmp67 + 16*x4), xmask, eviction_policy='evict_last')
    tmp75 = tmp42 & tmp60
    tmp76 = tmp41 & tmp75
    tmp77 = tl.where(tmp76, tmp63, tmp25)
    tmp78 = tmp77 + tmp27
    tmp79 = tmp77 < 0
    tmp80 = tl.where(tmp79, tmp78, tmp77)
    tl.device_assert(((0 <= tmp80) & (tmp80 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp80 < 4")
    tmp82 = tl.where(tmp76, tmp50, tmp25)
    tmp83 = tmp82 + tmp27
    tmp84 = tmp82 < 0
    tmp85 = tl.where(tmp84, tmp83, tmp82)
    tl.device_assert(((0 <= tmp85) & (tmp85 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp85 < 4")
    tmp87 = tl.load(in_ptr1 + (tmp85 + 4*tmp80 + 16*x4), xmask, eviction_policy='evict_last')
    tmp88 = tmp40 - tmp8
    tmp89 = tmp57 - tmp17
    tmp90 = tmp88 * tmp89
    tmp91 = tl.where(tmp23, tmp90, tmp5)
    tmp92 = tmp8 - tmp9
    tmp93 = tmp92 * tmp89
    tmp94 = tl.where(tmp44, tmp93, tmp5)
    tmp95 = tmp17 - tmp18
    tmp96 = tmp88 * tmp95
    tmp97 = tl.where(tmp62, tmp96, tmp5)
    tmp98 = tmp92 * tmp95
    tmp99 = tl.where(tmp76, tmp98, tmp5)
    tmp101 = tmp38 * tmp91
    tmp102 = tmp56 * tmp94
    tmp103 = tmp101 + tmp102
    tmp104 = tmp74 * tmp97
    tmp105 = tmp103 + tmp104
    tmp106 = tmp87 * tmp99
    tmp107 = tmp105 + tmp106
    tmp108 = tmp100 + tmp107
    tmp109 = tl_math.abs(tmp108)
    tl.store(in_out_ptr0 + (x3), tmp109, xmask)
