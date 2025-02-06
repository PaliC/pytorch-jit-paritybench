
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    x4 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x2), xmask, eviction_policy='evict_last')
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
    tmp24 = 1.0
    tmp25 = tmp9 + tmp24
    tmp26 = tmp25 - tmp8
    tmp27 = tmp18 + tmp24
    tmp28 = tmp27 - tmp17
    tmp29 = tmp26 * tmp28
    tmp30 = tl.where(tmp23, tmp29, tmp5)
    tmp31 = tmp25 >= tmp5
    tmp32 = tmp25 < tmp11
    tmp33 = tmp32 & tmp21
    tmp34 = tmp31 & tmp33
    tmp35 = tmp8 - tmp9
    tmp36 = tmp35 * tmp28
    tmp37 = tl.where(tmp34, tmp36, tmp5)
    tmp38 = tmp27 >= tmp5
    tmp39 = tmp27 < tmp11
    tmp40 = tmp38 & tmp39
    tmp41 = tmp12 & tmp40
    tmp42 = tmp10 & tmp41
    tmp43 = tmp17 - tmp18
    tmp44 = tmp26 * tmp43
    tmp45 = tl.where(tmp42, tmp44, tmp5)
    tmp46 = tmp18.to(tl.int64)
    tmp47 = tl.full([1], 0, tl.int64)
    tmp48 = tl.where(tmp23, tmp46, tmp47)
    tmp49 = tl.full([XBLOCK], 4, tl.int32)
    tmp50 = tmp48 + tmp49
    tmp51 = tmp48 < 0
    tmp52 = tl.where(tmp51, tmp50, tmp48)
    tl.device_assert(((0 <= tmp52) & (tmp52 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp52 < 4")
    tmp54 = tmp9.to(tl.int64)
    tmp55 = tl.where(tmp23, tmp54, tmp47)
    tmp56 = tmp55 + tmp49
    tmp57 = tmp55 < 0
    tmp58 = tl.where(tmp57, tmp56, tmp55)
    tl.device_assert(((0 <= tmp58) & (tmp58 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp58 < 4")
    tmp60 = tl.load(in_ptr1 + (tmp58 + 4*tmp52 + 16*x4), xmask, eviction_policy='evict_last')
    tmp61 = tl.where(tmp34, tmp46, tmp47)
    tmp62 = tmp61 + tmp49
    tmp63 = tmp61 < 0
    tmp64 = tl.where(tmp63, tmp62, tmp61)
    tl.device_assert(((0 <= tmp64) & (tmp64 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp64 < 4")
    tmp66 = tmp25.to(tl.int64)
    tmp67 = tl.where(tmp34, tmp66, tmp47)
    tmp68 = tmp67 + tmp49
    tmp69 = tmp67 < 0
    tmp70 = tl.where(tmp69, tmp68, tmp67)
    tl.device_assert(((0 <= tmp70) & (tmp70 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp70 < 4")
    tmp72 = tl.load(in_ptr1 + (tmp70 + 4*tmp64 + 16*x4), xmask, eviction_policy='evict_last')
    tmp73 = tmp27.to(tl.int64)
    tmp74 = tl.where(tmp42, tmp73, tmp47)
    tmp75 = tmp74 + tmp49
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tl.device_assert(((0 <= tmp77) & (tmp77 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp77 < 4")
    tmp79 = tl.where(tmp42, tmp54, tmp47)
    tmp80 = tmp79 + tmp49
    tmp81 = tmp79 < 0
    tmp82 = tl.where(tmp81, tmp80, tmp79)
    tl.device_assert(((0 <= tmp82) & (tmp82 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp82 < 4")
    tmp84 = tl.load(in_ptr1 + (tmp82 + 4*tmp77 + 16*x4), xmask, eviction_policy='evict_last')
    tmp85 = tmp32 & tmp40
    tmp86 = tmp31 & tmp85
    tmp87 = tl.where(tmp86, tmp73, tmp47)
    tmp88 = tmp87 + tmp49
    tmp89 = tmp87 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp87)
    tl.device_assert(((0 <= tmp90) & (tmp90 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp90 < 4")
    tmp92 = tl.where(tmp86, tmp66, tmp47)
    tmp93 = tmp92 + tmp49
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tl.device_assert(((0 <= tmp95) & (tmp95 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp95 < 4")
    tmp97 = tl.load(in_ptr1 + (tmp95 + 4*tmp90 + 16*x4), xmask, eviction_policy='evict_last')
    tmp98 = tmp35 * tmp43
    tmp99 = tl.where(tmp86, tmp98, tmp5)
    tmp100 = tmp60 * tmp30
    tmp101 = tmp72 * tmp37
    tmp102 = tmp100 + tmp101
    tmp103 = tmp84 * tmp45
    tmp104 = tmp102 + tmp103
    tmp105 = tmp97 * tmp99
    tmp106 = tmp104 + tmp105
    tl.store(in_out_ptr0 + (x3), tmp106, xmask)
