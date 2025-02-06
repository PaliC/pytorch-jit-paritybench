
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
    x4 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x2), xmask, eviction_policy='evict_last')
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tmp3 = 1.5
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 4.0
    tmp9 = tmp5 < tmp8
    tmp11 = tmp10 * tmp1
    tmp12 = tmp11 + tmp3
    tmp13 = libdevice.floor(tmp12)
    tmp14 = tmp13 >= tmp6
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp9 & tmp16
    tmp18 = tmp7 & tmp17
    tmp19 = tmp13.to(tl.int64)
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.full([XBLOCK], 4, tl.int32)
    tmp23 = tmp21 + tmp22
    tmp24 = tmp21 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp21)
    tl.device_assert(((0 <= tmp25) & (tmp25 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp25 < 4")
    tmp27 = tmp5.to(tl.int64)
    tmp28 = tl.where(tmp18, tmp27, tmp20)
    tmp29 = tmp28 + tmp22
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tl.device_assert(((0 <= tmp31) & (tmp31 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp31 < 4")
    tmp33 = tl.load(in_ptr1 + (tmp31 + 4*tmp25 + 16*x4), xmask, eviction_policy='evict_last')
    tmp34 = 1.0
    tmp35 = tmp5 + tmp34
    tmp36 = tmp35 - tmp4
    tmp37 = tmp13 + tmp34
    tmp38 = tmp37 - tmp12
    tmp39 = tmp36 * tmp38
    tmp40 = tl.where(tmp18, tmp39, tmp6)
    tmp41 = tmp33 * tmp40
    tmp42 = tmp35 >= tmp6
    tmp43 = tmp35 < tmp8
    tmp44 = tmp43 & tmp16
    tmp45 = tmp42 & tmp44
    tmp46 = tl.where(tmp45, tmp19, tmp20)
    tmp47 = tmp46 + tmp22
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp49 < 4")
    tmp51 = tmp35.to(tl.int64)
    tmp52 = tl.where(tmp45, tmp51, tmp20)
    tmp53 = tmp52 + tmp22
    tmp54 = tmp52 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp52)
    tl.device_assert(((0 <= tmp55) & (tmp55 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tl.load(in_ptr1 + (tmp55 + 4*tmp49 + 16*x4), xmask, eviction_policy='evict_last')
    tmp58 = tmp4 - tmp5
    tmp59 = tmp58 * tmp38
    tmp60 = tl.where(tmp45, tmp59, tmp6)
    tmp61 = tmp57 * tmp60
    tmp62 = tmp37 >= tmp6
    tmp63 = tmp37 < tmp8
    tmp64 = tmp62 & tmp63
    tmp65 = tmp9 & tmp64
    tmp66 = tmp7 & tmp65
    tmp67 = tmp37.to(tl.int64)
    tmp68 = tl.where(tmp66, tmp67, tmp20)
    tmp69 = tmp68 + tmp22
    tmp70 = tmp68 < 0
    tmp71 = tl.where(tmp70, tmp69, tmp68)
    tl.device_assert(((0 <= tmp71) & (tmp71 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp71 < 4")
    tmp73 = tl.where(tmp66, tmp27, tmp20)
    tmp74 = tmp73 + tmp22
    tmp75 = tmp73 < 0
    tmp76 = tl.where(tmp75, tmp74, tmp73)
    tl.device_assert(((0 <= tmp76) & (tmp76 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp76 < 4")
    tmp78 = tl.load(in_ptr1 + (tmp76 + 4*tmp71 + 16*x4), xmask, eviction_policy='evict_last')
    tmp79 = tmp12 - tmp13
    tmp80 = tmp36 * tmp79
    tmp81 = tl.where(tmp66, tmp80, tmp6)
    tmp82 = tmp78 * tmp81
    tmp83 = tmp43 & tmp64
    tmp84 = tmp42 & tmp83
    tmp85 = tl.where(tmp84, tmp67, tmp20)
    tmp86 = tmp85 + tmp22
    tmp87 = tmp85 < 0
    tmp88 = tl.where(tmp87, tmp86, tmp85)
    tl.device_assert(((0 <= tmp88) & (tmp88 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp88 < 4")
    tmp90 = tl.where(tmp84, tmp51, tmp20)
    tmp91 = tmp90 + tmp22
    tmp92 = tmp90 < 0
    tmp93 = tl.where(tmp92, tmp91, tmp90)
    tl.device_assert(((0 <= tmp93) & (tmp93 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp93 < 4")
    tmp95 = tl.load(in_ptr1 + (tmp93 + 4*tmp88 + 16*x4), xmask, eviction_policy='evict_last')
    tmp96 = tmp58 * tmp79
    tmp97 = tl.where(tmp84, tmp96, tmp6)
    tmp98 = tmp95 * tmp97
    tmp99 = tmp41 + tmp61
    tmp100 = tmp99 + tmp82
    tmp101 = tmp100 + tmp98
    tl.store(in_out_ptr0 + (x3), tmp101, xmask)
