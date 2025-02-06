
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 6, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tmp5.to(tl.float64)
    tmp7 = tl.full([1], 2.0, tl.float64)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([1], 12.25, tl.float64)
    tmp10 = tmp9 - tmp8
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 3.5, tl.float64)
    tmp13 = tmp12 - tmp11
    tmp14 = libdevice.floor(tmp13)
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 12, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = (-6) + x0
    tmp24 = tmp23.to(tl.float64)
    tmp25 = tl.full([1], 2.0, tl.float64)
    tmp26 = tmp24 * tmp25
    tmp27 = tl.full([1], 12.25, tl.float64)
    tmp28 = tmp27 - tmp26
    tmp29 = libdevice.sqrt(tmp28)
    tmp30 = tl.full([1], 3.5, tl.float64)
    tmp31 = tmp30 - tmp29
    tmp32 = libdevice.floor(tmp31)
    tmp33 = tl.full([1], 5.0, tl.float64)
    tmp34 = tmp33 - tmp32
    tmp35 = tmp34 * tmp32
    tmp36 = tl.full([1], 0.5, tl.float64)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp24 - tmp37
    tmp39 = libdevice.floor(tmp38)
    tmp40 = tmp39.to(tl.int64)
    tmp41 = tl.full([1], 1, tl.int64)
    tmp42 = tmp40 + tmp41
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp20, tmp42, tmp43)
    tmp45 = tl.where(tmp4, tmp19, tmp44)
    tmp46 = tl.full([XBLOCK], 4, tl.int32)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp45 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp45)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp49 < 4")
    tmp51 = 6 + x0
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = 6 + x0
    tmp55 = tmp54.to(tl.float64)
    tmp56 = tl.full([1], 2.0, tl.float64)
    tmp57 = tmp55 * tmp56
    tmp58 = tl.full([1], 12.25, tl.float64)
    tmp59 = tmp58 - tmp57
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tl.full([1], 3.5, tl.float64)
    tmp62 = tmp61 - tmp60
    tmp63 = libdevice.floor(tmp62)
    tmp64 = tmp63.to(tl.int64)
    tmp65 = tl.full([1], 0, tl.int64)
    tmp66 = tmp64 + tmp65
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp53, tmp66, tmp67)
    tmp69 = tmp51 >= tmp3
    tmp70 = tmp51 < tmp21
    tmp71 = x0
    tmp72 = tmp71.to(tl.float64)
    tmp73 = tl.full([1], 2.0, tl.float64)
    tmp74 = tmp72 * tmp73
    tmp75 = tl.full([1], 12.25, tl.float64)
    tmp76 = tmp75 - tmp74
    tmp77 = libdevice.sqrt(tmp76)
    tmp78 = tl.full([1], 3.5, tl.float64)
    tmp79 = tmp78 - tmp77
    tmp80 = libdevice.floor(tmp79)
    tmp81 = tl.full([1], 5.0, tl.float64)
    tmp82 = tmp81 - tmp80
    tmp83 = tmp82 * tmp80
    tmp84 = tl.full([1], 0.5, tl.float64)
    tmp85 = tmp83 * tmp84
    tmp86 = tmp72 - tmp85
    tmp87 = libdevice.floor(tmp86)
    tmp88 = tmp87.to(tl.int64)
    tmp89 = tl.full([1], 1, tl.int64)
    tmp90 = tmp88 + tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp69, tmp90, tmp91)
    tmp93 = tl.where(tmp53, tmp68, tmp92)
    tmp94 = tmp93 + tmp46
    tmp95 = tmp93 < 0
    tmp96 = tl.where(tmp95, tmp94, tmp93)
    tl.device_assert(((0 <= tmp96) & (tmp96 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp96 < 4")
    tmp98 = tl.load(in_ptr0 + (tmp96 + 4*tmp49 + 16*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp98, xmask)
