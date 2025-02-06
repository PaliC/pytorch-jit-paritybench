
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_3(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    x4 = xindex // 16
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 32*x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + 32*x2), None, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.3333333333333333
    tmp7 = tmp5 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 - tmp8
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp11 * tmp4
    tmp13 = 1.5
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 0.0
    tmp17 = tmp15 >= tmp16
    tmp18 = 4.0
    tmp19 = tmp15 < tmp18
    tmp20 = tmp1 == tmp1
    tmp21 = tl.where(tmp20, tmp9, tmp3)
    tmp22 = tmp21 * tmp4
    tmp23 = tmp22 + tmp13
    tmp24 = libdevice.floor(tmp23)
    tmp25 = tmp24 >= tmp16
    tmp26 = tmp24 < tmp18
    tmp27 = tmp25 & tmp26
    tmp28 = tmp19 & tmp27
    tmp29 = tmp17 & tmp28
    tmp30 = tmp24.to(tl.int64)
    tmp31 = tl.full([1], 0, tl.int64)
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = tmp15.to(tl.int64)
    tmp34 = tl.where(tmp29, tmp33, tmp31)
    tmp35 = tmp15 + tmp8
    tmp36 = tmp35 - tmp14
    tmp37 = tmp24 + tmp8
    tmp38 = tmp37 - tmp23
    tmp39 = tmp36 * tmp38
    tmp40 = tl.where(tmp29, tmp39, tmp16)
    tmp41 = tmp35 < tmp18
    tmp42 = tmp37 >= tmp16
    tmp43 = tmp37 < tmp18
    tmp44 = tmp42 & tmp43
    tmp45 = tmp41 & tmp44
    tmp46 = tmp19 & tmp44
    tmp47 = tmp17 & tmp46
    tmp48 = tmp35 >= tmp16
    tmp49 = tmp41 & tmp27
    tmp50 = tmp48 & tmp49
    tmp51 = tl.where(tmp50, tmp30, tmp31)
    tmp52 = tl.full([XBLOCK], 4, tl.int32)
    tmp53 = tmp51 + tmp52
    tmp54 = tmp51 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp51)
    tl.device_assert((0 <= tmp55) & (tmp55 < 4), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tmp35.to(tl.int64)
    tmp58 = tl.where(tmp50, tmp57, tmp31)
    tmp59 = tmp58 + tmp52
    tmp60 = tmp58 < 0
    tmp61 = tl.where(tmp60, tmp59, tmp58)
    tl.device_assert((0 <= tmp61) & (tmp61 < 4), "index out of bounds: 0 <= tmp61 < 4")
    tmp63 = tl.load(in_ptr1 + (tmp61 + 4*tmp55 + 16*x4), None, eviction_policy='evict_last')
    tmp64 = tmp14 - tmp15
    tmp65 = tmp64 * tmp38
    tmp66 = tl.where(tmp50, tmp65, tmp16)
    tmp67 = tmp63 * tmp66
    tmp68 = tmp37.to(tl.int64)
    tmp69 = tl.where(tmp47, tmp68, tmp31)
    tmp70 = tmp69 + tmp52
    tmp71 = tmp69 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp69)
    tl.device_assert((0 <= tmp72) & (tmp72 < 4), "index out of bounds: 0 <= tmp72 < 4")
    tmp74 = tl.where(tmp47, tmp33, tmp31)
    tmp75 = tmp74 + tmp52
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tl.device_assert((0 <= tmp77) & (tmp77 < 4), "index out of bounds: 0 <= tmp77 < 4")
    tmp79 = tl.load(in_ptr1 + (tmp77 + 4*tmp72 + 16*x4), None, eviction_policy='evict_last')
    tmp80 = tmp23 - tmp24
    tmp81 = tmp36 * tmp80
    tmp82 = tl.where(tmp47, tmp81, tmp16)
    tmp83 = tmp79 * tmp82
    tmp84 = tmp48 & tmp45
    tmp85 = tmp64 * tmp80
    tmp86 = tl.where(tmp84, tmp85, tmp16)
    tmp87 = tl.where(tmp84, tmp68, tmp31)
    tmp88 = tmp87 + tmp52
    tmp89 = tmp87 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp87)
    tl.device_assert((0 <= tmp90) & (tmp90 < 4), "index out of bounds: 0 <= tmp90 < 4")
    tmp92 = tl.where(tmp84, tmp57, tmp31)
    tmp93 = tmp92 + tmp52
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tl.device_assert((0 <= tmp95) & (tmp95 < 4), "index out of bounds: 0 <= tmp95 < 4")
    tmp97 = tl.load(in_ptr1 + (tmp95 + 4*tmp90 + 16*x4), None, eviction_policy='evict_last')
    tmp98 = tmp97 * tmp86
    tl.store(out_ptr0 + (x3), tmp32, None)
    tl.store(out_ptr1 + (x3), tmp34, None)
    tl.store(out_ptr2 + (x3), tmp40, None)
    tl.store(in_out_ptr0 + (x3), tmp67, None)
    tl.store(in_out_ptr1 + (x3), tmp83, None)
    tl.store(in_out_ptr2 + (x3), tmp98, None)
