
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_sub_6', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_sub_6(in_out_ptr1, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 256)
    x0 = (xindex % 256)
    x2 = xindex // 65536
    x6 = xindex
    x4 = ((xindex // 65536) % 3)
    tmp44 = tl.load(in_out_ptr1 + (x6), None)
    tmp45 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12 * tmp4
    tmp14 = tmp13 - tmp2
    tmp15 = triton_helpers.maximum(tmp14, tmp7)
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.load(in_ptr0 + (tmp16 + 64*tmp9 + 4096*x2), None, eviction_policy='evict_last')
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full([1], 63, tl.int64)
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tl.load(in_ptr0 + (tmp21 + 64*tmp9 + 4096*x2), None, eviction_policy='evict_last')
    tmp23 = tmp22 - tmp17
    tmp24 = tmp16.to(tl.float32)
    tmp25 = tmp15 - tmp24
    tmp26 = triton_helpers.maximum(tmp25, tmp7)
    tmp27 = 1.0
    tmp28 = triton_helpers.minimum(tmp26, tmp27)
    tmp29 = tmp23 * tmp28
    tmp30 = tmp17 + tmp29
    tmp31 = tmp9 + tmp18
    tmp32 = triton_helpers.minimum(tmp31, tmp20)
    tmp33 = tl.load(in_ptr0 + (tmp21 + 64*tmp32 + 4096*x2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (tmp16 + 64*tmp32 + 4096*x2), None, eviction_policy='evict_last')
    tmp35 = tmp33 - tmp34
    tmp36 = tmp35 * tmp28
    tmp37 = tmp34 + tmp36
    tmp38 = tmp37 - tmp30
    tmp39 = tmp9.to(tl.float32)
    tmp40 = tmp8 - tmp39
    tmp41 = triton_helpers.maximum(tmp40, tmp7)
    tmp42 = triton_helpers.minimum(tmp41, tmp27)
    tmp43 = tmp38 * tmp42
    tmp46 = tmp44 + tmp45
    tmp47 = tmp30 + tmp43
    tmp48 = tmp46 + tmp47
    tl.store(in_out_ptr1 + (x6), tmp48, None)
