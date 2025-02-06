
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
    triton_meta={'signature': {'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_relu_sub_2', 'mutated_arg_names': ['in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_relu_sub_2(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x6 = xindex
    x4 = ((xindex // 256) % 4)
    tmp44 = tl.load(in_out_ptr1 + (x6), None)
    tmp45 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (x6), None)
    tmp48 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 7, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 + tmp2
    tmp16 = tmp15 * tmp2
    tmp17 = tmp16 - tmp2
    tmp18 = triton_helpers.maximum(tmp17, tmp6)
    tmp19 = tmp18.to(tl.int32)
    tmp20 = tmp19 + tmp9
    tmp21 = triton_helpers.minimum(tmp20, tmp11)
    tmp22 = tl.load(in_ptr0 + (tmp21 + 8*tmp12 + 64*x2), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (tmp19 + 8*tmp12 + 64*x2), None, eviction_policy='evict_last')
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19.to(tl.float32)
    tmp26 = tmp18 - tmp25
    tmp27 = triton_helpers.maximum(tmp26, tmp6)
    tmp28 = 1.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp24 * tmp29
    tmp31 = tmp23 + tmp30
    tmp32 = tl.load(in_ptr0 + (tmp19 + 8*tmp8 + 64*x2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (tmp21 + 8*tmp8 + 64*x2), None, eviction_policy='evict_last')
    tmp34 = tmp33 - tmp32
    tmp35 = tmp34 * tmp29
    tmp36 = tmp32 + tmp35
    tmp37 = tmp31 - tmp36
    tmp38 = tmp8.to(tl.float32)
    tmp39 = tmp7 - tmp38
    tmp40 = triton_helpers.maximum(tmp39, tmp6)
    tmp41 = triton_helpers.minimum(tmp40, tmp28)
    tmp42 = tmp37 * tmp41
    tmp43 = tmp36 + tmp42
    tmp46 = tmp44 + tmp45
    tmp49 = tmp47 + tmp48
    tmp50 = tmp46 + tmp49
    tmp51 = tmp50 + tmp43
    tmp52 = tl.full([1], 0, tl.int32)
    tmp53 = triton_helpers.maximum(tmp52, tmp51)
    tl.store(in_out_ptr1 + (x6), tmp51, None)
    tl.store(out_ptr0 + (x6), tmp53, None)
