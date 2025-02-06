
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr2': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_14', 'mutated_arg_names': ['in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_14(in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 256)
    x0 = (xindex % 256)
    x6 = xindex // 65536
    x2 = ((xindex // 65536) % 256)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_out_ptr2 + (x4), None)
    tmp48 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr10 + (x4), None)
    tmp51 = tl.load(in_ptr11 + (x2), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 128, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 128*tmp4 + 16384*x6), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr4 + (tmp8 + 128*tmp4 + 16384*x6), None, eviction_policy='evict_last')
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp17 = tmp16 + tmp1
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr2 + (tmp8 + 128*tmp19 + 16384*x6), None, eviction_policy='evict_last')
    tmp21 = tmp20 + tmp10
    tmp22 = tl.load(in_ptr4 + (tmp8 + 128*tmp19 + 16384*x6), None, eviction_policy='evict_last')
    tmp23 = tmp22 + tmp13
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 + tmp1
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr2 + (tmp28 + 128*tmp19 + 16384*x6), None, eviction_policy='evict_last')
    tmp30 = tmp29 + tmp10
    tmp31 = tl.load(in_ptr4 + (tmp28 + 128*tmp19 + 16384*x6), None, eviction_policy='evict_last')
    tmp32 = tmp31 + tmp13
    tmp33 = tmp30 + tmp32
    tmp34 = tmp33 - tmp24
    tmp36 = tmp34 * tmp35
    tmp37 = tmp24 + tmp36
    tmp38 = tl.load(in_ptr2 + (tmp28 + 128*tmp4 + 16384*x6), None, eviction_policy='evict_last')
    tmp39 = tmp38 + tmp10
    tmp40 = tl.load(in_ptr4 + (tmp28 + 128*tmp4 + 16384*x6), None, eviction_policy='evict_last')
    tmp41 = tmp40 + tmp13
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42 - tmp15
    tmp44 = tmp43 * tmp35
    tmp45 = tmp15 + tmp44
    tmp46 = tmp45 - tmp37
    tmp49 = tmp47 + tmp48
    tmp52 = tmp50 + tmp51
    tmp53 = tmp49 + tmp52
    tmp55 = tmp46 * tmp54
    tmp56 = tmp37 + tmp55
    tmp57 = tmp53 + tmp56
    tmp58 = tl.full([1], 0, tl.int32)
    tmp59 = triton_helpers.maximum(tmp58, tmp57)
    tl.store(in_out_ptr2 + (x4), tmp57, None)
    tl.store(out_ptr0 + (x4), tmp59, None)
