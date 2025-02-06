
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = xindex
    x1 = ((xindex // 4) % 4)
    x4 = ((xindex // 2) % 2)
    x3 = (xindex % 2)
    x5 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x6), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x5), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x4), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr11 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK], 1, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tmp9 = tmp8 + tmp4
    tmp10 = tmp8 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp8)
    tmp14 = tmp13 + tmp4
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tmp17 = tmp12 - tmp12
    tmp19 = tmp17 * tmp18
    tmp20 = tmp12 + tmp19
    tmp22 = tmp2 - tmp21
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.sqrt(tmp25)
    tmp27 = tl.full([1], 1, tl.int32)
    tmp28 = tmp27 / tmp26
    tmp29 = 1.0
    tmp30 = tmp28 * tmp29
    tmp31 = tmp22 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0
    tmp37 = triton_helpers.maximum(tmp35, tmp36)
    tmp38 = 6.0
    tmp39 = triton_helpers.minimum(tmp37, tmp38)
    tmp41 = tmp40 + tmp4
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp44 = tmp20 - tmp20
    tmp46 = tmp44 * tmp45
    tmp47 = tmp20 + tmp46
    tmp48 = tmp39 + tmp47
    tl.store(in_out_ptr0 + (x6), tmp2, xmask)
    tl.store(in_out_ptr1 + (x6), tmp48, xmask)
