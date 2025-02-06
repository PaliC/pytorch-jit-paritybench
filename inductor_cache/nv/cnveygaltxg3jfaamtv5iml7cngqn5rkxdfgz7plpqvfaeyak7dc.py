
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_native_batch_norm_backward_relu_sub_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_native_batch_norm_backward_relu_sub_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = ((xindex // 14) % 14)
    x1 = (xindex % 14)
    x3 = xindex // 196
    x5 = ((xindex // 196) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr11 + (x5), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr12 + (x5), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr13 + (x5), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr14 + (x5), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tl.full([XBLOCK], 7, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp11 = tmp10 + tmp6
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr4 + (tmp13 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (tmp13 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (tmp13 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp20 = tmp19 + tmp6
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr4 + (tmp13 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (tmp13 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (tmp13 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp26 = tmp24 + tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp28 + tmp6
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tmp32 = tl.load(in_ptr4 + (tmp31 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (tmp31 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (tmp31 + 7*tmp22 + 49*x3), xmask, eviction_policy='evict_last')
    tmp35 = tmp33 + tmp34
    tmp36 = tmp32 + tmp35
    tmp37 = tmp36 - tmp27
    tmp39 = tmp37 * tmp38
    tmp40 = tmp27 + tmp39
    tmp41 = tl.load(in_ptr4 + (tmp31 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (tmp31 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (tmp31 + 7*tmp9 + 49*x3), xmask, eviction_policy='evict_last')
    tmp44 = tmp42 + tmp43
    tmp45 = tmp41 + tmp44
    tmp46 = tmp45 - tmp18
    tmp47 = tmp46 * tmp38
    tmp48 = tmp18 + tmp47
    tmp49 = tmp48 - tmp40
    tmp51 = tmp49 * tmp50
    tmp52 = tmp40 + tmp51
    tmp53 = tmp52 + tmp4
    tmp55 = tmp53 - tmp54
    tmp57 = 1e-05
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp66 = tmp64 * tmp65
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr0 + (x0), tmp70, xmask)
    tl.store(out_ptr1 + (x0), tmp55, xmask)
