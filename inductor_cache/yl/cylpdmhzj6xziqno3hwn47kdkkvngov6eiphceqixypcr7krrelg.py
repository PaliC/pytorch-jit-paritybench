
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_convolution_hardtanh_mul_sub_14(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x6 = xindex
    x4 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_out_ptr1 + (x6), xmask)
    tmp33 = tl.load(in_ptr7 + (x4), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr11 + (x4), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr13 + (x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x4), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr15 + (x4), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr16 + (x4), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr17 + (x4), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr18 + (x1), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr19 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 2*tmp4 + 4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (tmp8 + 2*tmp22 + 4*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (tmp13 + 2*tmp22 + 4*x2), xmask, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tmp34 = tmp32 + tmp33
    tmp36 = tl.full([XBLOCK], 1, tl.int32)
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tmp41 = tmp40 + tmp36
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp46 = tmp44 + tmp45
    tmp48 = tmp47 + tmp36
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tmp51 = tmp46 - tmp46
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 + tmp53
    tmp56 = tmp34 - tmp55
    tmp58 = 1e-05
    tmp59 = tmp57 + tmp58
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tl.full([1], 1, tl.int32)
    tmp62 = tmp61 / tmp60
    tmp63 = 1.0
    tmp64 = tmp62 * tmp63
    tmp65 = tmp56 * tmp64
    tmp67 = tmp65 * tmp66
    tmp69 = tmp67 + tmp68
    tmp70 = 0.0
    tmp71 = triton_helpers.maximum(tmp69, tmp70)
    tmp72 = 6.0
    tmp73 = triton_helpers.minimum(tmp71, tmp72)
    tmp74 = tmp31 * tmp73
    tmp76 = tmp75 + tmp36
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tmp79 = tmp54 - tmp54
    tmp81 = tmp79 * tmp80
    tmp82 = tmp54 + tmp81
    tmp83 = tmp74 + tmp82
    tl.store(in_out_ptr0 + (x6), tmp31, xmask)
    tl.store(in_out_ptr1 + (x6), tmp34, xmask)
    tl.store(in_out_ptr2 + (x6), tmp83, xmask)
