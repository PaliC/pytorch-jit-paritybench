
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 17, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = xindex
    x1 = ((xindex // 16) % 72)
    x4 = ((xindex // 4) % 4)
    x3 = (xindex % 4)
    x5 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x6), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x6), xmask)
    tmp19 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr10 + (x4), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + (x3), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr13 + (x3), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr14 + (x3), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr15 + (x6), xmask)
    tmp53 = tl.load(in_ptr16 + (x4), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr17 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp7 / tmp23
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp17 + tmp30
    tmp33 = tl.full([XBLOCK], 2, tl.int32)
    tmp34 = tmp32 + tmp33
    tmp35 = tmp32 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp32)
    tmp38 = tmp37 + tmp33
    tmp39 = tmp37 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tmp41 = tl.load(in_ptr12 + (tmp40 + 2*tmp36 + 4*x5), xmask, eviction_policy='evict_last')
    tmp43 = tmp42 + tmp33
    tmp44 = tmp42 < 0
    tmp45 = tl.where(tmp44, tmp43, tmp42)
    tmp46 = tl.load(in_ptr12 + (tmp45 + 2*tmp36 + 4*x5), xmask, eviction_policy='evict_last')
    tmp47 = tmp46 - tmp41
    tmp49 = tmp47 * tmp48
    tmp50 = tmp41 + tmp49
    tmp52 = tmp31 + tmp51
    tmp54 = tmp53 + tmp33
    tmp55 = tmp53 < 0
    tmp56 = tl.where(tmp55, tmp54, tmp53)
    tmp57 = tl.load(in_ptr12 + (tmp40 + 2*tmp56 + 4*x5), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr12 + (tmp45 + 2*tmp56 + 4*x5), xmask, eviction_policy='evict_last')
    tmp59 = tmp58 - tmp57
    tmp60 = tmp59 * tmp48
    tmp61 = tmp57 + tmp60
    tmp62 = tmp61 - tmp50
    tmp64 = tmp62 * tmp63
    tmp65 = tmp50 + tmp64
    tmp66 = tmp52 + tmp65
    tmp67 = tl.full([1], 0, tl.int32)
    tmp68 = triton_helpers.maximum(tmp67, tmp66)
    tl.store(in_out_ptr0 + (x6), tmp68, xmask)
