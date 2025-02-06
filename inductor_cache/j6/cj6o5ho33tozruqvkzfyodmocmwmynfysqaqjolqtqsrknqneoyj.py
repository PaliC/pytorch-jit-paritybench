
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_native_group_norm_relu_sub_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_native_group_norm_relu_sub_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x6 = xindex
    x4 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr9 + (x6), None)
    tmp61 = tl.load(in_ptr10 + (x2 // 4), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr11 + (x2 // 4), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr13 + (x4), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp19 = tl.load(in_ptr5 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (tmp13 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp21 = tmp20 - tmp19
    tmp22 = tmp21 * tmp16
    tmp23 = tmp19 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tmp28 = tl.load(in_ptr5 + (tmp8 + 8*tmp27 + 64*x2), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr5 + (tmp13 + 8*tmp27 + 64*x2), None, eviction_policy='evict_last')
    tmp30 = tmp29 - tmp28
    tmp31 = tmp30 * tmp16
    tmp32 = tmp28 + tmp31
    tmp33 = tmp32 - tmp23
    tmp35 = tmp33 * tmp34
    tmp36 = tmp23 + tmp35
    tmp37 = tl.load(in_ptr2 + (tmp8 + 8*tmp27 + 64*x2), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr2 + (tmp13 + 8*tmp27 + 64*x2), None, eviction_policy='evict_last')
    tmp39 = tmp38 - tmp37
    tmp40 = tmp39 * tmp16
    tmp41 = tmp37 + tmp40
    tmp42 = tmp41 - tmp18
    tmp43 = tmp42 * tmp34
    tmp44 = tmp18 + tmp43
    tmp45 = tmp36 + tmp44
    tmp46 = tl.load(in_ptr8 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr8 + (tmp13 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp48 = tmp47 - tmp46
    tmp49 = tmp48 * tmp16
    tmp50 = tmp46 + tmp49
    tmp51 = tl.load(in_ptr8 + (tmp8 + 8*tmp27 + 64*x2), None, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr8 + (tmp13 + 8*tmp27 + 64*x2), None, eviction_policy='evict_last')
    tmp53 = tmp52 - tmp51
    tmp54 = tmp53 * tmp16
    tmp55 = tmp51 + tmp54
    tmp56 = tmp55 - tmp50
    tmp57 = tmp56 * tmp34
    tmp58 = tmp50 + tmp57
    tmp59 = tmp45 + tmp58
    tmp62 = tmp60 - tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 * tmp65
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tmp59 + tmp70
    tl.store(in_out_ptr0 + (x6), tmp71, None)
