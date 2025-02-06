
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4) % 1280)
    x3 = xindex // 5120
    x4 = (xindex % 4)
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 4*(x2) + 1024*x3), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 1, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (256*x3 + (x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.load(in_ptr5 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 + tmp7
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tmp19 - tmp19
    tmp25 = tl.load(in_ptr6 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp19 + tmp26
    tmp28 = tmp27 - tmp5
    tmp29 = tl.load(in_ptr7 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp5 + tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp4, tmp31, tmp32)
    tmp34 = tmp0 >= tmp3
    tmp35 = tl.full([1], 512, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tl.load(in_ptr8 + (x4 + 4*((-256) + x2) + 1024*x3), tmp37, other=0.0)
    tmp39 = tl.load(in_ptr9 + ((-256) + x2), tmp37, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 + tmp39
    tmp41 = tl.full([1], 0, tl.int32)
    tmp42 = triton_helpers.maximum(tmp41, tmp40)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp37, tmp42, tmp43)
    tmp45 = tmp0 >= tmp35
    tmp46 = tl.full([1], 768, tl.int64)
    tmp47 = tmp0 < tmp46
    tmp48 = tmp45 & tmp47
    tmp49 = tl.load(in_ptr10 + (x4 + 4*((-512) + x2) + 1024*x3), tmp48, other=0.0)
    tmp50 = tl.load(in_ptr11 + ((-512) + x2), tmp48, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 + tmp50
    tmp52 = tl.full([1], 0, tl.int32)
    tmp53 = triton_helpers.maximum(tmp52, tmp51)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp48, tmp53, tmp54)
    tmp56 = tmp0 >= tmp46
    tmp57 = tl.full([1], 1024, tl.int64)
    tmp58 = tmp0 < tmp57
    tmp59 = tmp56 & tmp58
    tmp60 = tl.load(in_ptr12 + (x4 + 4*((-768) + x2) + 1024*x3), tmp59, other=0.0)
    tmp61 = tl.load(in_ptr13 + ((-768) + x2), tmp59, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp60 + tmp61
    tmp63 = tl.full([1], 0, tl.int32)
    tmp64 = triton_helpers.maximum(tmp63, tmp62)
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp59, tmp64, tmp65)
    tmp67 = tmp0 >= tmp57
    tmp68 = tl.full([1], 1280, tl.int64)
    tmp69 = tmp0 < tmp68
    tmp70 = tl.load(in_ptr14 + (x4 + 4*((-1024) + x2) + 1024*x3), tmp67, other=0.0)
    tmp71 = tl.load(in_ptr15 + ((-1024) + x2), tmp67, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = tl.full([1], 0, tl.int32)
    tmp74 = triton_helpers.maximum(tmp73, tmp72)
    tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
    tmp76 = tl.where(tmp67, tmp74, tmp75)
    tmp77 = tl.where(tmp59, tmp66, tmp76)
    tmp78 = tl.where(tmp48, tmp55, tmp77)
    tmp79 = tl.where(tmp37, tmp44, tmp78)
    tmp80 = tl.where(tmp4, tmp33, tmp79)
    tl.store(out_ptr0 + (x5), tmp80, None)
