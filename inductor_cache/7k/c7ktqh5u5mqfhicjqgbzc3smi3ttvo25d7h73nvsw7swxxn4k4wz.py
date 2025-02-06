
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr2': '*fp32', 'out_ptr5': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_relu_sub_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr2, out_ptr5, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr9 + (0))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp67 = tl.load(in_ptr11 + (0))
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK])
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp12 = tmp9 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = tmp15 + tmp1
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr2 + (tmp18 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp20 = tmp19 + tmp11
    tmp21 = triton_helpers.maximum(tmp13, tmp20)
    tmp22 = tmp21 - tmp14
    tmp24 = tmp22 * tmp23
    tmp25 = tmp14 + tmp24
    tmp27 = tmp26 + tmp1
    tmp28 = tmp26 < 0
    tmp29 = tl.where(tmp28, tmp27, tmp26)
    tmp30 = tl.load(in_ptr2 + (tmp8 + 8*tmp29 + 64*x2), None, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp11
    tmp32 = triton_helpers.maximum(tmp13, tmp31)
    tmp33 = tl.load(in_ptr2 + (tmp18 + 8*tmp29 + 64*x2), None, eviction_policy='evict_last')
    tmp34 = tmp33 + tmp11
    tmp35 = triton_helpers.maximum(tmp13, tmp34)
    tmp36 = tmp35 - tmp32
    tmp37 = tmp36 * tmp23
    tmp38 = tmp32 + tmp37
    tmp39 = tmp38 - tmp25
    tmp41 = tmp39 * tmp40
    tmp42 = tmp25 + tmp41
    tmp43 = tl.load(in_ptr8 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp46 = tmp43 + tmp45
    tmp47 = triton_helpers.maximum(tmp13, tmp46)
    tmp48 = tl.load(in_ptr8 + (tmp18 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp49 = tmp48 + tmp45
    tmp50 = triton_helpers.maximum(tmp13, tmp49)
    tmp51 = tmp50 - tmp47
    tmp52 = tmp51 * tmp23
    tmp53 = tmp47 + tmp52
    tmp54 = tl.load(in_ptr8 + (tmp8 + 8*tmp29 + 64*x2), None, eviction_policy='evict_last')
    tmp55 = tmp54 + tmp45
    tmp56 = triton_helpers.maximum(tmp13, tmp55)
    tmp57 = tl.load(in_ptr8 + (tmp18 + 8*tmp29 + 64*x2), None, eviction_policy='evict_last')
    tmp58 = tmp57 + tmp45
    tmp59 = triton_helpers.maximum(tmp13, tmp58)
    tmp60 = tmp59 - tmp56
    tmp61 = tmp60 * tmp23
    tmp62 = tmp56 + tmp61
    tmp63 = tmp62 - tmp53
    tmp64 = tmp63 * tmp40
    tmp65 = tmp53 + tmp64
    tmp66 = tl.load(in_ptr10 + (tmp8 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp69 = tmp66 + tmp68
    tmp70 = triton_helpers.maximum(tmp13, tmp69)
    tmp71 = tl.load(in_ptr10 + (tmp18 + 8*tmp4 + 64*x2), None, eviction_policy='evict_last')
    tmp72 = tmp71 + tmp68
    tmp73 = triton_helpers.maximum(tmp13, tmp72)
    tmp74 = tmp73 - tmp70
    tmp75 = tmp74 * tmp23
    tmp76 = tmp70 + tmp75
    tmp77 = tl.load(in_ptr10 + (tmp8 + 8*tmp29 + 64*x2), None, eviction_policy='evict_last')
    tmp78 = tmp77 + tmp68
    tmp79 = triton_helpers.maximum(tmp13, tmp78)
    tmp80 = tl.load(in_ptr10 + (tmp18 + 8*tmp29 + 64*x2), None, eviction_policy='evict_last')
    tmp81 = tmp80 + tmp68
    tmp82 = triton_helpers.maximum(tmp13, tmp81)
    tmp83 = tmp82 - tmp79
    tmp84 = tmp83 * tmp23
    tmp85 = tmp79 + tmp84
    tmp86 = tmp85 - tmp76
    tmp87 = tmp86 * tmp40
    tmp88 = tmp76 + tmp87
    tl.store(out_ptr2 + (13*x3), tmp42, None)
    tl.store(out_ptr5 + (13*x3), tmp65, None)
    tl.store(out_ptr8 + (13*x3), tmp88, None)
