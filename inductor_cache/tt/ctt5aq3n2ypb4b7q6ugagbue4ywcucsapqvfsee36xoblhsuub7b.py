
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*i64', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = xindex // 64
    x5 = xindex
    x3 = ((xindex // 64) % 96)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x5), None)
    tmp20 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr10 + (x5), None)
    tmp37 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr16 + (x0), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr17 + (x0), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr18 + (x1), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr19 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 4*tmp4 + 16*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp21 = tmp19 - tmp20
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp36 = tmp34 + tmp35
    tmp38 = tmp37 + tmp1
    tmp39 = tmp37 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp37)
    tmp41 = tl.load(in_ptr2 + (tmp8 + 4*tmp40 + 16*x2), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr2 + (tmp13 + 4*tmp40 + 16*x2), None, eviction_policy='evict_last')
    tmp43 = tmp42 - tmp41
    tmp44 = tmp43 * tmp16
    tmp45 = tmp41 + tmp44
    tmp46 = tmp45 - tmp18
    tmp48 = tmp46 * tmp47
    tmp49 = tmp18 + tmp48
    tmp50 = tmp36 + tmp49
    tmp52 = tl.full([XBLOCK], 2, tl.int32)
    tmp53 = tmp51 + tmp52
    tmp54 = tmp51 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp51)
    tmp57 = tmp56 + tmp52
    tmp58 = tmp56 < 0
    tmp59 = tl.where(tmp58, tmp57, tmp56)
    tmp60 = tl.load(in_ptr15 + (tmp59 + 2*tmp55 + 4*x2), None, eviction_policy='evict_last')
    tmp62 = tmp61 + tmp52
    tmp63 = tmp61 < 0
    tmp64 = tl.where(tmp63, tmp62, tmp61)
    tmp65 = tl.load(in_ptr15 + (tmp64 + 2*tmp55 + 4*x2), None, eviction_policy='evict_last')
    tmp66 = tmp65 - tmp60
    tmp68 = tmp66 * tmp67
    tmp69 = tmp60 + tmp68
    tmp71 = tmp70 + tmp52
    tmp72 = tmp70 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp70)
    tmp74 = tl.load(in_ptr15 + (tmp59 + 2*tmp73 + 4*x2), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr15 + (tmp64 + 2*tmp73 + 4*x2), None, eviction_policy='evict_last')
    tmp76 = tmp75 - tmp74
    tmp77 = tmp76 * tmp67
    tmp78 = tmp74 + tmp77
    tmp79 = tmp78 - tmp69
    tmp81 = tmp79 * tmp80
    tmp82 = tmp69 + tmp81
    tmp83 = tmp50 + tmp82
    tmp84 = tl.full([1], 0, tl.int32)
    tmp85 = triton_helpers.maximum(tmp84, tmp83)
    tmp86 = 0.0
    tmp87 = tmp85 <= tmp86
    tl.store(in_out_ptr0 + (x5), tmp83, None)
    tl.store(out_ptr1 + (x5), tmp87, None)
