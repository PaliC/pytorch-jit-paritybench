
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*i64', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x6 = xindex
    x1 = ((xindex // 256) % 4)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x5 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x6), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x6), None)
    tmp20 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x4), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr13 + (x4), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr14 + (x3), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x3), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr17 + (x3), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr19 + (x4), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr8 + (tmp28 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp21
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tmp34 = tl.load(in_ptr8 + (tmp33 + 8*tmp24 + 64*x5), None, eviction_policy='evict_last')
    tmp35 = tmp34 - tmp29
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 + tmp37
    tmp40 = tmp39 + tmp21
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.load(in_ptr8 + (tmp28 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr8 + (tmp33 + 8*tmp42 + 64*x5), None, eviction_policy='evict_last')
    tmp45 = tmp44 - tmp43
    tmp46 = tmp45 * tmp36
    tmp47 = tmp43 + tmp46
    tmp48 = tmp47 - tmp38
    tmp50 = tmp48 * tmp49
    tmp51 = tmp38 + tmp50
    tmp52 = tmp19 + tmp51
    tmp54 = tl.full([XBLOCK], 4, tl.int32)
    tmp55 = tmp53 + tmp54
    tmp56 = tmp53 < 0
    tmp57 = tl.where(tmp56, tmp55, tmp53)
    tmp59 = tmp58 + tmp54
    tmp60 = tmp58 < 0
    tmp61 = tl.where(tmp60, tmp59, tmp58)
    tmp62 = tl.load(in_ptr15 + (tmp61 + 4*tmp57 + 16*x5), None, eviction_policy='evict_last')
    tmp64 = tmp63 + tmp54
    tmp65 = tmp63 < 0
    tmp66 = tl.where(tmp65, tmp64, tmp63)
    tmp67 = tl.load(in_ptr15 + (tmp66 + 4*tmp57 + 16*x5), None, eviction_policy='evict_last')
    tmp68 = tmp67 - tmp62
    tmp70 = tmp68 * tmp69
    tmp71 = tmp62 + tmp70
    tmp73 = tmp72 + tmp54
    tmp74 = tmp72 < 0
    tmp75 = tl.where(tmp74, tmp73, tmp72)
    tmp76 = tl.load(in_ptr15 + (tmp61 + 4*tmp75 + 16*x5), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr15 + (tmp66 + 4*tmp75 + 16*x5), None, eviction_policy='evict_last')
    tmp78 = tmp77 - tmp76
    tmp79 = tmp78 * tmp69
    tmp80 = tmp76 + tmp79
    tmp81 = tmp80 - tmp71
    tmp83 = tmp81 * tmp82
    tmp84 = tmp71 + tmp83
    tmp85 = tmp52 + tmp84
    tmp86 = triton_helpers.maximum(tmp18, tmp85)
    tl.store(out_ptr0 + (x6), tmp19, None)
    tl.store(in_out_ptr0 + (x6), tmp86, None)
