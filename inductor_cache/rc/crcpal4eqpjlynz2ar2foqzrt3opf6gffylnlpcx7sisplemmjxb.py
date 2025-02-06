
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x1 = ((xindex // 256) % 32)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x6 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x5), None)
    tmp22 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr12 + (x4), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (x3), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr14 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr15 + (x1), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr17 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr18 + (x4), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr18 + (x3), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr20 + (x1), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr21 + (x1), None, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr22 + (x1), None, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr23 + (x1), None, eviction_policy='evict_last')
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
    tmp20 = 0.0
    tmp21 = tmp19 + tmp20
    tmp23 = tl.full([XBLOCK], 8, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp22 < 0
    tmp26 = tl.where(tmp25, tmp24, tmp22)
    tmp28 = tmp27 + tmp23
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr7 + (tmp30 + 8*tmp26 + 64*x6), None, eviction_policy='evict_last')
    tmp33 = tmp31 - tmp32
    tmp35 = tmp34 + tmp4
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tmp7 / tmp36
    tmp38 = tmp37 * tmp9
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tmp21 + tmp43
    tmp46 = tl.full([XBLOCK], 4, tl.int32)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp45 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp45)
    tmp51 = tmp50 + tmp46
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr13 + (tmp53 + 4*tmp49 + 16*x6), None, eviction_policy='evict_last')
    tmp56 = tmp54 - tmp55
    tmp58 = tmp57 + tmp4
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tmp7 / tmp59
    tmp61 = tmp60 * tmp9
    tmp62 = tmp56 * tmp61
    tmp64 = tmp62 * tmp63
    tmp66 = tmp64 + tmp65
    tmp67 = tmp44 + tmp66
    tmp69 = tl.full([XBLOCK], 2, tl.int32)
    tmp70 = tmp68 + tmp69
    tmp71 = tmp68 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp68)
    tmp74 = tmp73 + tmp69
    tmp75 = tmp73 < 0
    tmp76 = tl.where(tmp75, tmp74, tmp73)
    tmp77 = tl.load(in_ptr19 + (tmp76 + 2*tmp72 + 4*x6), None, eviction_policy='evict_last')
    tmp79 = tmp77 - tmp78
    tmp81 = tmp80 + tmp4
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tmp7 / tmp82
    tmp84 = tmp83 * tmp9
    tmp85 = tmp79 * tmp84
    tmp87 = tmp85 * tmp86
    tmp89 = tmp87 + tmp88
    tmp90 = tmp67 + tmp89
    tmp91 = triton_helpers.maximum(tmp18, tmp90)
    tl.store(out_ptr0 + (x5), tmp19, None)
    tl.store(in_out_ptr0 + (x5), tmp91, None)
