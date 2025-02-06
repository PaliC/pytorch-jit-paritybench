
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_max_pool2d_with_indices_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x0 = (xindex % 1024)
    x3 = ((xindex // 8192) % 8)
    x2 = ((xindex // 1024) % 8)
    x7 = xindex // 8192
    tmp0 = tl.load(in_out_ptr0 + (x5), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = (-1) + 2*x3
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 16, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = (-1) + 2*x2
    tmp10 = tmp9 >= tmp4
    tmp11 = tmp9 < tmp6
    tmp12 = tmp10 & tmp11
    tmp13 = tmp8 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-17408) + x0 + 2048*x2 + 32768*x7), tmp13, other=float("-inf"))
    tmp15 = 2*x2
    tmp16 = tmp15 >= tmp4
    tmp17 = tmp15 < tmp6
    tmp18 = tmp16 & tmp17
    tmp19 = tmp8 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-16384) + x0 + 2048*x2 + 32768*x7), tmp19, other=float("-inf"))
    tmp21 = triton_helpers.maximum(tmp20, tmp14)
    tmp22 = 1 + 2*x2
    tmp23 = tmp22 >= tmp4
    tmp24 = tmp22 < tmp6
    tmp25 = tmp23 & tmp24
    tmp26 = tmp8 & tmp25
    tmp27 = tl.load(in_ptr1 + ((-15360) + x0 + 2048*x2 + 32768*x7), tmp26, other=float("-inf"))
    tmp28 = triton_helpers.maximum(tmp27, tmp21)
    tmp29 = 2*x3
    tmp30 = tmp29 >= tmp4
    tmp31 = tmp29 < tmp6
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp12
    tmp34 = tl.load(in_ptr1 + ((-1024) + x0 + 2048*x2 + 32768*x7), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp28)
    tmp36 = tmp32 & tmp18
    tmp37 = tl.load(in_ptr1 + (x0 + 2048*x2 + 32768*x7), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = tmp32 & tmp25
    tmp40 = tl.load(in_ptr1 + (1024 + x0 + 2048*x2 + 32768*x7), tmp39, other=float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp38)
    tmp42 = 1 + 2*x3
    tmp43 = tmp42 >= tmp4
    tmp44 = tmp42 < tmp6
    tmp45 = tmp43 & tmp44
    tmp46 = tmp45 & tmp12
    tmp47 = tl.load(in_ptr1 + (15360 + x0 + 2048*x2 + 32768*x7), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp41)
    tmp49 = tmp45 & tmp18
    tmp50 = tl.load(in_ptr1 + (16384 + x0 + 2048*x2 + 32768*x7), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp45 & tmp25
    tmp53 = tl.load(in_ptr1 + (17408 + x0 + 2048*x2 + 32768*x7), tmp52, other=float("-inf"))
    tmp54 = triton_helpers.maximum(tmp53, tmp51)
    tmp55 = tmp20 > tmp14
    tmp56 = tl.full([1], 1, tl.int8)
    tmp57 = tl.full([1], 0, tl.int8)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp27 > tmp21
    tmp60 = tl.full([1], 2, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp28
    tmp63 = tl.full([1], 3, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 4, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp40 > tmp38
    tmp69 = tl.full([1], 5, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp41
    tmp72 = tl.full([1], 6, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 7, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp53 > tmp51
    tmp78 = tl.full([1], 8, tl.int8)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp81 = tmp2 - tmp80
    tmp83 = 1e-05
    tmp84 = tmp82 + tmp83
    tmp85 = libdevice.sqrt(tmp84)
    tmp86 = tl.full([1], 1, tl.int32)
    tmp87 = tmp86 / tmp85
    tmp88 = 1.0
    tmp89 = tmp87 * tmp88
    tmp90 = tmp81 * tmp89
    tmp92 = tmp90 * tmp91
    tmp94 = tmp92 + tmp93
    tmp95 = tmp94 + tmp54
    tl.store(in_out_ptr0 + (x5), tmp2, None)
    tl.store(out_ptr0 + (x5), tmp79, None)
    tl.store(in_out_ptr1 + (x5), tmp95, None)
