
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_clone_relu_threshold_backward_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_clone_relu_threshold_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x4 = xindex // 4
    x6 = xindex
    x2 = ((xindex // 4) % 4)
    x3 = xindex // 16
    x5 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (4 + x0 + 3*x1 + 9*x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x6), xmask)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr6 + (x6), xmask)
    tmp78 = tl.load(in_ptr7 + (x6), xmask)
    tmp79 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.001
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tmp18 = (-1) + x1
    tmp19 = tl.full([1], 0, tl.int64)
    tmp20 = tmp18 >= tmp19
    tmp21 = tl.full([1], 2, tl.int64)
    tmp22 = tmp18 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = (-1) + x0
    tmp25 = tmp24 >= tmp19
    tmp26 = tmp24 < tmp21
    tmp27 = tmp25 & tmp26
    tmp28 = tmp23 & tmp27
    tmp29 = tl.load(in_ptr6 + ((-3) + x6), tmp28 & xmask, other=0.0)
    tmp30 = x0
    tmp31 = tmp30 >= tmp19
    tmp32 = tmp30 < tmp21
    tmp33 = tmp31 & tmp32
    tmp34 = tmp23 & tmp33
    tmp35 = tl.load(in_ptr6 + ((-2) + x6), tmp34 & xmask, other=0.0)
    tmp36 = tmp35 + tmp29
    tmp37 = 1 + x0
    tmp38 = tmp37 >= tmp19
    tmp39 = tmp37 < tmp21
    tmp40 = tmp38 & tmp39
    tmp41 = tmp23 & tmp40
    tmp42 = tl.load(in_ptr6 + ((-1) + x6), tmp41 & xmask, other=0.0)
    tmp43 = tmp42 + tmp36
    tmp44 = x1
    tmp45 = tmp44 >= tmp19
    tmp46 = tmp44 < tmp21
    tmp47 = tmp45 & tmp46
    tmp48 = tmp47 & tmp27
    tmp49 = tl.load(in_ptr6 + ((-1) + x6), tmp48 & xmask, other=0.0)
    tmp50 = tmp49 + tmp43
    tmp51 = tmp47 & tmp33
    tmp52 = tl.load(in_ptr6 + (x6), tmp51 & xmask, other=0.0)
    tmp53 = tmp52 + tmp50
    tmp54 = tmp47 & tmp40
    tmp55 = tl.load(in_ptr6 + (1 + x6), tmp54 & xmask, other=0.0)
    tmp56 = tmp55 + tmp53
    tmp57 = 1 + x1
    tmp58 = tmp57 >= tmp19
    tmp59 = tmp57 < tmp21
    tmp60 = tmp58 & tmp59
    tmp61 = tmp60 & tmp27
    tmp62 = tl.load(in_ptr6 + (1 + x6), tmp61 & xmask, other=0.0)
    tmp63 = tmp62 + tmp56
    tmp64 = tmp60 & tmp33
    tmp65 = tl.load(in_ptr6 + (2 + x6), tmp64 & xmask, other=0.0)
    tmp66 = tmp65 + tmp63
    tmp67 = tmp60 & tmp40
    tmp68 = tl.load(in_ptr6 + (3 + x6), tmp67 & xmask, other=0.0)
    tmp69 = tmp68 + tmp66
    tmp70 = 4 + ((-2)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) + ((-2)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))
    tmp71 = tmp69 / tmp70
    tmp72 = tmp71 + tmp17
    tmp74 = tl.full([1], 0, tl.int32)
    tmp75 = triton_helpers.maximum(tmp74, tmp73)
    tmp76 = 0.0
    tmp77 = tmp75 <= tmp76
    tmp80 = tmp78 - tmp79
    tmp82 = tmp81 + tmp5
    tmp83 = libdevice.sqrt(tmp82)
    tmp84 = tmp8 / tmp83
    tmp85 = tmp84 * tmp10
    tmp86 = tmp80 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp91 = tmp90 + tmp0
    tl.store(out_ptr0 + (x5 + 64*x3), tmp17, xmask)
    tl.store(out_ptr2 + (x5 + 64*x3), tmp72, xmask)
    tl.store(out_ptr3 + (x6), tmp77, xmask)
    tl.store(out_ptr4 + (x5 + 64*x3), tmp91, xmask)
