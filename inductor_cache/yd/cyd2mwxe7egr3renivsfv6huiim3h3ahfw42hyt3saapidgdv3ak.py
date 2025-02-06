
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_cat_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x8 = xindex
    x3 = (xindex % 64)
    x4 = xindex // 64
    x6 = ((xindex // 16) % 4)
    tmp72 = tl.load(in_ptr1 + (x8), xmask)
    tmp75 = tl.load(in_ptr2 + (x8), xmask)
    tmp76 = tl.load(in_ptr3 + (x6), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr4 + (x6), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr5 + (x6), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr6 + (x6), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr0 + (x8), xmask)
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + x8), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + x8), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + x8), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x8), tmp30 & xmask, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x8), tmp33 & xmask, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x8), tmp36 & xmask, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + x8), tmp43 & xmask, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + x8), tmp46 & xmask, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + x8), tmp49 & xmask, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0))) + ((4) * ((4) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (4)))*((4) * ((4) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (4))) + ((-1)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((4) * ((4) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (4)))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((4) * ((4) <= (2 + x0)) + (2 + x0) * ((2 + x0) < (4))))
    tmp53 = tmp51 / tmp52
    tmp54 = tl.load(in_ptr1 + ((-5) + x8), tmp10 & xmask, other=0.0)
    tmp55 = tl.load(in_ptr1 + ((-4) + x8), tmp16 & xmask, other=0.0)
    tmp56 = tmp55 + tmp54
    tmp57 = tl.load(in_ptr1 + ((-3) + x8), tmp23 & xmask, other=0.0)
    tmp58 = tmp57 + tmp56
    tmp59 = tl.load(in_ptr1 + ((-1) + x8), tmp30 & xmask, other=0.0)
    tmp60 = tmp59 + tmp58
    tmp61 = tl.load(in_ptr1 + (x8), tmp33 & xmask, other=0.0)
    tmp62 = tmp61 + tmp60
    tmp63 = tl.load(in_ptr1 + (1 + x8), tmp36 & xmask, other=0.0)
    tmp64 = tmp63 + tmp62
    tmp65 = tl.load(in_ptr1 + (3 + x8), tmp43 & xmask, other=0.0)
    tmp66 = tmp65 + tmp64
    tmp67 = tl.load(in_ptr1 + (4 + x8), tmp46 & xmask, other=0.0)
    tmp68 = tmp67 + tmp66
    tmp69 = tl.load(in_ptr1 + (5 + x8), tmp49 & xmask, other=0.0)
    tmp70 = tmp69 + tmp68
    tmp71 = tmp70 / tmp52
    tmp73 = tmp53 + tmp72
    tmp74 = tmp71 + tmp71
    tmp77 = tmp75 - tmp76
    tmp79 = 0.001
    tmp80 = tmp78 + tmp79
    tmp81 = libdevice.sqrt(tmp80)
    tmp82 = tl.full([1], 1, tl.int32)
    tmp83 = tmp82 / tmp81
    tmp84 = 1.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp77 * tmp85
    tmp88 = tmp86 * tmp87
    tmp90 = tmp88 + tmp89
    tmp92 = tmp90 + tmp91
    tl.store(out_ptr2 + (x3 + 384*x4), tmp72, xmask)
    tl.store(out_ptr3 + (x3 + 384*x4), tmp73, xmask)
    tl.store(out_ptr4 + (x3 + 384*x4), tmp74, xmask)
    tl.store(out_ptr5 + (x3 + 384*x4), tmp92, xmask)
