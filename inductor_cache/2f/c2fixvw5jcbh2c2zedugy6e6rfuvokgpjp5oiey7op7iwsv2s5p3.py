
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 1088) % 8)
    x1 = ((xindex // 64) % 17)
    x5 = (xindex % 1088)
    x6 = xindex // 1088
    x7 = xindex
    x0 = (xindex % 64)
    tmp31 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-64) + x5 + 2048*x6), tmp10 & xmask, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (x5 + 2048*x6), tmp16 & xmask, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x2
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp22 & tmp9
    tmp24 = tl.load(in_ptr0 + (960 + x5 + 2048*x6), tmp23 & xmask, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = tmp22 & tmp15
    tmp27 = tl.load(in_ptr0 + (1024 + x5 + 2048*x6), tmp26 & xmask, other=0.0)
    tmp28 = tmp27 + tmp25
    tmp29 = ((-2)*x2) + ((16) * ((16) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (16)))*((17) * ((17) <= (1 + x1)) + (1 + x1) * ((1 + x1) < (17))) + ((-1)*x1*((16) * ((16) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (16)))) + ((-2)*x2*((17) * ((17) <= (1 + x1)) + (1 + x1) * ((1 + x1) < (17)))) + 2*x1*x2 + ((16) * ((16) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (16)))
    tmp30 = tmp28 / tmp29
    tmp32 = tmp30 - tmp31
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.sqrt(tmp35)
    tmp37 = tl.full([1], 1, tl.int32)
    tmp38 = tmp37 / tmp36
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp32 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tl.full([1], 0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tmp49 = tmp30 - tmp48
    tmp51 = tmp50 + tmp34
    tmp52 = libdevice.sqrt(tmp51)
    tmp53 = tmp37 / tmp52
    tmp54 = tmp53 * tmp39
    tmp55 = tmp49 * tmp54
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tl.store(out_ptr0 + (x7), tmp30, xmask)
    tl.store(out_ptr1 + (x7), tmp47, xmask)
    tl.store(out_ptr2 + (x7), tmp59, xmask)
