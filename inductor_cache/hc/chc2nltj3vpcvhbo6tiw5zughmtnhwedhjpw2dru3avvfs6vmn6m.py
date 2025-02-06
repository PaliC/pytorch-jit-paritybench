
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_elu_mul_sub_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_elu_mul_sub_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr9 + (x3), None)
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 1.0
    tmp17 = tmp13 * tmp16
    tmp18 = libdevice.expm1(tmp17)
    tmp19 = tmp18 * tmp16
    tmp20 = tl.where(tmp15, tmp17, tmp19)
    tmp22 = tmp21 + tmp1
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tl.load(in_ptr2 + (tmp8 + 32*tmp24 + 1024*x2), None, eviction_policy='evict_last')
    tmp26 = tmp25 - tmp10
    tmp27 = tmp26 * tmp12
    tmp28 = tmp27 > tmp14
    tmp29 = tmp27 * tmp16
    tmp30 = libdevice.expm1(tmp29)
    tmp31 = tmp30 * tmp16
    tmp32 = tl.where(tmp28, tmp29, tmp31)
    tmp34 = tmp33 + tmp1
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tmp37 = tl.load(in_ptr2 + (tmp36 + 32*tmp24 + 1024*x2), None, eviction_policy='evict_last')
    tmp38 = tmp37 - tmp10
    tmp39 = tmp38 * tmp12
    tmp40 = tmp39 > tmp14
    tmp41 = tmp39 * tmp16
    tmp42 = libdevice.expm1(tmp41)
    tmp43 = tmp42 * tmp16
    tmp44 = tl.where(tmp40, tmp41, tmp43)
    tmp45 = tmp44 - tmp32
    tmp47 = tmp45 * tmp46
    tmp48 = tmp32 + tmp47
    tmp49 = tl.load(in_ptr2 + (tmp36 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp50 = tmp49 - tmp10
    tmp51 = tmp50 * tmp12
    tmp52 = tmp51 > tmp14
    tmp53 = tmp51 * tmp16
    tmp54 = libdevice.expm1(tmp53)
    tmp55 = tmp54 * tmp16
    tmp56 = tl.where(tmp52, tmp53, tmp55)
    tmp57 = tmp56 - tmp20
    tmp58 = tmp57 * tmp46
    tmp59 = tmp20 + tmp58
    tmp60 = tmp59 - tmp48
    tmp62 = tmp60 * tmp61
    tmp63 = tmp48 + tmp62
    tmp65 = tmp63 + tmp64
    tl.store(in_out_ptr0 + (x3), tmp65, None)
