
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 4)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x3 = xindex // 16384
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr9 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tl.full([XBLOCK], 16, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp11 = tmp10 + tmp6
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr3 + (tmp13 + 16*tmp9 + 256*tmp4 + 1024*x3), None, eviction_policy='evict_last')
    tmp16 = tmp15 + tmp6
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr3 + (tmp18 + 16*tmp9 + 256*tmp4 + 1024*x3), None, eviction_policy='evict_last')
    tmp20 = tmp19 - tmp14
    tmp22 = tmp20 * tmp21
    tmp23 = tmp14 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tmp28 = tl.load(in_ptr3 + (tmp13 + 16*tmp9 + 256*tmp27 + 1024*x3), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (tmp18 + 16*tmp9 + 256*tmp27 + 1024*x3), None, eviction_policy='evict_last')
    tmp30 = tmp29 - tmp28
    tmp31 = tmp30 * tmp21
    tmp32 = tmp28 + tmp31
    tmp34 = tmp33 + tmp6
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tmp37 = tl.load(in_ptr3 + (tmp13 + 16*tmp36 + 256*tmp27 + 1024*x3), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr3 + (tmp18 + 16*tmp36 + 256*tmp27 + 1024*x3), None, eviction_policy='evict_last')
    tmp39 = tmp38 - tmp37
    tmp40 = tmp39 * tmp21
    tmp41 = tmp37 + tmp40
    tmp42 = tmp41 - tmp32
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr3 + (tmp13 + 16*tmp36 + 256*tmp4 + 1024*x3), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr3 + (tmp18 + 16*tmp36 + 256*tmp4 + 1024*x3), None, eviction_policy='evict_last')
    tmp47 = tmp46 - tmp45
    tmp48 = tmp47 * tmp21
    tmp49 = tmp45 + tmp48
    tmp50 = tmp49 - tmp23
    tmp51 = tmp50 * tmp43
    tmp52 = tmp32 + tmp44
    tmp53 = tmp23 + tmp51
    tmp54 = tmp53 - tmp52
    tmp56 = tmp54 * tmp55
    tmp57 = tmp52 + tmp56
    tl.store(in_out_ptr0 + (x6), tmp57, None)
