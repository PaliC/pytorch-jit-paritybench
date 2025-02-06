
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_max_pool2d_with_indices_mul_sub_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = ((xindex // 256) % 16)
    x3 = xindex // 4096
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (x2 + 32*tmp8 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (16 + x2 + 32*tmp8 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.load(in_ptr2 + (512 + x2 + 32*tmp8 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.load(in_ptr2 + (528 + x2 + 32*tmp8 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tmp17 = tmp16 + tmp1
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr2 + (x2 + 32*tmp8 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (16 + x2 + 32*tmp8 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.load(in_ptr2 + (512 + x2 + 32*tmp8 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tl.load(in_ptr2 + (528 + x2 + 32*tmp8 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp28 = tmp27 + tmp1
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr2 + (x2 + 32*tmp30 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (16 + x2 + 32*tmp30 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tl.load(in_ptr2 + (512 + x2 + 32*tmp30 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tmp36 = tl.load(in_ptr2 + (528 + x2 + 32*tmp30 + 1024*tmp19 + 16384*x3), None, eviction_policy='evict_last')
    tmp37 = triton_helpers.maximum(tmp36, tmp35)
    tmp38 = tmp37 - tmp26
    tmp40 = tmp38 * tmp39
    tmp41 = tmp26 + tmp40
    tmp42 = tl.load(in_ptr2 + (x2 + 32*tmp30 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr2 + (16 + x2 + 32*tmp30 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp44 = triton_helpers.maximum(tmp43, tmp42)
    tmp45 = tl.load(in_ptr2 + (512 + x2 + 32*tmp30 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.load(in_ptr2 + (528 + x2 + 32*tmp30 + 1024*tmp4 + 16384*x3), None, eviction_policy='evict_last')
    tmp48 = triton_helpers.maximum(tmp47, tmp46)
    tmp49 = tmp48 - tmp15
    tmp50 = tmp49 * tmp39
    tmp51 = tmp15 + tmp50
    tmp52 = tmp51 - tmp41
    tl.store(in_out_ptr0 + (x5), tmp41, None)
    tl.store(in_out_ptr1 + (x5), tmp52, None)
