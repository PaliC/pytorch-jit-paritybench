
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel__unsafe_index_add_mul_sub_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
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
    tmp12 = tl.load(in_ptr3 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp14 = tmp13 * tmp9
    tmp15 = tl.where(tmp11, tmp9, tmp14)
    tmp17 = tmp16 + tmp1
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr2 + (tmp19 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp21 = tmp20 > tmp10
    tmp22 = tmp13 * tmp20
    tmp23 = tl.where(tmp21, tmp20, tmp22)
    tmp24 = tmp23 - tmp15
    tmp26 = tmp24 * tmp25
    tmp27 = tmp15 + tmp26
    tmp29 = tmp28 + tmp1
    tmp30 = tmp28 < 0
    tmp31 = tl.where(tmp30, tmp29, tmp28)
    tmp32 = tl.load(in_ptr2 + (tmp8 + 32*tmp31 + 1024*x2), None, eviction_policy='evict_last')
    tmp33 = tmp32 > tmp10
    tmp34 = tmp13 * tmp32
    tmp35 = tl.where(tmp33, tmp32, tmp34)
    tmp36 = tl.load(in_ptr2 + (tmp19 + 32*tmp31 + 1024*x2), None, eviction_policy='evict_last')
    tmp37 = tmp36 > tmp10
    tmp38 = tmp13 * tmp36
    tmp39 = tl.where(tmp37, tmp36, tmp38)
    tmp40 = tmp39 - tmp35
    tmp41 = tmp40 * tmp25
    tmp42 = tmp35 + tmp41
    tmp43 = tmp42 - tmp27
    tmp45 = tmp43 * tmp44
    tmp46 = tmp27 + tmp45
    tl.store(in_out_ptr0 + (x3), tmp46, None)
