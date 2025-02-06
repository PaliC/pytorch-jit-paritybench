
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_index_relu_rsub_sum_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_index_relu_rsub_sum_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (5))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp18 = tl.load(in_ptr1 + (1))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp28 = tl.load(in_ptr0 + (10))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp31 = tl.load(in_ptr1 + (2))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp41 = tl.load(in_ptr0 + (15))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr1 + (3))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp2 = 0.3
    tmp3 = tmp2 - tmp1
    tmp6 = tl.full([XBLOCK], 4, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert((0 <= tmp9) & (tmp9 < 4), "index out of bounds: 0 <= tmp9 < 4")
    tmp11 = tl.load(in_ptr0 + (tmp9), None, eviction_policy='evict_last')
    tmp12 = tmp3 + tmp11
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp17 = tmp2 - tmp16
    tmp20 = tmp19 + tmp6
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tl.device_assert((0 <= tmp22) & (tmp22 < 4), "index out of bounds: 0 <= tmp22 < 4")
    tmp24 = tl.load(in_ptr0 + (4 + tmp22), None, eviction_policy='evict_last')
    tmp25 = tmp17 + tmp24
    tmp26 = triton_helpers.maximum(tmp13, tmp25)
    tmp27 = tmp14 + tmp26
    tmp30 = tmp2 - tmp29
    tmp33 = tmp32 + tmp6
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tl.device_assert((0 <= tmp35) & (tmp35 < 4), "index out of bounds: 0 <= tmp35 < 4")
    tmp37 = tl.load(in_ptr0 + (8 + tmp35), None, eviction_policy='evict_last')
    tmp38 = tmp30 + tmp37
    tmp39 = triton_helpers.maximum(tmp13, tmp38)
    tmp40 = tmp27 + tmp39
    tmp43 = tmp2 - tmp42
    tmp46 = tmp45 + tmp6
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tl.device_assert((0 <= tmp48) & (tmp48 < 4), "index out of bounds: 0 <= tmp48 < 4")
    tmp50 = tl.load(in_ptr0 + (12 + tmp48), None, eviction_policy='evict_last')
    tmp51 = tmp43 + tmp50
    tmp52 = triton_helpers.maximum(tmp13, tmp51)
    tmp53 = tmp40 + tmp52
    tmp54 = 0.25
    tmp55 = tmp53 * tmp54
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp55, None)
