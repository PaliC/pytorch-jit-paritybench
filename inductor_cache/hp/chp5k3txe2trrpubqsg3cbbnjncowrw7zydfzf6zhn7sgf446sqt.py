
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_index_mean_sum_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_index_mean_sum_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp16 = tl.load(in_ptr0 + (1))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp31 = tl.load(in_ptr0 + (2))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp46 = tl.load(in_ptr0 + (3))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp2 = tl.full([XBLOCK], 96, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert((0 <= tmp5) & (tmp5 < 96), "index out of bounds: 0 <= tmp5 < 96")
    tmp7 = tl.load(in_ptr1 + (tmp5), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (96 + tmp5), None, eviction_policy='evict_last')
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr1 + (192 + tmp5), None, eviction_policy='evict_last')
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr1 + (288 + tmp5), None, eviction_policy='evict_last')
    tmp13 = tmp11 + tmp12
    tmp14 = 4.0
    tmp15 = tmp13 / tmp14
    tmp18 = tmp17 + tmp2
    tmp19 = tmp17 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp17)
    tl.device_assert((0 <= tmp20) & (tmp20 < 96), "index out of bounds: 0 <= tmp20 < 96")
    tmp22 = tl.load(in_ptr1 + (384 + tmp20), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (480 + tmp20), None, eviction_policy='evict_last')
    tmp24 = tmp22 + tmp23
    tmp25 = tl.load(in_ptr1 + (576 + tmp20), None, eviction_policy='evict_last')
    tmp26 = tmp24 + tmp25
    tmp27 = tl.load(in_ptr1 + (672 + tmp20), None, eviction_policy='evict_last')
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28 / tmp14
    tmp30 = tmp15 + tmp29
    tmp33 = tmp32 + tmp2
    tmp34 = tmp32 < 0
    tmp35 = tl.where(tmp34, tmp33, tmp32)
    tl.device_assert((0 <= tmp35) & (tmp35 < 96), "index out of bounds: 0 <= tmp35 < 96")
    tmp37 = tl.load(in_ptr1 + (768 + tmp35), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr1 + (864 + tmp35), None, eviction_policy='evict_last')
    tmp39 = tmp37 + tmp38
    tmp40 = tl.load(in_ptr1 + (960 + tmp35), None, eviction_policy='evict_last')
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr1 + (1056 + tmp35), None, eviction_policy='evict_last')
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43 / tmp14
    tmp45 = tmp30 + tmp44
    tmp48 = tmp47 + tmp2
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tl.device_assert((0 <= tmp50) & (tmp50 < 96), "index out of bounds: 0 <= tmp50 < 96")
    tmp52 = tl.load(in_ptr1 + (1152 + tmp50), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr1 + (1248 + tmp50), None, eviction_policy='evict_last')
    tmp54 = tmp52 + tmp53
    tmp55 = tl.load(in_ptr1 + (1344 + tmp50), None, eviction_policy='evict_last')
    tmp56 = tmp54 + tmp55
    tmp57 = tl.load(in_ptr1 + (1440 + tmp50), None, eviction_policy='evict_last')
    tmp58 = tmp56 + tmp57
    tmp59 = tmp58 / tmp14
    tmp60 = tmp45 + tmp59
    tmp61 = 0.25
    tmp62 = tmp60 * tmp61
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp62, None)
