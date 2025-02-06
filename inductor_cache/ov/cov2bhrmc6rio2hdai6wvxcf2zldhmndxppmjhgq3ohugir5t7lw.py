
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_neg_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_neg_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp3 = -tmp2
    tmp4 = tmp1 > tmp3
    tmp5 = tmp1 == tmp3
    tmp6 = tmp1 != tmp1
    tmp7 = tmp3 != tmp3
    tmp8 = tmp6 > tmp7
    tmp9 = tmp4 | tmp8
    tmp10 = tmp6 & tmp7
    tmp11 = tmp5 | tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tl.full([1], 1, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp9 | tmp15
    tmp17 = tl.where(tmp16, tmp1, tmp3)
    tmp18 = tl.where(tmp16, tmp12, tmp13)
    tmp20 = -tmp19
    tmp21 = tmp17 > tmp20
    tmp22 = tmp17 == tmp20
    tmp23 = tmp17 != tmp17
    tmp24 = tmp20 != tmp20
    tmp25 = tmp23 > tmp24
    tmp26 = tmp21 | tmp25
    tmp27 = tmp23 & tmp24
    tmp28 = tmp22 | tmp27
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp18 < tmp29
    tmp31 = tmp28 & tmp30
    tmp32 = tmp26 | tmp31
    tmp33 = tl.where(tmp32, tmp17, tmp20)
    tmp34 = tl.where(tmp32, tmp18, tmp29)
    tmp36 = -tmp35
    tmp37 = tmp33 > tmp36
    tmp38 = tmp33 == tmp36
    tmp39 = tmp33 != tmp33
    tmp40 = tmp36 != tmp36
    tmp41 = tmp39 > tmp40
    tmp42 = tmp37 | tmp41
    tmp43 = tmp39 & tmp40
    tmp44 = tmp38 | tmp43
    tmp45 = tl.full([1], 3, tl.int64)
    tmp46 = tmp34 < tmp45
    tmp47 = tmp44 & tmp46
    tmp48 = tmp42 | tmp47
    tmp49 = tl.where(tmp48, tmp33, tmp36)
    tmp50 = tl.where(tmp48, tmp34, tmp45)
    tl.store(out_ptr0 + (x0), tmp50, xmask)
