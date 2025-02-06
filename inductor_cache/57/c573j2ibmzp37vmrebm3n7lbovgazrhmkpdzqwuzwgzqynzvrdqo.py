
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp0 > tmp1
    tmp8 = tmp0 == tmp1
    tmp9 = tmp0 != tmp0
    tmp10 = tmp1 != tmp1
    tmp11 = tmp9 > tmp10
    tmp12 = tmp7 | tmp11
    tmp13 = tmp9 & tmp10
    tmp14 = tmp8 | tmp13
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp12 | tmp18
    tmp20 = tl.where(tmp19, tmp0, tmp1)
    tmp21 = tl.where(tmp19, tmp15, tmp16)
    tmp22 = tmp20 > tmp3
    tmp23 = tmp20 == tmp3
    tmp24 = tmp20 != tmp20
    tmp25 = tmp3 != tmp3
    tmp26 = tmp24 > tmp25
    tmp27 = tmp22 | tmp26
    tmp28 = tmp24 & tmp25
    tmp29 = tmp23 | tmp28
    tmp30 = tl.full([1], 2, tl.int64)
    tmp31 = tmp21 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tmp27 | tmp32
    tmp34 = tl.where(tmp33, tmp20, tmp3)
    tmp35 = tl.where(tmp33, tmp21, tmp30)
    tmp36 = tmp34 > tmp5
    tmp37 = tmp34 == tmp5
    tmp38 = tmp34 != tmp34
    tmp39 = tmp5 != tmp5
    tmp40 = tmp38 > tmp39
    tmp41 = tmp36 | tmp40
    tmp42 = tmp38 & tmp39
    tmp43 = tmp37 | tmp42
    tmp44 = tl.full([1], 3, tl.int64)
    tmp45 = tmp35 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tmp41 | tmp46
    tmp48 = tl.where(tmp47, tmp34, tmp5)
    tmp49 = tl.where(tmp47, tmp35, tmp44)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp49, xmask)
