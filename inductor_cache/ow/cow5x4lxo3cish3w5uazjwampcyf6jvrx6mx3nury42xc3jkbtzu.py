
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + 64*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (33 + 64*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (34 + 64*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (35 + 64*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (36 + 64*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (37 + 64*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (38 + 64*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (39 + 64*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (40 + 64*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (41 + 64*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (42 + 64*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (43 + 64*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (44 + 64*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (45 + 64*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (46 + 64*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (47 + 64*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = triton_helpers.maximum(tmp6, tmp13)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp14, tmp21)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp29 = triton_helpers.maximum(tmp27, tmp28)
    tmp30 = triton_helpers.maximum(tmp22, tmp29)
    tl.store(out_ptr0 + (x0), tmp30, xmask)
