
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 16*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 16*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + 16*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (6 + 16*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (7 + 16*x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (8 + 16*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (10 + 16*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (11 + 16*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (12 + 16*x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (14 + 16*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (15 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 / tmp7
    tmp17 = tmp8 + tmp16
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 / tmp7
    tmp26 = tmp17 + tmp25
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33 / tmp7
    tmp35 = tmp26 + tmp34
    tmp36 = tmp35 / tmp7
    tl.store(out_ptr0 + (x0 + 161*x1), tmp36, xmask)
