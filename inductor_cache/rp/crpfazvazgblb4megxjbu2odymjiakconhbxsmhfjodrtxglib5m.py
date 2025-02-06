
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + 16*x1), xmask)
    tmp6 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (4 + x0 + 16*x1), xmask)
    tmp13 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (8 + x0 + 16*x1), xmask)
    tmp20 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (12 + x0 + 16*x1), xmask)
    tmp1 = tmp0 - tmp0
    tmp2 = tl_math.exp(tmp1)
    tmp3 = tmp2 / tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tmp8 / tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 + tmp11
    tmp14 = tmp13 - tmp13
    tmp15 = tl_math.exp(tmp14)
    tmp16 = tmp15 / tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 + tmp18
    tmp21 = tmp20 - tmp20
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp22 / tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tmp19 + tmp25
    tl.store(out_ptr0 + (x2), tmp26, xmask)
