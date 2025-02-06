
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_pow_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_pow_sub_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 16)
    x2 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (16 + x1 + 64*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (32 + x1 + 64*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (48 + x1 + 64*x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tmp3 * tmp3
    tmp7 = tmp5 - tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp4 + tmp8
    tmp12 = tmp10 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tmp9 + tmp13
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp14 + tmp18
    tmp20 = tmp0 * tmp19
    tl.store(out_ptr0 + (x4), tmp20, xmask)
