
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_sum_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mul_sum_5(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y4 = yindex // 2
    y2 = yindex // 8
    y5 = (yindex % 8)
    y1 = ((yindex // 2) % 4)
    y6 = yindex
    tmp0 = tl.load(in_ptr0 + (x3 + 16*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y5 + 8*x3 + 128*y2), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (2*y1 + 8*x3 + 128*y2), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (1 + 2*y1 + 8*x3 + 128*y2), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.full([1, 1], 0, tl.int32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp7 = triton_helpers.maximum(tmp2, tmp6)
    tmp8 = tmp7 + tmp4
    tmp10 = triton_helpers.maximum(tmp2, tmp9)
    tmp11 = tmp10 + tmp4
    tmp12 = tmp8 + tmp11
    tmp13 = tmp5 / tmp12
    tmp14 = tmp0 * tmp13
    tl.store(out_ptr0 + (x3 + 16*y6), tmp14, xmask & ymask)
