
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex // 3
    x2 = (xindex % 3)
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 2048)
    y1 = yindex // 2048
    tmp0 = (4*x3) // 3
    tmp1 = 2 + ((4*x3) // 3)
    tmp2 = tmp0 < tmp1
    tmp3 = (4*x2) // 3
    tmp4 = 2 + ((4*x2) // 3)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (4*((4*x3) // 3) + 16*y4 + ((4*x2) // 3)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = 1 + ((4*x2) // 3)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 4*((4*x3) // 3) + 16*y4 + ((4*x2) // 3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 1 + ((4*x3) // 3)
    tmp14 = tmp13 < tmp1
    tmp15 = tmp14 & tmp5
    tmp16 = tl.load(in_ptr0 + (4 + 4*((4*x3) // 3) + 16*y4 + ((4*x2) // 3)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp14 & tmp9
    tmp19 = tl.load(in_ptr0 + (5 + 4*((4*x3) // 3) + 16*y4 + ((4*x2) // 3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp19 + tmp17
    tmp21 = 1.0
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = 1.0
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tmp26 + tmp23
    tmp28 = 1.0
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp15, tmp28, tmp29)
    tmp31 = tmp30 + tmp27
    tmp32 = 1.0
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp18, tmp32, tmp33)
    tmp35 = tmp34 + tmp31
    tmp36 = tmp20 / tmp35
    tl.store(out_ptr0 + (y0 + 2048*x5 + 18432*y1), tmp36, xmask)
