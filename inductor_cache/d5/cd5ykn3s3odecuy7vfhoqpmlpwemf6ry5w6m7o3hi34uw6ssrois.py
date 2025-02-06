
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex // 4
    x1 = (xindex % 4)
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 512)
    y4 = yindex // 512
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 21.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1, 1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1, 1], 63, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = x1
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp2
    tmp14 = triton_helpers.maximum(tmp13, tmp4)
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + 64*tmp10 + 4096*y0), xmask, eviction_policy='evict_last')
    tmp17 = tmp15 + tmp7
    tmp18 = triton_helpers.minimum(tmp17, tmp9)
    tmp19 = tl.load(in_ptr0 + (tmp18 + 64*tmp10 + 4096*y0), xmask, eviction_policy='evict_last')
    tmp20 = tmp19 - tmp16
    tmp21 = tmp15.to(tl.float32)
    tmp22 = tmp14 - tmp21
    tmp23 = triton_helpers.maximum(tmp22, tmp4)
    tmp24 = 1.0
    tmp25 = triton_helpers.minimum(tmp23, tmp24)
    tmp26 = tmp20 * tmp25
    tmp27 = tmp16 + tmp26
    tmp28 = tl.load(in_ptr0 + (tmp15 + 64*tmp6 + 4096*y0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (tmp18 + 64*tmp6 + 4096*y0), xmask, eviction_policy='evict_last')
    tmp30 = tmp29 - tmp28
    tmp31 = tmp30 * tmp25
    tmp32 = tmp28 + tmp31
    tmp33 = tmp27 - tmp32
    tmp34 = tmp6.to(tl.float32)
    tmp35 = tmp5 - tmp34
    tmp36 = triton_helpers.maximum(tmp35, tmp4)
    tmp37 = triton_helpers.minimum(tmp36, tmp24)
    tmp38 = tmp33 * tmp37
    tmp39 = tmp32 + tmp38
    tl.store(out_ptr1 + (y3 + 512*x5 + 8192*y4), tmp39, xmask)
