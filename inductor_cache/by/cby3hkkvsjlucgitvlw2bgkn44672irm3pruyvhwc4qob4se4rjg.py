
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_clone_mul_sub_view_as_complex_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_clone_mul_sub_view_as_complex_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 12
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex // 3
    x2 = (xindex % 3)
    y0 = (yindex % 2)
    y1 = yindex // 2
    x4 = xindex
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 14, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tl.full([XBLOCK, YBLOCK], 8, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp10 = tl.load(in_ptr2 + (y0 + 2*tmp9 + 16*tmp4 + 224*y1), xmask & ymask)
    tmp12 = tmp11 + tmp6
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr2 + (y0 + 2*tmp14 + 16*tmp4 + 224*y1), xmask & ymask)
    tmp16 = tmp15 - tmp10
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 + tmp18
    tmp21 = tmp20 + tmp1
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr2 + (y0 + 2*tmp9 + 16*tmp23 + 224*y1), xmask & ymask)
    tmp25 = tl.load(in_ptr2 + (y0 + 2*tmp14 + 16*tmp23 + 224*y1), xmask & ymask)
    tmp26 = tmp25 - tmp24
    tmp27 = tmp26 * tmp17
    tmp28 = tmp24 + tmp27
    tmp29 = tmp28 - tmp19
    tmp31 = tmp29 * tmp30
    tmp32 = tmp19 + tmp31
    tl.store(out_ptr1 + (y0 + 2*x4 + 24*y1), tmp32, xmask & ymask)
