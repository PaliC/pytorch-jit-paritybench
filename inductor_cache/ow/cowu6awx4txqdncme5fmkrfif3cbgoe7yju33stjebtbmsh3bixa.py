
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sigmoid_sub_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sigmoid_sub_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex // 16
    x2 = (xindex % 16)
    y0 = (yindex % 128)
    y1 = yindex // 128
    x4 = xindex
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (y0 + 128*tmp8 + 512*tmp4 + 2048*y1), xmask & ymask)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (y0 + 128*tmp13 + 512*tmp4 + 2048*y1), xmask & ymask)
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (y0 + 128*tmp8 + 512*tmp22 + 2048*y1), xmask & ymask)
    tmp24 = tl.load(in_ptr2 + (y0 + 128*tmp13 + 512*tmp22 + 2048*y1), xmask & ymask)
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tmp32 = tl.sigmoid(tmp31)
    tmp33 = tl.load(in_ptr7 + (y0 + 128*tmp8 + 512*tmp4 + 2048*y1), xmask & ymask)
    tmp34 = tl.load(in_ptr8 + (y0 + 128*tmp8 + 512*tmp4 + 2048*y1), xmask & ymask)
    tmp35 = tl.sigmoid(tmp34)
    tmp36 = tmp33 * tmp35
    tmp37 = tl.load(in_ptr7 + (y0 + 128*tmp13 + 512*tmp4 + 2048*y1), xmask & ymask)
    tmp38 = tl.load(in_ptr8 + (y0 + 128*tmp13 + 512*tmp4 + 2048*y1), xmask & ymask)
    tmp39 = tl.sigmoid(tmp38)
    tmp40 = tmp37 * tmp39
    tmp41 = tmp40 - tmp36
    tmp42 = tmp41 * tmp16
    tmp43 = tmp36 + tmp42
    tmp44 = tl.load(in_ptr7 + (y0 + 128*tmp8 + 512*tmp22 + 2048*y1), xmask & ymask)
    tmp45 = tl.load(in_ptr8 + (y0 + 128*tmp8 + 512*tmp22 + 2048*y1), xmask & ymask)
    tmp46 = tl.sigmoid(tmp45)
    tmp47 = tmp44 * tmp46
    tmp48 = tl.load(in_ptr7 + (y0 + 128*tmp13 + 512*tmp22 + 2048*y1), xmask & ymask)
    tmp49 = tl.load(in_ptr8 + (y0 + 128*tmp13 + 512*tmp22 + 2048*y1), xmask & ymask)
    tmp50 = tl.sigmoid(tmp49)
    tmp51 = tmp48 * tmp50
    tmp52 = tmp51 - tmp47
    tmp53 = tmp52 * tmp16
    tmp54 = tmp47 + tmp53
    tmp55 = tmp54 - tmp43
    tl.store(out_ptr1 + (y0 + 128*x4 + 32768*y1), tmp32, xmask & ymask)
    tl.store(out_ptr2 + (x4 + 256*y5), tmp43, xmask & ymask)
    tl.store(out_ptr3 + (x4 + 256*y5), tmp55, xmask & ymask)
