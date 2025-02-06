
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*fp32', 'in_ptr15': '*i64', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*i64', 'in_ptr21': '*fp32', 'out_ptr2': '*i1', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_relu_sub_threshold_backward_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 19, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_relu_sub_threshold_backward_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex // 64
    x1 = (xindex % 64)
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 512)
    y4 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (y3 + 512*x5 + 2097152*y4), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr8 + (x2), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr13 + (x2), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr14 + (x2), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr15 + (x2), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr16 + (x1), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr18 + (x1), None, eviction_policy='evict_last')
    tmp83 = tl.load(in_ptr19 + (x1), None, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr20 + (x2), None, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr21 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*y0), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 32*tmp4 + 1024*y0), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp21 = tmp20 + tmp1
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr2 + (tmp8 + 32*tmp23 + 1024*y0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (tmp13 + 32*tmp23 + 1024*y0), None, eviction_policy='evict_last')
    tmp26 = tmp25 - tmp24
    tmp27 = tmp26 * tmp16
    tmp28 = tmp24 + tmp27
    tmp29 = tmp28 - tmp18
    tmp31 = tmp29 * tmp30
    tmp32 = tmp18 + tmp31
    tmp33 = tmp19 + tmp32
    tmp35 = tl.full([XBLOCK, YBLOCK], 10, tl.int32)
    tmp36 = tmp34 + tmp35
    tmp37 = tmp34 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp34)
    tmp40 = tmp39 + tmp35
    tmp41 = tmp39 < 0
    tmp42 = tl.where(tmp41, tmp40, tmp39)
    tmp43 = tl.load(in_ptr10 + (tmp42 + 10*tmp38 + 100*y0), None, eviction_policy='evict_last')
    tmp45 = tmp44 + tmp35
    tmp46 = tmp44 < 0
    tmp47 = tl.where(tmp46, tmp45, tmp44)
    tmp48 = tl.load(in_ptr10 + (tmp47 + 10*tmp38 + 100*y0), None, eviction_policy='evict_last')
    tmp49 = tmp48 - tmp43
    tmp51 = tmp49 * tmp50
    tmp52 = tmp43 + tmp51
    tmp54 = tmp53 + tmp35
    tmp55 = tmp53 < 0
    tmp56 = tl.where(tmp55, tmp54, tmp53)
    tmp57 = tl.load(in_ptr10 + (tmp42 + 10*tmp56 + 100*y0), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr10 + (tmp47 + 10*tmp56 + 100*y0), None, eviction_policy='evict_last')
    tmp59 = tmp58 - tmp57
    tmp60 = tmp59 * tmp50
    tmp61 = tmp57 + tmp60
    tmp62 = tmp61 - tmp52
    tmp64 = tmp62 * tmp63
    tmp65 = tmp52 + tmp64
    tmp66 = tmp33 + tmp65
    tmp68 = tl.full([XBLOCK, YBLOCK], 5, tl.int32)
    tmp69 = tmp67 + tmp68
    tmp70 = tmp67 < 0
    tmp71 = tl.where(tmp70, tmp69, tmp67)
    tmp73 = tmp72 + tmp68
    tmp74 = tmp72 < 0
    tmp75 = tl.where(tmp74, tmp73, tmp72)
    tmp76 = tl.load(in_ptr17 + (tmp75 + 5*tmp71 + 25*y0), None, eviction_policy='evict_last')
    tmp78 = tmp77 + tmp68
    tmp79 = tmp77 < 0
    tmp80 = tl.where(tmp79, tmp78, tmp77)
    tmp81 = tl.load(in_ptr17 + (tmp80 + 5*tmp71 + 25*y0), None, eviction_policy='evict_last')
    tmp82 = tmp81 - tmp76
    tmp84 = tmp82 * tmp83
    tmp85 = tmp76 + tmp84
    tmp87 = tmp86 + tmp68
    tmp88 = tmp86 < 0
    tmp89 = tl.where(tmp88, tmp87, tmp86)
    tmp90 = tl.load(in_ptr17 + (tmp75 + 5*tmp89 + 25*y0), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr17 + (tmp80 + 5*tmp89 + 25*y0), None, eviction_policy='evict_last')
    tmp92 = tmp91 - tmp90
    tmp93 = tmp92 * tmp83
    tmp94 = tmp90 + tmp93
    tmp95 = tmp94 - tmp85
    tmp97 = tmp95 * tmp96
    tmp98 = tmp85 + tmp97
    tmp99 = tmp66 + tmp98
    tmp100 = tl.full([1, 1], 0, tl.int32)
    tmp101 = triton_helpers.maximum(tmp100, tmp99)
    tmp102 = 0.0
    tmp103 = tmp101 <= tmp102
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + 4096*y0), tmp101, None)
    tl.store(out_ptr2 + (y3 + 512*x5 + 2097152*y4), tmp103, None)
