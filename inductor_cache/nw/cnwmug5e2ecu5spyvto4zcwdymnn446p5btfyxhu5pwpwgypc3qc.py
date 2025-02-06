
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_clone_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 30, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_clone_max_pool2d_with_indices_0(in_ptr0, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex // 4
    x1 = (xindex % 4)
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 4)
    y4 = yindex // 4
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-2) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6) + x1 + 8*x2 + 16*y0), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = (-1) + x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5) + x1 + 8*x2 + 16*y0), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-4) + x1 + 8*x2 + 16*y0), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 1 + x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + ((-3) + x1 + 8*x2 + 16*y0), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = 2 + x1
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp5 & tmp36
    tmp38 = tl.load(in_ptr0 + ((-2) + x1 + 8*x2 + 16*y0), tmp37 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38 + tmp32
    tmp40 = 2*x2
    tmp41 = tmp40 >= tmp1
    tmp42 = tmp40 < tmp3
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp9
    tmp45 = tl.load(in_ptr0 + ((-2) + x1 + 8*x2 + 16*y0), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp39
    tmp47 = tmp43 & tmp15
    tmp48 = tl.load(in_ptr0 + ((-1) + x1 + 8*x2 + 16*y0), tmp47 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 + tmp46
    tmp50 = tmp43 & tmp22
    tmp51 = tl.load(in_ptr0 + (x1 + 8*x2 + 16*y0), tmp50 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 + tmp49
    tmp53 = tmp43 & tmp29
    tmp54 = tl.load(in_ptr0 + (1 + x1 + 8*x2 + 16*y0), tmp53 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 + tmp52
    tmp56 = tmp43 & tmp36
    tmp57 = tl.load(in_ptr0 + (2 + x1 + 8*x2 + 16*y0), tmp56 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp58 = tmp57 + tmp55
    tmp59 = 1 + 2*x2
    tmp60 = tmp59 >= tmp1
    tmp61 = tmp59 < tmp3
    tmp62 = tmp60 & tmp61
    tmp63 = tmp62 & tmp9
    tmp64 = tl.load(in_ptr0 + (2 + x1 + 8*x2 + 16*y0), tmp63 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp64 + tmp58
    tmp66 = tmp62 & tmp15
    tmp67 = tl.load(in_ptr0 + (3 + x1 + 8*x2 + 16*y0), tmp66 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp67 + tmp65
    tmp69 = tmp62 & tmp22
    tmp70 = tl.load(in_ptr0 + (4 + x1 + 8*x2 + 16*y0), tmp69 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp71 = tmp70 + tmp68
    tmp72 = tmp62 & tmp29
    tmp73 = tl.load(in_ptr0 + (5 + x1 + 8*x2 + 16*y0), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp74 = tmp73 + tmp71
    tmp75 = tmp62 & tmp36
    tmp76 = tl.load(in_ptr0 + (6 + x1 + 8*x2 + 16*y0), tmp75 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp76 + tmp74
    tmp78 = 2 + ((-1)*x1) + ((-4)*x2) + 2*((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5))) + ((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5)))*((6) * ((6) <= (3 + x1)) + (3 + x1) * ((3 + x1) < (6))) + ((-1)*x1*((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5)))) + ((-2)*x2*((6) * ((6) <= (3 + x1)) + (3 + x1) * ((3 + x1) < (6)))) + 2*x1*x2 + ((6) * ((6) <= (3 + x1)) + (3 + x1) * ((3 + x1) < (6)))
    tmp79 = tmp77 / tmp78
    tmp80 = tl.load(in_ptr0 + ((-6) + x1 + 8*x2 + 16*y0), tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp81 = tl.load(in_ptr0 + ((-5) + x1 + 8*x2 + 16*y0), tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp82 = triton_helpers.maximum(tmp81, tmp80)
    tmp83 = tl.load(in_ptr0 + ((-4) + x1 + 8*x2 + 16*y0), tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp82)
    tmp85 = tl.load(in_ptr0 + ((-3) + x1 + 8*x2 + 16*y0), tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp86 = triton_helpers.maximum(tmp85, tmp84)
    tmp87 = tl.load(in_ptr0 + ((-2) + x1 + 8*x2 + 16*y0), tmp37 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp88 = triton_helpers.maximum(tmp87, tmp86)
    tmp89 = tl.load(in_ptr0 + ((-2) + x1 + 8*x2 + 16*y0), tmp44 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp90 = triton_helpers.maximum(tmp89, tmp88)
    tmp91 = tl.load(in_ptr0 + ((-1) + x1 + 8*x2 + 16*y0), tmp47 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp92 = triton_helpers.maximum(tmp91, tmp90)
    tmp93 = tl.load(in_ptr0 + (x1 + 8*x2 + 16*y0), tmp50 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp94 = triton_helpers.maximum(tmp93, tmp92)
    tmp95 = tl.load(in_ptr0 + (1 + x1 + 8*x2 + 16*y0), tmp53 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp94)
    tmp97 = tl.load(in_ptr0 + (2 + x1 + 8*x2 + 16*y0), tmp56 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp98 = triton_helpers.maximum(tmp97, tmp96)
    tmp99 = tl.load(in_ptr0 + (2 + x1 + 8*x2 + 16*y0), tmp63 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp100 = triton_helpers.maximum(tmp99, tmp98)
    tmp101 = tl.load(in_ptr0 + (3 + x1 + 8*x2 + 16*y0), tmp66 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp102 = triton_helpers.maximum(tmp101, tmp100)
    tmp103 = tl.load(in_ptr0 + (4 + x1 + 8*x2 + 16*y0), tmp69 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp104 = triton_helpers.maximum(tmp103, tmp102)
    tmp105 = tl.load(in_ptr0 + (5 + x1 + 8*x2 + 16*y0), tmp72 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp106 = triton_helpers.maximum(tmp105, tmp104)
    tmp107 = tl.load(in_ptr0 + (6 + x1 + 8*x2 + 16*y0), tmp75 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp108 = triton_helpers.maximum(tmp107, tmp106)
    tmp109 = tmp79 + tmp108
    tmp110 = 0.5
    tmp111 = tmp109 * tmp110
    tl.store(out_ptr2 + (y3 + 4*x5 + 32*y4), tmp111, xmask & ymask)
