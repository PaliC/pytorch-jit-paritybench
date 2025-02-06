
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = triton_helpers.div_floor_integer(4*(((((-16) + x0) // 3) % 3)),  3)
    tmp11 = 2 + (triton_helpers.div_floor_integer(4*(((((-16) + x0) // 3) % 3)),  3))
    tmp12 = tmp10 < tmp11
    tmp13 = triton_helpers.div_floor_integer(4*((((-16) + x0) % 3)),  3)
    tmp14 = 2 + (triton_helpers.div_floor_integer(4*((((-16) + x0) % 3)),  3))
    tmp15 = tmp13 < tmp14
    tmp16 = tmp12 & tmp15
    tmp17 = tmp16 & tmp9
    tmp18 = tl.load(in_ptr1 + (4*(triton_helpers.div_floor_integer(4*(((((-16) + x0) // 3) % 3)),  3)) + 16*(((((-16) + x0) // 9) % 16)) + (triton_helpers.div_floor_integer(4*((((-16) + x0) % 3)),  3))), tmp17 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp19 = 1 + (triton_helpers.div_floor_integer(4*((((-16) + x0) % 3)),  3))
    tmp20 = tmp19 < tmp14
    tmp21 = tmp12 & tmp20
    tmp22 = tmp21 & tmp9
    tmp23 = tl.load(in_ptr1 + (1 + 4*(triton_helpers.div_floor_integer(4*(((((-16) + x0) // 3) % 3)),  3)) + 16*(((((-16) + x0) // 9) % 16)) + (triton_helpers.div_floor_integer(4*((((-16) + x0) % 3)),  3))), tmp22 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp24 = triton_helpers.maximum(tmp23, tmp18)
    tmp25 = 1 + (triton_helpers.div_floor_integer(4*(((((-16) + x0) // 3) % 3)),  3))
    tmp26 = tmp25 < tmp11
    tmp27 = tmp26 & tmp15
    tmp28 = tmp27 & tmp9
    tmp29 = tl.load(in_ptr1 + (4 + 4*(triton_helpers.div_floor_integer(4*(((((-16) + x0) // 3) % 3)),  3)) + 16*(((((-16) + x0) // 9) % 16)) + (triton_helpers.div_floor_integer(4*((((-16) + x0) % 3)),  3))), tmp28 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp30 = triton_helpers.maximum(tmp29, tmp24)
    tmp31 = tmp26 & tmp20
    tmp32 = tmp31 & tmp9
    tmp33 = tl.load(in_ptr1 + (5 + 4*(triton_helpers.div_floor_integer(4*(((((-16) + x0) // 3) % 3)),  3)) + 16*(((((-16) + x0) // 9) % 16)) + (triton_helpers.div_floor_integer(4*((((-16) + x0) % 3)),  3))), tmp32 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp34 = triton_helpers.maximum(tmp33, tmp30)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp9, tmp34, tmp35)
    tmp37 = tmp0 >= tmp7
    tmp38 = tl.full([1], 736, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = triton_helpers.div_floor_integer(2*(((((-160) + x0) // 6) % 6)),  3)
    tmp42 = triton_helpers.div_floor_integer(9 + 4*(((((-160) + x0) // 6) % 6)),  6)
    tmp43 = tmp41 < tmp42
    tmp44 = triton_helpers.div_floor_integer(2*((((-160) + x0) % 6)),  3)
    tmp45 = triton_helpers.div_floor_integer(9 + 4*((((-160) + x0) % 6)),  6)
    tmp46 = tmp44 < tmp45
    tmp47 = tmp43 & tmp46
    tmp48 = tmp47 & tmp40
    tmp49 = tl.load(in_ptr1 + (4*(triton_helpers.div_floor_integer(2*(((((-160) + x0) // 6) % 6)),  3)) + 16*(((((-160) + x0) // 36) % 16)) + (triton_helpers.div_floor_integer(2*((((-160) + x0) % 6)),  3))), tmp48 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp50 = 1 + (triton_helpers.div_floor_integer(2*((((-160) + x0) % 6)),  3))
    tmp51 = tmp50 < tmp45
    tmp52 = tmp43 & tmp51
    tmp53 = tmp52 & tmp40
    tmp54 = tl.load(in_ptr1 + (1 + 4*(triton_helpers.div_floor_integer(2*(((((-160) + x0) // 6) % 6)),  3)) + 16*(((((-160) + x0) // 36) % 16)) + (triton_helpers.div_floor_integer(2*((((-160) + x0) % 6)),  3))), tmp53 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp49)
    tmp56 = 1 + (triton_helpers.div_floor_integer(2*(((((-160) + x0) // 6) % 6)),  3))
    tmp57 = tmp56 < tmp42
    tmp58 = tmp57 & tmp46
    tmp59 = tmp58 & tmp40
    tmp60 = tl.load(in_ptr1 + (4 + 4*(triton_helpers.div_floor_integer(2*(((((-160) + x0) // 6) % 6)),  3)) + 16*(((((-160) + x0) // 36) % 16)) + (triton_helpers.div_floor_integer(2*((((-160) + x0) % 6)),  3))), tmp59 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp61 = triton_helpers.maximum(tmp60, tmp55)
    tmp62 = tmp57 & tmp51
    tmp63 = tmp62 & tmp40
    tmp64 = tl.load(in_ptr1 + (5 + 4*(triton_helpers.div_floor_integer(2*(((((-160) + x0) // 6) % 6)),  3)) + 16*(((((-160) + x0) // 36) % 16)) + (triton_helpers.div_floor_integer(2*((((-160) + x0) % 6)),  3))), tmp63 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp65 = triton_helpers.maximum(tmp64, tmp61)
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp40, tmp65, tmp66)
    tmp68 = tmp0 >= tmp38
    tmp69 = tl.full([1], 1760, tl.int64)
    tmp70 = tmp0 < tmp69
    tmp71 = triton_helpers.div_floor_integer(((((-736) + x0) // 8) % 8),  2)
    tmp72 = triton_helpers.div_floor_integer(11 + 4*(((((-736) + x0) // 8) % 8)),  8)
    tmp73 = tmp71 < tmp72
    tmp74 = triton_helpers.div_floor_integer((((-736) + x0) % 8),  2)
    tmp75 = triton_helpers.div_floor_integer(11 + 4*((((-736) + x0) % 8)),  8)
    tmp76 = tmp74 < tmp75
    tmp77 = tmp73 & tmp76
    tmp78 = tmp77 & tmp68
    tmp79 = tl.load(in_ptr1 + (4*(triton_helpers.div_floor_integer(((((-736) + x0) // 8) % 8),  2)) + 16*(((((-736) + x0) // 64) % 16)) + (triton_helpers.div_floor_integer((((-736) + x0) % 8),  2))), tmp78 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp80 = 1 + (triton_helpers.div_floor_integer((((-736) + x0) % 8),  2))
    tmp81 = tmp80 < tmp75
    tmp82 = tmp73 & tmp81
    tmp83 = tmp82 & tmp68
    tmp84 = tl.load(in_ptr1 + (1 + 4*(triton_helpers.div_floor_integer(((((-736) + x0) // 8) % 8),  2)) + 16*(((((-736) + x0) // 64) % 16)) + (triton_helpers.div_floor_integer((((-736) + x0) % 8),  2))), tmp83 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp85 = triton_helpers.maximum(tmp84, tmp79)
    tmp86 = 1 + (triton_helpers.div_floor_integer(((((-736) + x0) // 8) % 8),  2))
    tmp87 = tmp86 < tmp72
    tmp88 = tmp87 & tmp76
    tmp89 = tmp88 & tmp68
    tmp90 = tl.load(in_ptr1 + (4 + 4*(triton_helpers.div_floor_integer(((((-736) + x0) // 8) % 8),  2)) + 16*(((((-736) + x0) // 64) % 16)) + (triton_helpers.div_floor_integer((((-736) + x0) % 8),  2))), tmp89 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp91 = triton_helpers.maximum(tmp90, tmp85)
    tmp92 = tmp87 & tmp81
    tmp93 = tmp92 & tmp68
    tmp94 = tl.load(in_ptr1 + (5 + 4*(triton_helpers.div_floor_integer(((((-736) + x0) // 8) % 8),  2)) + 16*(((((-736) + x0) // 64) % 16)) + (triton_helpers.div_floor_integer((((-736) + x0) % 8),  2))), tmp93 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp95 = triton_helpers.maximum(tmp94, tmp91)
    tmp96 = tl.full(tmp95.shape, 0.0, tmp95.dtype)
    tmp97 = tl.where(tmp68, tmp95, tmp96)
    tmp98 = tl.where(tmp40, tmp67, tmp97)
    tmp99 = tl.where(tmp9, tmp36, tmp98)
    tmp100 = tl.where(tmp4, tmp5, tmp99)
    tl.store(out_ptr0 + (x0), tmp100, xmask)
