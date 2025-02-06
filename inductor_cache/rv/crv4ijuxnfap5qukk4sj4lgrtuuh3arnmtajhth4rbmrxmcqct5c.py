
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 3)
    x0 = (xindex % 3)
    x2 = xindex // 9
    x4 = xindex
    tmp0 = (8*x1) // 3
    tmp1 = (10 + 8*x1) // 3
    tmp2 = tmp0 < tmp1
    tmp3 = (8*x0) // 3
    tmp4 = (10 + 8*x0) // 3
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = 1 + ((8*x0) // 3)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 2 + ((8*x0) // 3)
    tmp14 = tmp13 < tmp4
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + (2 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = 3 + ((8*x0) // 3)
    tmp19 = tmp18 < tmp4
    tmp20 = tmp2 & tmp19
    tmp21 = tl.load(in_ptr0 + (3 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = 1 + ((8*x1) // 3)
    tmp24 = tmp23 < tmp1
    tmp25 = tmp24 & tmp5
    tmp26 = tl.load(in_ptr0 + (8 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp26 + tmp22
    tmp28 = tmp24 & tmp9
    tmp29 = tl.load(in_ptr0 + (9 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp27
    tmp31 = tmp24 & tmp14
    tmp32 = tl.load(in_ptr0 + (10 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp30
    tmp34 = tmp24 & tmp19
    tmp35 = tl.load(in_ptr0 + (11 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = 2 + ((8*x1) // 3)
    tmp38 = tmp37 < tmp1
    tmp39 = tmp38 & tmp5
    tmp40 = tl.load(in_ptr0 + (16 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 + tmp36
    tmp42 = tmp38 & tmp9
    tmp43 = tl.load(in_ptr0 + (17 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp43 + tmp41
    tmp45 = tmp38 & tmp14
    tmp46 = tl.load(in_ptr0 + (18 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp46 + tmp44
    tmp48 = tmp38 & tmp19
    tmp49 = tl.load(in_ptr0 + (19 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 + tmp47
    tmp51 = 3 + ((8*x1) // 3)
    tmp52 = tmp51 < tmp1
    tmp53 = tmp52 & tmp5
    tmp54 = tl.load(in_ptr0 + (24 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp53 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 + tmp50
    tmp56 = tmp52 & tmp9
    tmp57 = tl.load(in_ptr0 + (25 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tmp57 + tmp55
    tmp59 = tmp52 & tmp14
    tmp60 = tl.load(in_ptr0 + (26 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp60 + tmp58
    tmp62 = tmp52 & tmp19
    tmp63 = tl.load(in_ptr0 + (27 + 8*((8*x1) // 3) + 64*x2 + ((8*x0) // 3)), tmp62 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp63 + tmp61
    tmp65 = 1.0
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp6, tmp65, tmp66)
    tmp68 = 1.0
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp10, tmp68, tmp69)
    tmp71 = tmp70 + tmp67
    tmp72 = 1.0
    tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
    tmp74 = tl.where(tmp15, tmp72, tmp73)
    tmp75 = tmp74 + tmp71
    tmp76 = 1.0
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp20, tmp76, tmp77)
    tmp79 = tmp78 + tmp75
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp25, tmp80, tmp81)
    tmp83 = tmp82 + tmp79
    tmp84 = 1.0
    tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
    tmp86 = tl.where(tmp28, tmp84, tmp85)
    tmp87 = tmp86 + tmp83
    tmp88 = 1.0
    tmp89 = tl.full(tmp88.shape, 0.0, tmp88.dtype)
    tmp90 = tl.where(tmp31, tmp88, tmp89)
    tmp91 = tmp90 + tmp87
    tmp92 = 1.0
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp34, tmp92, tmp93)
    tmp95 = tmp94 + tmp91
    tmp96 = 1.0
    tmp97 = tl.full(tmp96.shape, 0.0, tmp96.dtype)
    tmp98 = tl.where(tmp39, tmp96, tmp97)
    tmp99 = tmp98 + tmp95
    tmp100 = 1.0
    tmp101 = tl.full(tmp100.shape, 0.0, tmp100.dtype)
    tmp102 = tl.where(tmp42, tmp100, tmp101)
    tmp103 = tmp102 + tmp99
    tmp104 = 1.0
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp45, tmp104, tmp105)
    tmp107 = tmp106 + tmp103
    tmp108 = 1.0
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp48, tmp108, tmp109)
    tmp111 = tmp110 + tmp107
    tmp112 = 1.0
    tmp113 = tl.full(tmp112.shape, 0.0, tmp112.dtype)
    tmp114 = tl.where(tmp53, tmp112, tmp113)
    tmp115 = tmp114 + tmp111
    tmp116 = 1.0
    tmp117 = tl.full(tmp116.shape, 0.0, tmp116.dtype)
    tmp118 = tl.where(tmp56, tmp116, tmp117)
    tmp119 = tmp118 + tmp115
    tmp120 = 1.0
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp59, tmp120, tmp121)
    tmp123 = tmp122 + tmp119
    tmp124 = 1.0
    tmp125 = tl.full(tmp124.shape, 0.0, tmp124.dtype)
    tmp126 = tl.where(tmp62, tmp124, tmp125)
    tmp127 = tmp126 + tmp123
    tmp128 = tmp64 / tmp127
    tl.store(out_ptr0 + (x4), tmp128, xmask)
