
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = ((x0) % 2)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*((((x0) // 2) % 32))
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 0.5
    tmp15 = tmp13 * tmp14
    tmp16 = libdevice.floor(tmp15)
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp19 = 0.015625
    tmp20 = tmp18 * tmp19
    tmp21 = 10000.0
    tmp22 = libdevice.pow(tmp21, tmp20)
    tmp23 = tmp11 / tmp22
    tmp24 = tl_math.sin(tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tmp5 >= tmp8
    tmp28 = tl.full([1], 2, tl.int64)
    tmp29 = tmp5 < tmp28
    tmp30 = tmp27 & tmp4
    tmp31 = tl.load(in_ptr0 + (x1), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = 1 + 2*((((x0) // 2) % 32))
    tmp33 = tmp32.to(tl.float32)
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = libdevice.floor(tmp35)
    tmp37 = 2.0
    tmp38 = tmp36 * tmp37
    tmp39 = 0.015625
    tmp40 = tmp38 * tmp39
    tmp41 = 10000.0
    tmp42 = libdevice.pow(tmp41, tmp40)
    tmp43 = tmp31 / tmp42
    tmp44 = tl_math.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp30, tmp44, tmp45)
    tmp47 = tl.where(tmp9, tmp26, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp4, tmp47, tmp48)
    tmp50 = tmp0 >= tmp3
    tmp51 = tl.full([1], 128, tl.int64)
    tmp52 = tmp0 < tmp51
    tmp53 = (((-64) + x0) % 2)
    tmp54 = tl.full([1], 0, tl.int64)
    tmp55 = tmp53 >= tmp54
    tmp56 = tl.full([1], 1, tl.int64)
    tmp57 = tmp53 < tmp56
    tmp58 = tmp57 & tmp50
    tmp59 = tl.load(in_ptr1 + (x1), tmp58, eviction_policy='evict_last', other=0.0)
    tmp60 = 2*(((((-64) + x0) // 2) % 32))
    tmp61 = tmp60.to(tl.float32)
    tmp62 = 0.5
    tmp63 = tmp61 * tmp62
    tmp64 = libdevice.floor(tmp63)
    tmp65 = 2.0
    tmp66 = tmp64 * tmp65
    tmp67 = 0.015625
    tmp68 = tmp66 * tmp67
    tmp69 = 10000.0
    tmp70 = libdevice.pow(tmp69, tmp68)
    tmp71 = tmp59 / tmp70
    tmp72 = tl_math.sin(tmp71)
    tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
    tmp74 = tl.where(tmp58, tmp72, tmp73)
    tmp75 = tmp53 >= tmp56
    tmp76 = tl.full([1], 2, tl.int64)
    tmp77 = tmp53 < tmp76
    tmp78 = tmp75 & tmp50
    tmp79 = tl.load(in_ptr1 + (x1), tmp78, eviction_policy='evict_last', other=0.0)
    tmp80 = 1 + 2*(((((-64) + x0) // 2) % 32))
    tmp81 = tmp80.to(tl.float32)
    tmp82 = 0.5
    tmp83 = tmp81 * tmp82
    tmp84 = libdevice.floor(tmp83)
    tmp85 = 2.0
    tmp86 = tmp84 * tmp85
    tmp87 = 0.015625
    tmp88 = tmp86 * tmp87
    tmp89 = 10000.0
    tmp90 = libdevice.pow(tmp89, tmp88)
    tmp91 = tmp79 / tmp90
    tmp92 = tl_math.cos(tmp91)
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp78, tmp92, tmp93)
    tmp95 = tl.where(tmp57, tmp74, tmp94)
    tmp96 = tl.full(tmp95.shape, 0.0, tmp95.dtype)
    tmp97 = tl.where(tmp50, tmp95, tmp96)
    tmp98 = tl.where(tmp4, tmp49, tmp97)
    tl.store(out_ptr0 + (x2), tmp98, None)
