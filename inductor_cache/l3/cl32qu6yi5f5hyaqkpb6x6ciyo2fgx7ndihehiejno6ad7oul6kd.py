
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x2 = ((xindex // 512) % 4)
    x1 = ((xindex // 128) % 4)
    x7 = xindex
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
    tmp11 = 2*((((x0) // 2) % 32))
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.trunc(tmp14)
    tmp16 = 2.0
    tmp17 = tmp15 * tmp16
    tmp18 = 0.015625
    tmp19 = tmp17 * tmp18
    tmp20 = 10000.0
    tmp21 = libdevice.pow(tmp20, tmp19)
    tmp22 = 1 + x2
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 / tmp21
    tmp25 = tl_math.sin(tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp10, tmp25, tmp26)
    tmp28 = tmp5 >= tmp8
    tmp29 = tl.full([1], 2, tl.int64)
    tmp30 = tmp5 < tmp29
    tmp31 = tmp28 & tmp4
    tmp32 = 1 + 2*((((x0) // 2) % 32))
    tmp33 = tmp32.to(tl.float32)
    tmp34 = 0.5
    tmp35 = tmp33 * tmp34
    tmp36 = libdevice.trunc(tmp35)
    tmp37 = 2.0
    tmp38 = tmp36 * tmp37
    tmp39 = 0.015625
    tmp40 = tmp38 * tmp39
    tmp41 = 10000.0
    tmp42 = libdevice.pow(tmp41, tmp40)
    tmp43 = 1 + x2
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 / tmp42
    tmp46 = tl_math.cos(tmp45)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp31, tmp46, tmp47)
    tmp49 = tl.where(tmp9, tmp27, tmp48)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp4, tmp49, tmp50)
    tmp52 = tmp0 >= tmp3
    tmp53 = tl.full([1], 128, tl.int64)
    tmp54 = tmp0 < tmp53
    tmp55 = (((-64) + x0) % 2)
    tmp56 = tl.full([1], 0, tl.int64)
    tmp57 = tmp55 >= tmp56
    tmp58 = tl.full([1], 1, tl.int64)
    tmp59 = tmp55 < tmp58
    tmp60 = tmp59 & tmp52
    tmp61 = 2*(((((-64) + x0) // 2) % 32))
    tmp62 = tmp61.to(tl.float32)
    tmp63 = 0.5
    tmp64 = tmp62 * tmp63
    tmp65 = libdevice.trunc(tmp64)
    tmp66 = 2.0
    tmp67 = tmp65 * tmp66
    tmp68 = 0.015625
    tmp69 = tmp67 * tmp68
    tmp70 = 10000.0
    tmp71 = libdevice.pow(tmp70, tmp69)
    tmp72 = 1 + x1
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73 / tmp71
    tmp75 = tl_math.sin(tmp74)
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp60, tmp75, tmp76)
    tmp78 = tmp55 >= tmp58
    tmp79 = tl.full([1], 2, tl.int64)
    tmp80 = tmp55 < tmp79
    tmp81 = tmp78 & tmp52
    tmp82 = 1 + 2*(((((-64) + x0) // 2) % 32))
    tmp83 = tmp82.to(tl.float32)
    tmp84 = 0.5
    tmp85 = tmp83 * tmp84
    tmp86 = libdevice.trunc(tmp85)
    tmp87 = 2.0
    tmp88 = tmp86 * tmp87
    tmp89 = 0.015625
    tmp90 = tmp88 * tmp89
    tmp91 = 10000.0
    tmp92 = libdevice.pow(tmp91, tmp90)
    tmp93 = 1 + x1
    tmp94 = tmp93.to(tl.float32)
    tmp95 = tmp94 / tmp92
    tmp96 = tl_math.cos(tmp95)
    tmp97 = tl.full(tmp96.shape, 0.0, tmp96.dtype)
    tmp98 = tl.where(tmp81, tmp96, tmp97)
    tmp99 = tl.where(tmp59, tmp77, tmp98)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp52, tmp99, tmp100)
    tmp102 = tl.where(tmp4, tmp51, tmp101)
    tl.store(out_ptr0 + (x7), tmp102, None)
