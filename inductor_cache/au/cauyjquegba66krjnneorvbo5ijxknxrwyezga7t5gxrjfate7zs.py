
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (4*x1) // 3
    tmp1 = (13 + 8*x1) // 6
    tmp2 = tmp0 < tmp1
    tmp3 = (4*x0) // 3
    tmp4 = (13 + 8*x0) // 6
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp6, other=0.0)
    tmp8 = 1 + ((4*x0) // 3)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp10, other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 2 + ((4*x0) // 3)
    tmp14 = tmp13 < tmp4
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + (2 + 8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp15, other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = 1 + ((4*x1) // 3)
    tmp19 = tmp18 < tmp1
    tmp20 = tmp19 & tmp5
    tmp21 = tl.load(in_ptr0 + (8 + 8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp20, other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = tmp19 & tmp9
    tmp24 = tl.load(in_ptr0 + (9 + 8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp23, other=0.0)
    tmp25 = tmp24 + tmp22
    tmp26 = tmp19 & tmp14
    tmp27 = tl.load(in_ptr0 + (10 + 8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp26, other=0.0)
    tmp28 = tmp27 + tmp25
    tmp29 = 2 + ((4*x1) // 3)
    tmp30 = tmp29 < tmp1
    tmp31 = tmp30 & tmp5
    tmp32 = tl.load(in_ptr0 + (16 + 8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp31, other=0.0)
    tmp33 = tmp32 + tmp28
    tmp34 = tmp30 & tmp9
    tmp35 = tl.load(in_ptr0 + (17 + 8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp34, other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = tmp30 & tmp14
    tmp38 = tl.load(in_ptr0 + (18 + 8*((4*x1) // 3) + 64*x2 + ((4*x0) // 3)), tmp37, other=0.0)
    tmp39 = tmp38 + tmp36
    tmp40 = 1.0
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp6, tmp40, tmp41)
    tmp43 = 1.0
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp10, tmp43, tmp44)
    tmp46 = tmp45 + tmp42
    tmp47 = 1.0
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp15, tmp47, tmp48)
    tmp50 = tmp49 + tmp46
    tmp51 = 1.0
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp20, tmp51, tmp52)
    tmp54 = tmp53 + tmp50
    tmp55 = 1.0
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp23, tmp55, tmp56)
    tmp58 = tmp57 + tmp54
    tmp59 = 1.0
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp26, tmp59, tmp60)
    tmp62 = tmp61 + tmp58
    tmp63 = 1.0
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp31, tmp63, tmp64)
    tmp66 = tmp65 + tmp62
    tmp67 = 1.0
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp34, tmp67, tmp68)
    tmp70 = tmp69 + tmp66
    tmp71 = 1.0
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp37, tmp71, tmp72)
    tmp74 = tmp73 + tmp70
    tmp75 = tmp39 / tmp74
    tl.store(out_ptr0 + (x4), tmp75, None)
