
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5) + x4), tmp10 & xmask, other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp8 & tmp13
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-4) + x4), tmp16 & xmask, other=0.0)
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp8 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + ((-3) + x4), tmp23 & xmask, other=0.0)
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2 + x0
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp8 & tmp27
    tmp30 = tmp29 & tmp28
    tmp31 = tl.load(in_ptr0 + ((-2) + x4), tmp30 & xmask, other=0.0)
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = x1
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp36 & tmp6
    tmp38 = tmp37 & tmp7
    tmp39 = tl.load(in_ptr0 + ((-1) + x4), tmp38 & xmask, other=0.0)
    tmp40 = triton_helpers.maximum(tmp39, tmp32)
    tmp41 = tmp36 & tmp13
    tmp42 = tmp41 & tmp14
    tmp43 = tl.load(in_ptr0 + (x4), tmp42 & xmask, other=0.0)
    tmp44 = triton_helpers.maximum(tmp43, tmp40)
    tmp45 = tmp36 & tmp20
    tmp46 = tmp45 & tmp21
    tmp47 = tl.load(in_ptr0 + (1 + x4), tmp46 & xmask, other=0.0)
    tmp48 = triton_helpers.maximum(tmp47, tmp44)
    tmp49 = tmp36 & tmp27
    tmp50 = tmp49 & tmp28
    tmp51 = tl.load(in_ptr0 + (2 + x4), tmp50 & xmask, other=0.0)
    tmp52 = triton_helpers.maximum(tmp51, tmp48)
    tmp53 = 1 + x1
    tmp54 = tmp53 >= tmp1
    tmp55 = tmp53 < tmp3
    tmp56 = tmp54 & tmp55
    tmp57 = tmp56 & tmp6
    tmp58 = tmp57 & tmp7
    tmp59 = tl.load(in_ptr0 + (3 + x4), tmp58 & xmask, other=0.0)
    tmp60 = triton_helpers.maximum(tmp59, tmp52)
    tmp61 = tmp56 & tmp13
    tmp62 = tmp61 & tmp14
    tmp63 = tl.load(in_ptr0 + (4 + x4), tmp62 & xmask, other=0.0)
    tmp64 = triton_helpers.maximum(tmp63, tmp60)
    tmp65 = tmp56 & tmp20
    tmp66 = tmp65 & tmp21
    tmp67 = tl.load(in_ptr0 + (5 + x4), tmp66 & xmask, other=0.0)
    tmp68 = triton_helpers.maximum(tmp67, tmp64)
    tmp69 = tmp56 & tmp27
    tmp70 = tmp69 & tmp28
    tmp71 = tl.load(in_ptr0 + (6 + x4), tmp70 & xmask, other=0.0)
    tmp72 = triton_helpers.maximum(tmp71, tmp68)
    tmp73 = 2 + x1
    tmp74 = tmp73 >= tmp1
    tmp75 = tmp73 < tmp3
    tmp76 = tmp74 & tmp75
    tmp77 = tmp76 & tmp6
    tmp78 = tmp77 & tmp7
    tmp79 = tl.load(in_ptr0 + (7 + x4), tmp78 & xmask, other=0.0)
    tmp80 = triton_helpers.maximum(tmp79, tmp72)
    tmp81 = tmp76 & tmp13
    tmp82 = tmp81 & tmp14
    tmp83 = tl.load(in_ptr0 + (8 + x4), tmp82 & xmask, other=0.0)
    tmp84 = triton_helpers.maximum(tmp83, tmp80)
    tmp85 = tmp76 & tmp20
    tmp86 = tmp85 & tmp21
    tmp87 = tl.load(in_ptr0 + (9 + x4), tmp86 & xmask, other=0.0)
    tmp88 = triton_helpers.maximum(tmp87, tmp84)
    tmp89 = tmp76 & tmp27
    tmp90 = tmp89 & tmp28
    tmp91 = tl.load(in_ptr0 + (10 + x4), tmp90 & xmask, other=0.0)
    tmp92 = triton_helpers.maximum(tmp91, tmp88)
    tl.store(out_ptr0 + (x4), tmp92, xmask)
