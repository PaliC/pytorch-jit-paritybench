
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 67)
    x3 = xindex // 13132
    x2 = ((xindex // 938) % 14)
    x1 = ((xindex // 67) % 14)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 66, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 64, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = tmp13 & tmp13
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + (64*x3 + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 1.0
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp15, tmp17, tmp18)
    tmp20 = tmp16 / tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp10, tmp20, tmp21)
    tmp23 = tmp5 >= tmp8
    tmp24 = tl.full([1], 65, tl.int64)
    tmp25 = tmp5 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp4
    tmp28 = x2
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 0.07692307692307693
    tmp31 = tmp29 * tmp30
    tmp32 = 2.0
    tmp33 = tmp31 * tmp32
    tmp34 = 1.0
    tmp35 = tmp33 - tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp27, tmp35, tmp36)
    tmp38 = tmp5 >= tmp24
    tmp39 = tl.full([1], 66, tl.int64)
    tmp40 = tmp5 < tmp39
    tmp41 = tmp38 & tmp4
    tmp42 = x1
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 0.07692307692307693
    tmp45 = tmp43 * tmp44
    tmp46 = 2.0
    tmp47 = tmp45 * tmp46
    tmp48 = 1.0
    tmp49 = tmp47 - tmp48
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp41, tmp49, tmp50)
    tmp52 = tl.where(tmp26, tmp37, tmp51)
    tmp53 = tl.where(tmp9, tmp22, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp4, tmp53, tmp54)
    tmp56 = tmp0 >= tmp3
    tmp57 = tl.full([1], 67, tl.int64)
    tmp58 = tmp0 < tmp57
    tmp59 = x2
    tmp60 = tmp59.to(tl.float32)
    tmp61 = 0.07692307692307693
    tmp62 = tmp60 * tmp61
    tmp63 = 2.0
    tmp64 = tmp62 * tmp63
    tmp65 = 1.0
    tmp66 = tmp64 - tmp65
    tmp67 = 0.5
    tmp68 = tmp66 - tmp67
    tmp69 = tmp68 * tmp68
    tmp70 = x1
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp71 * tmp61
    tmp73 = tmp72 * tmp63
    tmp74 = tmp73 - tmp65
    tmp75 = tmp74 - tmp67
    tmp76 = tmp75 * tmp75
    tmp77 = tmp69 + tmp76
    tmp78 = libdevice.sqrt(tmp77)
    tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
    tmp80 = tl.where(tmp56, tmp78, tmp79)
    tmp81 = tl.where(tmp4, tmp55, tmp80)
    tl.store(out_ptr0 + (x6), tmp81, xmask)
