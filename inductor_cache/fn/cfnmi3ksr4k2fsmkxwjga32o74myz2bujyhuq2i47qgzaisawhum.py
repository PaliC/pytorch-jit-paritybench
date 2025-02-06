
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24)
    x1 = xindex // 24
    x2 = xindex
    tmp0 = x0 // 6
    tmp1 = (27 + 4*x0) // 24
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.full([1], 4, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (4*(x0 // 6) + 16*x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 4*(x0 // 6) + 16*x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp13 < tmp4
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + (2 + 4*(x0 // 6) + 16*x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tl.full([1], 3, tl.int64)
    tmp19 = tmp18 < tmp4
    tmp20 = tmp2 & tmp19
    tmp21 = tl.load(in_ptr0 + (3 + 4*(x0 // 6) + 16*x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = 1 + (x0 // 6)
    tmp24 = tmp23 < tmp1
    tmp25 = tmp24 & tmp5
    tmp26 = tl.load(in_ptr0 + (4 + 4*(x0 // 6) + 16*x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp26 + tmp22
    tmp28 = tmp24 & tmp9
    tmp29 = tl.load(in_ptr0 + (5 + 4*(x0 // 6) + 16*x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp27
    tmp31 = tmp24 & tmp14
    tmp32 = tl.load(in_ptr0 + (6 + 4*(x0 // 6) + 16*x1), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp30
    tmp34 = tmp24 & tmp19
    tmp35 = tl.load(in_ptr0 + (7 + 4*(x0 // 6) + 16*x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = 1.0
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp6, tmp37, tmp38)
    tmp40 = 1.0
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp10, tmp40, tmp41)
    tmp43 = tmp42 + tmp39
    tmp44 = 1.0
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp15, tmp44, tmp45)
    tmp47 = tmp46 + tmp43
    tmp48 = 1.0
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp20, tmp48, tmp49)
    tmp51 = tmp50 + tmp47
    tmp52 = 1.0
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp25, tmp52, tmp53)
    tmp55 = tmp54 + tmp51
    tmp56 = 1.0
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp28, tmp56, tmp57)
    tmp59 = tmp58 + tmp55
    tmp60 = 1.0
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp31, tmp60, tmp61)
    tmp63 = tmp62 + tmp59
    tmp64 = 1.0
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp34, tmp64, tmp65)
    tmp67 = tmp66 + tmp63
    tmp68 = tmp36 / tmp67
    tl.store(out_ptr0 + (x2), tmp68, None)
