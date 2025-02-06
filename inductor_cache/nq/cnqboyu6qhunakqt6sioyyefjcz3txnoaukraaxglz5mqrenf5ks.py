
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_affine_grid_generator_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_affine_grid_generator_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x5 = xindex
    x2 = ((xindex // 6) % 16)
    x1 = ((xindex // 3) % 2)
    x3 = xindex // 96
    x4 = (xindex % 6)
    tmp51 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = ((((x5 // 6) % 16)) % 4)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 2.0
    tmp6 = tmp4 < tmp5
    tmp7 = 0.5
    tmp8 = tmp4 * tmp7
    tmp9 = -0.75
    tmp10 = tmp8 + tmp9
    tmp11 = 3 + ((-1)*((x2 % 4)))
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp7
    tmp14 = 0.75
    tmp15 = tmp14 - tmp13
    tmp16 = tl.where(tmp6, tmp10, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = (-1) + x0
    tmp20 = tl.full([1], 0, tl.int64)
    tmp21 = tmp19 >= tmp20
    tmp22 = tmp19 < tmp1
    tmp23 = tmp21 & tmp22
    tmp24 = x2 // 4
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 2.0
    tmp27 = tmp25 < tmp26
    tmp28 = 0.5
    tmp29 = tmp25 * tmp28
    tmp30 = -0.75
    tmp31 = tmp29 + tmp30
    tmp32 = 3 + ((-1)*(x2 // 4))
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp28
    tmp35 = 0.75
    tmp36 = tmp35 - tmp34
    tmp37 = tl.where(tmp27, tmp31, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp23, tmp37, tmp38)
    tmp40 = tmp18 + tmp39
    tmp41 = (-2) + x0
    tmp42 = tmp41 >= tmp20
    tmp43 = 1.0
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp40 + tmp45
    tmp47 = x1
    tmp48 = tl.full([1], 0, tl.int32)
    tmp49 = tmp47 == tmp48
    tmp50 = tmp0 == tmp48
    tmp52 = 0.5
    tmp53 = tmp51 < tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = 2.0
    tmp56 = tmp54 * tmp55
    tmp57 = 1.0
    tmp58 = tmp56 - tmp57
    tmp60 = tl.where(tmp50, tmp58, tmp59)
    tmp62 = tl.where(tmp49, tmp60, tmp61)
    tmp63 = tmp46 * tmp62
    tl.store(out_ptr0 + (x5), tmp63, xmask)
