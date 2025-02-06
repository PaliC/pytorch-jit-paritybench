
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_cos_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_cos_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (((-1) + x0) % 2)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 == tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (1 + 2*(triton_helpers.div_floor_integer((-1) + x0,  2)) + 4*x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl_math.cos(tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = (x2 % 2)
    tmp12 = tmp11 == tmp4
    tmp13 = 2*(x0 // 2)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = libdevice.floor(tmp16)
    tmp18 = 2.0
    tmp19 = tmp17 * tmp18
    tmp20 = 0.25
    tmp21 = tmp19 * tmp20
    tmp22 = -9.210340371976184
    tmp23 = tmp21 * tmp22
    tmp24 = tl_math.exp(tmp23)
    tmp25 = x1
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp24
    tmp28 = tl_math.sin(tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp12, tmp28, tmp29)
    tmp31 = tmp0.to(tl.float32)
    tmp32 = 0.5
    tmp33 = tmp31 * tmp32
    tmp34 = libdevice.floor(tmp33)
    tmp35 = 2.0
    tmp36 = tmp34 * tmp35
    tmp37 = 0.25
    tmp38 = tmp36 * tmp37
    tmp39 = -9.210340371976184
    tmp40 = tmp38 * tmp39
    tmp41 = tl_math.exp(tmp40)
    tmp42 = x1
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp43 * tmp41
    tmp45 = tl.where(tmp12, tmp30, tmp44)
    tmp46 = tl.where(tmp6, tmp10, tmp45)
    tl.store(out_ptr0 + (x2), tmp46, xmask)
