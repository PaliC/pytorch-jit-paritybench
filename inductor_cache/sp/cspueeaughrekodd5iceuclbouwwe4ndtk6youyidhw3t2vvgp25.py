
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_repeat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_repeat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 4)
    x2 = ((xindex // 8) % 4)
    x3 = xindex // 32
    x7 = xindex
    tmp41 = tl.load(in_ptr0 + (x0 + 2*x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 6.0
    tmp8 = tmp6 < tmp7
    tmp9 = 0.16666666666666666
    tmp10 = tmp6 * tmp9
    tmp11 = -0.9166666666666666
    tmp12 = tmp10 + tmp11
    tmp13 = 11 + ((-1)*x1)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp9
    tmp16 = 0.9166666666666666
    tmp17 = tmp16 - tmp15
    tmp18 = tl.where(tmp8, tmp12, tmp17)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp4, tmp18, tmp19)
    tmp21 = tmp0 >= tmp3
    tmp22 = tl.full([1], 2, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = x2
    tmp25 = tmp24.to(tl.float32)
    tmp26 = 6.0
    tmp27 = tmp25 < tmp26
    tmp28 = 0.16666666666666666
    tmp29 = tmp25 * tmp28
    tmp30 = -0.9166666666666666
    tmp31 = tmp29 + tmp30
    tmp32 = 11 + ((-1)*x2)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp28
    tmp35 = 0.9166666666666666
    tmp36 = tmp35 - tmp34
    tmp37 = tl.where(tmp27, tmp31, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp20, tmp39)
    tmp42 = 0.16666666666666666
    tmp43 = tmp41 * tmp42
    tmp44 = tmp40 + tmp43
    tl.store(out_ptr0 + (x7), tmp44, xmask)
