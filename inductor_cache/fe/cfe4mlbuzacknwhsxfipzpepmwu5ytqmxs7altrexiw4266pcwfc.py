
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_neg_repeat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_neg_repeat_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 16) % 2)
    x3 = xindex // 32
    x4 = (xindex % 16)
    x5 = ((xindex // 4) % 8)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x6 = xindex // 16
    x7 = xindex
    tmp3 = tl.load(in_ptr0 + (x4 + 16*x3), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (x6), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = x5
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 4, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = x1 + 4*x2
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp8, tmp10, tmp11)
    tmp13 = tmp4 >= tmp7
    tmp14 = tl.full([1], 8, tl.int64)
    tmp15 = tmp4 < tmp14
    tmp16 = x0
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.where(tmp8, tmp12, tmp19)
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = -tmp23
    tmp25 = tmp20 + tmp24
    tmp26 = 0.25
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.where(tmp2, tmp3, tmp28)
    tl.store(out_ptr0 + (x7), tmp29, None)
