
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 16) % 64)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x3 = xindex // 1024
    x4 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 32.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.int32)
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11 * tmp7
    tmp13 = tmp12.to(tl.int32)
    tmp14 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp15 = tmp14.to(tl.float32)
    tmp16 = 0.5
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.int32)
    tmp19 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp16
    tmp22 = tmp21.to(tl.int32)
    tmp23 = tl.load(in_ptr0 + (tmp22 + 64*tmp18 + 4096*(x2) + 245760*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1], 64, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tl.load(in_ptr1 + (x4 + 16*((-60) + x2) + 64*x3), tmp24, other=0.0)
    tmp28 = tl.where(tmp4, tmp23, tmp27)
    tl.store(out_ptr0 + (x5), tmp28, None)
