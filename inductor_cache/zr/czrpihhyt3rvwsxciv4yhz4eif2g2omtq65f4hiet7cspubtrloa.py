
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_eq_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_eq_sum_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (1))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (2))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (3))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (1))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (2))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp33 = tl.load(in_ptr1 + (3))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp7 = tmp4 + tmp6
    tmp10 = tmp7 + tmp9
    tmp13 = tl.full([1], -100, tl.int64)
    tmp14 = tmp12 == tmp13
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 == tmp16
    tmp18 = tmp17.to(tl.int64)
    tmp21 = tmp20 == tmp13
    tmp22 = tmp21.to(tl.int64)
    tmp23 = tmp22 == tmp16
    tmp24 = tmp23.to(tl.int64)
    tmp25 = tmp18 + tmp24
    tmp28 = tmp27 == tmp13
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp29 == tmp16
    tmp31 = tmp30.to(tl.int64)
    tmp32 = tmp25 + tmp31
    tmp35 = tmp34 == tmp13
    tmp36 = tmp35.to(tl.int64)
    tmp37 = tmp36 == tmp16
    tmp38 = tmp37.to(tl.int64)
    tmp39 = tmp32 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp10 / tmp40
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp41, None)
