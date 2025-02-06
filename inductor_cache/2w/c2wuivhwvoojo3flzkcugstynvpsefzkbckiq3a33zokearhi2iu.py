
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x1 + 64*x2 + ((x2 + 4*x1) // 16)), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (16 + x0 + 4*x1 + 64*x2 + ((x2 + 4*x1) // 16)), xmask)
    tmp5 = tl.load(in_ptr1 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (32 + x0 + 4*x1 + 64*x2 + ((x2 + 4*x1) // 16)), xmask)
    tmp10 = tl.load(in_ptr1 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (48 + x0 + 4*x1 + 64*x2 + ((x2 + 4*x1) // 16)), xmask)
    tmp15 = tl.load(in_ptr1 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp7 = tmp4 * tmp6
    tmp8 = tmp3 + tmp7
    tmp12 = tmp9 * tmp11
    tmp13 = tmp8 + tmp12
    tmp17 = tmp14 * tmp16
    tmp18 = tmp13 + tmp17
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr0 + (x3), tmp21, xmask)
