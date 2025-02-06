
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x3 = xindex // 5
    x2 = xindex // 20
    x4 = (xindex % 20)
    x5 = xindex
    tmp11 = tl.load(in_ptr2 + (40 + x4 + 80*x2), xmask)
    tmp15 = tl.load(in_ptr2 + (60 + x4 + 80*x2), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 4*x3), tmp2 & xmask, other=0.0)
    tmp4 = tl.full([1], 3, tl.int32)
    tmp5 = tl.full([1], 2, tl.int32)
    tmp6 = tmp4 == tmp5
    tmp7 = tmp0 >= tmp1
    tmp8 = tl.load(in_ptr1 + ((-1) + x0 + 4*x3), tmp7 & xmask, other=0.0)
    tmp9 = tmp5 == tmp5
    tmp10 = tl.load(in_ptr1 + (x0 + 4*x3), tmp2 & xmask, other=0.0)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp9, tmp12, tmp11)
    tmp14 = tl.where(tmp7, tmp8, tmp13)
    tmp16 = tl.where(tmp6, tmp12, tmp15)
    tmp17 = tl.where(tmp6, tmp14, tmp16)
    tmp18 = tl.where(tmp2, tmp3, tmp17)
    tl.store(out_ptr0 + (x5), tmp18, xmask)
