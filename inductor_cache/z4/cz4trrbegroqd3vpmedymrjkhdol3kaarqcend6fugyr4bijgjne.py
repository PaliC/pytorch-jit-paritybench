
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'kernel_name': 'triton_for_fused_1', 'mutated_arg_names': [], 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_for_fused_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2):
    pid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    num_xblocks_0 = tl.cdiv(576, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(40, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(12, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        xnumel = 576
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = (xindex % 144)
        x1 = xindex // 144
        tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (x0 + 161*x1), tmp0, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        xnumel = 40
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = (xindex % 10)
        x3 = xindex // 10
        tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
        tl.store(out_ptr1 + (x2 + 161*x3), tmp1, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        xnumel = 12
        rnumel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = (xindex % 3)
        x5 = xindex // 3
        tmp2 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
        tl.store(out_ptr2 + (x4 + 161*x5), tmp2, xmask)
    else:
        pass
