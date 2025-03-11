import sys, atexit, pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from tinygrad.ops import UOp, Variable, Ops, GroupOp, PatternMatcher, UPat, graph_rewrite, graph_rewrite_map, track_rewrites, buffers
from tinygrad.ops import can_pad, identity_element, resolve, merge_views
from tinygrad.codegen.symbolic import symbolic_simple
from tinygrad.helpers import Context, ContextVar, Metadata, all_int, all_same, colored, diskcache_put, prod, dedup, unwrap, flatten, getenv, pluralize
from tinygrad.helpers import FUSE_CONV_BW, FUSE_ARANGE, DEBUG, CAPTURE_PROCESS_REPLAY, DONT_REALIZE_EXPAND, DONT_GROUP_REDUCES, SPLIT_REDUCEOP
from tinygrad.dtype import ImageDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.device import Buffer
from tinygrad.spec import type_verify, kernel_spec, contiguous_spec

# creation can recurse a lot
sys.setrecursionlimit(10000)

# **** schedule simplifier

def simplify_stride0_reduce(reduce:UOp, x:UOp):
  # must be unmasked (NOTE: can be relaxed if not masked on stride 0 axis)
  if any(v.mask is not None for v in unwrap(x.st).views): return None
  # must have all stride 0 in the relevant axis (NOTE: can do partial)
  if not all(unwrap(x.st).views[-1].strides[axis] == 0 for axis in reduce.arg[1]) or not all_int(x.shape): return None
  prshape = prod(x.shape[i] for i in reduce.arg[1])
  ret = x.shrink(tuple((0,s) if i not in reduce.arg[1] else (0,1) for i,s in enumerate(x.shape)))
  match reduce.arg[0]:
    case Ops.ADD: return ret*prshape
    case Ops.MUL: return ret.pow(prshape)
    case Ops.MAX: return ret # NOTE: Ops.MAX is passthrough

def split_reduceop(reduce:UOp, x:UOp):
  if not SPLIT_REDUCEOP or not all_int(x.shape) or (prod(x.shape)//prod(reduce.shape))<getenv("REDUCEOP_SPLIT_THRESHOLD", 32768): return None
  # if there are few globals, make some reduces into globals by splitting into two kernels
  # cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
  #   ~2**10 should be enough if GROUP is used
  # 256 split maximum should be "negligible reduce" for low prod(reduce.shape), 8 split minimum.
  # split is moved to the end to provide maximum locality for the second phase reduce.
  real_strides = unwrap(x.st).real_strides(ignore_valid=True)
  if not (split_candidates:=[(i,d) for i in reduce.arg[1] for d in range(min(256,2**getenv("REDUCEOP_SPLIT_SIZE",22)//prod(reduce.shape)),8-1,-1)
                             if x.shape[i]%d==0 and real_strides[i]!=0]): return None
  dim_to_split, divisor = split_candidates[0]
  splitted_shape = x.shape[:dim_to_split]+(divisor,)+(x.shape[dim_to_split]//divisor,)+x.shape[dim_to_split+1:]
  splitted = x.reshape(splitted_shape).permute(tuple([d for d in range(len(splitted_shape)) if d!=dim_to_split]+[dim_to_split]))
  if DEBUG >= 3: print(f"split {divisor}: {x.shape} -> {splitted.shape} -> {reduce.shape}")
  # reduce original axes, then split
  return splitted.r(*reduce.arg).r(reduce.arg[0], (len(reduce.shape),)).reshape(reduce.shape)

def found_contiguous(ctx:dict[UOp, UOp], contig:UOp, src:UOp):
  if (sti:=unwrap(src.st).invert(src.base.shape)) is not None: ctx[src.base] = contig.view(sti)
def replace_contiguous(ctx:dict[UOp, UOp], alu:UOp):
  new_src = list(alu.src)
  for i,s in enumerate(alu.src):
    if (replace_src:=ctx.get(s, None)) is not None: new_src[i] = replace_src
  if tuple(new_src) != alu.src: return alu.replace(src=tuple(new_src))

def create_buffer_view(tr:UOp, x:UOp):
  assert isinstance(tr.device, str), "device must be string"
  if not tr.device.startswith("DISK"): return None
  return UOp(Ops.BUFFER_VIEW, tr.dtype, (x.base,), (tr.size, unwrap(x.st).views[0].offset)).reshape(tr.shape)

sym = symbolic_simple+PatternMatcher([
  # UOp with size 0 is zero
  (UPat(GroupOp.All-{Ops.SINK}, name="root"), lambda root: root.const_like(0) if root.base.st is not None and root.size == 0 \
   and not (root.base.op is Ops.CONST and root.base.arg == 0) else None),
  # DETACH and CONTIGUOUS_BACKWARD are NOOPs here
  (UPat((Ops.DETACH, Ops.CONTIGUOUS_BACKWARD), name="x"), lambda x: x.src[0]),
  # reduce of size 0 is the identity element
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)),
   lambda reduce,x: reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) if x.size == 0 and reduce.size != 0 else None),
  # reduce on stride 0 is collapsed
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), simplify_stride0_reduce),
  # split_reduceop
  (UPat(Ops.REDUCE_AXIS, name="reduce", src=(UPat.var("x"),)), split_reduceop),
  # COPY(CONST) creates a new CONST on the destination device
  (UPat(Ops.COPY, name="root", src=(UPat(), UPat.cvar("x"),)), lambda root,x: root.const_like(x.arg)),
  # no COPY to same device, except clone (arg is True)
  (UPat(Ops.COPY, src=(UPat(), UPat.var("copyin")), name="copy"),
   lambda copyin,copy: copyin if copyin.device == copy.device and copy.arg is not True else None),
  # copyin must be base
  (UPat(Ops.COPY, src=(UPat(), UPat(Ops.VIEW, name="v")), name="copy"), lambda copy,v: v.contiguous().copy_to_device(copy.device) \
    if prod(v.shape) < prod(v.base.shape) else v.base.copy_to_device(copy.device, clone=copy.arg).view(v.st)),
  # remove cast to image when it's already a contiguous image
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm", src=(UPat(Ops.CONTIGUOUS, name="base"))),)),
   lambda cast,base,vm: base.view(vm.st) if isinstance(cast.dtype, ImageDType) and isinstance(base.dtype, ImageDType) else None),
  # put CAST to smaller dtype before EXPAND
  (UPat(Ops.CAST, name="cast", src=(UPat(Ops.VIEW, name="vm"),)), lambda cast,vm: vm.base.cast(cast.dtype).view(vm.st)
     if (not getenv("CAST_AFTER_EXPAND") or vm.base.op is not Ops.BUFFER) and cast.dtype.itemsize <= vm.dtype.itemsize
     and resolve(prod(vm.shape) > vm.st.real_size()) else None),
  # make things that can't be images not images
  (UPat(GroupOp.All-{Ops.BUFFER, Ops.VIEW, Ops.CONST, Ops.DEVICE}, name="u"), lambda u: u.replace(dtype=dt.base) if isinstance(dt:=u.dtype,ImageDType)
   and (prod(u.shape) != prod(dt.shape) or not any(u.shape[x]%4 == 0 for x in u.st.unit_stride_axes())) else None),
  # remove contiguous if we can just view the buffer
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat(Ops.VIEW, name="view", src=(UPat(Ops.BUFFER, name="buf"),)),)),
   lambda root,view,buf: view if view.st.contiguous and view.size == buf.size else None),
  # contiguous/buffer/copy is already contiguous
  (UPat(Ops.CONTIGUOUS, name="root", src=(UPat((Ops.CONTIGUOUS, Ops.BUFFER, Ops.COPY)),)), lambda root: root.src[0]),
  # support for using a contiguous permuted view instead of the parent view if one exists
  (UPat(Ops.CONTIGUOUS, name="contig", src=(UPat(Ops.VIEW, name="src"),)), found_contiguous),
  (UPat(GroupOp.ALU, name="alu"), replace_contiguous),
  # substitute BITCAST/CONTIGUOUS with BUFFER_VIEW on DISK
  (UPat((Ops.BITCAST, Ops.CONTIGUOUS), src=(UPat.var("x"),), name="tr"), create_buffer_view),
  # put UnaryOps before EXPANDs
  (UPat(GroupOp.Unary, src=UPat(Ops.VIEW, src=(UPat.var("inp"),), name="v"), name="alu"),
   lambda inp,v,alu: inp.alu(alu.op).view(v.st) if resolve(prod(alu.shape) > v.st.real_size()) else None),
  # put CAST after expanding BUFFER
  (UPat(Ops.VIEW, src=(UPat(Ops.CAST, src=(UPat.var("x"),)),), name="v"), lambda x,v: x.view(x.st+v.st).cast(v.dtype) if getenv("CAST_AFTER_EXPAND")
    and x.base.op is Ops.BUFFER and resolve(prod(v.shape) > prod(x.shape)) else None),
  # remove CONST/BIND/VIEW from SINK
  (UPat(Ops.SINK, name="x"), lambda x: x.replace(src=new_src)
    if (new_src:=tuple(dedup(s.base for s in x.src if s.op not in {Ops.CONST,Ops.BIND}))) != x.src else None),
])

# **** swizzler

GROUPED = {Ops.BUFFER, Ops.ASSIGN, Ops.CONTIGUOUS, Ops.COPY, Ops.CONST}

remove_sink_views = PatternMatcher([
  (UPat(Ops.SINK, name="x"), lambda x:x.replace(src=tuple(s.base for s in x.src)) if any(s.op is Ops.VIEW for s in x.src) else None),
])

def swizzle_reduceop(r:UOp, src:UOp, view:UOp):
  if (st:=unwrap(view.st)).contiguous: return None
  input_st = ShapeTracker.from_shape(src.shape)
  tmp = input_st.permute(tuple(i for i in range(len(input_st.shape)) if i not in r.axis_arg)+r.axis_arg)
  prshape = prod(rshape:=tmp.shape[-len(r.axis_arg):])
  strides = strides_for_shape(rshape)
  nv = [View.create(v.shape+rshape, tuple(x*prshape for x in v.strides)+strides,
                    v.offset*prshape, v.mask+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in st.views]
  # update input_st and axis
  new_input_st = tmp + ShapeTracker(tuple(nv))
  new_axis = tuple(range(len(st.shape), len(st.shape) + len(r.axis_arg)))
  return UOp(Ops.REDUCE_AXIS, r.dtype, (UOp(Ops.VIEW, src.dtype, (src,), new_input_st),), (r.arg[0], new_axis)).reshape(st.shape)

view_left = merge_views+PatternMatcher([
  # VIEW before elementwise/buffer ops
  (UPat(Ops.VIEW, src=(UPat(GroupOp.All-{*GROUPED, Ops.DEVICE, Ops.REDUCE_AXIS}, name="x"),), name="view"),
   lambda x,view: x.replace(src=tuple(UOp(Ops.VIEW, s.dtype, (s,), view.arg) for s in x.src))),
])

def passthrough_elementwise(root:UOp):
  if not (swizzles:=[x for x in root.src if x.op is Ops.VIEW and x.base.op not in GROUPED]): return None
  assert all_same([x.base.size for x in swizzles]), f"swizzle inputs must have the same size {swizzles}"
  # place view after applying the elementwise op
  new_shape = swizzles[0].base.shape
  ret = root.replace(src=tuple(x.base if x.base.shape == new_shape else x.reshape(new_shape) for x in root.src))
  # reshape to match downstream shapes
  return ret.reshape(root.shape)

def reduceop_view_right(src:UOp, v:UOp, r:UOp):
  assert unwrap(v.st).contiguous and v.size == src.size, f"can't compute new axis for {src.shape} -> {r.shape}"
  return src.r(r.arg[0], tuple(i for i,(s,u) in enumerate(zip(src.shape, r.shape)) if s != u)).view(ShapeTracker.from_shape(r.shape))

view_right = merge_views+PatternMatcher([
  # push expand through reduce by making the reduce on a larger dim
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), swizzle_reduceop),
  # passthrough elementwise ops
  (UPat(GroupOp.All-{*GROUPED, Ops.SINK}, name="root"), passthrough_elementwise),
  # reduce on the base
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.VIEW, src=(UPat(GroupOp.All-GROUPED, name="src"),), name="v"),), name="r"), reduceop_view_right),
])

# **** create kernels

@dataclass(frozen=True)
class Kernel:
  ast: UOp
  metadata: tuple[Metadata, ...] = ()
  def __repr__(self):
    return f"<Kernel {len(list(self.ast.toposort))} {[s.op for s in self.ast.src] if self.ast.op is Ops.SINK else self.ast.op} {self.metadata}>"

@dataclass(frozen=True)
class KernelContext:
  ops_metadata: dict[UOp, Metadata]

def create_kernel(ctx:KernelContext, x:UOp, b:UOp):
  kernel = UOp(Ops.KERNEL, src=(b,)+x.src, arg=Kernel(x.sink(), (m,) if (m:=ctx.ops_metadata.get(x)) else ()))
  buffer = b.base if b.size == b.base.size else UOp(Ops.BUFFER_VIEW, b.dtype, (b.base,), (b.size, b.arg.views[0].offset))
  return UOp(Ops.ASSIGN, x.dtype, (buffer, kernel)).reshape(x.shape)

DONT_PLACE_IN_KERNEL = {Ops.KERNEL, Ops.ASSIGN, Ops.BUFFER}
def append_to_kernel(ctx:KernelContext, x:UOp):
  new_srcs: list[UOp] = []
  metadata = dict.fromkeys(x.arg.metadata)
  for s in x.src:
    if s.op in DONT_PLACE_IN_KERNEL: new_srcs.append(s)
    else:
      new_srcs.extend(s.src)
      if (m:=ctx.ops_metadata.get(s)) is not None: metadata[m] = None
  if (new_src:=tuple(dedup(new_srcs))) != x.src: return x.replace(src=new_src, arg=Kernel(x.arg.ast, tuple(metadata)))

create_kernels = merge_views+remove_sink_views+PatternMatcher([
  # always give assign/contiguous a kernel
  (UPat.assign(UPat.var("b"), UPat(GroupOp.All-{Ops.KERNEL}), name="x"), create_kernel),
  (UPat(Ops.CONTIGUOUS, name="x"), lambda ctx,x: create_kernel(ctx, x, UOp.new_buffer(x.device, x.size, x.dtype))),
  # create a buffer for COPY on the new device
  (UPat(Ops.COPY, src=(UPat(Ops.DEVICE, name="d"), UPat()), name="x"), lambda ctx,d,x: create_kernel(ctx, x, UOp.new_buffer(d.arg, x.size, x.dtype))),
  # walk back the local graph until we reach a buffer/assign parent
  (UPat(Ops.KERNEL, name="x"), append_to_kernel),
  # all ops in SINK get a kernel
  (UPat(Ops.SINK, name="x"), lambda ctx,x: x.replace(src=new_src)
    if (new_src:=tuple(s if s.op in DONT_PLACE_IN_KERNEL else s.contiguous() for s in x.src)) != x.src else None),
])

# ** create buffer ops + enumerate buffers

add_buffer_ops = PatternMatcher([
  # LOAD
  (UPat(Ops.BUFFER, name="x"), lambda ctx,x: UOp(Ops.LOAD, x.dtype, (UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), ctx.index(x)), x.st.to_uop()))),
  (UPat(Ops.ASSIGN, src=(UPat.var("x"), UPat())), lambda x:x),
  # STORE (except for COPY/BUFFER_VIEW)
  (UPat(Ops.SINK, src=(UPat((Ops.COPY, Ops.BUFFER_VIEW), name="x"),)), lambda x:x),
  (UPat(Ops.SINK, src=(UPat(GroupOp.All-{Ops.STORE}, name="x"),)),
   lambda x: UOp.store(UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), (), 0), ShapeTracker.from_shape(x.shape).to_uop(), x).sink()),
  # CONST/VALID
  (UPat(Ops.CONST, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),), name="view"),), name="x"), lambda x,view:x.replace(src=(view.arg.to_uop(),))),
  (UPat(Ops.VIEW, src=(UPat(Ops.CONST, name="x"),), name="view"), lambda view,x:x.valid(view.arg)),
  # no contiguous
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x:x),
  # +merge_views
  (UPat(Ops.VIEW, src=(UPat(Ops.LOAD, name="x"),), name="view"), lambda view,x: x.replace(src=(x.src[0], (x.src[1].arg+view.arg).to_uop()))),
  (UPat(Ops.STORE, src=(UPat.var("b"), UPat.var("st"), UPat(Ops.VIEW, src=(UPat.var("base"),)))), lambda b,st,base: UOp.store(b, st.view(base.st), base)),
])+merge_views

def fix_kernel_ast(k:UOp, var_vals:dict[Variable, int], idx:int) -> UOp:
  assert k.op is Ops.KERNEL, f"kernel isn't kernel, it's {k}"
  # add buffer ops
  ast = graph_rewrite(k.arg.ast, add_buffer_ops, bufs:=tuple(s.buf_uop for s in k.src), bottom_up=True, name=f"ast_{idx}")
  if ast.op is Ops.SINK and not all_same(dev:=[x.device for x in bufs]): raise RuntimeError(f"all buffers must be on the same device: {dev}")
  # create subbuffer (TODO: this does not belong here)
  if ast.op is Ops.BUFFER_VIEW: buffers[bufs[0]] = (base:=bufs[1].buffer).view(ast.size, ast.dtype, ast.arg[1]*base.dtype.itemsize)
  return k.replace(arg=Kernel(ast, k.arg.metadata))

PROCESS_REPLAY_CAPTURE:dict[str, bytes] = {}
if CAPTURE_PROCESS_REPLAY:
  @atexit.register
  def save_process_replay():
    for k,v in PROCESS_REPLAY_CAPTURE.items(): diskcache_put("schedule_process_replay", k, v, prepickled=True)

# **** schedule creation and toposort

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]

@track_rewrites(name_fxn=lambda r: f"Schedule {pluralize('Kernel', len(r[0]))}"+(f" (with_{pluralize('Var', len(r[1]))})" if len(r[1]) != 0 else ""))
def create_schedule_with_vars(big_sink:UOp) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  # merge_views + sym
  tensor_map = graph_rewrite_map(big_sink, merge_views+sym, ctx={})

  # display the cleaned up tensor graph
  if getenv("VIZ"): graph_rewrite(tensor_map[big_sink], PatternMatcher([]), name="View Tensor Graph")

  # view_left + view_right
  sink = tensor_map[big_sink]
  vl_map = graph_rewrite_map(sink, view_left)
  vr_map = graph_rewrite_map(vl_map[sink], view_right+remove_sink_views)
  contiguous_sink = vr_map[vl_map[sink]]
  if getenv("VIZ"): graph_rewrite(contiguous_sink, PatternMatcher([]), name="View Contiguous Graph")
  type_verify(list(contiguous_sink.toposort), contiguous_spec)
  # create_kernels
  kernel_map = graph_rewrite_map(contiguous_sink, create_kernels, ctx=KernelContext({}))
  sched_sink = kernel_map[contiguous_sink]
  type_verify(list(sched_sink.toposort), kernel_spec)
  assert len(sched_sink.src) == len(sink.src)
  # track back to the tensors
  buffer_map: dict[UOp, UOp] = {}
  for s1,s2 in zip(contiguous_sink.src, sched_sink.src): kernel_map[s1] = s2
  for k,v in tensor_map.items():
    if v.base not in vl_map: continue
    vl = vl_map[v.base]
    if vl not in vr_map: continue
    vr = vr_map[vl]
    if vr.base not in kernel_map: continue
    kernel = kernel_map[vr.base]
    if (a:=kernel.base).op is Ops.ASSIGN: buffer_map[v] = a.src[0] if a.src[0].st == v.st else a.src[0].view(unwrap(v.st))

  # map tensors to buffer/const, optionally apply a VIEW on top
  becomes_map: dict[UOp, UOp] = {}
  for k,v in tensor_map.items():
    # ASSIGN always becomes the target buffer
    if v.op is Ops.ASSIGN: becomes_map[k] = v.src[0]
    # if we created a new buffer for this tensor, map it to the assigned buffer
    elif (buf_view:=buffer_map.get(v)) is not None: becomes_map[k] = buf_view
    # tensors can also simplify to an existing buffer/const
    else:
      if k is v: continue
      if v.base.op is Ops.BUFFER: becomes_map[k] = v
      if v.base.op is Ops.CONST and all_int(v.shape): becomes_map[k] = v

  # if a kernel depends on a buffer, and that buffer is later assigned to, make the assign depend on the kernel's assign
  kernel_assign: dict[UOp, UOp] = {}
  assign_rep: dict[UOp, UOp] = {}
  for u in sched_sink.toposort:
    if u.op is not Ops.ASSIGN: continue
    kernel_assign[u.buf_uop] = u
    for s in u.src[1].src:
      if s.op is not Ops.BUFFER or s is u.buf_uop or (a:=kernel_assign.get(s)) is None: continue
      if any(x.op is Ops.ASSIGN and x.buf_uop is s for x in u.toposort):
        raise RuntimeError(f"cycle detected in graph, kernel for {u.buf_uop} must either depend on ASSIGN or BUFFER")
      assign_rep[a] = kernel_assign[s] = a.replace(src=a.src+(u,))
  if assign_rep:
    sched_sink = sched_sink.substitute(assign_rep)
    type_verify(list(sched_sink.toposort), kernel_spec)

  # display the final graph
  if getenv("VIZ"): graph_rewrite(sched_sink, PatternMatcher([]), name="View Kernel Graph")

  # final toposort (bfs)
  children: dict[UOp, list[UOp]] = {}
  in_degree: dict[UOp, int] = {}
  for u in sched_sink.toposort:
    if u.op is not Ops.ASSIGN: continue
    in_degree[u] = 0
    for s in u.src[1].src:
      if s.op is not Ops.ASSIGN: continue
      children.setdefault(s, []).append(u)
      in_degree[u] += 1

  queue = deque(k for k,v in in_degree.items() if v == 0)
  schedule: list[ScheduleItem] = []
  var_vals: dict[Variable, int] = {}
  while queue:
    u = queue.popleft()
    # TODO: move this to create_kernels
    k = fix_kernel_ast(u.src[1], var_vals, idx=len(schedule))
    schedule.append(ScheduleItem(k.arg.ast, tuple(s.buf_uop.buffer for s in k.src), k.arg.metadata))
    # increment the refcount of the target buf (this is required by the JIT and memory planner) TODO: this does not belong here
    k.src[0].buffer.ref(1)
    for x in children.get(u, []):
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(x)

  # confirm everything was scheduled correctly
  if len(schedule) != (kc:=len(in_degree)): raise RuntimeError(f"cycle detected in graph, created {kc} kernels but only scheduled {len(schedule)}")
  if DEBUG >= 1 and len(schedule) >= 10: print(f"scheduled {len(schedule)} kernels")
  # capture process replay
  if CAPTURE_PROCESS_REPLAY:
    with Context(PICKLE_BUFFERS=0): PROCESS_REPLAY_CAPTURE[str(big_sink.key)] = pickle.dumps((big_sink, ContextVar._cache, [x.ast for x in schedule]))
  return schedule, var_vals, becomes_map
