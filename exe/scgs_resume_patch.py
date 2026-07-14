#!/usr/bin/env python3
# ============================================================================
# scgs_resume_patch.py -- make yihua7/SC-GS's train_gui.py crash / preemption
# safe on resume. Applied by exe/scgs_run.sh at launch against the installed
# copy at /opt/SC-GS/train_gui.py (arg 1 = path to train_gui.py).
#
# IDEMPOTENT: every edit is guarded by a '# [anchorflow-resume]' marker string,
# so re-running on an already-patched file is a no-op. Safe to apply on every
# GPU launch (the instance can be preempted at any time).
#
# What it changes (all faithful to the real train_gui.py control flow):
#   1. Injects two module-level helpers before `if __name__ == "__main__":`
#        _anchorflow_write_resume_state(model_path, it, phase)
#            atomically writes <model_path>/resume_state.json AS THE COMMIT
#            MARKER -- last, after point_cloud.ply + deform.pth are on disk.
#        _anchorflow_reconcile_checkpoints(model_path)
#            before any load, drops any iteration_<N> checkpoint newer than the
#            last atomically-committed one so that Scene(load_iteration=-1) and
#            DeformModel.load_weights(-1) -- which each blindly take the MAX
#            iteration_<N> folder (utils/system_utils.searchForMaxIteration) --
#            converge on the SAME, fully-written iteration even when a save was
#            interrupted mid-write by preemption.
#   2. Rewrites GUI.train() so the main phase is driven by self.iteration up to
#      opt.iterations (resume-safe) instead of the original fixed
#      `for i in tqdm.trange(iters + iterations_node_rendering)` which, on
#      resume (self.iteration = loaded_iter), overshot the target by ~loaded_iter
#      extra steps. The node-bootstrap phase is kept, but on resume __init__ has
#      already set self.iteration_node_rendering == iterations_node_rendering, so
#      it is correctly skipped.
#   3. Appends the resume_state.json commit-marker write to the train_step()
#      checkpoint block (right after scene.save + deform.save_weights).
#   4. Calls _anchorflow_reconcile_checkpoints(args.model_path) in __main__,
#      before the GUI is constructed (i.e. before anything loads a checkpoint).
#
# NOTE on the node-bootstrap phase: train_node_rendering_step() writes NO
# checkpoint (only train_step does), so a preemption during the <=
# iterations_node_rendering (default 10000) node phase loses that phase and it
# re-runs from scratch on restart -- by design, it is short and cheap. Only the
# MAIN phase is checkpointed/resumed.
#
# NOTE on the optimizer: SC-GS does NOT persist Adam optimizer state (only the
# canonical GaussianModel .ply and the deform state_dict). On resume the
# optimizers are re-initialised by training_setup/train_setting; Adam re-warms in
# a few hundred steps. The trained gaussians + deform network (the expensive
# state) are fully preserved -- that is what matters.
# ============================================================================
import sys

path = sys.argv[1]
with open(path, 'r') as f:
    src = f.read()
orig = src

# --- 1. module-level helpers -------------------------------------------------
HELPERS = '''
def _anchorflow_write_resume_state(model_path, iteration, phase):  # [anchorflow-resume]
    """Commit marker: written ATOMICALLY and LAST, after point_cloud.ply + deform.pth."""
    import os, json
    tmp = os.path.join(model_path, 'resume_state.json.tmp')
    dst = os.path.join(model_path, 'resume_state.json')
    with open(tmp, 'w') as f:
        json.dump({'iteration': int(iteration), 'phase': phase,
                   'node_bootstrap_done': True}, f)
    os.replace(tmp, dst)  # atomic on POSIX -> resume_state.json is never half-written


def _anchorflow_reconcile_checkpoints(model_path):  # [anchorflow-resume]
    """Crash-safe resume. Scene(load_iteration=-1) and deform.load_weights(-1) each
    take the MAX iteration_<N> folder name blindly (searchForMaxIteration), so a save
    interrupted by preemption (point_cloud written but deform not, or a truncated .ply)
    would make them load MISMATCHED iterations or crash. Drop every iteration_<N> newer
    than the last atomically-committed checkpoint so both loaders land on the same
    fully-written iteration. Returns that iteration (or None for a fresh start)."""
    import os, json, shutil
    pc_dir = os.path.join(model_path, 'point_cloud')
    df_dir = os.path.join(model_path, 'deform')

    def iters_in(d):
        if not os.path.isdir(d):
            return set()
        return {int(n.split('_')[-1]) for n in os.listdir(d)
                if n.startswith('iteration_') and n.split('_')[-1].isdigit()}

    def artifacts_ok(it):
        ply = os.path.join(pc_dir, 'iteration_%d' % it, 'point_cloud.ply')
        pth = os.path.join(df_dir, 'iteration_%d' % it, 'deform.pth')
        return (os.path.isfile(ply) and os.path.getsize(ply) > 0 and
                os.path.isfile(pth) and os.path.getsize(pth) > 0)

    # Authoritative commit pointer (written after both artifacts). Fall back to a
    # scan for the newest iteration whose ply AND deform.pth both exist & are non-empty.
    committed = None
    sj = os.path.join(model_path, 'resume_state.json')
    if os.path.isfile(sj):
        try:
            committed = int(json.load(open(sj)).get('iteration'))
        except Exception:
            committed = None
    if committed is not None and not artifacts_ok(committed):
        committed = None
    if committed is None:
        for it in sorted(iters_in(pc_dir) | iters_in(df_dir), reverse=True):
            if artifacts_ok(it):
                committed = it
                break

    # Prune every iteration dir newer than the committed one (partial/mismatched writes).
    for d in (pc_dir, df_dir):
        for it in iters_in(d):
            if committed is None or it > committed:
                shutil.rmtree(os.path.join(d, 'iteration_%d' % it), ignore_errors=True)

    if committed is None:
        print('[anchorflow-resume] no complete checkpoint; fresh start '
              '(node-bootstrap phase runs from scratch).')
    else:
        print('[anchorflow-resume] resuming MAIN phase from committed iteration %d '
              '(node-bootstrap skipped; pruned any partial newer saves).' % committed)
    return committed


'''
if '_anchorflow_reconcile_checkpoints' not in src:
    anchor = 'if __name__ == "__main__":'
    assert anchor in src, 'anchor `if __name__ == "__main__":` not found'
    src = src.replace(anchor, HELPERS + anchor, 1)

# --- 2. resume-safe GUI.train() loop ----------------------------------------
OLD_LOOP = '''    def train(self, iters=5000):
        if iters > 0:
            for i in tqdm.trange(iters+self.opt.iterations_node_rendering):
                if self.deform.name == 'node' and self.iteration_node_rendering < self.opt.iterations_node_rendering:
                    self.train_node_rendering_step()
                else:
                    self.train_step()'''
NEW_LOOP = '''    def train(self, iters=5000):
        if iters > 0:
            # [anchorflow-resume] node-bootstrap phase: run until iterations_node_rendering.
            # On resume, __init__ set self.iteration_node_rendering == iterations_node_rendering,
            # so this body never runs and the (never-checkpointed) bootstrap is skipped.
            while self.deform.name == 'node' and self.iteration_node_rendering < self.opt.iterations_node_rendering:
                self.train_node_rendering_step()
            # [anchorflow-resume] main phase: drive self.iteration (which starts at
            # scene.loaded_iter on resume, else 1) up to opt.iterations. Replaces the
            # original fixed `tqdm.trange(iters + iterations_node_rendering)`, which on
            # resume overshot the target by ~loaded_iter extra train_steps.
            while self.iteration <= self.opt.iterations:
                self.train_step()'''
if NEW_LOOP not in src:
    assert OLD_LOOP in src, 'GUI.train() loop anchor not found'
    src = src.replace(OLD_LOOP, NEW_LOOP, 1)

# --- 3. commit-marker write in the train_step() checkpoint block -------------
OLD_SAVE = '''                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)'''
NEW_SAVE = '''                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)
                _anchorflow_write_resume_state(self.args.model_path, self.iteration, 'main')  # [anchorflow-resume]'''
if NEW_SAVE not in src:
    assert OLD_SAVE in src, 'train_step() checkpoint block anchor not found'
    # 16-space indent matches ONLY the train_step() save block, not the 24-space
    # GUI callback_save one, so the first-occurrence replace is unambiguous.
    src = src.replace(OLD_SAVE, NEW_SAVE, 1)

# --- 4. reconcile before the GUI (any load) in __main__ ----------------------
OLD_MAIN = '''    print("Optimizing " + args.model_path)
    safe_state(args.quiet)'''
NEW_MAIN = '''    print("Optimizing " + args.model_path)
    _anchorflow_reconcile_checkpoints(args.model_path)  # [anchorflow-resume]
    safe_state(args.quiet)'''
if '_anchorflow_reconcile_checkpoints(args.model_path)' not in src:
    assert OLD_MAIN in src, '__main__ anchor not found'
    src = src.replace(OLD_MAIN, NEW_MAIN, 1)

if src != orig:
    with open(path, 'w') as f:
        f.write(src)
    print('[scgs_resume_patch] applied resume patch to %s' % path)
else:
    print('[scgs_resume_patch] resume patch already present; no change')
