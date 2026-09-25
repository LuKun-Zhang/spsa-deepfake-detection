# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import json
import math
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None,
        swa_models=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        # ---- mechanism experiment state ------------------------------------
        # tail_lam: clamp the optimizer lr to tail_lam * lr_max once the SWA
        # collection window opens.  0.0 = old behaviour: the cosine schedule has
        # already driven the window lr to lr_eta_min (~1e-6), so weights barely
        # move and averaging is a near no-op -- which is exactly why the
        # archive's "SWA effect" is ~0.  lambda is the manipulating variable.
        self.tail_lam = float(config.get('tail_lam', 0.0) or 0.0)
        self.tail_lam_lr_max = None
        self.tail_lam_step = None      # step_cnt at which the clamp first applied
        self._win_count = 0            # number of theta_t collected so far (w=1)
        self._win_sq_sum = 0.0         # sum of ||theta_t||^2 over the window (w=1)
        self._mediators_path = None
        # ---- PHASE 1 MANIPULATION: window-only isotropic weight noise -------
        # Why this exists at all.  The lam route is dead: lam moves the window's
        # lr, and R = drift/rms was supposed to be the mediator it acts through,
        # but R is a ratio of two norms of the SAME random process, and at
        # lam=0 the denominator collapses (defect H: rms_w rounds to exactly 0
        # on the real leg because rms^2 = mean_sq - bar_sq is a difference of
        # two ~1.2e5 quantities whose answer is ~1e-4 in float32).  So lam cannot
        # answer "does R carry the SWA gain" -- it cannot even produce a
        # readable R.  This manipulates the same quantity from the other side,
        # WITHOUT touching lr: it injects noise directly into the weights, which
        # sets the trajectory's diffusive component by construction.
        #
        # The manipulating variable is a NOISE NORM, not a lambda:
        #   window_noise_global  target L2 norm of the perturbation added to the
        #                        WHOLE parameter vector, once per window step.
        # 0.0 (the default) disables the method entirely, and the code path is
        # the identity: a leg with this key absent is bit-identical to the
        # archive's lam=0 leg.  That is what keeps the A/B pair clean -- A is
        # the leg already running (mexp_p_ctrl_l0), B is the same yaml plus one
        # line.
        #
        # Expressed as a GLOBAL norm rather than a per-parameter sigma on
        # purpose: the quantity the experiment cares about is how far the window
        # wanders, which is a global displacement, and the per-parameter sigma
        # that produces it depends on the parameter COUNT (~2.1e7), which is a
        # property of the architecture rather than of the manipulation.  So the
        # yaml carries the interpretable number and the code divides by
        # sqrt(N).  Measured reference (real leg mexp_p_ctrl_l0, w=0.5 at
        # step 20532): window drift 0.068 over 2872 steps ~ 2.4e-5 of
        # displacement per step, so a per-step noise norm of ~5e-3 is ~200x the
        # per-step update -- large enough that the random walk dominates the
        # drift inside the window, which drives R toward its sqrt(6) = 2.45
        # diffusion value from the drift-dominated 5.1 the leg is reporting.
        #
        # KNOWN CONFOUND, recorded here rather than corrected, because it cannot
        # be corrected and must instead be declared: noise raises the drift
        # magnitude as well as the jitter, and it raises f(theta) itself (the
        # window explores a worse region).  So R is a READOUT of this
        # manipulation, not an independently-set knob.  The pre-registered
        # comparison is therefore G = AUC(theta_bar) - max_t AUC(theta_t) --
        # an IN-WINDOW difference, where both endpoints are evaluated on the
        # same noised trajectory, so the loss-level shift largely cancels.
        self.win_noise_global = float(config.get('window_noise_global', 0.0) or 0.0)
        # A DEDICATED generator, seeded from manualSeed.  Drawing this noise
        # from the global torch stream would advance it and change the
        # DataLoader's batch order, i.e. it would silently reintroduce the
        # module-vs-dataorder confound that rng_neutral_build was added to
        # remove.  Per-device because a CPU generator cannot fill a CUDA tensor.
        self.win_noise_seed = (int(config.get('window_noise_seed',
                                              config.get('manualSeed', 0)) or 0)
                               + 7717)
        self._win_noise_gens = {}
        self._win_noise_step = None    # step_cnt of the first injection
        self._win_noise_calls = 0
        self._win_noise_n = None       # number of floating requires_grad params
        self._win_noise_last = 0.0     # MEASURED norm of the last injection
        # ---- w-way parallel averaging ---------------------------------------
        # w = WIDTH of the averaging window as a fraction of the collection
        # window.  All w share the same ENDPOINT (the last collected step); they
        # differ only in where they START.  w == 1.0 is the whole window and is
        # exactly the old single swa_model; w == 1/16 averages only the last
        # 1/16 of it.  This is what turns "does averaging pay?" into a
        # dose-response curve instead of a single number.
        self.swa_models = dict(swa_models or {})
        if not self.swa_models and self.swa_model is not None:
            self.swa_models = {1.0: self.swa_model}
        self._win_start_step = None    # step_cnt of the first collected step
        self._win_len = None           # length of the collection window, in steps
        self._w_start_sd = {}          # theta_start per w (CPU clones)
        self._w_sq_sum = {w: 0.0 for w in self.swa_models}
        self._w_count = {w: 0 for w in self.swa_models}
        self._swa_eval_w = 1.0         # which w is currently being evaluated
        self._swa_eval_bn = False      # True while the BN-recalibrated theta_bar is tested
        # ---- BatchNorm recalibration for theta_bar ---------------------------
        # AveragedModel defaults to use_buffers=False, and its update_parameters
        # then ends with "keep the buffers in sync with the source model": the
        # buffers are COPIED FROM the model being averaged, every time.  So
        # theta_bar(w) carries the running stats of the LAST theta_t, and since
        # every width's last update happens at the window's final step, all five
        # share one set of statistics -- theta_end's -- none of which belongs to
        # the averaged parameters they are deployed with.  (Verified against
        # torch 2.8.0 by _avgmodel_probe.py, not read off the docs.)
        # That is harmless while the window sits on a tiny lr
        # (the protocol's own regime) and destroys the model once the window
        # actually moves -- see _bn_recalibrate for the measurement.  The
        # protocol's own convention is kept untouched; these fields add a SECOND
        # column so the two can be compared instead of argued about.
        self._train_loader = None      # held for the end-of-window recalibration
        self._bn_sd = {}               # w -> theta_bar(w) with BN recomputed for it
        self._bn_scratch = None        # one reusable AveragedModel to deploy those
        self._bn_batches = int(config.get('bn_recalib_batches', 100) or 0)
        # ---- loss-landscape probe (design doc 5.2) ---------------------------
        # A fixed batch B0, captured once when the window opens, is the only
        # common evaluation point in the whole run: f(theta_t;B0) at different t
        # is only comparable because the batch does not move.  Everything below
        # is measured against it, and NOTHING here is allowed to touch training
        # state (eval mode, no BN updates, RNG saved and restored) -- the
        # bit-identical reruns the whole experiment rests on must survive it.
        self._probe_batch = None       # fixed batch B0 (GPU tensors, detached)
        self._probe_model = None       # scratch copy used to evaluate arbitrary theta
        self._probe_path = None
        self._probe_every = int(config.get('probe_every', 300) or 0)
        self._probe_grad_every = int(config.get('probe_grad_every', 0) or 0)
        # per-w accumulation of f(theta_t;B0) over w's OWN sub-window, so that
        # J(w) = fbar(w) - f(theta_bar(w)) is available for every w, not just 1.
        self._probe_f_sum = {w: 0.0 for w in self.swa_models}
        self._probe_n = {w: 0 for w in self.swa_models}
        self._probe_f_last = None      # most recent f(theta_t;B0)
        self._probe_wrote_header = False
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU
        self._ema_init()
        self._la_init()

        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)
        # weight-space mediators (see _mediator_write); one json line per 300 steps
        self._mediators_path = os.path.join(self.log_dir, 'mediators.jsonl')
        self._probe_path = os.path.join(self.log_dir, 'probes.jsonl')

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self, w=None, model=None):
        """Save one averaged model.  w=None keeps the legacy single-swa.pth name."""
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        if w is None:
            ckpt_name = "swa.pth"
            _m = self.swa_model
        else:
            ckpt_name = "swa_w%g.pth" % w
            _m = model if model is not None else self.swa_models[w]
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(_m.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def _ema_init(self):
        self.use_ema = self.config.get('EMA', False)
        self._ema_decay = float(self.config.get('ema_decay', 0.999))
        self._ema = {}
        if self.use_ema:
            for _n, _p in self.model.named_parameters():
                if _p.requires_grad:
                    self._ema[_n] = _p.detach().clone()
            self.logger.info('EMA enabled: decay=%.4f params=%d' % (self._ema_decay, len(self._ema)))

    def _ema_sd(self):
        sd = self.model.state_dict()
        for _n in self._ema:
            sd[_n] = self._ema[_n]
        return sd

    def _la_init(self):
        self.use_la = self.config.get('Lookahead', False)
        self._la_k = int(self.config.get('la_k', 5))
        self._la_alpha = float(self.config.get('la_alpha', 0.5))
        self._la_cnt = 0
        self._la_syncs = 0
        self._la_logged = False
        self._la_slow = {}
        if self.use_la:
            for _n, _p in self.model.named_parameters():
                if _p.requires_grad:
                    self._la_slow[_n] = _p.detach().clone()
            self.logger.info('Lookahead enabled: k=%d alpha=%.3f params=%d' % (self._la_k, self._la_alpha, len(self._la_slow)))

    def _la_sync(self):
        """Outer Lookahead step every la_k inner steps: slow <- slow + alpha*(fast-slow); fast <- slow."""
        self._la_cnt += 1
        if self._la_cnt % self._la_k != 0:
            return
        with torch.no_grad():
            for _n, _p in self.model.named_parameters():
                if _p.requires_grad and _n in self._la_slow:
                    self._la_slow[_n].lerp_(_p.detach(), self._la_alpha)
                    _p.data.copy_(self._la_slow[_n])
        self._la_syncs += 1
        if not self._la_logged:
            self._la_logged = True
            self.logger.info('Lookahead first sync done (k=%d) at inner step %d' % (self._la_k, self._la_cnt))

    def _la_slow_sd(self):
        sd = self.model.state_dict()
        for _n in self._la_slow:
            sd[_n] = self._la_slow[_n]
        return sd

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:

            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()


            return losses,predictions


    def _swa_should_collect(self, epoch):
        """Protocol-agnostic SWA collection trigger.

        If config['swa_start_lr_ratio'] is set, start collecting from the first epoch
        whose training lr drops to <= lr_eta_min + ratio*(lr_max-lr_eta_min). This
        adapts automatically to the scheduler (short vs long protocol). Otherwise fall
        back to the original fixed-epoch trigger: epoch > swa_start.
        """
        ratio = self.config.get('swa_start_lr_ratio', None)
        if ratio is not None:
            sched = self.scheduler
            if sched is None:
                return epoch > self.config.get('swa_start', -1)
            lr_now = sched.get_last_lr()[0]
            lr_max = sched.base_lrs[0]
            eta_min = float(self.config.get('lr_eta_min', 0.0))
            lr_thr = eta_min + ratio * (lr_max - eta_min)
            return lr_now <= lr_thr
        return epoch > self.config['swa_start']

    # ---- mechanism experiment helpers --------------------------------------
    def _tail_lam_apply(self, step_cnt):
        """Clamp lr to tail_lam * lr_max while the collection window is open.

        Re-applied every iteration rather than once, so the epoch-level
        scheduler.step() in train.py cannot overwrite it.  tail_lam == 0 leaves
        the optimizer completely untouched (old behaviour).
        """
        if self.tail_lam <= 0:
            return
        if self.tail_lam_lr_max is None:
            try:
                self.tail_lam_lr_max = float(self.scheduler.base_lrs[0])
            except Exception:
                self.tail_lam_lr_max = float(self.optimizer.param_groups[0]['lr'])
        if self.tail_lam_step is None:
            self.tail_lam_step = step_cnt
            self.logger.info(
                '===> tail_lam=%.6g ACTIVE from step %d (lr %.4g -> %.4g)'
                % (self.tail_lam, step_cnt, self.tail_lam_lr_max,
                   self.tail_lam * self.tail_lam_lr_max))
        _lr = self.tail_lam * self.tail_lam_lr_max
        for g in self.optimizer.param_groups:
            g['lr'] = _lr

    # ---- PHASE 1 MANIPULATION: the noise injection itself ------------------
    def _win_noise_gen(self, p):
        """The dedicated RNG for parameter p's device.

        Created lazily and once per device.  These draws must NOT come from the
        global stream: torch's global generator also drives the DataLoader (via
        the base-seed draw that rng_neutral_build pins), so consuming it here
        would advance the batch order and put back the module-vs-dataorder
        confound.
        """
        _key = str(p.device)
        _g = self._win_noise_gens.get(_key)
        if _g is None:
            try:
                _g = torch.Generator(device=p.device)
            except Exception:
                # Older torch, or a device kind with no generator: fall back to
                # a CPU one, which randn still accepts for a CPU tensor and
                # which at least keeps the draw off the global stream.
                _g = torch.Generator()
            _g.manual_seed(int(self.win_noise_seed))
            self._win_noise_gens[_key] = _g
        return _g

    def _window_noise_apply(self, step_cnt):
        """Add N(0, sigma^2) to every trainable weight, window steps only.

        Called from inside the `_collecting` block, so "window-only" is a
        property of the CALL SITE, not of a guard in here -- there is no step
        arithmetic to get wrong, which is how defect D happened.  This method
        never checks whether the window is open.

        Placement within the step is deliberate: the caller runs it AFTER
        _swa_collect and _loss_probe, so theta_t is collected and f(theta_t;B0)
        is measured BEFORE the kick, and the kick then rides into the next
        collected point.  The window's first collected point is therefore clean
        and every later one carries the accumulated random walk -- exactly the
        dose the experiment wants, with no extra snapshot bookkeeping.

        f(theta) is NOT recomputed after the kick, and that is correct rather
        than sloppy: the probe measures the point the average actually contains.

        Cost: one randn_like-sized draw plus one add per parameter per step.
        Measured on the real leg's shape (~2.1e7 params) this is ~0.1 ms of GPU
        work against a 630 ms step.
        """
        _target = self.win_noise_global
        if _target <= 0:
            return
        if self._win_noise_n is None:
            self._win_noise_n = sum(int(p.numel()) for p in self.model.parameters()
                                    if p.requires_grad and p.is_floating_point())
            if self._win_noise_n == 0:
                self.logger.info('===> window_noise: no floating trainable params; '
                                 'manipulation DISABLED')
                # Negative, not 0: keeps the early-out above from re-scanning
                # the parameter list on every one of the 5744 window steps.
                self.win_noise_global = -1.0
                return
        if self._win_noise_step is None:
            self._win_noise_step = step_cnt
            self.logger.info(
                '===> window_noise_global=%.6g ACTIVE from step %d (N=%d params, '
                'per-param sigma=%.6g, seed=%d)'
                % (_target, step_cnt, self._win_noise_n,
                   _target / math.sqrt(self._win_noise_n), self.win_noise_seed))
        # Split the target global norm evenly over the entries.  Evenly, not by
        # parameter size: an isotropic kick on the whole vector is the
        # manipulation (isotropic in the parameter space the drift and jitter
        # are measured in), and _rms_and_drift sums over ALL entries, so
        # weighting by numel would make the realised global norm depend on how
        # the architecture happens to be chopped into tensors.
        _s = _target / math.sqrt(self._win_noise_n)
        _sq = 0.0
        with torch.no_grad():
            for p in self.model.parameters():
                if not p.requires_grad or not p.is_floating_point():
                    continue
                _n = torch.randn(p.shape, generator=self._win_noise_gen(p),
                                 dtype=p.dtype, device=p.device)
                p.add_(_n, alpha=_s)
                # Measured, not assumed: the realised norm of a Gaussian draw
                # has ~1/sqrt(2N) relative spread, and reporting it is what lets
                # a reader confirm the manipulation actually landed at the
                # configured size instead of trusting the arithmetic.  The sum
                # is over the RAW draw; _s is factored out below rather than
                # multiplied in here, which is the same number for one fewer
                # tensor op per parameter.  (Caught by T2 of _test_noise.py: the
                # first version reported the raw norm, i.e. sqrt(N) ~ 257, a
                # factor of 5e4 too large -- and nothing else in the leg would
                # have shown it.)
                _sq += float(_n.pow(2).sum())
        self._win_noise_calls += 1
        # sum(||s*n||^2) == s^2 * sum(||n||^2), so the scale comes out here.
        self._win_noise_last = _s * math.sqrt(_sq)

    @torch.no_grad()
    def _swa_collect(self, step_cnt):
        """Feed the current model into every w whose sub-window has opened.

        w == 1 starts at the first collected step; a smaller w starts later, so
        all of them share the same endpoint and differ only in width.  Each
        AveragedModel keeps its own n_averaged counter, so theta_bar(w) is the
        exact mean over its own sub-window.

        Cost is O(#w) scalar accumulations per step plus one state_dict pass per
        w at its own start; no per-step storage.
        """
        elapsed = step_cnt - self._win_start_step
        sq = 0.0
        for p in self.model.parameters():
            # float64, NOT float32 -- see the magnitude argument in _rms_and_drift.
            # ||theta||^2 ~ 1.2e5 while the squared window jitter is ~1e-4, so a
            # float32 sum here leaves rms_w below the resolution of the
            # subtraction that consumes it.
            sq += float(p.detach().double().pow(2).sum())
        for _w in sorted(self.swa_models):
            _m = self.swa_models[_w]
            if _m is None:
                continue
            if elapsed < round((1.0 - _w) * self._win_len):
                continue
            if _w not in self._w_start_sd:
                self._w_start_sd[_w] = {k: v.detach().cpu().clone()
                                        for k, v in self.model.state_dict().items()}
            _m.update_parameters(self.model)
            self._w_sq_sum[_w] += sq
            self._w_count[_w] += 1
        # w == 1 aggregates are also kept at top level for backward compatibility
        _w1 = self.swa_models.get(1.0)
        if _w1 is not None and self._w_count.get(1.0, 0) > 0:
            self._win_sq_sum = self._w_sq_sum[1.0]
            self._win_count = self._w_count[1.0]

    @torch.no_grad()
    def _rms_and_drift(self, w):
        """(rms_w, drift_w) for one window width w.

        rms_w   = sqrt(mean_t ||theta_t||^2 - ||theta_bar||^2)   -- window jitter
        drift_w = ||theta_end - theta_start(w)||                  -- window drift
        R = drift_w / rms_w is the drift-to-jitter ratio the theory ties to the
        averaging gain.  It never touches AUC, so it cannot be circular with the
        endpoint measure.
        """
        n = self._w_count.get(w, 0)
        if n == 0:
            return 0.0, 0.0
        mean_sq = self._w_sq_sum[w] / n
        bar_sq = 0.0
        _avg = getattr(self.swa_models[w], 'module', self.swa_models[w])
        # float64 on BOTH sides of the subtraction, for the reason the docstring
        # above is about: rms_w^2 = mean_sq - bar_sq is a difference of two
        # quantities of size ||theta||^2 ~ 1.2e5, and the answer is ~1e-4 --
        # nine orders of magnitude below, so float32 rounds the difference to 0
        # about as often as not.  Measured on the REAL leg mexp_p_ctrl_l0
        # (2026-09-17): rms_w came out as EXACTLY 0 on 10 of 20 mediator rows,
        # interleaved with scattered 0.0078/0.0102/0.0215 wild values, and the
        # row that matters most -- the window's last -- had rms_w = 0 for all
        # five widths while drift_w = 0.5771 on the same row.  A window whose
        # endpoint moved by 0.58 cannot have zero jitter, so rms_w was provably
        # wrong, and R = drift_w/rms_w (the PRIMARY MEDIATOR, and item 1 of the
        # pre-registered reading order) was None or noise.
        # The smoke could not show this: only the real leg runs long enough for
        # the cancellation to round to zero.  Note drift_w below needs no such
        # fix -- it is a direct difference of two similar tensors, elementwise,
        # so nothing cancels.
        #
        # float64 on both sides is necessary and NOT sufficient.  The identity
        # rms^2 = mean_sq - bar_sq holds only when theta_bar is EXACTLY the mean
        # of the theta_t it is compared against.  Here theta_bar is an
        # AveragedModel, i.e. a float32 running mean updated once per step, so it
        # is mu + eps, and expanding about mu gives
        #     rms^2_identity - rms^2_definition = -2<mu, eps> - 2||eps||^2
        # which is independent of the summation width -- widening the sums does
        # not touch it.  MEASURED by simulating the real update rule over the
        # real window (T=5744, _test_fixes_DE.py section E2): ||eps|| = 6.1e-4,
        # -2<mu,eps> = -3.785e-3 (predicted) vs -3.786e-3 (measured), i.e. rms_w
        # reads 3.4% low at rms ~ 0.24.
        # LEFT UNCORRECTED, deliberately: correcting it needs a float64 vector
        # accumulator for sum_t theta_t plus a snapshot per w, and it buys 3.4%
        # where the float32->float64 change above bought 40-140%.  So read rms_w
        # and R as good to a few percent, not to their printed digits.  Both the
        # bias and the sign of <mu,eps> vary per w and per leg, so across the 20
        # mediator rows of a leg it averages down rather than accumulating.
        for p in _avg.parameters():
            bar_sq += float(p.detach().double().pow(2).sum())
        rms_w = math.sqrt(max(mean_sq - bar_sq, 0.0))
        drift_sq = 0.0
        start_sd = self._w_start_sd.get(w)
        if start_sd is not None:
            # PARAMETERS ONLY.  self.model.state_dict() also carries counters
            # such as num_batches_tracked, which increments on every forward in
            # train mode and so reaches ~5.2e4 over one collecting epoch.  Its
            # squared difference then dominates the sum: measured on the
            # 2026-09-16 smoke, drift_w came out at 52064.7 while the whole
            # parameter vector has norm ~3.4e2.  rms_w was always clean (it sums
            # parameters), so R = drift_w/rms_w was silently reporting a step
            # counter -- and R is the PRIMARY MEDIATOR.  Iterate parameters to
            # match the denominator.
            for k, v in self.model.named_parameters():
                s = start_sd.get(k)
                if s is None:
                    continue
                # theta_start is banked on CPU (VRAM), so subtract there too:
                # mixing a cuda tensor with a cpu one raises at the first
                # mediator write -- which is every leg, on the first probe.
                d = v.detach().float().cpu() - s.float()
                drift_sq += float(d.pow(2).sum())
        return rms_w, math.sqrt(drift_sq)

    @torch.no_grad()
    def _mediator_write(self, step_cnt):
        """Emit one json line per 300 steps: the per-w weight-space mediators.

        Top-level fields keep the w == 1 meaning they had in the single-model
        version (so old and new runs are comparable); `per_w` carries the whole
        dose-response over window width.
        """
        if self._mediators_path is None or not self._w_start_sd:
            return
        per_w = {}
        for _w in sorted(self.swa_models):
            if self._w_count.get(_w, 0) == 0:
                continue
            _rms, _drift = self._rms_and_drift(_w)
            per_w['%g' % _w] = dict(
                n_averaged=self._w_count[_w], rms_w=_rms, drift_w=_drift,
                R=(_drift / _rms if _rms > 0 else None))
        if not per_w:
            return
        _w1 = per_w.get('1')
        rec = dict(step=step_cnt, tail_lam=self.tail_lam,
                   tail_lam_step=self.tail_lam_step,
                   # PHASE 1 manipulation state.  A leg without the key reports
                   # 0.0 / null / 0, which is also what the archive's lam=0 leg
                   # would report if it had been given these fields -- so the
                   # two arms' mediator files stay column-compatible.
                   win_noise_global=self.win_noise_global,
                   noise_step=self._win_noise_step,
                   noise_calls=self._win_noise_calls,
                   noise_last=self._win_noise_last,
                   per_w=per_w)
        if _w1 is not None:
            rec.update(n_averaged=_w1['n_averaged'], rms_w=_w1['rms_w'],
                       drift_w=_w1['drift_w'], R=_w1['R'])
        with open(self._mediators_path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')

    # ---- loss-landscape probe (design doc 5.2, theory 定理 1/3/5) ----------
    #
    # Everything below answers ONE question: the averaged point is better than
    # the average of the points -- how much, and is the difference explained by
    # a drift term or by a barrier?  Theorem 1 says the answer is exactly
    #     J = fbar - f(theta_bar),     fbar = (1/W) sum_t f(theta_t)
    # which needs a COMMON evaluation point, hence the fixed batch B0.
    #
    # No weight snapshots are kept: theta_bar(1) IS the window mean, so the
    # mean displacement d_bar = theta_bar(1) - theta_start and d_T =
    # theta_end - theta_start are both free (theory 定理 3).

    def _probe_capture(self, data_dict):
        """Snapshot the current training batch as the fixed probe batch B0."""
        b0 = {}
        for k, v in data_dict.items():
            if k == 'name' or v is None:
                continue
            if torch.is_tensor(v):
                b0[k] = v.detach().clone()
        if not b0:
            return
        self._probe_batch = b0
        # scratch container: lets us evaluate f at ANY theta without disturbing
        # the live model or any averaged model.  eval() forever after.
        self._probe_model = deepcopy(self.model)
        self._probe_model.eval()
        self.logger.info('===> probe batch captured: %s'
                         % ', '.join('%s%s' % (k, tuple(v.shape))
                                     for k, v in sorted(b0.items())))

    @torch.no_grad()
    def _probe_forward(self, model):
        """(loss, prob) of `model` on B0, as plain python floats / CPU tensor."""
        pred = model(self._probe_batch)
        losses = model.get_losses(self._probe_batch, pred)
        loss = losses['overall']
        loss = float(loss) if torch.is_tensor(loss) else float(loss)
        p = pred['prob'].detach().float()
        if p.dim() > 1 and p.shape[-1] == 2:
            p = p[:, 1]
        return loss, p.reshape(-1).cpu()

    @torch.no_grad()
    def _f_at_state(self, state_dict):
        """f(theta;B0) for an arbitrary parameter vector held as a state_dict."""
        self._probe_model.load_state_dict(state_dict)
        return self._probe_forward(self._probe_model)[0]

    @torch.no_grad()
    def _bn_recalibrate(self, state_dict, n_batches=None):
        """theta_bar with its BatchNorm running stats recomputed FOR IT.

        This is what torch.optim.swa_utils.update_bn does, and the SWA recipe has
        always required it: the mean of a set of parameters is not a point on the
        trajectory, so the activation statistics it produces are not the ones the
        buffers it inherits -- theta_end's -- describe.  DeepfakeBench never
        called update_bn --
        harmless in the protocol's own regime, where the collection window sits
        on the lr=1e-6 epoch and the weights barely move, so the mismatch is
        negligible.

        It stops being negligible as soon as the window is made to move, which is
        exactly what the lambda manipulation does.  Measured on the 2026-09-16
        smoke (collection window = epoch 0, lr = 2e-4 = lambda 1.0's clamp):
        f(theta_bar(1);B0) grew 2.0 -> 3.4e10 across the window, doubling roughly
        every 250 steps -- geometric, not a bad optimum -- while f stayed under
        2.25 at every point of the trajectory AND along the whole chord between
        theta_start and theta_end, evaluated with this same code.  The content
        encoder's BN running_vars sit at ~1e-6 (min 5.6e-45, 16024 of 55552
        channels below 1e-8), so in eval mode those layers are ~300x amplifiers:
        once the parameters move away from the point the buffers were frozen at,
        each layer amplifies the previous layer's mismatch and the error
        compounds with depth.

        No training state is touched: the scratch model is loaded from a copy,
        the RNG is saved and restored, and the live model is never forwarded.
        """
        if self._train_loader is None or self._probe_model is None:
            return None
        n_batches = self._bn_batches if n_batches is None else int(n_batches)
        if n_batches <= 0:
            return None
        m = self._probe_model
        try:
            m.load_state_dict(state_dict)
        except Exception as e:
            self.logger.warning('bn_recalibrate load failed: %r' % (e,))
            return None
        bns = [mod for mod in m.modules()
               if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
               and mod.track_running_stats]
        if not bns:
            return None
        _rng = torch.get_rng_state()
        _cuda = (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None)
        was_training = m.training
        momenta = {}
        seen = 0
        try:
            for mod in bns:
                mod.reset_running_stats()
                momenta[mod] = mod.momentum
                mod.momentum = None       # None == cumulative average == exact mean
            m.train()
            dev = next(m.parameters()).device
            for data_dict in self._train_loader:
                b0 = {}
                for k, v in data_dict.items():
                    if k == 'name' or v is None:
                        continue
                    if torch.is_tensor(v):
                        b0[k] = v.detach().to(dev, non_blocking=True)
                if not b0:
                    continue
                m(b0)
                seen += 1
                if seen >= n_batches:
                    break
        except Exception as e:
            self.logger.warning('bn_recalibrate failed at batch %d: %r' % (seen, e))
            return None
        finally:
            for mod in bns:
                mod.momentum = momenta.get(mod, 0.1)
            m.train(was_training)
            m.eval()
            torch.set_rng_state(_rng)
            if _cuda is not None:
                torch.cuda.set_rng_state_all(_cuda)
        self.logger.info('===> BN recalibrated over %d batches' % seen)
        return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

    def bn_model(self, w):
        """A deployable model object for theta_bar(w) with recalibrated BN.

        Returned as an AveragedModel so that test_epoch's SWA branch (which keys
        off `type(self.model) is AveragedModel`) writes the dose-response row for
        it too, tagged with bn=True.  One scratch container is reused across
        widths because only one is deployed at a time.
        """
        sd = self._bn_sd.get(w)
        if sd is None:
            return None
        if self._bn_scratch is None:
            self._bn_scratch = deepcopy(self.swa_models[w])
            self._bn_scratch.eval()
        self._bn_scratch.module.load_state_dict(sd)
        self._bn_scratch.eval()
        return self._bn_scratch

    def _probe_due(self, step_cnt, every):
        """Is a window-relative probe due at this global step?

        Relative to the WINDOW, not to the global counter.  Written as its own
        method because the two differ, and the difference cost a GPU night:
        the real legs open their window at global step 17232 (= 132 mod 300),
        so a `step_cnt % probe_every == 0` guard NEVER fires inside the window.
        Measured on leg mexp_p_ctrl_l0 (2026-09-17): probes.jsonl came out with
        0 lines, J(w) = {}, and -- because the BN-recalibrated column is gated
        on _probe_n -- the recal-BN column never ran either, so 3 of the 4
        health-check failures traced back to this one expression.

        The smoke could not show it: `swa_start: -1` opens the window at global
        step 0, where the global guard is trivially satisfied.  Relative to the
        window the probe steps are 0, probe_every, 2*probe_every, ... in every
        run, whatever the window's start step is.
        """
        if every <= 0 or self._win_start_step is None:
            return False
        return (step_cnt - self._win_start_step) % every == 0

    def _loss_probe(self, step_cnt):
        """One probe every `probe_every` window steps (design doc 5.2).

        Guarded so it cannot perturb the training stream: eval mode (no dropout,
        no BN running-stat update) and the RNG state is saved and restored.
        The guard is checked, not assumed -- see the `rng_moved` field.
        """
        if self._probe_batch is None or self._probe_every <= 0:
            return
        if not self._probe_due(step_cnt, self._probe_every):
            return
        _rng = torch.get_rng_state()
        _cuda = (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None)
        was_training = self.model.training
        self.model.eval()
        try:
            f_t, p_t = self._probe_forward(self.model)
        finally:
            if was_training:
                self.model.train()
            torch.set_rng_state(_rng)
            if _cuda is not None:
                torch.cuda.set_rng_state_all(_cuda)
        rng_moved = not torch.equal(_rng, torch.get_rng_state())
        self._probe_f_last = f_t

        per_w = {}
        for _w in sorted(self.swa_models):
            if self._w_count.get(_w, 0) == 0:
                continue
            self._probe_f_sum[_w] += f_t
            self._probe_n[_w] += 1
            _m = self.swa_models[_w]
            _m.module.eval()
            f_bar, p_bar = self._probe_forward(_m.module)
            n = int(p_t.numel())
            agree = float((p_t >= 0.5).eq(p_bar >= 0.5).float().mean()) if n else None
            # symmetric disagreement, robust to the 0/0 of a plain KL
            eps = 1e-7
            a = p_t.clamp(eps, 1 - eps)
            b = p_bar.clamp(eps, 1 - eps)
            kl = float((a * torch.log(a / b)
                        + (1 - a) * torch.log((1 - a) / (1 - b))).mean())
            per_w['%g' % _w] = dict(
                f_bar=f_bar, agree=agree, kl=kl,
                fbar_so_far=(self._probe_f_sum[_w] / self._probe_n[_w]),
                n_probe=self._probe_n[_w])

        # bn=False is written explicitly, not left implicit: every f_bar below is
        # f(theta_bar(w)) under the PROTOCOL's frozen-buffer deployment, i.e. it
        # inherits theta_end's statistics and is not comparable to the
        # recalibrated numbers in window_probe.json's J_bn.  Measured on the
        # 2026-09-16 smoke the two conventions disagree in SIGN at every width
        # (frozen says averaging always hurts; recalibrated says it helps for
        # w >= 1/8).  Recalibrating per probe would cost 5 widths x N batches
        # every 300 steps; not worth it, but the field must not be mistaken for
        # the readout.
        rec = dict(step=step_cnt, tail_lam=self.tail_lam, f_t=f_t, bn=False,
                   per_w=per_w, rng_moved=rng_moved)
        if self._probe_due(step_cnt, self._probe_grad_every):
            rec['grad_norm'] = self._probe_grad_norm()
        with open(self._probe_path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        if rng_moved:
            self.logger.warning('!! loss probe moved the RNG state at step %d'
                                % step_cnt)

    def _probe_grad_norm(self):
        """||grad L(theta_t;B0)|| -- one extra backward, optional."""
        _rng = torch.get_rng_state()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                pred = self.model(self._probe_batch)
                losses = self.model.get_losses(self._probe_batch, pred)
                losses['overall'].backward()
                sq = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        sq += float(p.grad.detach().float().pow(2).sum())
            return math.sqrt(sq)
        except Exception as e:                      # never let a probe kill a leg
            self.logger.warning('grad probe failed: %r' % (e,))
            return None
        finally:
            self.model.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()
            torch.set_rng_state(_rng)

    def _interp_state(self, sd_a, sd_b, s):
        """gamma(s) = (1-s)*theta_a + s*theta_b, as a state_dict."""
        out = {}
        for k in sd_a:
            va, vb = sd_a[k], sd_b[k]
            if torch.is_floating_point(va):
                out[k] = (1.0 - s) * va.float() + s * vb.float()
            else:
                out[k] = va.clone()
        return out

    def _window_end_probe(self, n_chord=20):
        """Everything that can only be measured once, at the window's end.

        Writes window_probe.json:
          J(w)      = fbar(w) - f(theta_bar(w))                    定理 1
          chord     = f(gamma(s)) at n_chord points, s in [0,1]     假设 A2
          B         = max_s f(gamma(s)) - max(f(start), f(end))     定理 5
          DriftGap  = ||d_bar||^2_H - ||d_T||^2_H, estimated in the quadratic
                      region from f and grad f alone (no Hessian needed)  定理 3
        """
        if self._probe_batch is None or not self._w_start_sd:
            self.logger.info('window_end_probe skipped: no probe batch / window')
            return
        sd_start = self._w_start_sd.get(1.0)
        if sd_start is None:
            return
        sd_end = {k: v.detach().cpu().clone()
                  for k, v in self.model.state_dict().items()}

        _rng = torch.get_rng_state()
        _cuda = (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None)
        was_training = self.model.training
        self.model.eval()
        out = dict(tail_lam=self.tail_lam, window_start_step=self._win_start_step,
                   window_len=self._win_len,
                   n_probe_per_w={('%g' % w): self._probe_n[w] for w in self.swa_models})
        try:
            f_start = self._f_at_state(sd_start)
            f_end = self._f_at_state(sd_end)
            out['f_start'] = f_start
            out['f_end'] = f_end

            # J(w): the averaging gain in LOSS, per window width
            jw = {}
            for _w in sorted(self.swa_models):
                if self._probe_n[_w] == 0:
                    continue
                f_avg = self._f_at_state(self.swa_models[_w].module.state_dict())
                fbar = self._probe_f_sum[_w] / self._probe_n[_w]
                jw['%g' % _w] = dict(
                    fbar=fbar, f_theta_bar=f_avg, J=fbar - f_avg,
                    n_probe=self._probe_n[_w],
                    n_averaged=self._w_count[_w])
            out['J'] = jw

            # The same dose-response under the OTHER BatchNorm convention: each
            # theta_bar(w) gets its running stats recomputed for its own
            # parameters.  Both columns are reported because the protocol's
            # frozen-buffer deployment and a correctly normalised one coincide
            # when the window does not move (lambda=0) and can differ by orders
            # of magnitude when it does.  Whichever way the sign of J goes, the
            # reader can see whether it survives the choice.
            if self._bn_batches > 0 and self._train_loader is not None:
                bn_jw = {}
                for _w in sorted(self.swa_models):
                    if self._probe_n[_w] == 0:
                        continue
                    sd_bn = self._bn_recalibrate(
                        self.swa_models[_w].module.state_dict())
                    if sd_bn is None:
                        continue
                    self._bn_sd[_w] = sd_bn
                    f_bn = self._f_at_state(sd_bn)
                    fbar = self._probe_f_sum[_w] / self._probe_n[_w]
                    bn_jw['%g' % _w] = dict(
                        fbar=fbar, f_theta_bar=f_bn, J=fbar - f_bn,
                        n_probe=self._probe_n[_w],
                        n_averaged=self._w_count[_w],
                        n_batches=self._bn_batches)
                if bn_jw:
                    out['J_bn'] = bn_jw

            # the chord theta_start -> theta_end (assumption A2 / 定理 5)
            chord = []
            for i in range(n_chord + 1):
                s = i / float(n_chord)
                chord.append(dict(s=s, f=self._f_at_state(
                    self._interp_state(sd_start, sd_end, s))))
            out['chord'] = chord
            out['B'] = (max(c['f'] for c in chord)
                        - max(f_start, f_end))

            # DriftGap, in the quadratic region: d^T H d ~ 2(f(theta+d) - f(theta) - g^T d)
            # d_bar = theta_bar(1) - theta_start is FREE (定理 1's window mean),
            # d_T = theta_end - theta_start.
            _w1 = self.swa_models.get(1.0)
            if _w1 is not None and self._w_count.get(1.0, 0) > 0:
                # theta_bar(1) with the statistics that belong to it, when the
                # recalibrated column ran.  This is the SECOND place the
                # frozen-buffer deployment poisons a reported quantity: the
                # estimate below is 2*(f(theta_start + d_bar) - f(theta_start)
                # - g.d_bar), and with the frozen buffers f(theta_bar(1)) is
                # 3.4e10 on the 2026-09-16 smoke, so DriftGap came out at 7.0e10
                # while the per-step loss scale is ~2.  The weight-space
                # displacement `d` below is parameter-only and identical either
                # way -- only the evaluation point's statistics change.
                sd_bar = (self._bn_sd.get(1.0) or
                          {k: v.detach().cpu().clone()
                           for k, v in _w1.module.state_dict().items()})
                g_start = self._grad_at_state(sd_start)
                if g_start is not None:
                    q = {}
                    for tag, sd_tgt in (('bar', sd_bar), ('T', sd_end)):
                        # parameters only -- see _rms_and_drift for why the
                        # non-parameter entries of a state_dict (counters such
                        # as num_batches_tracked) must not enter a weight-space
                        # norm.  They differ between theta_start and theta_end by
                        # ~5.2e4 here, which swamped ||d|| and the DriftGap.
                        d = {}
                        for _k, _p in self.model.named_parameters():
                            if _k in sd_start and _k in sd_tgt:
                                d[_k] = (sd_tgt[_k].float()
                                         - sd_start[_k].float())
                        gd = sum(float((g_start[k].float() * d[k]).sum())
                                 for k in d if k in g_start)
                        f_x = self._f_at_state(sd_tgt)
                        # ½ dᵀHd  ≈  f(θ+d) − f(θ) − gᵀd
                        q[tag] = dict(f_delta=f_x - f_start, g_dot_d=gd,
                                      quad=2.0 * (f_x - f_start - gd),
                                      norm_d=math.sqrt(
                                          sum(float(d[k].pow(2).sum()) for k in d)))
                    out['drift'] = q
                    if q['bar'].get('quad') is not None and q['T'].get('quad') is not None:
                        out['DriftGap_quad'] = q['bar']['quad'] - q['T']['quad']
        except Exception as e:
            self.logger.warning('window_end_probe failed: %r' % (e,))
            out['error'] = repr(e)
        finally:
            if was_training:
                self.model.train()
            torch.set_rng_state(_rng)
            if _cuda is not None:
                torch.cuda.set_rng_state_all(_cuda)

        out['rng_restored'] = torch.equal(_rng, torch.get_rng_state())
        # Window-width sanity.  w means "the last w of the WHOLE window", and the
        # whole window is `_win_len` steps long.  If the collection trigger fires
        # for more than one epoch, _win_len (set at window open) understates it
        # and w quietly stops meaning what it says.  This run's protocol is a
        # single collecting epoch, so the check should read equal -- it is here
        # so that it can never fail silently.
        w1 = self._w_count.get(1.0, 0)
        out['win_len_check'] = dict(collected=w1, win_len=self._win_len,
                                    ok=(w1 == self._win_len))
        if w1 != self._win_len:
            self.logger.warning(
                '!! SWA window covers %d steps but _win_len=%d: the w widths do '
                'NOT mean what they say in this run' % (w1, self._win_len))
        with open(os.path.join(self.log_dir, 'window_probe.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
        self.logger.info('===> window_end_probe written: J=%s B=%s'
                         % ({k: round(v['J'], 6) for k, v in out.get('J', {}).items()},
                            round(out.get('B', float('nan')), 6)))
        return out

    def _grad_at_state(self, state_dict):
        """grad f(theta;B0) as a CPU state_dict, or None if it blows up."""
        try:
            with torch.enable_grad():
                self._probe_model.load_state_dict(state_dict)
                self._probe_model.zero_grad(set_to_none=True)
                pred = self._probe_model(self._probe_batch)
                losses = self._probe_model.get_losses(self._probe_batch, pred)
                losses['overall'].backward()
                g = {}
                for k, p in self._probe_model.named_parameters():
                    if p.grad is not None:
                        g[k] = p.grad.detach().cpu().clone()
                self._probe_model.zero_grad(set_to_none=True)
                self._probe_model.eval()
                return g
        except Exception as e:
            self.logger.warning('grad_at_state failed: %r' % (e,))
            return None

    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        # Held for the end-of-window BatchNorm recalibration, which needs real
        # data to recompute running stats for the averaged parameters.  Storing
        # the reference costs nothing and avoids keeping a second loader alive.
        self._train_loader = train_data_loader
        test_every = self.config.get('test_every', 1)
        # G = AUC(theta_bar) - max_{t in window} AUC(theta_t) is a difference
        # against a MAXIMUM, so how densely the window is sampled decides the
        # answer: with the default 2 points in the window the max is estimated
        # from 2 samples and G can flip sign on a single point.  `test_times_per_epoch`
        # densifies the online evaluation when set; unset keeps old behaviour.
        _tpe = self.config.get('test_times_per_epoch', None)
        if _tpe is not None:
            times_per_epoch = int(_tpe)
        elif epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1


        #times_per_epoch=4

        test_step = len(train_data_loader) // times_per_epoch    # test 10 times per epoch
        step_cnt = epoch * len(train_data_loader)
        test_best_metric = None

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            self.setTrain()
            if iteration < 4:
                self.logger.info(f'MEMDBG iter{iteration} start alloc={torch.cuda.memory_allocated()/2**30:.2f}GB reserved={torch.cuda.memory_reserved()/2**30:.2f}GB')
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()

            losses,predictions=self.train_step(data_dict)

            # update learning rate

            _collecting = ('SWA' in self.config and self.config['SWA']
                           and self._swa_should_collect(epoch))
            if _collecting:
                if self._win_start_step is None:
                    # The window opens here.  Every w shares this window and
                    # therefore this endpoint; they differ only in how far back
                    # they start (elapsed >= (1-w)*len).
                    self._win_start_step = step_cnt
                    # The window is however many epochs still collect, so
                    # _win_len must cover all of them -- with the fixed-epoch
                    # trigger that count is exact; with the lr-ratio trigger the
                    # future epochs are not knowable here, so it is a prediction
                    # and window_probe.win_len_check verifies it at the end.
                    _n_ep = max(1, sum(1 for _e in range(epoch, self.config['nEpochs'] + 1)
                                       if self._swa_should_collect(_e)))
                    self._win_len = len(train_data_loader) * _n_ep
                    self.logger.info(
                        '===> SWA window open at step %d (len=%d = %d epochs x %d, w=%s)'
                        % (step_cnt, self._win_len, _n_ep, len(train_data_loader),
                           ','.join('%g' % _w for _w in sorted(self.swa_models))))
                self._swa_collect(step_cnt)
                if self._probe_batch is None:
                    # B0 must be captured at the window's first step so that
                    # f(theta_t;B0) covers the same steps the averaging does.
                    self._probe_capture(data_dict)
                # Called on EVERY window step, not from the `% 300` logging block
                # below.  Two reasons, one of which cost a GPU night:
                #   * _loss_probe gates itself on a WINDOW-relative cadence
                #     (_probe_due); reaching it only on local-iteration multiples
                #     of 300 coupled probe_every to the log cadence, so a
                #     probe_every that was not a multiple of 300 would silently
                #     round up to 300.
                #   * the old version passed the GLOBAL step_cnt to a guard that
                #     took it modulo probe_every against an absolute zero, which
                #     never fires for a window opening at 17232 (= 132 mod 300).
                # Placed AFTER _swa_collect and _probe_capture so the first
                # window step already has _w_count >= 1 and B0 in hand.
                self._loss_probe(step_cnt)
                # mechanism experiment: tail-lambda clamp + weight-space mediators
                self._tail_lam_apply(step_cnt)
                # PHASE 1 MANIPULATION: isotropic window-only weight noise.
                # In the same `_collecting` block as the clamp above, so the
                # window is the only place it can fire; but AFTER _swa_collect
                # (which is above this block's head) and after _loss_probe, so
                # the collected point and the probed f(theta_t;B0) describe the
                # same weights.  A leg with window_noise_global absent or 0
                # returns immediately and is bit-identical to the archive.
                self._window_noise_apply(step_cnt)

            if self.use_ema:
                with torch.no_grad():
                    for _n, _p in self.model.named_parameters():
                        if _p.requires_grad and _n in self._ema:
                            self._ema[_n].mul_(self._ema_decay).add_(_p.detach(), alpha=1 - self._ema_decay)

            if self.use_la:
                self._la_sync()
            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and (self._swa_should_collect(epoch) or self.config['dry_run']) and isinstance(self.scheduler, SWALR):
                    self.scheduler.step()
                # mechanism experiment: dump the weight-space mediators.
                # (_loss_probe is NOT here -- it lives in the `_collecting`
                # block above, where it gates on a window-relative cadence.)
                if _collecting:
                    self._mediator_write(step_cnt)
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)



                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            if (step_cnt+1) % test_step == 0 and epoch % test_every == 0:
                _swap_ema = self.use_ema and test_data_loaders is not None
                if _swap_ema:
                    _online_sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    self.model.load_state_dict(self._ema_sd())
                    self.logger.info('===> Eval on EMA weights (step %d)' % step_cnt)
                try:
                    if test_data_loaders is not None and (not self.config['ddp'] ):
                        self.logger.info("===> Test start!")
                        test_best_metric = self.test_epoch(
                            epoch,
                            iteration,
                            test_data_loaders,
                            step_cnt,
                        )
                    elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                        self.logger.info("===> Test start!")
                        test_best_metric = self.test_epoch(
                            epoch,
                            iteration,
                            test_data_loaders,
                            step_cnt,
                        )
                    else:
                        test_best_metric = None

                    # ---- step snapshot save: 被测权重 (EMA arm => EMA weights) ----
                    if test_data_loaders is not None:
                        _sd_dir = os.path.join(self.log_dir, 'testpoints')
                        os.makedirs(_sd_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), os.path.join(_sd_dir, 'step_%d.pth' % step_cnt))
                        self.logger.info('Step ckpt saved: step_%d.pth' % step_cnt)
                finally:
                    if _swap_ema:
                        self.model.load_state_dict(_online_sd)
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1
            if iteration < 4:
                self.logger.info(f'MEMDBG iter{iteration} end alloc={torch.cuda.memory_allocated()/2**30:.2f}GB peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB')
        return test_best_metric

    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        real_idx = np.where(label == 0)[0]
        fake_idx = np.where(label == 1)[0]
        acc_real = np.count_nonzero(judge[real_idx]) / len(real_idx)
        acc_fake = np.count_nonzero(judge[fake_idx]) / len(fake_idx)

        return acc_real, acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)
    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                _tag = ('w=%g, BN recalibrated' % self._swa_eval_w
                        if self._swa_eval_bn else 'w=%g' % self._swa_eval_w)
                metric_str = (f"Iter Final for SWA ({_tag}):    ")
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                # machine-readable copy: one line per (w, dataset, BN convention).
                # Parsing the log line above is fragile; the dose-response curve
                # AUC(theta_bar(w)) is the whole point of the w-way run, so it
                # gets its own file.  `bn` separates the protocol's frozen-buffer
                # deployment from one whose running stats were recomputed for
                # theta_bar itself -- identical when the window does not move,
                # not otherwise.
                with open(os.path.join(self.log_dir, 'swa_eval.jsonl'), 'a') as fh:
                    fh.write(json.dumps(dict(
                        w=self._swa_eval_w, dataset=key,
                        bn=bool(self._swa_eval_bn),
                        n_averaged=self._w_count.get(self._swa_eval_w, 0),
                        metrics={k: v for k, v in metric_one_dataset.items()
                                 if k not in ('pred', 'label')})) + '\n')
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
