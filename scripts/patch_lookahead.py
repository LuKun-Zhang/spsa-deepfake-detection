#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Add Lookahead (Zhang et al. 2019, fast/slow dual weights) support to the controlled
# protocol, mirroring the EMA patch's structure so the two live side by side:
#   - Trainer keeps slow copies of all requires_grad params.
#   - Every la_k inner Adam steps: slow <- slow + alpha*(fast-slow), then fast <- slow
#     (standard Lookahead outer step).
#   - The 7 training testpoints evaluate the ONLINE (fast) weights -- same metric
#     convention as CTRL/SAM, so the arm is fairly comparable (unlike EMA, whose 7
#     points run on the averaged weights and carry the early-lag artifact).
#   - Final test runs on the slow weights (= the Lookahead deployment artifact),
#     mirroring Final-SWA / Final-EMA.
# Lookahead arm = spsa:false SWA:false + Adam (same hparams as CTRL) + Lookahead.
import shutil, time

TRAINER = '/root/autodl-tmp/DeepfakeBench/training/trainer/trainer.py'
TRAIN = '/root/autodl-tmp/DeepfakeBench/training/train.py'
ts = time.strftime('%Y%m%d_%H%M%S')

def patch_file(path, edits):
    shutil.copy(path, path + '.bak_la_' + ts)
    s = open(path, encoding='utf-8').read()
    for name, old, new in edits:
        n = s.count(old)
        assert n == 1, ('[%s] anchor %r matched %d times' % (path, name, n))
        s = s.replace(old, new, 1)
        print('OK  %-16s -> %s' % (name, path))
    open(path, 'w', encoding='utf-8').write(s)
    print('PATCHED %s' % path)

# ---------------- trainer.py ----------------
_la_helpers = '''
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

'''

_trainer_edits = [
    ('la_init_call',
     "        self.speed_up()  # move model to GPU\n        self._ema_init()\n",
     "        self.speed_up()  # move model to GPU\n        self._ema_init()\n        self._la_init()\n"),
    ('la_helpers',
     "    def _ema_sd(self):\n        sd = self.model.state_dict()\n        for _n in self._ema:\n            sd[_n] = self._ema[_n]\n        return sd\n\n\n    def save_feat(self, phase, fea, dataset_key):",
     "    def _ema_sd(self):\n        sd = self.model.state_dict()\n        for _n in self._ema:\n            sd[_n] = self._ema[_n]\n        return sd\n" + _la_helpers + "    def save_feat(self, phase, fea, dataset_key):"),
    ('la_sync_call',
     "            if self.use_ema:\n                with torch.no_grad():\n                    for _n, _p in self.model.named_parameters():\n                        if _p.requires_grad and _n in self._ema:\n                            self._ema[_n].mul_(self._ema_decay).add_(_p.detach(), alpha=1 - self._ema_decay)\n",
     "            if self.use_ema:\n                with torch.no_grad():\n                    for _n, _p in self.model.named_parameters():\n                        if _p.requires_grad and _n in self._ema:\n                            self._ema[_n].mul_(self._ema_decay).add_(_p.detach(), alpha=1 - self._ema_decay)\n\n            if self.use_la:\n                self._la_sync()\n"),
]

# ---------------- train.py ----------------
_train_final_ema = """    # 最终测试：用 EMA 平均权重（若启用）
    if config.get('EMA', False) and getattr(trainer, 'use_ema', False):
        trainer.model.load_state_dict(trainer._ema_sd())
        trainer.model.epoch = config['nEpochs'] + 1
        trainer.test_epoch(config['nEpochs'] + 1, -1, test_data_loaders, 0)
        logger.info("===> Final EMA testing done!")"""
_train_final_la = _train_final_ema + """
    # 最终测试：用 Lookahead slow 权重（若启用，Lookahead 的部署物）
    if config.get('Lookahead', False) and getattr(trainer, 'use_la', False):
        torch.save(trainer._la_slow_sd(), os.path.join(trainer.log_dir, 'la_slow.pth'))
        trainer.model.load_state_dict(trainer._la_slow_sd())
        trainer.model.epoch = config['nEpochs'] + 1
        trainer.test_epoch(config['nEpochs'] + 1, -1, test_data_loaders, 0)
        logger.info("===> Final Lookahead testing done!")"""

_train_edits = [
    ('final_la_test', _train_final_ema, _train_final_la),
]

if __name__ == '__main__':
    patch_file(TRAINER, _trainer_edits)
    patch_file(TRAIN, _train_edits)
    print('ALL_PATCHED ts=%s' % ts)
