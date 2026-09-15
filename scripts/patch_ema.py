#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Add EMA (exponential moving average of weights) support to the controlled protocol:
#  - Trainer maintains EMA copies of all requires_grad params (decay ema_decay).
#  - At every testpoint the EVAL + step-snapshot run on the EMA weights (mirrors how
#    SWA arms report; CTRL arms unchanged because use_ema False).
#  - train.py adds a final "Final EMA" test (mirrors Final SWA).
# EMA arm = spsa:false SWA:false + EMA:true (pure controlled detector + EMA averaging).
import re, shutil, time

TRAINER = '/root/autodl-tmp/DeepfakeBench/training/trainer/trainer.py'
TRAIN = '/root/autodl-tmp/DeepfakeBench/training/train.py'
ts = time.strftime('%Y%m%d_%H%M%S')

def patch_file(path, edits):
    shutil.copy(path, path + '.bak_ema_' + ts)
    s = open(path, encoding='utf-8').read()
    for name, old, new in edits:
        n = s.count(old)
        assert n == 1, ('[%s] anchor %r matched %d times' % (path, name, n))
        s = s.replace(old, new, 1)
        print('OK  %-14s -> %s' % (name, path))
    open(path, 'w', encoding='utf-8').write(s)
    print('PATCHED %s' % path)

# ---------------- trainer.py ----------------
_ema_helpers = '''
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

'''

_run_test_old = '''            # run test
            if (step_cnt+1) % test_step == 0 and epoch % test_every == 0:
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

                    # total_end_time = time.time()
                # ---- step snapshot save for cross-domain 7-point (Task #10) ----
                if test_data_loaders is not None:
                    _sd_dir = os.path.join(self.log_dir, 'testpoints')
                    os.makedirs(_sd_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(_sd_dir, 'step_%d.pth' % step_cnt))
                    self.logger.info('Step ckpt saved: step_%d.pth' % step_cnt)
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1'''

_run_test_new = '''            # run test
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
            step_cnt += 1'''

_ema_update_new = """            if 'SWA' in self.config and self.config['SWA'] and self._swa_should_collect(epoch):
                self.swa_model.update_parameters(self.model)

            if self.use_ema:
                with torch.no_grad():
                    for _n, _p in self.model.named_parameters():
                        if _p.requires_grad and _n in self._ema:
                            self._ema[_n].mul_(self._ema_decay).add_(_p.detach(), alpha=1 - self._ema_decay)"""

trainer_edits = [
    ('ema_init_call',
     "        self.speed_up()  # move model to GPU\n",
     "        self.speed_up()  # move model to GPU\n        self._ema_init()\n"),
    ('ema_helpers',
     "\n    def save_feat(self, phase, fea, dataset_key):",
     _ema_helpers + "\n    def save_feat(self, phase, fea, dataset_key):"),
    ('ema_update_loop',
     "            if 'SWA' in self.config and self.config['SWA'] and self._swa_should_collect(epoch):\n                self.swa_model.update_parameters(self.model)\n",
     _ema_update_new),
    ('eval_on_ema',
     _run_test_old,
     _run_test_new),
]

# ---------------- train.py ----------------
_train_final_old = """    # 最终测试：用 SWA 平均模型
    if config.get('SWA', False) and trainer.swa_model is not None:
        trainer.save_swa_ckpt()
        trainer.model = trainer.swa_model
        trainer.model.epoch = config['nEpochs'] + 1
        trainer.test_epoch(config['nEpochs'] + 1, -1, test_data_loaders, 0)
        logger.info("===> Final SWA testing done!")"""
_train_final_new = _train_final_old + """
    # 最终测试：用 EMA 平均权重（若启用）
    if config.get('EMA', False) and getattr(trainer, 'use_ema', False):
        trainer.model.load_state_dict(trainer._ema_sd())
        trainer.model.epoch = config['nEpochs'] + 1
        trainer.test_epoch(config['nEpochs'] + 1, -1, test_data_loaders, 0)
        logger.info("===> Final EMA testing done!")"""

train_edits = [
    ('final_ema_test', _train_final_old, _train_final_new),
]

if __name__ == '__main__':
    patch_file(TRAINER, trainer_edits)
    patch_file(TRAIN, train_edits)
    print('ALL_PATCHED ts=%s' % ts)
