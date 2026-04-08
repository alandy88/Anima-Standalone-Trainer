from tqdm import tqdm
import time
import torch

class StepProfiler:
    def __init__(self, accelerator, enabled=False):
        self.accelerator = accelerator
        self.enabled = enabled
        self._cum_fwd = 0.0
        self._cum_bwd = 0.0
        self._t0_step = None
        self._t0 = None # start of current micro-batch
        self._t1 = None # end of fwd
        self._t2 = None # end of bwd
        self._t3 = None # end of comm
        
    def on_batch_start(self):
        if not self.enabled: return
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self._t0_step is None:
            self._t0_step = now
        self._t0 = now
        
    def on_fwd_done(self):
        if not self.enabled: return
        torch.cuda.synchronize()
        self._t1 = time.perf_counter()
        self._cum_fwd += (self._t1 - self._t0) * 1000
        
    def on_bwd_done(self):
        if not self.enabled: return
        torch.cuda.synchronize()
        self._t2 = time.perf_counter()
        self._cum_bwd += (self._t2 - self._t1) * 1000
        # Default t3 to t2 so that if on_comm_done isn't called, comm time is 0
        self._t3 = self._t2
        
    def on_comm_done(self):
        if not self.enabled: return
        torch.cuda.synchronize()
        self._t3 = time.perf_counter()
        
    def on_step_done(self, global_step):
        if not self.enabled: return
        
        # Only print summary when accumulation is complete
        if not self.accelerator.sync_gradients: return
        
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        
        ms_comm = (self._t3 - self._t2) * 1000
        ms_opt  = (t4 - self._t3) * 1000
        ms_wall = (t4 - self._t0_step) * 1000
        ms_fwd  = self._cum_fwd
        ms_bwd  = self._cum_bwd
        
        # Gather all metrics to the main process to print them together
        metrics = torch.tensor([ms_wall, ms_fwd, ms_bwd, ms_comm, ms_opt], device=self.accelerator.device)
        gathered = self.accelerator.gather(metrics)
        
        if self.accelerator.is_main_process:
            # Build a single multi-line string to ensure it prints as a single block
            output_lines = [f"[PROFILE step {global_step}]"]
            for i in range(self.accelerator.num_processes):
                m = gathered[i*5 : (i+1)*5].tolist()
                output_lines.append(
                    f"  rank {i}: wall={m[0]:.1f}ms  "
                    f"fwd={m[1]:.1f}ms  "
                    f"bwd={m[2]:.1f}ms  "
                    f"comm={m[3]:.1f}ms  "
                    f"opt={m[4]:.1f}ms"
                )
            tqdm.write("\n" + "\n".join(output_lines))
        
        # Reset for next global optimization step
        self._cum_fwd = 0.0
        self._cum_bwd = 0.0
        self._t0_step = None
