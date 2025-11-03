# Gated Phase Transition (GAPT)
# -----------------------------

class GatedPhaseTransition:
    """
    Gated Phase Transition (GAPT) : https://arxiv.org/pdf/2505.08727
    with percentage-based thresholds.
    """
    def __init__(self, tau_plateau: float = 0.01, tau_spike: float = 0.1, 
                 p_m: int = 5, p_a: int = 5):
        """
        Args:
            tau_plateau: Relative threshold for detecting plateau (e.g., 0.01 = 1% improvement)
            tau_spike: Relative threshold for detecting spike (e.g., 0.1 = 10% degradation)
            p_m: Patience for main objective (steps without improvement)
            p_c: Patience for auxiliary objective (steps without improvement)
        """
        self.tau_plateau = tau_plateau  # % improvement needed to avoid plateau
        self.tau_spike = tau_spike      # % degradation that triggers phase switch
        self.p_m = p_m
        self.p_a = p_a

        self.phi = 1  # 1 for main phase, 2 for compression phase
        self.s_m = 0  # steps since improvement in main
        self.s_a = 0  # steps since improvement in auxiliary

        self.min_m = float('inf')
        self.min_a = float('inf')

    def _relative_gain(self, current_loss: float, min_loss: float) -> float:
        """Calculate percentage improvement (negative = degradation)"""
        if min_loss == float('inf') or min_loss == 0:
            return 0.0
        return (min_loss - current_loss) / min_loss.clamp(min=1e-6)
    
    def step(self, main_loss: float, auxiliary_loss: float, verbose: bool = False):
        """
        Update phase based on loss dynamics.
        
        Returns:
            phi: Current phase (1=main, 2=compression)
        """
        gain_m = self._relative_gain(main_loss, self.min_m)
        gain_a = self._relative_gain(auxiliary_loss, self.min_a)
        
        self.min_m = min(self.min_m, main_loss)
        self.min_a = min(self.min_a, auxiliary_loss)

        prev_phi = self.phi

        if self.phi == 1:  # Main objective phase
            if gain_m > self.tau_plateau: 
                self.s_m = 0
            else: 
                self.s_m += 1
            
            if self.s_m >= self.p_m:
                self.s_m = 0
                self.phi = 2

        elif self.phi == 2:  # Main + Auxiliary phase
            if gain_m < -self.tau_spike:  
                if verbose:
                    print(f"  [GAPT] Main loss spiked: {gain_m:.3f} < {-self.tau_spike:.3f}")
                self.s_a = 0
                self.phi = 1
            else:
                if gain_a > self.tau_plateau:  
                    self.s_a = 0
                else: 
                    self.s_a += 1
                
                if self.s_a >= self.p_a:
                    if verbose:
                        print(f"  [GAPT] Auxiliary loss plateaued for {self.p_a} steps")
                    self.s_a = 0
                    self.phi = 1
                    
        
        if verbose and prev_phi != self.phi:
            print(f"  [GAPT] Phase transition: {prev_phi} → {self.phi}")
            print(f"         main_loss={main_loss:.4f}, aux_loss={auxiliary_loss:.4f}")
        
        return self.phi