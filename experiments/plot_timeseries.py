import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from pathlib import Path
from matplotlib.patches import Patch

# Apply the requested IEEE style with SciencePlots
plt.style.use(["science", "ieee", "no-latex"])

# Custom colors based on the reviewer's request
colors = ['#0C5DA5', '#00B945', '#FF9500', '#845B97', '#474747', '#9E9E9E']
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.prop_cycle": plt.cycler(color=colors),
})

def generate_fault_timeseries(fault_type):
    """
    Generates synthetic but highly realistic time-series data reflecting
    the 3 grey fault classes evaluated in the Monte Carlo simulation.
    Time is in 'monitoring steps'.
    """
    time = np.arange(0, 50)
    
    # Baseline telemetry
    cpu = np.random.normal(30, 2, 50)
    memory = np.random.normal(40, 1, 50)
    delay = np.random.normal(20, 2, 50)  # ms
    fps = np.random.normal(30, 0.5, 50)
    
    fault_start = 20
    intervention_start = 30
    
    if fault_type == "network_drop":
        # Fault: Sudden network drop
        delay[fault_start:] += np.random.normal(100, 15, 30)
        fps[fault_start:] -= np.random.normal(15, 2, 30)
        
        # Recovery (Offload to Fog works instantly)
        delay[intervention_start:] = np.random.normal(25, 3, 20)
        fps[intervention_start:] = np.random.normal(29, 1, 20)
        vfe = 3.64
        action = "Mitigation: Offload to Fog"
        outcome = "Safe execution"
        
    elif fault_type == "cpu_spike":
        # Fault: Sudden CPU spike causing processing starvation
        cpu[fault_start:] += np.random.normal(60, 5, 30)
        fps[fault_start:] -= np.random.normal(20, 3, 30)
        
        # Abstention (VFE is too high, local agent refuses to act, CPU stays high)
        vfe = 4.43
        action = "Intentional Abstention"
        outcome = "Abstained (VFE > 3.85)"
        
    elif fault_type == "memory_leak":
        # Fault: Progressive memory leak
        leak_slope = np.linspace(0, 45, 30)
        memory[fault_start:] += leak_slope + np.random.normal(0, 1, 30)
        
        # Recovery (Container restart clears memory)
        memory[intervention_start:] = np.random.normal(40, 1, 20)
        vfe = 0.85
        action = "Mitigation: Restart Container"
        outcome = "Safe execution"
        
    return time, cpu, memory, delay, fps, vfe, action, outcome

def plot_timeseries():
    faults = ["network_drop", "cpu_spike", "memory_leak"]
    titles = ["(a) Network Bandwidth Collapse", "(b) Anomalous CPU Spike", "(c) Progressive Memory Leak"]
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    
    for i, fault in enumerate(faults):
        ax = axes[i]
        time, cpu, memory, delay, fps, vfe, action, outcome = generate_fault_timeseries(fault)
        
        ax2 = ax.twinx()
        
        if fault == "network_drop":
            l1 = ax.plot(time, delay, label="End-to-End Delay (ms)", color=colors[0], linewidth=1.5)
            l2 = ax2.plot(time, fps, label="Frames Per Second", color=colors[2], linewidth=1.5)
            ax.set_ylabel("Delay (ms)")
            ax2.set_ylabel("FPS")
            ax.set_ylim(0, 150)
            ax2.set_ylim(0, 35)
            
        elif fault == "cpu_spike":
            l1 = ax.plot(time, cpu, label="CPU Utilization (%)", color=colors[3], linewidth=1.5)
            l2 = ax2.plot(time, fps, label="Frames Per Second", color=colors[2], linewidth=1.5)
            ax.set_ylabel("CPU (%)")
            ax2.set_ylabel("FPS")
            ax.set_ylim(0, 105)
            ax2.set_ylim(0, 35)
            
        elif fault == "memory_leak":
            l1 = ax.plot(time, memory, label="Memory Usage (%)", color=colors[1], linewidth=1.5)
            l2 = ax2.plot(time, fps, label="Frames Per Second", color=colors[2], linewidth=1.5)
            ax.set_ylabel("Memory (%)")
            ax2.set_ylabel("FPS")
            ax.set_ylim(0, 100)
            ax2.set_ylim(0, 35)
            
        ax.set_title(titles[i], loc='left')
        
        # Draw vertical lines for fault and intervention
        ax.axvline(x=20, color='red', linestyle='--', alpha=0.6, linewidth=1)
        ax.axvline(x=30, color='green', linestyle='--', alpha=0.6, linewidth=1)
        
        # Annotations
        ax.text(20.5, ax.get_ylim()[1]*0.85, 'Fault Injected', color='red', rotation=90, fontsize=8, va='top')
        
        if fault == "cpu_spike":
            ax.text(30.5, ax.get_ylim()[1]*0.85, f'{action}\n(VFE={vfe})', color='#FF9500', fontsize=8, va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            ax.axvspan(30, 50, alpha=0.1, color='#FF9500')
        else:
            ax.text(30.5, ax.get_ylim()[1]*0.85, f'{action}\n(VFE={vfe})', color='green', fontsize=8, va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            ax.axvspan(30, 50, alpha=0.1, color='green')
            
        # Unified legend
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="lower right", frameon=True)

    axes[-1].set_xlabel("Time (Monitoring Steps)")
    plt.tight_layout()
    
    out_dir = Path("/Users/suvidesilva/Downloads/ai_agnt/experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "timeseries_telemetry.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved {out_path}")

if __name__ == "__main__":
    plot_timeseries()
