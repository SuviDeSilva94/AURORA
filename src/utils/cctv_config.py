"""
CCTV Camera Configuration for Computing Continuum
Based on   requirements: Raspberry Pi processing live CCTV footage
"""

# CCTV SLO Requirements (from  )
CCTV_SLOS = {
    'fps': 30,              # Target: 30 FPS
    'network_delay': 33,    # Target: < 33 ms
    'throughput': 1.6,      # Target: > 1.6 MB/s (megabytes per second)
    'success_rate': 0.95,   # Target: ≥ 95%
}

# CCTV Fault Types (exact scenarios from  )
CCTV_FAULT_TYPES = {
    'excessive_frames': {
        'description': 'Camera #3 sending excessive frames',
        'root_cause': 'camera_excessive',
        'symptoms': ['fps_drop', 'bandwidth_saturated', 'delay_spike'],
        'correct_action': 'rate_limit_camera'
    },
    'bandwidth_drop': {
        'description': 'Network bandwidth drop on specific link',
        'root_cause': 'network_quality',
        'symptoms': ['throughput_low', 'delay_high', 'packet_loss'],
        'correct_action': 'offload_to_fog'
    },
    'node_overload': {
        'description': 'Edge node (Raspberry Pi) CPU/memory saturated',
        'root_cause': 'node_resources',
        'symptoms': ['cpu_high', 'memory_high', 'processing_delay'],
        'correct_action': 'redistribute_load'
    },
    'stream_conflict': {
        'description': 'Multiple cameras competing on same network link',
        'root_cause': 'link_contention',
        'symptoms': ['all_fps_drop', 'jitter', 'packet_collisions'],
        'correct_action': 'separate_streams'
    }
}

# CCTV Recovery Actions (self-healing)
CCTV_RECOVERY_ACTIONS = {
    'rate_limit_camera': 'Apply rate limiting to excessive camera',
    'offload_to_fog': 'Offload processing to fog node',
    'redistribute_load': 'Redistribute cameras across edge nodes',
    'separate_streams': 'Route cameras through different network paths',
    'switch_wifi': 'Switch to alternative WiFi channel/AP',
    'reduce_quality': 'Temporarily reduce video quality',
    'isolate_node': 'Isolate problematic node for diagnostics'
}

# Device Types
DEVICE_TYPES = {
    'raspberry_pi_4': {'cpu_cores': 4, 'memory_gb': 4, 'network_mbps': 1000},
    'raspberry_pi_zero': {'cpu_cores': 1, 'memory_gb': 0.5, 'network_mbps': 100},
    'edge_server': {'cpu_cores': 8, 'memory_gb': 16, 'network_mbps': 10000}
}

# CCTV Cameras
CAMERA_CONFIGS = {
    'camera_1': {'resolution': '1080p', 'fps_max': 30, 'bitrate_mbps': 2.5},
    'camera_2': {'resolution': '1080p', 'fps_max': 30, 'bitrate_mbps': 2.5},
    'camera_3': {'resolution': '4K', 'fps_max': 60, 'bitrate_mbps': 8.0},  # Problematic!
    'camera_4': {'resolution': '720p', 'fps_max': 25, 'bitrate_mbps': 1.5}
}

