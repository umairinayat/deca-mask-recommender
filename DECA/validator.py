#!/usr/bin/env python3
"""
CPAP Measurement Validator
Visualizes the 3 critical measurements across multiple captures
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

class CPAPValidator:
    def __init__(self, session_index=None):
        # Use script directory as base for results
        self.results_dir = Path(__file__).parent.parent / 'results'
        self.session_index = session_index
        self.session_dir = None
        
    def get_latest_session(self):
        """Get the latest session index"""
        if not self.results_dir.exists():
            return None
        
        existing_sessions = [d for d in self.results_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not existing_sessions:
            return None
        
        return max([int(d.name) for d in existing_sessions])
    
    def load_measurements(self):
        """Load all measurements from the session"""
        if self.session_index is None:
            self.session_index = self.get_latest_session()
            if self.session_index is None:
                print("[ERROR] No sessions found in results directory")
                return None
        
        self.session_dir = self.results_dir / str(self.session_index)
        
        if not self.session_dir.exists():
            print(f"[ERROR] Session {self.session_index} not found")
            return None
        
        # Load all JSON files
        json_files = sorted(self.session_dir.glob('measurement_*.json'))
        
        if not json_files:
            print(f"[ERROR] No measurements found in session {self.session_index}")
            return None
        
        measurements = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    measurements.append(data)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping corrupted file {json_file.name}: {e}")
        
        if not measurements:
            print(f"[ERROR] No valid measurements found in session {self.session_index}")
            return None
            
        return measurements
    
    @staticmethod
    def _safe_cv(values):
        """Calculate coefficient of variation safely"""
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        return (np.std(values) / mean) * 100
    
    def calculate_statistics(self, measurements):
        """Calculate statistics for each measurement type"""
        nose_widths = [m['measurements']['nose_width'] for m in measurements]
        cheekbone_widths = [m['measurements']['cheekbone_width'] for m in measurements]
        nose_to_chins = [m['measurements']['nose_to_chin'] for m in measurements]
        
        stats = {
            'nose_width': {
                'mean': np.mean(nose_widths),
                'std': np.std(nose_widths),
                'cv': self._safe_cv(nose_widths),
                'min': np.min(nose_widths),
                'max': np.max(nose_widths),
                'count': len(nose_widths)
            },
            'cheekbone_width': {
                'mean': np.mean(cheekbone_widths),
                'std': np.std(cheekbone_widths),
                'cv': self._safe_cv(cheekbone_widths),
                'min': np.min(cheekbone_widths),
                'max': np.max(cheekbone_widths),
                'count': len(cheekbone_widths)
            },
            'nose_to_chin': {
                'mean': np.mean(nose_to_chins),
                'std': np.std(nose_to_chins),
                'cv': self._safe_cv(nose_to_chins),
                'min': np.min(nose_to_chins),
                'max': np.max(nose_to_chins),
                'count': len(nose_to_chins)
            }
        }
        
        return stats
    
    def print_statistics(self, measurements, stats):
        """Print measurement statistics"""
        print("\n" + "=" * 70)
        print(f"📊 CPAP Measurement Statistics - Session {self.session_index}")
        print("=" * 70)
        print(f"Total measurements: {len(measurements)}")
        print(f"Session directory: {self.session_dir}")
        print("\n" + "-" * 70)
        
        print("\n[1] NOSE WIDTH (Alar Base) - Primary for Nasal/Pillow Masks")
        print(f"   Mean:  {stats['nose_width']['mean']:.6f} FLAME units")
        print(f"   Std:   {stats['nose_width']['std']:.6f} FLAME units")
        print(f"   CV:    {stats['nose_width']['cv']:.2f}%")
        print(f"   Range: [{stats['nose_width']['min']:.6f}, {stats['nose_width']['max']:.6f}]")
        
        print("\n[2] CHEEKBONE WIDTH (Zygion-Zygion) - Primary for Full-Face Masks")
        print(f"   Mean:  {stats['cheekbone_width']['mean']:.6f} FLAME units")
        print(f"   Std:   {stats['cheekbone_width']['std']:.6f} FLAME units")
        print(f"   CV:    {stats['cheekbone_width']['cv']:.2f}%")
        print(f"   Range: [{stats['cheekbone_width']['min']:.6f}, {stats['cheekbone_width']['max']:.6f}]")
        
        print("\n[3] NOSE-TO-CHIN (Subnasale -> Menton) - Secondary Check")
        print(f"   Mean:  {stats['nose_to_chin']['mean']:.6f} FLAME units")
        print(f"   Std:   {stats['nose_to_chin']['std']:.6f} FLAME units")
        print(f"   CV:    {stats['nose_to_chin']['cv']:.2f}%")
        print(f"   Range: [{stats['nose_to_chin']['min']:.6f}, {stats['nose_to_chin']['max']:.6f}]")
        
        print("\n" + "=" * 70)
        
        # Consistency assessment
        print("\n[*] Consistency Assessment:")
        if stats['nose_width']['cv'] < 2.0:
            print("   [OK] Nose Width: EXCELLENT consistency (<2% CV)")
        elif stats['nose_width']['cv'] < 5.0:
            print("   [OK] Nose Width: GOOD consistency (<5% CV)")
        else:
            print("   [WARN] Nose Width: Moderate consistency (>5% CV)")
        
        if stats['cheekbone_width']['cv'] < 2.0:
            print("   [OK] Cheekbone Width: EXCELLENT consistency (<2% CV)")
        elif stats['cheekbone_width']['cv'] < 5.0:
            print("   [OK] Cheekbone Width: GOOD consistency (<5% CV)")
        else:
            print("   [WARN] Cheekbone Width: Moderate consistency (>5% CV)")
        
        if stats['nose_to_chin']['cv'] < 2.0:
            print("   [OK] Nose-to-Chin: EXCELLENT consistency (<2% CV)")
        elif stats['nose_to_chin']['cv'] < 5.0:
            print("   [OK] Nose-to-Chin: GOOD consistency (<5% CV)")
        else:
            print("   [WARN] Nose-to-Chin: Moderate consistency (>5% CV)")
        
        print("=" * 70 + "\n")
    
    def create_visualizations(self, measurements, stats):
        """Create 3 graphs showing measurement trends"""
        # Extract data
        sequences = [m['measurement_number'] for m in measurements]
        nose_widths = [m['measurements']['nose_width'] for m in measurements]
        cheekbone_widths = [m['measurements']['cheekbone_width'] for m in measurements]
        nose_to_chins = [m['measurements']['nose_to_chin'] for m in measurements]
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'CPAP Measurements - Session {self.session_index}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Nose Width
        ax1 = axes[0]
        ax1.plot(sequences, nose_widths, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.axhline(y=stats['nose_width']['mean'], color='red', linestyle='--', 
                    label=f"Mean: {stats['nose_width']['mean']:.6f}")
        ax1.fill_between(sequences, 
                         stats['nose_width']['mean'] - stats['nose_width']['std'],
                         stats['nose_width']['mean'] + stats['nose_width']['std'],
                         alpha=0.2, color='red')
        ax1.set_xlabel('Measurement Number', fontsize=11)
        ax1.set_ylabel('FLAME Units', fontsize=11)
        ax1.set_title(f'1. Nose Width (Alar Base) - CV: {stats["nose_width"]["cv"]:.2f}%', 
                      fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cheekbone Width
        ax2 = axes[1]
        ax2.plot(sequences, cheekbone_widths, 'o-', linewidth=2, markersize=8, color='#A23B72')
        ax2.axhline(y=stats['cheekbone_width']['mean'], color='red', linestyle='--',
                    label=f"Mean: {stats['cheekbone_width']['mean']:.6f}")
        ax2.fill_between(sequences,
                         stats['cheekbone_width']['mean'] - stats['cheekbone_width']['std'],
                         stats['cheekbone_width']['mean'] + stats['cheekbone_width']['std'],
                         alpha=0.2, color='red')
        ax2.set_xlabel('Measurement Number', fontsize=11)
        ax2.set_ylabel('FLAME Units', fontsize=11)
        ax2.set_title(f'2. Cheekbone Width (Zygion-Zygion) - CV: {stats["cheekbone_width"]["cv"]:.2f}%',
                      fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Nose-to-Chin
        ax3 = axes[2]
        ax3.plot(sequences, nose_to_chins, 'o-', linewidth=2, markersize=8, color='#F18F01')
        ax3.axhline(y=stats['nose_to_chin']['mean'], color='red', linestyle='--',
                    label=f"Mean: {stats['nose_to_chin']['mean']:.6f}")
        ax3.fill_between(sequences,
                         stats['nose_to_chin']['mean'] - stats['nose_to_chin']['std'],
                         stats['nose_to_chin']['mean'] + stats['nose_to_chin']['std'],
                         alpha=0.2, color='red')
        ax3.set_xlabel('Measurement Number', fontsize=11)
        ax3.set_ylabel('FLAME Units', fontsize=11)
        ax3.set_title(f'3. Nose-to-Chin Distance - CV: {stats["nose_to_chin"]["cv"]:.2f}%',
                      fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.session_dir / f'session_{self.session_index}_visualization.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"[OK] Visualization saved to: {output_file}")
        
        # Show plot
        plt.show()
    
    def validate(self):
        """Run validation and visualization"""
        print("[*] Loading measurements...")
        
        measurements = self.load_measurements()
        if measurements is None:
            return
        
        print(f"[OK] Loaded {len(measurements)} measurements from session {self.session_index}")
        
        # Calculate statistics
        stats = self.calculate_statistics(measurements)
        
        # Print statistics
        self.print_statistics(measurements, stats)
        
        # Create visualizations
        print("[*] Generating visualizations...")
        self.create_visualizations(measurements, stats)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate CPAP measurements')
    parser.add_argument('session', type=int, nargs='?', default=None,
                       help='Session index to validate (default: latest)')
    
    args = parser.parse_args()
    
    validator = CPAPValidator(session_index=args.session)
    validator.validate()

if __name__ == "__main__":
    main()
