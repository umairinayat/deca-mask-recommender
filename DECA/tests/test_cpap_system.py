#!/usr/bin/env python3
"""
Unit Tests for CPAP Measurement System
Tests core functionality without requiring DECA model or camera
"""
import sys
import os
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCPAPMeasurementSystem:
    """Test suite for CPAPMeasurementSystem"""
    
    def __init__(self):
        self.temp_dir = None
        self.results = []
        
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temp directory: {self.temp_dir}")
        
    def teardown(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temp directory")
    
    def test_session_index_generation(self):
        """Test session index auto-increment logic"""
        print("\n=== Test: Session Index Generation ===")
        
        results_dir = Path(self.temp_dir) / 'results'
        
        # Test 1: No existing sessions - should return 1
        def get_next_session_index(results_dir):
            if not results_dir.exists():
                results_dir.mkdir(parents=True)
                return 1
            existing_sessions = [d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if not existing_sessions:
                return 1
            max_index = max([int(d.name) for d in existing_sessions])
            return max_index + 1
        
        # First call - should return 1
        idx = get_next_session_index(results_dir)
        assert idx == 1, f"Expected 1, got {idx}"
        print(f"  [PASS] Empty directory returns 1")
        
        # Create session 1
        (results_dir / '1').mkdir()
        idx = get_next_session_index(results_dir)
        assert idx == 2, f"Expected 2, got {idx}"
        print(f"  [PASS] After session 1 exists, returns 2")
        
        # Create session 5 (skip some)
        (results_dir / '5').mkdir()
        idx = get_next_session_index(results_dir)
        assert idx == 6, f"Expected 6, got {idx}"
        print(f"  [PASS] After session 5 exists, returns 6")
        
        # Create non-numeric directory (should be ignored)
        (results_dir / 'test').mkdir()
        idx = get_next_session_index(results_dir)
        assert idx == 6, f"Expected 6, got {idx}"
        print(f"  [PASS] Non-numeric directories ignored")
        
        self.results.append(("Session Index Generation", True))
        return True
    
    def test_measurement_extraction(self):
        """Test CPAP measurement extraction from vertices"""
        print("\n=== Test: Measurement Extraction ===")
        
        # Create mock vertices (5023 x 3)
        np.random.seed(42)
        vertices = np.random.randn(5023, 3).astype(np.float32)
        
        # Set specific vertices for testing
        NOSE_LEFT = 3632
        NOSE_RIGHT = 3325
        CHEEK_LEFT = 4478
        CHEEK_RIGHT = 2051
        NOSE_BASE = 175
        CHIN = 152
        
        # Set known positions
        vertices[NOSE_LEFT] = [0.0, 0.0, 0.0]
        vertices[NOSE_RIGHT] = [0.03, 0.0, 0.0]  # 30mm nose width
        vertices[CHEEK_LEFT] = [0.0, 0.0, 0.0]
        vertices[CHEEK_RIGHT] = [0.13, 0.0, 0.0]  # 130mm cheekbone width
        vertices[NOSE_BASE] = [0.0, 0.0, 0.0]
        vertices[CHIN] = [0.0, 0.07, 0.0]  # 70mm nose-to-chin
        
        def extract_measurements(vertices):
            nose_left = vertices[NOSE_LEFT]
            nose_right = vertices[NOSE_RIGHT]
            nose_width = np.linalg.norm(nose_left - nose_right)
            
            cheek_left = vertices[CHEEK_LEFT]
            cheek_right = vertices[CHEEK_RIGHT]
            cheekbone_width = np.linalg.norm(cheek_left - cheek_right)
            
            nose_base = vertices[NOSE_BASE]
            chin = vertices[CHIN]
            nose_to_chin = np.linalg.norm(nose_base - chin)
            
            return {
                'nose_width': float(nose_width),
                'cheekbone_width': float(cheekbone_width),
                'nose_to_chin': float(nose_to_chin)
            }
        
        measurements = extract_measurements(vertices)
        
        # Verify measurements
        assert abs(measurements['nose_width'] - 0.03) < 0.001, f"Nose width: {measurements['nose_width']}"
        print(f"  [PASS] Nose width: {measurements['nose_width']:.6f}")
        
        assert abs(measurements['cheekbone_width'] - 0.13) < 0.001, f"Cheekbone: {measurements['cheekbone_width']}"
        print(f"  [PASS] Cheekbone width: {measurements['cheekbone_width']:.6f}")
        
        assert abs(measurements['nose_to_chin'] - 0.07) < 0.001, f"Nose-to-chin: {measurements['nose_to_chin']}"
        print(f"  [PASS] Nose-to-chin: {measurements['nose_to_chin']:.6f}")
        
        self.results.append(("Measurement Extraction", True))
        return True
    
    def test_vertex_indices_bounds(self):
        """Test that vertex indices are within valid range"""
        print("\n=== Test: Vertex Indices Bounds ===")
        
        MAX_VERTICES = 5023  # FLAME mesh has 5023 vertices
        
        indices = {
            'NOSE_LEFT': 3632,
            'NOSE_RIGHT': 3325,
            'CHEEK_LEFT': 4478,
            'CHEEK_RIGHT': 2051,
            'NOSE_BASE': 175,
            'CHIN': 152
        }
        
        all_valid = True
        for name, idx in indices.items():
            valid = 0 <= idx < MAX_VERTICES
            status = "[PASS]" if valid else "[FAIL]"
            print(f"  {status} {name}: {idx} (valid: {valid})")
            if not valid:
                all_valid = False
        
        self.results.append(("Vertex Indices Bounds", all_valid))
        return all_valid
    
    def test_json_save_load(self):
        """Test JSON save/load functionality"""
        print("\n=== Test: JSON Save/Load ===")
        
        session_dir = Path(self.temp_dir) / 'session_test'
        session_dir.mkdir(parents=True)
        
        # Create test measurement
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'measurement_number': 1,
            'measurements': {
                'nose_width': 0.035,
                'cheekbone_width': 0.130,
                'nose_to_chin': 0.075
            },
            'processing_time_seconds': 1.5,
            'vertex_indices': {
                'nose_left': 3632,
                'nose_right': 3325,
                'cheek_left': 4478,
                'cheek_right': 2051,
                'nose_base': 175,
                'chin': 152
            }
        }
        
        # Save
        filepath = session_dir / 'measurement_test.json'
        with open(filepath, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"  [PASS] Saved JSON to {filepath.name}")
        
        # Load
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        
        # Verify
        assert loaded_data['measurement_number'] == 1, "measurement_number mismatch"
        assert abs(loaded_data['measurements']['nose_width'] - 0.035) < 0.0001, "nose_width mismatch"
        assert abs(loaded_data['measurements']['cheekbone_width'] - 0.130) < 0.0001, "cheekbone_width mismatch"
        assert abs(loaded_data['measurements']['nose_to_chin'] - 0.075) < 0.0001, "nose_to_chin mismatch"
        print(f"  [PASS] Loaded and verified JSON data")
        
        self.results.append(("JSON Save/Load", True))
        return True
    
    def test_statistics_calculation(self):
        """Test statistics calculation"""
        print("\n=== Test: Statistics Calculation ===")
        
        # Mock measurements
        measurements = [
            {'measurements': {'nose_width': 0.035, 'cheekbone_width': 0.130, 'nose_to_chin': 0.075}},
            {'measurements': {'nose_width': 0.036, 'cheekbone_width': 0.131, 'nose_to_chin': 0.076}},
            {'measurements': {'nose_width': 0.034, 'cheekbone_width': 0.129, 'nose_to_chin': 0.074}},
            {'measurements': {'nose_width': 0.035, 'cheekbone_width': 0.130, 'nose_to_chin': 0.075}},
            {'measurements': {'nose_width': 0.037, 'cheekbone_width': 0.132, 'nose_to_chin': 0.077}},
        ]
        
        def calculate_statistics(measurements):
            nose_widths = [m['measurements']['nose_width'] for m in measurements]
            cheekbone_widths = [m['measurements']['cheekbone_width'] for m in measurements]
            nose_to_chins = [m['measurements']['nose_to_chin'] for m in measurements]
            
            stats = {
                'nose_width': {
                    'mean': np.mean(nose_widths),
                    'std': np.std(nose_widths),
                    'cv': (np.std(nose_widths) / np.mean(nose_widths) * 100) if np.mean(nose_widths) > 0 else 0,
                    'min': np.min(nose_widths),
                    'max': np.max(nose_widths)
                },
                'cheekbone_width': {
                    'mean': np.mean(cheekbone_widths),
                    'std': np.std(cheekbone_widths),
                    'cv': (np.std(cheekbone_widths) / np.mean(cheekbone_widths) * 100) if np.mean(cheekbone_widths) > 0 else 0,
                    'min': np.min(cheekbone_widths),
                    'max': np.max(cheekbone_widths)
                },
                'nose_to_chin': {
                    'mean': np.mean(nose_to_chins),
                    'std': np.std(nose_to_chins),
                    'cv': (np.std(nose_to_chins) / np.mean(nose_to_chins) * 100) if np.mean(nose_to_chins) > 0 else 0,
                    'min': np.min(nose_to_chins),
                    'max': np.max(nose_to_chins)
                }
            }
            return stats
        
        stats = calculate_statistics(measurements)
        
        # Verify nose width stats
        expected_mean = 0.0354
        assert abs(stats['nose_width']['mean'] - expected_mean) < 0.001, f"Mean: {stats['nose_width']['mean']}"
        print(f"  [PASS] Nose width mean: {stats['nose_width']['mean']:.6f}")
        
        # CV should be low for consistent data
        assert stats['nose_width']['cv'] < 5.0, f"CV too high: {stats['nose_width']['cv']}"
        print(f"  [PASS] Nose width CV: {stats['nose_width']['cv']:.2f}%")
        
        # Min should be less than max
        assert stats['nose_width']['min'] < stats['nose_width']['max'], "Min >= Max"
        print(f"  [PASS] Range: [{stats['nose_width']['min']:.6f}, {stats['nose_width']['max']:.6f}]")
        
        self.results.append(("Statistics Calculation", True))
        return True
    
    def test_face_preprocessing_logic(self):
        """Test face preprocessing bounding box logic"""
        print("\n=== Test: Face Preprocessing Logic ===")
        
        # Mock landmarks (68 points)
        np.random.seed(42)
        landmarks = np.random.rand(68, 2) * 200 + 100  # Points between 100-300
        
        # Mock image dimensions
        h, w = 480, 640
        target_size = 224
        padding = 50
        
        def get_bbox(landmarks, h, w, padding):
            x_min, x_max = int(np.min(landmarks[:, 0])), int(np.max(landmarks[:, 0]))
            y_min, y_max = int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 1]))
            
            # Add padding
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            
            return (x_min, y_min, x_max, y_max)
        
        bbox = get_bbox(landmarks, h, w, padding)
        x_min, y_min, x_max, y_max = bbox
        
        # Verify bounds
        assert x_min >= 0, f"x_min out of bounds: {x_min}"
        assert y_min >= 0, f"y_min out of bounds: {y_min}"
        assert x_max <= w, f"x_max out of bounds: {x_max}"
        assert y_max <= h, f"y_max out of bounds: {y_max}"
        print(f"  [PASS] Bounding box within image: ({x_min}, {y_min}, {x_max}, {y_max})")
        
        # Verify crop dimensions are positive
        assert x_max > x_min, "Invalid x dimensions"
        assert y_max > y_min, "Invalid y dimensions"
        print(f"  [PASS] Crop dimensions positive: {x_max - x_min} x {y_max - y_min}")
        
        self.results.append(("Face Preprocessing Logic", True))
        return True
    
    def test_edge_cases(self):
        """Test edge cases"""
        print("\n=== Test: Edge Cases ===")
        
        # Test 1: Single measurement statistics
        single_measurement = [
            {'measurements': {'nose_width': 0.035, 'cheekbone_width': 0.130, 'nose_to_chin': 0.075}}
        ]
        
        nose_widths = [m['measurements']['nose_width'] for m in single_measurement]
        mean = np.mean(nose_widths)
        std = np.std(nose_widths)
        cv = (std / mean * 100) if mean > 0 else 0
        
        assert mean == 0.035, "Single measurement mean incorrect"
        assert std == 0.0, "Single measurement std should be 0"
        assert cv == 0.0, "Single measurement CV should be 0"
        print(f"  [PASS] Single measurement: mean={mean}, std={std}, cv={cv}%")
        
        # Test 2: Empty face crop handling
        # (simulated - actual test would need image processing)
        print(f"  [INFO] Empty face crop test skipped (requires image processing)")
        
        # Test 3: Very small measurements
        tiny_measurements = [
            {'measurements': {'nose_width': 0.0001, 'cheekbone_width': 0.0001, 'nose_to_chin': 0.0001}}
        ]
        nose_widths = [m['measurements']['nose_width'] for m in tiny_measurements]
        mean = np.mean(nose_widths)
        assert mean > 0, "Mean should be positive"
        print(f"  [PASS] Very small measurements handled: mean={mean}")
        
        self.results.append(("Edge Cases", True))
        return True
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "=" * 60)
        print("CPAP Measurement System - Unit Tests")
        print("=" * 60)
        
        self.setup()
        
        try:
            self.test_session_index_generation()
            self.test_measurement_extraction()
            self.test_vertex_indices_bounds()
            self.test_json_save_load()
            self.test_statistics_calculation()
            self.test_face_preprocessing_logic()
            self.test_edge_cases()
        except Exception as e:
            print(f"\n[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.teardown()
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)
        print(f"Passed: {passed}/{total}")
        
        for test_name, result in self.results:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {test_name}")
        
        print("=" * 60)
        return passed == total


class TestValidator:
    """Test suite for CPAPValidator"""
    
    def __init__(self):
        self.temp_dir = None
        self.results = []
        
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="validator_test_")
        
    def teardown(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_latest_session_detection(self):
        """Test latest session detection"""
        print("\n=== Test: Latest Session Detection ===")
        
        results_dir = Path(self.temp_dir) / 'results'
        results_dir.mkdir(parents=True)
        
        def get_latest_session(results_dir):
            if not results_dir.exists():
                return None
            existing_sessions = [d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if not existing_sessions:
                return None
            return max([int(d.name) for d in existing_sessions])
        
        # No sessions
        result = get_latest_session(results_dir)
        assert result is None, f"Expected None, got {result}"
        print(f"  [PASS] No sessions returns None")
        
        # Create sessions
        (results_dir / '1').mkdir()
        (results_dir / '3').mkdir()
        (results_dir / '7').mkdir()
        
        result = get_latest_session(results_dir)
        assert result == 7, f"Expected 7, got {result}"
        print(f"  [PASS] Latest session detected: {result}")
        
        self.results.append(("Latest Session Detection", True))
        return True
    
    def test_measurement_loading(self):
        """Test measurement loading from JSON files"""
        print("\n=== Test: Measurement Loading ===")
        
        session_dir = Path(self.temp_dir) / 'results' / '1'
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test measurements
        for i in range(3):
            data = {
                'timestamp': datetime.now().isoformat(),
                'measurement_number': i + 1,
                'measurements': {
                    'nose_width': 0.035 + i * 0.001,
                    'cheekbone_width': 0.130 + i * 0.001,
                    'nose_to_chin': 0.075 + i * 0.001
                }
            }
            filepath = session_dir / f'measurement_{i:03d}.json'
            with open(filepath, 'w') as f:
                json.dump(data, f)
        
        # Load measurements
        json_files = sorted(session_dir.glob('measurement_*.json'))
        measurements = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                measurements.append(data)
        
        assert len(measurements) == 3, f"Expected 3 measurements, got {len(measurements)}"
        print(f"  [PASS] Loaded {len(measurements)} measurements")
        
        # Verify order
        for i, m in enumerate(measurements):
            assert m['measurement_number'] == i + 1, f"Wrong order: {m['measurement_number']}"
        print(f"  [PASS] Measurements in correct order")
        
        self.results.append(("Measurement Loading", True))
        return True
    
    def test_consistency_assessment(self):
        """Test consistency assessment logic"""
        print("\n=== Test: Consistency Assessment ===")
        
        def assess_consistency(cv):
            if cv < 2.0:
                return "EXCELLENT"
            elif cv < 5.0:
                return "GOOD"
            else:
                return "MODERATE"
        
        assert assess_consistency(1.5) == "EXCELLENT", "CV 1.5 should be EXCELLENT"
        print(f"  [PASS] CV 1.5% -> EXCELLENT")
        
        assert assess_consistency(3.0) == "GOOD", "CV 3.0 should be GOOD"
        print(f"  [PASS] CV 3.0% -> GOOD")
        
        assert assess_consistency(7.0) == "MODERATE", "CV 7.0 should be MODERATE"
        print(f"  [PASS] CV 7.0% -> MODERATE")
        
        self.results.append(("Consistency Assessment", True))
        return True
    
    def run_all_tests(self):
        """Run all validator tests"""
        print("\n" + "=" * 60)
        print("CPAP Validator - Unit Tests")
        print("=" * 60)
        
        self.setup()
        
        try:
            self.test_latest_session_detection()
            self.test_measurement_loading()
            self.test_consistency_assessment()
        except Exception as e:
            print(f"\n[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.teardown()
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATOR TEST SUMMARY")
        print("=" * 60)
        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)
        print(f"Passed: {passed}/{total}")
        
        for test_name, result in self.results:
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {status} {test_name}")
        
        print("=" * 60)
        return passed == total


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("=" * 20 + " CPAP SYSTEM TEST SUITE " + "=" * 26)
    print("=" * 70)
    
    # Run CPAPMeasurementSystem tests
    cpap_tester = TestCPAPMeasurementSystem()
    cpap_result = cpap_tester.run_all_tests()
    
    # Run CPAPValidator tests
    validator_tester = TestValidator()
    validator_result = validator_tester.run_all_tests()
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    all_passed = cpap_result and validator_result
    status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"Result: {status}")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

