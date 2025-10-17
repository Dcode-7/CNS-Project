"""
COMPLETE ECC STEGANOGRAPHY IMPLEMENTATION
==========================================
This script provides an integrated runner for all three phases with validation.

Requirements:
pip install cryptography pillow numpy matplotlib psutil

Directory Structure:
ecc_stego/
├── run_all_phases.py (this file)
├── phase1_baseline.py
├── phase2_optimized.py
├── phase3_benchmark.py
├── keys/
├── images/
├── output/
└── benchmarks/
"""

import os
import sys
import time
import subprocess
import hashlib
from pathlib import Path


class ColorPrint:
    """Terminal color printing"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def print_header(text):
        print(f"\n{ColorPrint.HEADER}{ColorPrint.BOLD}{text}{ColorPrint.ENDC}")
    
    @staticmethod
    def print_success(text):
        print(f"{ColorPrint.OKGREEN}✓ {text}{ColorPrint.ENDC}")
    
    @staticmethod
    def print_error(text):
        print(f"{ColorPrint.FAIL}✗ {text}{ColorPrint.ENDC}")
    
    @staticmethod
    def print_warning(text):
        print(f"{ColorPrint.WARNING}⚠ {text}{ColorPrint.ENDC}")
    
    @staticmethod
    def print_info(text):
        print(f"{ColorPrint.OKCYAN}ℹ {text}{ColorPrint.ENDC}")


class ProjectSetup:
    """Handles project initialization and dependency checks"""
    
    @staticmethod
    def check_dependencies():
        """Check if required packages are installed"""
        ColorPrint.print_header("CHECKING DEPENDENCIES")
        
        required_packages = [
            'cryptography',
            'PIL',
            'numpy',
            'matplotlib',
            'psutil'
        ]
        
        missing = []
        for package in required_packages:
            try:
                if package == 'PIL':
                    __import__('PIL')
                else:
                    __import__(package)
                ColorPrint.print_success(f"{package} installed")
            except ImportError:
                ColorPrint.print_error(f"{package} NOT installed")
                missing.append(package if package != 'PIL' else 'pillow')
        
        if missing:
            ColorPrint.print_warning(f"\nMissing packages: {', '.join(missing)}")
            print(f"\nInstall with: pip install {' '.join(missing)}")
            return False
        
        ColorPrint.print_success("All dependencies satisfied")
        return True
    
    @staticmethod
    def create_directory_structure():
        """Create necessary directories"""
        ColorPrint.print_header("CREATING DIRECTORY STRUCTURE")
        
        dirs = ['keys', 'images', 'output', 'benchmarks', 'benchmarks/stego_output']
        
        for dir_name in dirs:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            ColorPrint.print_success(f"Created/verified: {dir_name}/")
    
    @staticmethod
    def verify_python_version():
        """Check Python version"""
        ColorPrint.print_header("CHECKING PYTHON VERSION")
        
        version = sys.version_info
        print(f"Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major >= 3 and version.minor >= 8:
            ColorPrint.print_success("Python version OK (3.8+)")
            return True
        else:
            ColorPrint.print_error("Python 3.8+ required")
            return False


class PhaseRunner:
    """Executes each phase with validation"""
    
    def __init__(self):
        self.results = {}
    
    def phase3_benxhmark(self, num_tests=10):
        """Fallback Phase 3 benchmarking using enhanced_phase3/dataset_manager"""
        try:
            # Import enhanced fallback utilities
            from enhanced_phase3 import EnhancedBenchmarkRunner, ComprehensiveReportGenerator
            from dataset_manager import DatasetManager
            import phase1_baseline
            import phase2_optimized
            from pathlib import Path
            
            ColorPrint.print_info("Using fallback Phase 3 (enhanced_phase3)")
            
            # Prepare or load dataset
            dataset_mgr = DatasetManager("dataset")
            if not dataset_mgr.metadata_file.exists():
                # If dataset not present, generate a small synthetic set
                dataset_mgr.generate_synthetic_dataset(num_images=max(10, num_tests))
            image_paths, text_messages = dataset_mgr.load_dataset()
            
            # Truncate to requested number of tests
            image_paths = image_paths[:num_tests]
            text_messages = text_messages[:num_tests]
            
            # Load receiver keys
            receiver_priv, receiver_pub = phase2_optimized.load_or_generate_keys()
            
            # Setup systems
            phase1_system = {
                'crypto': phase1_baseline.ECCCrypto(),
                'lsb': phase1_baseline.LSBSteganography()
            }
            phase2_system = phase2_optimized.OptimizedStegoSystem()
            
            # Run comprehensive benchmark
            output_dir = Path("benchmarks/stego_output")
            runner = EnhancedBenchmarkRunner(phase1_system, phase2_system, (receiver_priv, receiver_pub))
            runner.run_batch_benchmark(image_paths, text_messages, output_dir)
            
            # Generate comprehensive reports
            reports_dir = Path("benchmarks")
            ComprehensiveReportGenerator.generate_all_reports(
                runner.phase1_metrics,
                runner.phase2_metrics,
                reports_dir
            )
            
            # Collect stats for final summary compatibility
            phase1_stats = runner.phase1_metrics.get_statistics()
            phase2_stats = runner.phase2_metrics.get_statistics()
            
            self.results['phase3'] = {
                'status': 'SUCCESS',
                'phase1_stats': phase1_stats,
                'phase2_stats': phase2_stats,
                'num_tests': len(image_paths)
            }
            
            ColorPrint.print_success("Phase 3 completed successfully (fallback)")
            return True
        except Exception as e:
            ColorPrint.print_error(f"Phase 3 fallback error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results['phase3'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def run_phase1(self):
        """Execute Phase 1: Baseline Implementation"""
        ColorPrint.print_header("=" * 70)
        ColorPrint.print_header("PHASE 1: BASELINE ECC + LSB STEGANOGRAPHY")
        ColorPrint.print_header("=" * 70)
        
        try:
            # Import and run Phase 1
            import phase1_baseline
            
            print("\nExecuting baseline implementation...")
            start_time = time.time()
            
            # Run Phase 1 main
            phase1_baseline.main()
            
            elapsed = time.time() - start_time
            
            # Validate output
            if os.path.exists("output/stego_phase1.png"):
                ColorPrint.print_success(f"Phase 1 completed in {elapsed:.2f}s")
                self.results['phase1'] = {
                    'status': 'SUCCESS',
                    'time': elapsed,
                    'output': 'output/stego_phase1.png'
                }
                return True
            else:
                ColorPrint.print_error("Phase 1 failed: Output file not created")
                self.results['phase1'] = {'status': 'FAILED', 'error': 'No output file'}
                return False
                
        except Exception as e:
            ColorPrint.print_error(f"Phase 1 error: {str(e)}")
            self.results['phase1'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def run_phase2(self):
        """Execute Phase 2: Optimized Implementation"""
        ColorPrint.print_header("=" * 70)
        ColorPrint.print_header("PHASE 2: OPTIMIZED LIGHTWEIGHT IMPLEMENTATION")
        ColorPrint.print_header("=" * 70)
        
        try:
            import phase2_optimized
            
            print("\nExecuting optimized implementation...")
            start_time = time.time()
            
            phase2_optimized.main()
            
            elapsed = time.time() - start_time
            
            if os.path.exists("output/stego_phase2.png"):
                ColorPrint.print_success(f"Phase 2 completed in {elapsed:.2f}s")
                self.results['phase2'] = {
                    'status': 'SUCCESS',
                    'time': elapsed,
                    'output': 'output/stego_phase2.png'
                }
                return True
            else:
                ColorPrint.print_error("Phase 2 failed: Output file not created")
                self.results['phase2'] = {'status': 'FAILED', 'error': 'No output file'}
                return False
                
        except Exception as e:
            ColorPrint.print_error(f"Phase 2 error: {str(e)}")
            self.results['phase2'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def run_phase3(self, num_tests=10):
        """Execute Phase 3: Comprehensive Benchmarking"""
        ColorPrint.print_header("=" * 70)
        ColorPrint.print_header("PHASE 3: COMPREHENSIVE BENCHMARKING")
        ColorPrint.print_header("=" * 70)
        
        try:
            try:
                import importlib
                phase3_benchmark = importlib.import_module('phase3_benchmark')
                DatasetPreparation = getattr(phase3_benchmark, 'DatasetPreparation')
                ComparativeBenchmark = getattr(phase3_benchmark, 'ComparativeBenchmark')
                ResultsVisualizer = getattr(phase3_benchmark, 'ResultsVisualizer')
            except ModuleNotFoundError:
                # Use fallback implementation
                return self.phase3_benxhmark(num_tests)
            
            # Import Phase 1 and 2 systems
            import phase1_baseline
            import phase2_optimized
            
            print(f"\nPreparing dataset ({num_tests} images)...")
            dataset_prep = DatasetPreparation()
            images = dataset_prep.create_test_images("images/test_dataset", num_tests)
            messages = dataset_prep.generate_test_messages(num_tests)
            
            # Load receiver keys
            receiver_priv, receiver_pub = phase2_optimized.load_or_generate_keys()
            
            # Setup systems
            phase1_system = {
                'crypto': phase1_baseline.ECCCrypto(),
                'lsb': phase1_baseline.LSBSteganography()
            }
            phase2_system = phase2_optimized.OptimizedStegoSystem()
            
            # Run benchmark
            benchmark = ComparativeBenchmark(
                phase1_system, 
                phase2_system, 
                (receiver_priv, receiver_pub)
            )
            
            benchmark.run_batch_benchmark(images, messages, "benchmarks/stego_output")
            
            # Get statistics
            phase1_stats = benchmark.phase1_metrics.get_statistics()
            phase2_stats = benchmark.phase2_metrics.get_statistics()
            
            # Generate visualizations
            ColorPrint.print_header("\nGENERATING REPORTS")
            ResultsVisualizer.generate_comparison_charts(
                phase1_stats, phase2_stats, "benchmarks"
            )
            ResultsVisualizer.generate_report(
                phase1_stats, phase2_stats, "benchmarks"
            )
            
            # Store results
            self.results['phase3'] = {
                'status': 'SUCCESS',
                'phase1_stats': phase1_stats,
                'phase2_stats': phase2_stats,
                'num_tests': num_tests
            }
            
            ColorPrint.print_success("Phase 3 completed successfully")
            return True
            
        except Exception as e:
            ColorPrint.print_error(f"Phase 3 error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results['phase3'] = {'status': 'ERROR', 'error': str(e)}
            return False
    
    def print_final_summary(self):
        """Print final summary of all phases"""
        ColorPrint.print_header("=" * 70)
        ColorPrint.print_header("FINAL SUMMARY")
        ColorPrint.print_header("=" * 70)
        
        print("\nPhase Results:")
        for phase, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                ColorPrint.print_success(f"{phase.upper()}: {status}")
            else:
                ColorPrint.print_error(f"{phase.upper()}: {status}")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
        
        if 'phase3' in self.results and self.results['phase3']['status'] == 'SUCCESS':
            print("\n" + "=" * 70)
            print("KEY FINDINGS:")
            print("=" * 70)
            
            p1 = self.results['phase3']['phase1_stats']
            p2 = self.results['phase3']['phase2_stats']
            
            # Support both original and enhanced metric keys
            def _mean(stats: dict, key_fallbacks):
                for k in key_fallbacks:
                    if k in stats and isinstance(stats[k], dict) and 'mean' in stats[k]:
                        return stats[k]['mean']
                return 0.0

            p1_enc = _mean(p1, ['encode', 't_total_encode'])
            p2_enc = _mean(p2, ['encode', 't_total_encode'])
            p1_mem = _mean(p1, ['memory', 'peak_memory_encode'])
            p2_mem = _mean(p2, ['memory', 'peak_memory_encode'])

            enc_improvement = ((p1_enc - p2_enc) / p1_enc * 100) if p1_enc != 0 else 0.0
            mem_improvement = ((p1_mem - p2_mem) / p1_mem * 100) if p1_mem != 0 else 0.0
            
            print(f"\nEncoding Time:")
            print(f"  Phase 1: {p1_enc:.2f}ms")
            print(f"  Phase 2: {p2_enc:.2f}ms")
            print(f"  Improvement: {enc_improvement:+.1f}%")
            
            print(f"\nMemory Usage:")
            print(f"  Phase 1: {p1_mem:.2f}MB")
            print(f"  Phase 2: {p2_mem:.2f}MB")
            print(f"  Improvement: {mem_improvement:+.1f}%")
            
            print(f"\nSuccess Rate:")
            print(f"  Phase 1: {p1['success_rate']:.1f}%")
            print(f"  Phase 2: {p2['success_rate']:.1f}%")
            
            print("\n" + "=" * 70)
            ColorPrint.print_info("Detailed results saved in benchmarks/ directory")
            print("  - comparison_charts.png")
            print("  - benchmark_report.txt")


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print(" " * 15 + "ECC STEGANOGRAPHY PROJECT")
    print(" " * 10 + "Lightweight Implementation & Benchmarking")
    print("=" * 70)
    
    # Step 1: Check Python version
    if not ProjectSetup.verify_python_version():
        return
    
    # Step 2: Check dependencies
    if not ProjectSetup.check_dependencies():
        ColorPrint.print_error("\nPlease install missing dependencies and try again")
        return
    
    # Step 3: Setup directories
    ProjectSetup.create_directory_structure()
    
    # Step 4: Run all phases
    runner = PhaseRunner()
    
    # Ask user which phases to run
    ColorPrint.print_header("\nSELECT EXECUTION MODE")
    print("1. Run all phases (complete workflow)")
    print("2. Run Phase 1 only (baseline)")
    print("3. Run Phase 2 only (optimized)")
    print("4. Run Phase 3 only (benchmarking)")
    print("5. Run Phases 1+2 (skip benchmarking)")
    
    try:
        choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"
        
        if choice in ['1', '2', '5']:
            if runner.run_phase1():
                ColorPrint.print_success("\n✓ Phase 1 passed validation")
            else:
                ColorPrint.print_error("\n✗ Phase 1 failed")
                if choice != '1':
                    return
        
        if choice in ['1', '3', '5']:
            if runner.run_phase2():
                ColorPrint.print_success("\n✓ Phase 2 passed validation")
            else:
                ColorPrint.print_error("\n✗ Phase 2 failed")
                if choice != '1':
                    return
        
        if choice in ['1', '4']:
            num_tests = input("\nNumber of benchmark tests [default: 10]: ").strip()
            num_tests = int(num_tests) if num_tests.isdigit() else 10
            
            if runner.run_phase3(num_tests):
                ColorPrint.print_success("\n✓ Phase 3 completed")
            else:
                ColorPrint.print_error("\n✗ Phase 3 failed")
        
        # Print final summary
        runner.print_final_summary()
        
        ColorPrint.print_success("\n✓ All requested phases completed!")
        ColorPrint.print_info("\nNext steps for research paper:")
        print("  1. Analyze benchmarks/comparison_charts.png")
        print("  2. Review benchmarks/benchmark_report.txt")
        print("  3. Document Big-O complexity improvements")
        print("  4. Include statistical significance tests")
        print("  5. Discuss real-world applicability")
        
    except KeyboardInterrupt:
        ColorPrint.print_warning("\n\nExecution interrupted by user")
    except Exception as e:
        ColorPrint.print_error(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()