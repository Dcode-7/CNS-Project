"""
COMPLETE RESEARCH PIPELINE
===========================
One-click execution of entire research workflow:
1. Dataset preparation
2. Phase 1 baseline
3. Phase 2 optimized
4. Comprehensive benchmarking
5. Report generation
6. Validation
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime


class ResearchPipeline:
    """Complete pipeline orchestrator"""
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
        self.start_time = time.time()
    
    def run_complete_pipeline(self):
        """Execute complete research workflow"""
        print("\n" + "="*80)
        print(" " * 20 + "RESEARCH PIPELINE EXECUTION")
        print(" " * 15 + "ECC Steganography Optimization Study")
        print("="*80)
        print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {json.dumps(self.config, indent=2)}\n")
        
        try:
            # Stage 1: Environment Setup
            if not self._check_environment():
                return False
            
            # Stage 2: Dataset Preparation
            if not self._prepare_dataset():
                return False
            
            # Stage 3: Run Phase 1 Baseline
            if not self._run_phase1():
                return False
            
            # Stage 4: Run Phase 2 Optimized
            if not self._run_phase2():
                return False
            
            # Stage 5: Comprehensive Benchmarking
            if not self._run_comprehensive_benchmark():
                return False
            
            # Stage 6: Generate Reports
            if not self._generate_reports():
                return False
            
            # Stage 7: Validate Results
            if not self._validate_results():
                return False
            
            # Final Summary
            self._print_final_summary()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠ Pipeline interrupted by user")
            return False
        except Exception as e:
            print(f"\n\n✗ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_environment(self) -> bool:
        """Stage 1: Check environment and dependencies"""
        print("\n" + "="*80)
        print("STAGE 1: ENVIRONMENT SETUP")
        print("="*80)
        
        # Check Python version
        print("\n[1.1] Checking Python version...")
        version = sys.version_info
        if version.major < 3 or version.minor < 8:
            print(f"✗ Python 3.8+ required (found {version.major}.{version.minor})")
            return False
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        
        # Check dependencies
        print("\n[1.2] Checking dependencies...")
        required = ['cryptography', 'PIL', 'numpy', 'matplotlib', 'psutil']
        missing = []
        
        for pkg in required:
            try:
                if pkg == 'PIL':
                    __import__('PIL')
                else:
                    __import__(pkg)
                print(f"  ✓ {pkg}")
            except ImportError:
                print(f"  ✗ {pkg} NOT FOUND")
                missing.append('pillow' if pkg == 'PIL' else pkg)
        
        if missing:
            print(f"\n✗ Missing packages: {', '.join(missing)}")
            print(f"Install with: pip install {' '.join(missing)}")
            return False
        
        # Create directory structure
        print("\n[1.3] Creating directories...")
        dirs = ['keys', 'images', 'output', 'benchmarks', 'benchmarks/reports', 
                'benchmarks/stego_output', 'dataset', 'dataset/images', 'dataset/texts']
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        print("✓ Directory structure created")
        
        print("\n✓ STAGE 1 COMPLETE: Environment ready")
        return True
    
    def _prepare_dataset(self) -> bool:
        """Stage 2: Prepare benchmark dataset"""
        print("\n" + "="*80)
        print("STAGE 2: DATASET PREPARATION")
        print("="*80)
        
        from dataset_manager import DatasetManager
        
        dataset_mgr = DatasetManager("dataset")
        
        # Check if dataset exists
        if dataset_mgr.metadata_file.exists():
            print("\n✓ Dataset already exists")
            with open(dataset_mgr.metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  Images: {len(metadata['images'])}")
            print(f"  Texts: {len(metadata['texts'])}")
            
            use_existing = input("\nUse existing dataset? [Y/n]: ").strip().lower()
            if use_existing != 'n':
                self.results['dataset'] = metadata
                return True
        
        # Generate new dataset
        num_images = self.config.get('num_images', 200)
        print(f"\n[2.1] Generating {num_images} synthetic images...")
        
        metadata = dataset_mgr.prepare_standard_benchmark_set(
            num_images=num_images,
            image_size=(512, 512)
        )
        
        self.results['dataset'] = metadata
        
        print("\n✓ STAGE 2 COMPLETE: Dataset prepared")
        return True
    
    def _run_phase1(self) -> bool:
        """Stage 3: Run Phase 1 baseline"""
        print("\n" + "="*80)
        print("STAGE 3: PHASE 1 BASELINE")
        print("="*80)
        
        if self.config.get('skip_phase1', False):
            print("\n⊘ Skipping Phase 1 (configured)")
            return True
        
        try:
            print("\n[3.1] Executing Phase 1 baseline implementation...")
            import phase1_baseline
            
            # Run with output capture
            phase1_baseline.main()
            
            if Path("output/stego_phase1.png").exists():
                print("✓ Phase 1 output generated")
                self.results['phase1'] = {'status': 'SUCCESS'}
            else:
                print("✗ Phase 1 output not found")
                return False
            
        except Exception as e:
            print(f"✗ Phase 1 failed: {str(e)}")
            return False
        
        print("\n✓ STAGE 3 COMPLETE: Phase 1 baseline executed")
        return True
    
    def _run_phase2(self) -> bool:
        """Stage 4: Run Phase 2 optimized"""
        print("\n" + "="*80)
        print("STAGE 4: PHASE 2 OPTIMIZED")
        print("="*80)
        
        if self.config.get('skip_phase2', False):
            print("\n⊘ Skipping Phase 2 (configured)")
            return True
        
        try:
            print("\n[4.1] Executing Phase 2 optimized implementation...")
            import phase2_optimized
            
            phase2_optimized.main()
            
            if Path("output/stego_phase2.png").exists():
                print("✓ Phase 2 output generated")
                self.results['phase2'] = {'status': 'SUCCESS'}
            else:
                print("✗ Phase 2 output not found")
                return False
            
        except Exception as e:
            print(f"✗ Phase 2 failed: {str(e)}")
            return False
        
        print("\n✓ STAGE 4 COMPLETE: Phase 2 optimized executed")
        return True
    
    def _run_comprehensive_benchmark(self) -> bool:
        """Stage 5: Run comprehensive benchmarking"""
        print("\n" + "="*80)
        print("STAGE 5: COMPREHENSIVE BENCHMARKING")
        print("="*80)
        
        try:
            from dataset_manager import DatasetManager
            from enhanced_phase3 import (
                EnhancedBenchmarkRunner,
                ComprehensiveReportGenerator
            )
            import phase1_baseline
            import phase2_optimized
            
            print("\n[5.1] Loading dataset...")
            dataset_mgr = DatasetManager("dataset")
            images, texts = dataset_mgr.load_dataset()
            
            # Limit dataset if configured
            max_tests = self.config.get('max_benchmark_tests', len(images))
            images = images[:max_tests]
            texts = texts[:max_tests]
            
            print(f"  Using {len(images)} images for benchmarking")
            
            print("\n[5.2] Setting up benchmark systems...")
            receiver_priv, receiver_pub = phase2_optimized.load_or_generate_keys()
            
            phase1_system = {
                'crypto': phase1_baseline.ECCCrypto(),
                'lsb': phase1_baseline.LSBSteganography()
            }
            phase2_system = phase2_optimized.OptimizedStegoSystem()
            
            print("\n[5.3] Running batch benchmark...")
            print("  This may take 15-60 minutes depending on dataset size...")
            
            benchmark = EnhancedBenchmarkRunner(
                phase1_system,
                phase2_system,
                (receiver_priv, receiver_pub)
            )
            
            benchmark.run_batch_benchmark(
                images,
                texts,
                Path("benchmarks/stego_output")
            )
            
            print("\n[5.4] Generating comprehensive reports...")
            ComprehensiveReportGenerator.generate_all_reports(
                benchmark.phase1_metrics,
                benchmark.phase2_metrics,
                Path("benchmarks/reports")
            )
            
            # Store metrics
            self.results['phase1_metrics'] = benchmark.phase1_metrics.get_statistics()
            self.results['phase2_metrics'] = benchmark.phase2_metrics.get_statistics()
            
        except Exception as e:
            print(f"✗ Benchmark failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n✓ STAGE 5 COMPLETE: Comprehensive benchmarking done")
        return True
    
    def _generate_reports(self) -> bool:
        """Stage 6: Generate final reports"""
        print("\n" + "="*80)
        print("STAGE 6: REPORT GENERATION")
        print("="*80)
        
        print("\n[6.1] Reports generated:")
        reports = [
            "benchmarks/reports/comprehensive_comparison.png",
            "benchmarks/reports/comprehensive_report.txt",
            "benchmarks/reports/results_table.tex",
            "benchmarks/reports/phase1_metrics.json",
            "benchmarks/reports/phase2_metrics.json"
        ]
        
        for report in reports:
            if Path(report).exists():
                print(f"  ✓ {report}")
            else:
                print(f"  ⚠ {report} not found")
        
        print("\n✓ STAGE 6 COMPLETE: Reports generated")
        return True
    
    def _validate_results(self) -> bool:
        """Stage 7: Validate lightweight criteria"""
        print("\n" + "="*80)
        print("STAGE 7: RESULTS VALIDATION")
        print("="*80)
        
        if 'phase2_metrics' not in self.results:
            print("⚠ No metrics to validate")
            return True
        
        p1 = self.results.get('phase1_metrics', {})
        p2 = self.results['phase2_metrics']
        
        print("\n[7.1] Checking lightweight criteria...")
        
        criteria = []
        
        def _get_mean(stats: dict, key: str) -> float:
            try:
                return float(stats[key]['mean'])
            except Exception:
                return 0.0
        
        def _safe_improvement(baseline: float, improved: float) -> float:
            return ((baseline - improved) / baseline * 100) if baseline != 0 else 0.0
        
        # Time criteria
        if 't_total_encode' in p2:
            enc_time = _get_mean(p2, 't_total_encode')
            criteria.append(("Encode time < 100ms", enc_time < 100, f"{enc_time:.2f}ms"))
            
            if 't_total_encode' in p1:
                p1_enc = _get_mean(p1, 't_total_encode')
                improvement = _safe_improvement(p1_enc, enc_time)
                criteria.append(("Speed improvement ≥ 30%", improvement >= 30, 
                               f"{improvement:.1f}%"))
        
        # Memory criteria
        if 'peak_memory_encode' in p2:
            mem = _get_mean(p2, 'peak_memory_encode')
            criteria.append(("Memory < 50MB", mem < 50, f"{mem:.2f}MB"))
            
            if 'peak_memory_encode' in p1:
                p1_mem = _get_mean(p1, 'peak_memory_encode')
                mem_imp = _safe_improvement(p1_mem, mem)
                criteria.append(("Memory improvement ≥ 25%", mem_imp >= 25, 
                               f"{mem_imp:.1f}%"))
        
        # Quality criteria
        if 'psnr_db' in p2:
            psnr = p2['psnr_db']['mean']
            criteria.append(("PSNR > 50dB", psnr > 50, f"{psnr:.2f}dB"))
        
        if 'ssim' in p2:
            ssim = p2['ssim']['mean']
            criteria.append(("SSIM > 0.99", ssim > 0.99, f"{ssim:.4f}"))
        
        # Print results
        passed_count = 0
        for criterion, passed, value in criteria:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {criterion} ({value})")
            if passed:
                passed_count += 1
        
        print(f"\n[7.2] Validation Summary: {passed_count}/{len(criteria)} criteria met")
        
        all_passed = passed_count == len(criteria)
        if all_passed:
            print("\n✓ STAGE 7 COMPLETE: All validation criteria MET")
            print("  🎉 Implementation successfully proven lightweight!")
        else:
            print("\n⚠ STAGE 7 COMPLETE: Some criteria not met")
            print("  Consider further optimization or discuss in paper limitations")
        
        self.results['validation'] = {
            'criteria_passed': passed_count,
            'criteria_total': len(criteria),
            'all_passed': all_passed
        }
        
        return True
    
    def _print_final_summary(self):
        """Print final pipeline summary"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*80)
        print(" " * 25 + "PIPELINE COMPLETE")
        print("="*80)
        
        print(f"\nExecution Time: {elapsed/60:.1f} minutes")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nResults Summary:")
        print(f"  Dataset: {self.results.get('dataset', {}).get('num_images', 'N/A')} images")
        print(f"  Phase 1: {self.results.get('phase1', {}).get('status', 'N/A')}")
        print(f"  Phase 2: {self.results.get('phase2', {}).get('status', 'N/A')}")
        
        if 'validation' in self.results:
            val = self.results['validation']
            print(f"  Validation: {val['criteria_passed']}/{val['criteria_total']} criteria met")
        
        print("\nOutput Files:")
        print("  benchmarks/reports/comprehensive_comparison.png")
        print("  benchmarks/reports/comprehensive_report.txt")
        print("  benchmarks/reports/results_table.tex")
        print("  benchmarks/reports/phase1_metrics.json")
        print("  benchmarks/reports/phase2_metrics.json")
        
        print("\nNext Steps for Research Paper:")
        print("  1. Review comprehensive_report.txt for detailed statistics")
        print("  2. Include comprehensive_comparison.png in paper figures")
        print("  3. Use results_table.tex in paper results section")
        print("  4. Document methodology from dataset metadata")
        print("  5. Discuss improvements and validation in paper")
        
        if self.results.get('validation', {}).get('all_passed', False):
            print("\n✓ 🎉 SUCCESS: Lightweight implementation validated!")
        else:
            print("\n⚠ Note: Review validation criteria and consider optimization")
        
        print("\n" + "="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Complete Research Pipeline for ECC Steganography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with 200 images
  python complete_pipeline.py --num-images 200
  
  # Quick test with 50 images
  python complete_pipeline.py --num-images 50 --max-tests 50
  
  # Skip Phase 1, only run Phase 2 and benchmark
  python complete_pipeline.py --skip-phase1
        """
    )
    
    parser.add_argument(
        '--num-images',
        type=int,
        default=200,
        help='Number of images to generate in dataset (default: 200)'
    )
    
    parser.add_argument(
        '--max-tests',
        type=int,
        default=None,
        help='Maximum number of benchmark tests to run (default: all images)'
    )
    
    parser.add_argument(
        '--skip-phase1',
        action='store_true',
        help='Skip Phase 1 baseline execution'
    )
    
    parser.add_argument(
        '--skip-phase2',
        action='store_true',
        help='Skip Phase 2 optimized execution'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: 50 images, 50 tests'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'num_images': 50 if args.quick_test else args.num_images,
        'max_benchmark_tests': 50 if args.quick_test else args.max_tests,
        'skip_phase1': args.skip_phase1,
        'skip_phase2': args.skip_phase2
    }
    
    # Run pipeline
    pipeline = ResearchPipeline(config)
    success = pipeline.run_complete_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()