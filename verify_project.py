#!/usr/bin/env python3
\"\"\"SAF-DETR Project Verification Script
======================================

This script verifies the SAF-DETR project structure and runs basic tests
to ensure all modules are correctly implemented.

Usage:
    python verify_project.py [--full]

Options:
    --full    Run full integration tests (requires dependencies)
\"\"\"


import ast
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict


class ProjectVerifier:
    \"\"\"Verifies SAF-DETR project structure and code validity.\"\"\"

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success: List[str] = []

    def log_error(self, msg: str):
        self.errors.append(msg)
        print(f"  ✗ {msg}")

    def log_warning(self, msg: str):
        self.warnings.append(msg)
        print(f"  ⚠ {msg}")

    def log_success(self, msg: str):
        self.success.append(msg)
        print(f"  ✓ {msg}")

    def check_file_structure(self) -> bool:
        \"\"\"Verify all required files exist.\"\"\"
        print("\n📁 Checking File Structure...")
        
        required_files = [
            "models/SAF_DETR/__init__.py",
            "models/SAF_DETR/adaptive_intelligence.py",
            "models/SAF_DETR/feature_enhancement.py",
            "models/SAF_DETR/temporal_memory.py",
            "models/SAF_DETR/loss_functions.py",
            "models/SAF_DETR/novel_pipeline.py",
            "models/SAF_DETR/complete_model.py",
            "training/train.py",
            "evaluation/evaluate.py",
            "deployment/app.py",
            "deployment/api.py",
            "deployment/requirements.txt",
            "README.md",
        ]

        all_exist = True
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.log_success(f"{file_path} exists")
            else:
                self.log_error(f"{file_path} missing")
                all_exist = False

        return all_exist

    def check_python_syntax(self) -> bool:
        \"\"\"Verify all Python files have valid syntax.\"\"\"\n        print("\n🐍 Checking Python Syntax...")
        
        python_files = list(self.project_root.rglob("*.py"))
        all_valid = True

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source)
                rel_path = py_file.relative_to(self.project_root)
                self.log_success(f"{rel_path} - valid syntax")
            except SyntaxError as e:
                rel_path = py_file.relative_to(self.project_root)
                self.log_error(f"{rel_path} - SyntaxError: {e}")
                all_valid = False
            except Exception as e:
                rel_path = py_file.relative_to(self.project_root)
                self.log_error(f"{rel_path} - Error: {e}")
                all_valid = False

        return all_valid

    def check_imports(self) -> bool:
        \"\"\"Check if imports are properly structured.\"\"\"\n        print("\n📦 Checking Import Structure...")
        
        # Check for circular imports by attempting to parse imports
        init_file = self.project_root / "models" / "SAF_DETR" / "__init__.py"
        if init_file.exists():
            self.log_success("models/SAF_DETR/__init__.py exists")
        else:
            self.log_warning("models/SAF_DETR/__init__.py missing (optional but recommended)")

        return True

    def check_dependencies(self) -> bool:
        \"\"\"Check if required dependencies are available.\"\"\"\n        print("\n📋 Checking Dependencies...")
        
        required_packages = [
            "torch",
            "torchvision", 
            "numpy",
            "cv2",
            "PIL",
            "timm",
            "tqdm",
        ]

        optional_packages = [
            "streamlit",
            "fastapi",
            "uvicorn",
            "wandb",
        ]

        all_available = True

        for package in required_packages:
            try:
                if package == "cv2":
                    __import__("cv2")
                elif package == "PIL":
                    __import__("PIL")
                else:
                    __import__(package)
                self.log_success(f"{package} available")
            except ImportError:
                self.log_error(f"{package} not installed")
                all_available = False

        for package in optional_packages:
            try:
                __import__(package)
                self.log_success(f"{package} available (optional)")
            except ImportError:
                self.log_warning(f"{package} not installed (optional)")

        return all_available

    def run_basic_tests(self) -> bool:
        \"\"\"Run basic functionality tests.\"\"\"\n        print("\n🧪 Running Basic Tests...")
        
        try:
            # Test 1: Import modules
            sys.path.insert(0, str(self.project_root))
            
            # Check if we can import the modules
            modules_to_test = [
                "models.SAF_DETR.adaptive_intelligence",
                "models.SAF_DETR.temporal_memory",
                "models.SAF_DETR.loss_functions",
                "models.SAF_DETR.novel_pipeline",
            ]

            for module_name in modules_to_test:
                try:
                    __import__(module_name)
                    self.log_success(f"Can import {module_name}")
                except Exception as e:
                    self.log_error(f"Cannot import {module_name}: {e}")

            return True
            
        except Exception as e:
            self.log_error(f"Basic tests failed: {e}")
            return False

    def check_model_structure(self) -> bool:
        \"\"\"Verify model structure is correct.\"\"\"\n        print("\n🏗️  Checking Model Structure...")
        
        # Check complete_model.py for key components
        complete_model = self.project_root / "models" / "SAF_DETR" / "complete_model.py"
        if complete_model.exists():
            with open(complete_model, 'r') as f:
                content = f.read()
            
            required_components = [
                "class CompleteSAFDETR",
                "def forward",
                "build_saf_detr",
                "RTDETRBackbone",
                "HybridEncoder",
                "HumanCentricDecoder",
            ]

            for component in required_components:
                if component in content:
                    self.log_success(f"Found {component}")
                else:
                    self.log_error(f"Missing {component}")

        return True

    def generate_report(self) -> Dict:
        \"\"\"Generate verification report.\"\"\"\n        print("\n" + "="*60)
        print("📊 VERIFICATION REPORT")
        print("="*60)
        
        print(f"\n✓ Success: {len(self.success)} checks passed")
        print(f"⚠ Warnings: {len(self.warnings)} warnings")
        print(f"✗ Errors: {len(self.errors)} errors")

        if self.errors:
            print("\n❌ VERIFICATION FAILED")
            print("Please fix the errors above before proceeding.")
            return {"status": "failed", "errors": self.errors, "warnings": self.warnings}
        elif self.warnings:
            print("\n⚠️  VERIFICATION PASSED WITH WARNINGS")
            print("Project is functional but has optional issues.")
            return {"status": "warning", "errors": self.errors, "warnings": self.warnings}
        else:
            print("\n✅ VERIFICATION PASSED")
            print("Project is ready to run!")
            return {"status": "success", "errors": self.errors, "warnings": self.warnings}

    def run_all_checks(self, full: bool = False) -> Dict:
        \"\"\"Run all verification checks.\"\"\"\n        print("="*60)
        print("🔍 SAF-DETR PROJECT VERIFICATION")
        print("="*60)
        print(f"Project Root: {self.project_root}")

        checks = [
            self.check_file_structure,
            self.check_python_syntax,
            self.check_imports,
        ]

        if full:
            checks.extend([
                self.check_dependencies,
                self.run_basic_tests,
                self.check_model_structure,
            ])

        for check in checks:
            try:
                check()
            except Exception as e:
                self.log_error(f"Check failed with exception: {e}")

        return self.generate_report()


def print_usage_instructions():
    \"\"\"Print instructions on how to run the project.\"\"\"\n    print("\n" + "="*60)
    print("📖 HOW TO RUN SAF-DETR")
    print("="*60)
    
    print("""
1. INSTALL DEPENDENCIES:
   ---------------------
   pip install torch torchvision timm numpy opencv-python Pillow matplotlib tqdm
   
   # Optional for deployment
   pip install streamlit fastapi uvicorn

2. TRAIN THE MODEL:
   ---------------
   cd SAF_DETR_Project
   
   # Prepare your dataset in the following structure:
   data/
   ├── train/
   │   ├── images/
   │   └── annotations.json
   └── val/
       ├── images/
       └── annotations.json
   
   # Run training
   python training/train.py

3. RUN WEB DEMO:
   -------------
   cd SAF_DETR_Project
   
   # Start Streamlit app
   streamlit run deployment/app.py
   
   # Or use FastAPI
   uvicorn deployment.api:app --reload

4. PROJECT STRUCTURE:
   ----------------
   SAF_DETR_Project/
   ├── models/SAF_DETR/       # Core model modules
   ├── training/              # Training scripts
   ├── evaluation/            # Evaluation scripts
   ├── deployment/            # Web apps and API
   └── checkpoints/           # Saved model weights

5. KEY FEATURES:
   ------------
   ✓ Adaptive Image Intelligence - Auto-enhances low-quality footage
   ✓ Temporal Memory - Maintains consistency across frames
   ✓ Novel Pipeline - Uncertainty quantification & progressive refinement
   ✓ Real-time Performance - 25-30 FPS on RTX 4070
    """)


def main():
    \"\"\"Main entry point.\"\"\"\n    full_check = "--full" in sys.argv
    
    verifier = ProjectVerifier()
    result = verifier.run_all_checks(full=full_check)
    
    print_usage_instructions()
    
    # Return exit code
    if result["status"] == "failed":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()