#!/usr/bin/env python3
"""
HydroSim-RF Installation Checker
Verifica se todas as dependências estão corretamente instaladas.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Verifica versão do Python"""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python 3.9+ required (found {version.major}.{version.minor})")
        return False

def check_package(package_name, import_name=None):
    """Verifica se um pacote está instalado"""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} (not installed)")
        return False

def check_packages():
    """Verifica pacotes Python"""
    print("✓ Checking Python packages...")
    
    packages = [
        ("streamlit", "streamlit"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("rasterio", "rasterio"),
        ("geopandas", "geopandas"),
        ("contextily", "contextily"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("Pillow", "PIL"),
        ("shapely", "shapely"),
        ("fiona", "fiona"),
        ("pyproj", "pyproj"),
    ]
    
    results = []
    for package, import_name in packages:
        results.append(check_package(package, import_name))
    
    return all(results)

def check_gdal():
    """Verifica GDAL (system dependency)"""
    print("✓ Checking system dependencies...")
    
    try:
        result = subprocess.run(
            ["gdalinfo", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✓ GDAL: {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("  ⚠ GDAL not found (may require system installation)")
    return False

def check_docker():
    """Verifica Docker"""
    print("✓ Checking Docker availability...")
    
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✓ Docker: {version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("  ⚠ Docker not found (optional for containerization)")
    return False

def check_directories():
    """Verifica estrutura de diretórios"""
    print("✓ Checking project structure...")
    
    base_path = Path(__file__).parent
    required_dirs = ["data/input", "data/output", "logs"]
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ⚠ {dir_name}/ (creating...)")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return True

def main():
    print("=" * 50)
    print("HydroSim-RF Installation Check")
    print("=" * 50)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Packages", check_packages),
        ("System Dependencies", check_gdal),
        ("Docker", check_docker),
        ("Project Structure", check_directories),
    ]
    
    results = {}
    for check_name, check_func in checks:
        print()
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"✗ Error checking {check_name}: {e}")
            results[check_name] = False
    
    print()
    print("=" * 50)
    print("Installation Summary")
    print("=" * 50)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("✓ All checks passed! You can run: make run")
    else:
        print("⚠ Some checks failed. Review the output above.")
        print()
        print("For help, see README.md")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
