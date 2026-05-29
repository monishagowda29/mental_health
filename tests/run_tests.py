import unittest
import sys
import os
import json
import time

# Adjust search path so that the runner can load 'src' modules correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test_suite():
    print("=" * 60)
    print("        MINDSCAN BACKEND SERVICE - AUTOMATED TEST SUITE")
    print("=" * 60)
    print(f"Timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Ver  : {sys.version.split()[0]}")
    print(f"Platform    : {sys.platform.upper()}")
    print("-" * 60)
    
    # Discover and load unit tests
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern="test_*.py"
    )
    
    # Run tests using a custom TextTestRunner to capture metrics
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("-" * 60)
    print("                    TEST SUITE EXECUTION SUMMARY")
    print("-" * 60)
    print(f"Tests Run         : {result.testsRun}")
    print(f"Passed            : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures          : {len(result.failures)}")
    print(f"Errors            : {len(result.errors)}")
    
    # Format and save a detailed JSON execution log
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "platform": sys.platform,
        "metrics": {
            "tests_run": result.testsRun,
            "passed": result.testsRun - len(result.failures) - len(result.errors),
            "failures": len(result.failures),
            "errors": len(result.errors)
        },
        "success": result.wasSuccessful(),
        "failures_details": [
            {"test": str(test), "message": err}
            for test, err in result.failures
        ],
        "errors_details": [
            {"test": str(test), "message": err}
            for test, err in result.errors
        ]
    }
    
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "backend_test_report.json")
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print(f"\n[REPORT] Saved raw test metrics report JSON to: {report_path}")
    print("=" * 60)
    
    if not result.wasSuccessful():
        print("[FAIL] One or more test suites failed! Check details above.")
        sys.exit(1)
    else:
        print("[SUCCESS] All test suites completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    run_test_suite()
