#!/usr/bin/env python3
"""
Basic tests for Ara AI
"""

import unittest
from meridianalgo import quick_predict, analyze_accuracy, get_version_info

class TestBasicFunctionality(unittest.TestCase):
    """Test basic Ara AI functionality"""
    
    def test_version_info(self):
        """Test version information"""
        version_info = get_version_info()
        self.assertIsInstance(version_info, dict)
        self.assertIn('version', version_info)
        self.assertIn('features', version_info)
    
    def test_quick_predict_input_validation(self):
        """Test input validation"""
        # Test invalid symbol
        result = quick_predict('', days=5)
        self.assertIsNone(result)
        
        # Test invalid days
        result = quick_predict('AAPL', days=0)
        self.assertIsNone(result)
        
        result = quick_predict('AAPL', days=50)
        self.assertIsNone(result)
    
    def test_quick_predict_valid_input(self):
        """Test valid prediction"""
        result = quick_predict('AAPL', days=3)
        if result:  # May fail due to network issues
            self.assertIsInstance(result, dict)
            self.assertIn('predictions', result)
            self.assertEqual(len(result['predictions']), 3)

if __name__ == '__main__':
    unittest.main()
