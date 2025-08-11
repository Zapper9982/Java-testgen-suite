#!/usr/bin/env python3
"""
Test script to demonstrate the incremental merging functionality.
This script shows how the Copilot-style merging works with sample batch files.
"""

import os
import tempfile
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.llm.test_case_generator import TestCaseGenerator

def create_sample_batch_files():
    """Create sample batch files for testing the merge functionality."""
    
    # Sample batch 1 content
    batch1_content = """package com.example.service;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest_Batch1 {
    @Mock
    private UserRepository userRepository;
    
    @InjectMocks
    private UserService userService;
    
    @Test
    void testFindUserById() {
        when(userRepository.findById(1L)).thenReturn(new User("John"));
        User result = userService.findUserById(1L);
        assertNotNull(result);
        assertEquals("John", result.getName());
    }
    
    @Test
    void testCreateUser() {
        User user = new User("Jane");
        when(userRepository.save(any(User.class))).thenReturn(user);
        User result = userService.createUser(user);
        assertNotNull(result);
        assertEquals("Jane", result.getName());
    }
}"""

    # Sample batch 2 content
    batch2_content = """package com.example.service;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class UserServiceTest_Batch2 {
    @Mock
    private EmailService emailService;
    
    @InjectMocks
    private UserService userService;
    
    @Test
    void testSendWelcomeEmail() {
        User user = new User("Alice");
        doNothing().when(emailService).sendWelcomeEmail(any(User.class));
        userService.sendWelcomeEmail(user);
        verify(emailService).sendWelcomeEmail(user);
    }
    
    @Test
    void testDeleteUser() {
        when(userRepository.deleteById(1L)).thenReturn(true);
        boolean result = userService.deleteUser(1L);
        assertTrue(result);
    }
}"""

    return batch1_content, batch2_content

def test_incremental_merge():
    """Test the incremental merging functionality."""
    
    print("=== Testing Incremental Merge Functionality ===\n")
    
    # Create sample batch content
    batch1_content, batch2_content = create_sample_batch_files()
    
    print("Batch 1 Content:")
    print(batch1_content)
    print("\n" + "="*50 + "\n")
    
    print("Batch 2 Content:")
    print(batch2_content)
    print("\n" + "="*50 + "\n")
    
    # Initialize test generator (we won't actually run tests, just test the merge)
    try:
        test_generator = TestCaseGenerator(collection_name="test_collection", build_tool="maven")
        
        print("Testing merge functionality...")
        
        # Test the merge method directly
        merged_result = test_generator.merge_batch_with_existing_test_class(
            existing_test_class=batch1_content,
            new_batch_code=batch2_content,
            target_class_name="UserService",
            target_package_name="com.example.service",
            test_type="service"
        )
        
        print("=== MERGED RESULT ===")
        print(merged_result)
        print("\n=== END MERGED RESULT ===")
        
        # Check if merge was successful
        if "testFindUserById" in merged_result and "testSendWelcomeEmail" in merged_result:
            print("\n✅ SUCCESS: Merge appears to have worked correctly!")
            print("   - Both batch methods are present in merged result")
            print("   - Class structure maintained")
        else:
            print("\n❌ FAILURE: Merge may not have worked as expected")
            
    except Exception as e:
        print(f"\n❌ ERROR: Failed to test merge functionality: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_incremental_merge()
