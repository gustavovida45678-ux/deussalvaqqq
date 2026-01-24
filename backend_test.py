import requests
import sys
import json
from datetime import datetime
import time
import base64
import io
from PIL import Image

class ChatAPITester:
    def __init__(self, base_url="https://chatbot-pt-1.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Response: {data}"
            self.log_test("Root Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("Root Endpoint", False, str(e))
            return False

    def test_chat_endpoint(self, message="Olá, como você está?"):
        """Test the chat endpoint with a message"""
        try:
            payload = {"message": message}
            response = requests.post(
                f"{self.api_url}/chat", 
                json=payload, 
                headers={'Content-Type': 'application/json'},
                timeout=30  # Longer timeout for AI response
            )
            
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                # Validate response structure
                required_fields = ['user_message', 'assistant_message']
                has_required_fields = all(field in data for field in required_fields)
                
                if has_required_fields:
                    user_msg = data['user_message']
                    ai_msg = data['assistant_message']
                    
                    # Validate message structure
                    user_valid = all(field in user_msg for field in ['id', 'role', 'content', 'timestamp'])
                    ai_valid = all(field in ai_msg for field in ['id', 'role', 'content', 'timestamp'])
                    
                    if user_valid and ai_valid and user_msg['role'] == 'user' and ai_msg['role'] == 'assistant':
                        details += f", AI Response: {ai_msg['content'][:100]}..."
                    else:
                        success = False
                        details += ", Invalid message structure"
                else:
                    success = False
                    details += ", Missing required fields"
            else:
                try:
                    error_data = response.json()
                    details += f", Error: {error_data}"
                except:
                    details += f", Response: {response.text[:200]}"
            
            self.log_test("Chat Endpoint", success, details)
            return success, response.json() if success else None
            
        except Exception as e:
            self.log_test("Chat Endpoint", False, str(e))
            return False, None

    def test_get_messages(self):
        """Test getting all messages"""
        try:
            response = requests.get(f"{self.api_url}/messages", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                messages = response.json()
                details += f", Messages count: {len(messages)}"
                
                # Validate message structure if messages exist
                if messages:
                    first_msg = messages[0]
                    required_fields = ['id', 'role', 'content', 'timestamp']
                    if not all(field in first_msg for field in required_fields):
                        success = False
                        details += ", Invalid message structure"
            
            self.log_test("Get Messages", success, details)
            return success, response.json() if success else None
            
        except Exception as e:
            self.log_test("Get Messages", False, str(e))
            return False, None

    def test_clear_messages(self):
        """Test clearing all messages"""
        try:
            response = requests.delete(f"{self.api_url}/messages", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            
            if success:
                data = response.json()
                if 'deleted_count' in data:
                    details += f", Deleted: {data['deleted_count']} messages"
                else:
                    success = False
                    details += ", Missing deleted_count field"
            
            self.log_test("Clear Messages", success, details)
            return success
            
        except Exception as e:
            self.log_test("Clear Messages", False, str(e))
            return False

    def test_chat_persistence(self):
        """Test that chat messages are properly saved and retrieved"""
        print("\n🔄 Testing chat persistence...")
        
        # Clear existing messages first
        self.test_clear_messages()
        
        # Send a test message
        test_message = f"Test message at {datetime.now().strftime('%H:%M:%S')}"
        chat_success, chat_data = self.test_chat_endpoint(test_message)
        
        if not chat_success:
            self.log_test("Chat Persistence", False, "Chat endpoint failed")
            return False
        
        # Wait a moment for database write
        time.sleep(1)
        
        # Retrieve messages
        get_success, messages = self.test_get_messages()
        
        if not get_success:
            self.log_test("Chat Persistence", False, "Get messages failed")
            return False
        
        # Verify messages were saved
        if len(messages) >= 2:  # Should have user + assistant message
            user_msgs = [m for m in messages if m['role'] == 'user']
            ai_msgs = [m for m in messages if m['role'] == 'assistant']
            
            if len(user_msgs) >= 1 and len(ai_msgs) >= 1:
                # Check if our test message is there
                test_msg_found = any(test_message in msg['content'] for msg in user_msgs)
                if test_msg_found:
                    self.log_test("Chat Persistence", True, f"Messages saved correctly ({len(messages)} total)")
                    return True
                else:
                    self.log_test("Chat Persistence", False, "Test message not found in saved messages")
                    return False
            else:
                self.log_test("Chat Persistence", False, f"Incorrect message roles: {len(user_msgs)} user, {len(ai_msgs)} AI")
                return False
        else:
            self.log_test("Chat Persistence", False, f"Expected 2+ messages, got {len(messages)}")
            return False

    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Chat API Backend Tests")
        print(f"Testing against: {self.base_url}")
        print("=" * 50)
        
        # Test basic connectivity
        if not self.test_root_endpoint():
            print("❌ Root endpoint failed - stopping tests")
            return self.get_summary()
        
        # Test chat functionality
        self.test_chat_endpoint()
        
        # Test message retrieval
        self.test_get_messages()
        
        # Test message clearing
        self.test_clear_messages()
        
        # Test end-to-end persistence
        self.test_chat_persistence()
        
        return self.get_summary()

    def get_summary(self):
        """Get test summary"""
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        
        summary = {
            "total_tests": self.tests_run,
            "passed_tests": self.tests_passed,
            "failed_tests": self.tests_run - self.tests_passed,
            "success_rate": f"{success_rate:.1f}%",
            "test_results": self.test_results
        }
        
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        return summary

def main():
    tester = ChatAPITester()
    summary = tester.run_all_tests()
    
    # Save results to file
    with open('/app/backend_test_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Return appropriate exit code
    return 0 if summary['passed_tests'] == summary['total_tests'] else 1

if __name__ == "__main__":
    sys.exit(main())