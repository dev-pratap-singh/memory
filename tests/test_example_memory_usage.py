"""
Complete example demonstrating the LLM Memory Management System
Shows context compression, memory state tracking, task management, and more
"""

import os
from dotenv import load_dotenv
from src.memory_manager import (
    LLMMemoryManager,
    CompressionStrategy,
    TaskStatus
)

# Load environment variables
load_dotenv()

def simulate_debugging_workflow():
    """
    Simulate a debugging workflow where user provides multiple large files
    and the system automatically manages memory
    """

    print("=" * 80)
    print("LLM MEMORY MANAGEMENT SYSTEM - DEBUGGING WORKFLOW EXAMPLE")
    print("=" * 80)

    # Initialize manager
    manager = LLMMemoryManager(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME", "memory_db"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "password"),
        model_name=os.getenv("PRIMARY_LLM_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        memory_approach=os.getenv("MEMORY_APPROACH", "external_llm"),
        use_slm=os.getenv("USE_SLM", "false").lower() == "true",
        use_lora_finetuning=os.getenv("USE_LORA_FINETUNING", "false").lower() == "true",
        slm_model_name=os.getenv("SLM_MODEL_NAME", "microsoft/phi-2")
    )

    print(f"\n✓ Initialized memory manager for session: {manager.session_id}")
    print(f"  Model: {manager.model_name}")
    print(f"  Context length: {manager.model_config['context_length']} tokens")
    print(f"  Memory approach: {manager.memory_approach.value}")
    print(f"  Using SLM: {manager.use_slm}")

    # Step 1: User provides frontend code
    print("\n" + "-" * 80)
    print("STEP 1: Adding frontend code to memory")
    print("-" * 80)

    frontend_code = """
    // React Frontend - App.js
    import React, { useState, useEffect } from 'react';
    import { BrowserRouter, Routes, Route } from 'react-router-dom';
    import Login from './components/Login';
    import Dashboard from './components/Dashboard';
    import Profile from './components/Profile';

    function App() {
      const [user, setUser] = useState(null);
      const [isAuthenticated, setIsAuthenticated] = useState(false);

      useEffect(() => {
        const token = localStorage.getItem('authToken');
        if (token) {
          validateToken(token);
        }
      }, []);

      const validateToken = async (token) => {
        try {
          const response = await fetch('/api/auth/validate', {
            headers: { 'Authorization': `Bearer ${token}` }
          });
          if (response.ok) {
            const userData = await response.json();
            setUser(userData);
            setIsAuthenticated(true);
          }
        } catch (error) {
          console.error('Token validation failed:', error);
        }
      };

      return (
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/dashboard" element={<Dashboard user={user} />} />
            <Route path="/profile" element={<Profile user={user} />} />
          </Routes>
        </BrowserRouter>
      );
    }

    export default App;
    """ * 10  # Simulate large file

    frontend_id = manager.add_memory_item(
        content=frontend_code,
        content_type="frontend",
        priority=8,
        metadata={"file": "App.js", "language": "javascript"}
    )

    print(f"✓ Added frontend code (ID: {frontend_id[:8]}...)")
    print(f"  Tokens: {manager.count_tokens(frontend_code)}")

    # Step 2: User provides backend code
    print("\n" + "-" * 80)
    print("STEP 2: Adding backend code to memory")
    print("-" * 80)

    backend_code = """
    // Node.js Express Backend - auth.routes.js
    const express = require('express');
    const jwt = require('jsonwebtoken');
    const bcrypt = require('bcrypt');
    const User = require('../models/User');

    const router = express.Router();
    const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

    // Login route
    router.post('/login', async (req, res) => {
      try {
        const { email, password } = req.body;

        const user = await User.findOne({ email });
        if (!user) {
          return res.status(401).json({ error: 'Invalid credentials' });
        }

        const isValidPassword = await bcrypt.compare(password, user.password);
        if (!isValidPassword) {
          return res.status(401).json({ error: 'Invalid credentials' });
        }

        const token = jwt.sign(
          { userId: user._id, email: user.email },
          JWT_SECRET,
          { expiresIn: '24h' }
        );

        res.json({ token, user: { id: user._id, email: user.email } });
      } catch (error) {
        res.status(500).json({ error: 'Server error' });
      }
    });

    // Token validation route
    router.get('/validate', async (req, res) => {
      try {
        const token = req.headers.authorization?.replace('Bearer ', '');

        if (!token) {
          return res.status(401).json({ error: 'No token provided' });
        }

        const decoded = jwt.verify(token, JWT_SECRET);
        const user = await User.findById(decoded.userId);

        if (!user) {
          return res.status(401).json({ error: 'User not found' });
        }

        res.json({ user: { id: user._id, email: user.email } });
      } catch (error) {
        res.status(401).json({ error: 'Invalid token' });
      }
    });

    module.exports = router;
    """ * 8  # Simulate large file

    backend_id = manager.add_memory_item(
        content=backend_code,
        content_type="backend",
        priority=8,
        metadata={"file": "auth.routes.js", "language": "javascript"}
    )

    print(f"✓ Added backend code (ID: {backend_id[:8]}...)")
    print(f"  Tokens: {manager.count_tokens(backend_code)}")

    # Step 3: Check memory state
    print("\n" + "=" * 80)
    print("MEMORY STATE AFTER ADDING CONTENT")
    print("=" * 80)

    state = manager.get_memory_state()
    print(f"\nContext Utilization: {state['context_utilization_percentage']:.2f}%")
    print(f"Used: {state['used_context_length']} tokens")
    print(f"Available: {state['available_context_length']} tokens")
    print(f"Total: {state['total_context_length']} tokens")

    # Step 4: Get working context (automatically manages compression)
    print("\n" + "-" * 80)
    print("STEP 3: Getting working context")
    print("-" * 80)

    working_context = manager.get_working_context()

    print(f"\nFull items in context: {len(working_context['full_items'])}")
    for item in working_context['full_items']:
        print(f"  - {item['type']}: {item['tokens']} tokens")

    print(f"\nCompressed items: {len(working_context['compressed_items'])}")
    for item in working_context['compressed_items']:
        print(f"  - {item['type']}: {item['tokens']} tokens (compressed)")
        print(f"    Retrieval hint: {item['retrieval_hint']}")

    print(f"\nTotal tokens in working context: {working_context['total_tokens']}")
    print(f"Available tokens: {working_context['available_tokens']}")

    # Step 5: Create debugging tasks
    print("\n" + "-" * 80)
    print("STEP 4: Creating debugging tasks")
    print("-" * 80)

    task1_id = manager.create_task(
        title="Analyze token validation logic",
        description="Review backend auth.routes.js for token validation issues",
        priority=9,
        metadata={"relates_to": backend_id}
    )

    task2_id = manager.create_task(
        title="Check frontend token storage",
        description="Verify localStorage usage and token retrieval in frontend",
        priority=8,
        metadata={"relates_to": frontend_id}
    )

    print(f"✓ Created 2 debugging tasks")

    # Step 6: Create agents
    print("\n" + "-" * 80)
    print("STEP 5: Creating specialized agents")
    print("-" * 80)

    debug_agent = manager.create_agent(
        agent_name="Debug Agent",
        agent_type="debugging",
        capabilities=["code_analysis", "error_detection", "solution_generation"],
        assigned_tools=["semantic_search", "retrieve_memory", "create_task"]
    )

    print(f"✓ Created Debug Agent (ID: {debug_agent[:8]}...)")

    # Step 7: Export final state
    print("\n" + "=" * 80)
    print("FINAL MEMORY STATE EXPORT")
    print("=" * 80)

    state_json = manager.export_memory_state_json()

    # Save to file for UI display
    with open('memory_state_export.json', 'w') as f:
        f.write(state_json)

    print("\n✓ Memory state exported to: memory_state_export.json")

    # Print summary
    import json
    state_data = json.loads(state_json)

    print("\n" + "-" * 80)
    print("SESSION SUMMARY")
    print("-" * 80)
    print(f"Session ID: {state_data['session_id']}")
    print(f"Model: {state_data['model']}")
    print(f"Memory Approach: {state_data['memory_approach']}")
    print(f"Using SLM: {state_data['use_slm']}")
    print(f"Active memory items: {len(state_data['active_memory_items'])}")
    print(f"Tasks created: {len(state_data['tasks'])}")
    print(f"Agents created: {len(state_data['agents'])}")

    memory_state = state_data['memory_state']
    print(f"\nContext utilization: {memory_state['utilization_percentage']:.2f}%")
    print(f"Total tokens used: {memory_state['token_usage']['input_tokens'] + memory_state['token_usage']['output_tokens']}")
    print(f"Total cost: ${memory_state['token_usage']['total_cost']:.4f}")

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE!")
    print("=" * 80)

    # Cleanup
    manager.close()


if __name__ == "__main__":
    simulate_debugging_workflow()
