# Tool Calling / Function Calling

This document covers how to enable LLMs to use external tools and functions, enabling capabilities like web search, calculations, API calls, and more.

---

## Table of Contents

1. [What is Tool Calling?](#what-is-tool-calling)
2. [The ReAct Pattern](#the-react-pattern)
3. [Tool Schema Definition](#tool-schema-definition)
4. [Prompt Engineering for Tools](#prompt-engineering-for-tools)
5. [Parsing Tool Calls](#parsing-tool-calls)
6. [Multi-Turn Conversations](#multi-turn-conversations)
7. [Advanced Patterns](#advanced-patterns)
8. [Code Walkthrough](#code-walkthrough)

---

## What is Tool Calling?

LLMs are trained on text and can only generate text. But many tasks require:
- **Real-time data** (weather, stock prices)
- **Computation** (math, code execution)
- **External actions** (send email, book meeting)
- **Database access** (lookup records)

**Tool calling** lets the LLM request these capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Tool Calling Flow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User: "What's the weather in Paris?"                                      │
│            ↓                                                                │
│   ┌────────────────────────────────────┐                                   │
│   │           LLM                       │                                   │
│   │                                     │                                   │
│   │   "I need to call get_weather..."   │                                   │
│   └────────────────────────────────────┘                                   │
│            ↓                                                                │
│   {"name": "get_weather", "arguments": {"location": "Paris"}}              │
│            ↓                                                                │
│   ┌────────────────────────────────────┐                                   │
│   │   Your Application                  │                                   │
│   │   (executes the tool)               │                                   │
│   └────────────────────────────────────┘                                   │
│            ↓                                                                │
│   {"temp": 22, "condition": "Sunny"}                                       │
│            ↓                                                                │
│   ┌────────────────────────────────────┐                                   │
│   │           LLM                       │                                   │
│   │                                     │                                   │
│   │   "The weather in Paris is 22°C    │                                   │
│   │    and sunny!"                      │                                   │
│   └────────────────────────────────────┘                                   │
│            ↓                                                                │
│   Final response to user                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Tools?

| Limitation | Tool Solution |
|------------|---------------|
| No real-time data | Web search, APIs |
| Math errors | Calculator tool |
| No file access | File read/write tools |
| No actions | Email, calendar, etc. |
| Hallucinations | Database lookup for facts |

---

## The ReAct Pattern

**ReAct** (Reason + Act) is a prompting pattern where the model:
1. **Thinks** about what to do
2. **Acts** by calling a tool
3. **Observes** the result
4. **Repeats** until done

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ReAct Pattern                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User: "What's 15% tip on a $67.50 bill?"                                 │
│                                                                             │
│   Thought: I need to calculate 15% of 67.50                                │
│   Action: calculate(expression="67.50 * 0.15")                             │
│   Observation: {"result": 10.125}                                          │
│                                                                             │
│   Thought: The tip is $10.13 (rounded). Total would be $77.63              │
│   Action: None (I can answer directly now)                                 │
│                                                                             │
│   Answer: The 15% tip on $67.50 is $10.13, making the total $77.63.       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tool Schema Definition

### OpenAI Function Format (Standard)

Most LLMs and frameworks use this format:

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

### Schema Components

| Field | Purpose | Example |
|-------|---------|---------|
| `name` | Function identifier | `"get_weather"` |
| `description` | Helps LLM decide when to use | `"Get current weather..."` |
| `parameters` | JSON Schema for arguments | `{type: object, properties: {...}}` |
| `required` | Mandatory parameters | `["location"]` |
| `enum` | Allowed values | `["celsius", "fahrenheit"]` |

### Parameter Types

| Type | Example | Description |
|------|---------|-------------|
| `string` | `"Paris"` | Text |
| `number` | `42.5` | Integer or float |
| `integer` | `42` | Whole numbers only |
| `boolean` | `true` | True/false |
| `array` | `["a", "b"]` | List of items |
| `object` | `{key: value}` | Nested structure |

---

## Prompt Engineering for Tools

### System Prompt Structure

```python
def create_tool_prompt(user_message: str, tools: list) -> str:
    tools_json = json.dumps(tools, indent=2)
    
    return f"""<|im_start|>system
You are a helpful assistant with access to the following tools:

{tools_json}

When you need to use a tool, respond with a JSON object in this format:
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}

Only use tools when necessary. If you can answer directly, do so.
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
"""
```

### Key Prompt Elements

| Element | Purpose |
|---------|---------|
| List available tools | LLM knows what's available |
| Describe each tool | LLM knows when to use each |
| Specify output format | JSON parsing works reliably |
| Say when NOT to use | Avoid unnecessary tool calls |

### Model-Specific Formats

| Model | Chat Format |
|-------|-------------|
| **Qwen** | `<\|im_start\|>role ... <\|im_end\|>` |
| **Llama 3** | `<\|begin_of_text\|><\|start_header_id\|>role<\|end_header_id\|>` |
| **GPT** | Native function calling API |
| **Claude** | XML-style tool use |

---

## Parsing Tool Calls

### Basic JSON Extraction

```python
def parse_tool_call(response: str) -> dict | None:
    """Extract tool call JSON from model response"""
    try:
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        
        if start != -1 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            
            # Validate structure
            if "name" in data and "arguments" in data:
                return data
    except json.JSONDecodeError:
        pass
    
    return None
```

### Executing Tools

```python
# Tool implementations
def get_weather(location: str, unit: str = "celsius") -> dict:
    # Call real weather API
    return {"location": location, "temp": 22, "condition": "Sunny"}

def calculate(expression: str) -> dict:
    # Safe math evaluation
    result = eval(expression)  # Use safer eval in production!
    return {"expression": expression, "result": result}

# Tool registry
TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# Execute
tool_call = parse_tool_call(model_response)
if tool_call:
    tool_fn = TOOL_MAP[tool_call["name"]]
    result = tool_fn(**tool_call["arguments"])
```

### Error Handling

```python
def execute_tool_safely(name: str, arguments: dict) -> dict:
    """Execute tool with error handling"""
    if name not in TOOL_MAP:
        return {"error": f"Unknown tool: {name}"}
    
    try:
        result = TOOL_MAP[name](**arguments)
        return {"success": True, "result": result}
    except TypeError as e:
        return {"error": f"Invalid arguments: {e}"}
    except Exception as e:
        return {"error": f"Tool failed: {e}"}
```

---

## Multi-Turn Conversations

### Feeding Results Back

After the LLM requests a tool, you need to:
1. Execute the tool
2. Add the result to the conversation
3. Let the LLM continue

```python
class Conversation:
    def __init__(self, model, tokenizer, tools):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.messages = []
    
    def chat(self, user_message: str, max_tool_calls: int = 3) -> str:
        self.messages.append({"role": "user", "content": user_message})
        
        tool_calls = 0
        while tool_calls < max_tool_calls:
            # Generate response
            prompt = self._format_prompt()
            response = generate(self.model, self.tokenizer, prompt=prompt)
            
            # Check for tool call
            tool_call = parse_tool_call(response)
            
            if tool_call:
                tool_calls += 1
                
                # Execute tool
                result = execute_tool(tool_call["name"], tool_call["arguments"])
                
                # Add to conversation history
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({
                    "role": "user",  # or "tool" in some formats
                    "content": f"Tool result: {json.dumps(result)}"
                })
            else:
                # No tool call, return final response
                self.messages.append({"role": "assistant", "content": response})
                return response
        
        return "Max tool calls reached"
```

### Conversation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Turn Tool Calling                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Turn 1:                                                                   │
│   User: "Find electronics under $50"                                       │
│   Assistant: {"name": "search_products", "arguments": {...}}               │
│   Tool Result: [{"id": 2, "name": "Bluetooth Speaker", "price": 49.99}]    │
│                                                                             │
│   Turn 2:                                                                   │
│   Assistant: "I found a Bluetooth Speaker for $49.99. Want details?"       │
│   User: "Yes, tell me more"                                                │
│   Assistant: {"name": "get_product_details", "arguments": {"id": 2}}       │
│   Tool Result: {"name": "...", "description": "...", "rating": 4.2}        │
│                                                                             │
│   Turn 3:                                                                   │
│   Assistant: "The Bluetooth Speaker has a 4.2 rating and 12-hour battery"  │
│   User: "Add it to my cart"                                                │
│   Assistant: {"name": "add_to_cart", "arguments": {"product_id": 2}}       │
│   Tool Result: {"status": "added"}                                         │
│                                                                             │
│   Turn 4:                                                                   │
│   Assistant: "Done! Added to your cart."                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Advanced Patterns

### Tool Registry Pattern

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    function: Callable
    
    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def execute(self, **kwargs) -> dict:
        try:
            return {"success": True, "result": self.function(**kwargs)}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}
    
    def register(self, name: str, description: str, parameters: dict):
        """Decorator to register a tool"""
        def decorator(func: Callable):
            self.tools[name] = Tool(name, description, parameters, func)
            return func
        return decorator
    
    def get_schemas(self) -> list[dict]:
        return [tool.to_schema() for tool in self.tools.values()]
    
    def execute(self, name: str, arguments: dict) -> dict:
        if name not in self.tools:
            return {"error": f"Unknown tool: {name}"}
        return self.tools[name].execute(**arguments)


# Usage
registry = ToolRegistry()

@registry.register(
    name="get_weather",
    description="Get weather for a location",
    parameters={"type": "object", "properties": {"location": {"type": "string"}}}
)
def get_weather(location: str) -> dict:
    return {"temp": 22, "condition": "Sunny"}
```

### Parallel Tool Execution

Sometimes the LLM needs multiple tools at once:

```python
import asyncio

async def execute_parallel_tools(tool_calls: list[dict]) -> list[dict]:
    """Execute multiple tools concurrently"""
    async def execute_one(tc):
        result = await asyncio.to_thread(
            registry.execute, 
            tc["name"], 
            tc["arguments"]
        )
        return {"tool": tc["name"], "result": result}
    
    return await asyncio.gather(*[execute_one(tc) for tc in tool_calls])

# Example: Compare two products simultaneously
results = asyncio.run(execute_parallel_tools([
    {"name": "get_product_details", "arguments": {"product_id": 1}},
    {"name": "get_product_details", "arguments": {"product_id": 2}},
]))
```

### Agent Loop Pattern

For complex tasks, run in a loop until done:

```python
def agent_loop(user_request: str, max_iterations: int = 10) -> str:
    """Simple agent that uses tools until task is complete"""
    messages = [{"role": "user", "content": user_request}]
    
    for i in range(max_iterations):
        # Generate
        response = generate_with_tools(messages)
        
        # Check if done (no tool call)
        tool_call = parse_tool_call(response)
        if not tool_call:
            return response
        
        # Execute tool
        result = registry.execute(tool_call["name"], tool_call["arguments"])
        
        # Add to history
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "tool", "content": json.dumps(result)})
    
    return "Max iterations reached"
```

---

## Code Walkthrough

### File: `04_tool_calling_basic.py`

Basic tool calling demonstration:

```python
import json
from mlx_lm import load, generate

# Define tools
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "calculate",
            "description": "Math calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

# Tool implementations
def get_weather(location, unit="celsius"):
    return {"location": location, "temp": 22, "condition": "Sunny"}

def calculate(expression):
    return {"result": eval(expression)}

TOOL_MAP = {"get_weather": get_weather, "calculate": calculate}

# Create prompt
def create_tool_prompt(user_message, tools):
    return f"""<|im_start|>system
You have these tools: {json.dumps(tools)}
Use: {{"name": "...", "arguments": {{...}}}}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
"""

# Parse response
def parse_tool_call(response):
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1:
            data = json.loads(response[start:end])
            if "name" in data and "arguments" in data:
                return data
    except:
        pass
    return None

# Main loop
model, tokenizer = load("Qwen/Qwen2.5-1.5B-Instruct")

query = "What's the weather in Paris?"
prompt = create_tool_prompt(query, TOOLS)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)

tool_call = parse_tool_call(response)
if tool_call:
    result = TOOL_MAP[tool_call["name"]](**tool_call["arguments"])
    print(f"Tool: {tool_call['name']}, Result: {result}")
```

### File: `05_tool_calling_advanced.py`

Advanced patterns with registry and multi-turn:

```python
# Tool Registry (see pattern above)
registry = ToolRegistry()

@registry.register(name="search_products", description="...", parameters={...})
def search_products(query, category=None):
    return [{"id": 1, "name": "Product", "price": 49.99}]

@registry.register(name="add_to_cart", description="...", parameters={...})
def add_to_cart(product_id, quantity=1):
    return {"status": "added", "product_id": product_id}

# Conversation Manager
class Conversation:
    def __init__(self, model, tokenizer, tools):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.messages = []
    
    def chat(self, user_message, max_tool_calls=3):
        self.messages.append({"role": "user", "content": user_message})
        
        for _ in range(max_tool_calls):
            response = self._generate()
            tool_call = self._parse_tool_call(response)
            
            if tool_call:
                result = registry.execute(tool_call["name"], tool_call["arguments"])
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({"role": "user", "content": f"Result: {result}"})
            else:
                return response
        
        return "Max calls reached"

# Usage
conv = Conversation(model, tokenizer, registry.get_schemas())
response = conv.chat("Find electronics under $50")
response = conv.chat("Add the first one to my cart")
```

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Tool Calling** | LLM requests external capabilities via JSON |
| **ReAct** | Think → Act → Observe → Repeat |
| **Schema** | OpenAI function format (type, name, parameters) |
| **Parsing** | Extract JSON from response, validate structure |
| **Multi-Turn** | Feed tool results back, continue conversation |
| **Registry** | Decorator pattern for clean tool management |
| **Parallel** | Execute independent tools concurrently |

---

## Next Steps

- [05_deployment.md](05_deployment.md) - Production deployment with vLLM

