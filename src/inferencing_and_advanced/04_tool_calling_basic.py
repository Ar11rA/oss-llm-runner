"""
Basic Tool Calling with Qwen2.5-Instruct

Demonstrates how to:
1. Define tools/functions for the model
2. Get the model to call tools
3. Execute tools and return results

Model: Qwen/Qwen2.5-1.5B-Instruct (supports function calling)
"""

import json
from mlx_lm import load, generate

# ============================================================================
# Configuration
# ============================================================================
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ============================================================================
# Define Tools
# ============================================================================
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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2 * 3'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# ============================================================================
# Tool Implementations
# ============================================================================
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulated weather API"""
    # In real use, call actual weather API
    weather_data = {
        "San Francisco, CA": {"temp": 18, "condition": "Foggy"},
        "New York, NY": {"temp": 22, "condition": "Sunny"},
        "London, UK": {"temp": 15, "condition": "Rainy"},
    }
    data = weather_data.get(location, {"temp": 20, "condition": "Unknown"})
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
    return {"location": location, "unit": unit, **data}


def calculate(expression: str) -> dict:
    """Safe math evaluation"""
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return {"expression": expression, "result": result}
        return {"error": "Invalid expression"}
    except Exception as e:
        return {"error": str(e)}


TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# ============================================================================
# Load Model
# ============================================================================
print(f"Loading {MODEL}...")
model, tokenizer = load(MODEL)
print("✅ Model loaded!\n")


# ============================================================================
# Format Prompt with Tools (Qwen format)
# ============================================================================
def create_tool_prompt(user_message: str, tools: list) -> str:
    """Create a prompt with tool definitions for Qwen2.5"""
    tools_json = json.dumps(tools, indent=2)
    
    prompt = f"""<|im_start|>system
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
    return prompt


def parse_tool_call(response: str) -> dict | None:
    """Extract tool call from model response"""
    try:
        # Try to find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            data = json.loads(json_str)
            if "name" in data and "arguments" in data:
                return data
    except json.JSONDecodeError:
        pass
    return None


# ============================================================================
# Test Tool Calling
# ============================================================================
test_queries = [
    "What's the weather like in San Francisco?",
    "Calculate 15 * 7 + 23",
    "What is 2 to the power of 10?",  # Should use calculate
    "Hello, how are you?",  # No tool needed
]

for query in test_queries:
    print("=" * 60)
    print(f"USER: {query}")
    print("=" * 60)
    
    prompt = create_tool_prompt(query, TOOLS)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
    
    print(f"\n🤖 MODEL RESPONSE:\n{response}")
    
    # Check if model wants to call a tool
    tool_call = parse_tool_call(response)
    
    if tool_call:
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
        
        print(f"\n🔧 TOOL CALL: {tool_name}({tool_args})")
        
        if tool_name in TOOL_MAP:
            result = TOOL_MAP[tool_name](**tool_args)
            print(f"📤 TOOL RESULT: {result}")
    else:
        print("\n(No tool call detected - direct response)")
    
    print()

print("=" * 60)
print("COMPLETE")
print("=" * 60)

