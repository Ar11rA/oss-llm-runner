"""
Advanced Tool Calling with Qwen2.5-Instruct

Demonstrates:
1. Multi-turn conversations with tool results
2. Chaining multiple tool calls
3. Parallel tool execution
4. Structured output validation

Model: Qwen/Qwen2.5-1.5B-Instruct
"""

import json
from dataclasses import dataclass
from typing import Callable
from mlx_lm import load, generate

# ============================================================================
# Configuration
# ============================================================================
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ============================================================================
# Tool Registry with Validation
# ============================================================================
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
            return {"success": False, "error": f"Unknown tool: {name}"}
        return self.tools[name].execute(**arguments)


# Create registry
registry = ToolRegistry()


# ============================================================================
# Register Tools
# ============================================================================
@registry.register(
    name="search_products",
    description="Search for products in the catalog",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "category": {"type": "string", "description": "Product category", "enum": ["electronics", "clothing", "books", "home"]},
            "max_price": {"type": "number", "description": "Maximum price filter"}
        },
        "required": ["query"]
    }
)
def search_products(query: str, category: str = None, max_price: float = None) -> list:
    """Simulated product search"""
    products = [
        {"id": 1, "name": "Wireless Headphones", "category": "electronics", "price": 79.99},
        {"id": 2, "name": "Bluetooth Speaker", "category": "electronics", "price": 49.99},
        {"id": 3, "name": "USB-C Cable", "category": "electronics", "price": 12.99},
        {"id": 4, "name": "Python Cookbook", "category": "books", "price": 45.00},
        {"id": 5, "name": "ML Engineering", "category": "books", "price": 55.00},
    ]
    
    results = [p for p in products if query.lower() in p["name"].lower()]
    if category:
        results = [p for p in results if p["category"] == category]
    if max_price:
        results = [p for p in results if p["price"] <= max_price]
    return results


@registry.register(
    name="get_product_details",
    description="Get detailed information about a specific product",
    parameters={
        "type": "object",
        "properties": {
            "product_id": {"type": "integer", "description": "Product ID"}
        },
        "required": ["product_id"]
    }
)
def get_product_details(product_id: int) -> dict:
    """Get product details by ID"""
    details = {
        1: {"id": 1, "name": "Wireless Headphones", "price": 79.99, "stock": 50, "rating": 4.5, "description": "High-quality wireless headphones with noise cancellation"},
        2: {"id": 2, "name": "Bluetooth Speaker", "price": 49.99, "stock": 30, "rating": 4.2, "description": "Portable speaker with 12-hour battery"},
        3: {"id": 3, "name": "USB-C Cable", "price": 12.99, "stock": 200, "rating": 4.8, "description": "Fast charging USB-C cable, 2m length"},
    }
    return details.get(product_id, {"error": "Product not found"})


@registry.register(
    name="add_to_cart",
    description="Add a product to the shopping cart",
    parameters={
        "type": "object",
        "properties": {
            "product_id": {"type": "integer", "description": "Product ID to add"},
            "quantity": {"type": "integer", "description": "Quantity to add", "default": 1}
        },
        "required": ["product_id"]
    }
)
def add_to_cart(product_id: int, quantity: int = 1) -> dict:
    """Add item to cart"""
    return {"status": "added", "product_id": product_id, "quantity": quantity, "message": f"Added {quantity} item(s) to cart"}


@registry.register(
    name="check_order_status",
    description="Check the status of an order",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Order ID to check"}
        },
        "required": ["order_id"]
    }
)
def check_order_status(order_id: str) -> dict:
    """Check order status"""
    orders = {
        "ORD-001": {"status": "shipped", "tracking": "1Z999AA10123456784", "eta": "2024-01-15"},
        "ORD-002": {"status": "processing", "tracking": None, "eta": "2024-01-18"},
    }
    return orders.get(order_id, {"error": "Order not found"})


# ============================================================================
# Conversation Manager
# ============================================================================
class Conversation:
    def __init__(self, model, tokenizer, tools: list[dict]):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.messages = []
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        tools_json = json.dumps(self.tools, indent=2)
        return f"""You are a helpful shopping assistant with access to these tools:

{tools_json}

To use a tool, respond ONLY with a JSON object:
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}

After receiving tool results, provide a helpful response to the user.
If no tool is needed, respond directly to the user."""
    
    def _format_prompt(self) -> str:
        prompt = f"<|im_start|>system\n{self.system_prompt}\n<|im_end|>\n"
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def _parse_tool_call(self, response: str) -> dict | None:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response[start:end])
                if "name" in data and "arguments" in data:
                    return data
        except json.JSONDecodeError:
            pass
        return None
    
    def chat(self, user_message: str, max_tool_calls: int = 3) -> str:
        """Process user message, potentially calling tools"""
        self.messages.append({"role": "user", "content": user_message})
        
        tool_calls = 0
        while tool_calls < max_tool_calls:
            prompt = self._format_prompt()
            response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=300)
            
            tool_call = self._parse_tool_call(response)
            
            if tool_call:
                tool_calls += 1
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                
                print(f"  🔧 Tool: {tool_name}({json.dumps(tool_args)})")
                
                result = registry.execute(tool_name, tool_args)
                
                print(f"  📤 Result: {json.dumps(result, indent=2)}")
                
                # Add tool call and result to conversation
                self.messages.append({"role": "assistant", "content": response})
                self.messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}: {json.dumps(result)}"
                })
            else:
                # No tool call, return response
                self.messages.append({"role": "assistant", "content": response})
                return response
        
        return "Max tool calls reached"


# ============================================================================
# Load Model
# ============================================================================
print(f"Loading {MODEL}...")
model, tokenizer = load(MODEL)
print("✅ Model loaded!\n")


# ============================================================================
# Interactive Demo
# ============================================================================
print("=" * 60)
print("ADVANCED TOOL CALLING DEMO")
print("=" * 60)

conv = Conversation(model, tokenizer, registry.get_schemas())

# Test scenarios
scenarios = [
    "Find me some electronics under $50",
    "Tell me more about product 2",
    "Add it to my cart",
    "What's the status of order ORD-001?",
]

for user_input in scenarios:
    print(f"\n{'='*60}")
    print(f"👤 USER: {user_input}")
    print("-" * 60)
    
    response = conv.chat(user_input)
    
    print(f"\n🤖 ASSISTANT: {response}")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)


# ============================================================================
# Parallel Tool Calls Example
# ============================================================================
print("\n" + "=" * 60)
print("PARALLEL TOOL CALLS (Manual)")
print("=" * 60)

# Sometimes you want to execute multiple tools in parallel
# This requires parsing multiple tool calls from the response

def execute_parallel_tools(tool_calls: list[dict]) -> list[dict]:
    """Execute multiple tool calls (could be async in production)"""
    results = []
    for tc in tool_calls:
        result = registry.execute(tc["name"], tc["arguments"])
        results.append({"tool": tc["name"], "result": result})
    return results

# Example: Compare two products
print("\nComparing products 1 and 2...")
results = execute_parallel_tools([
    {"name": "get_product_details", "arguments": {"product_id": 1}},
    {"name": "get_product_details", "arguments": {"product_id": 2}},
])

for r in results:
    print(f"\n{r['tool']}: {json.dumps(r['result'], indent=2)}")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

