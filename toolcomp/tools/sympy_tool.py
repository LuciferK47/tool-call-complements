"""
SymPy Tool for Exact Arithmetic Evaluation (M1 Tool).
Complies with Decision D2 (c=1), Decision A-25 (cost charged even on error/timeout).
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional
import sympy

@dataclass
class ToolResult:
    output: str
    success: bool
    cost: float
    latency_s: float
    error_msg: Optional[str] = None

class SymPyTool:
    def __init__(self, timeout_s: float = 2.0, cost: float = 1.0):
        self.timeout_s = timeout_s
        self.cost = cost

    def execute(self, expression: str) -> ToolResult:
        """
        Executes an arithmetic expression using SymPy in a sandboxed, timed worker.
        """
        start_time = time.perf_counter()
        result_holder = {"output": None, "error": None}

        def worker():
            try:
                # Sanitize input: allow digits, whitespace, and basic arithmetic symbols
                cleaned = "".join(c for c in expression if c in "0123456789+-*/() .")
                if not cleaned.strip():
                    result_holder["error"] = "Empty expression"
                    return
                # Evaluate with SymPy
                parsed = sympy.sympify(cleaned, evaluate=True)
                # Format integer if whole number
                if getattr(parsed, "is_Integer", False):
                    result_holder["output"] = str(int(parsed))      # exact, no float round-trip
                else:
                    val = float(parsed)
                    result_holder["output"] = str(int(val)) if val.is_integer() else str(round(val, 6))
            except Exception as e:
                result_holder["error"] = str(e)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=self.timeout_s)

        elapsed = time.perf_counter() - start_time

        if thread.is_alive():
            # Timeout occurred
            return ToolResult(
                output="TOOL_TIMEOUT",
                success=False,
                cost=self.cost,  # Decision A-25: cost charged on timeout
                latency_s=elapsed,
                error_msg=f"Execution timed out after {self.timeout_s}s"
            )

        if result_holder["error"] is not None:
            return ToolResult(
                output="TOOL_ERROR",
                success=False,
                cost=self.cost,  # Decision A-25: cost charged on error
                latency_s=elapsed,
                error_msg=result_holder["error"]
            )

        return ToolResult(
            output=result_holder["output"],
            success=True,
            cost=self.cost,
            latency_s=elapsed
        )
