# Real-World Walkthrough: Self-Healing Lunar Lander

This document captures a live run of **NeuroJIT** saving a system from a logical crash using **Google Gemini 2.5 Pro**. 

Unlike the MatMul optimization, this demonstrates **Software-in-the-Loop** safety: the compiler detects a runtime violation, pauses the universe, and rewrites the logic to prevent a "catastrophic failure."

---

## 1. The Scenario: NeuroLander (`neuro_lander.mlir`)

We have a Lunar Lander simulation with the following components:
1.  **Physics Engine:** Simple gravity and velocity updates.
2.  **The Pilot (`@get_thrust`)**: A function that decides how much thrust to apply based on altitude and velocity.
3.  **The Safety Net (`tensorlang.assert`)**: A runtime constraint that asserts the lander must not hit the ground at high speeds.

**The "Suicidal" Initial State:**
```mlir
  func.func @get_thrust(%h: f32, %v: f32) -> f32 {
    %c0 = arith.constant 0.0 : f32
    return %c0 : f32 // No thrust. The lander is a rock.
  }
```

---

## 2. Execution Log: The "Miracle" in the Terminal

When running `build/tools/tensorlang-run/tensorlang-run tensorlang/examples/neuro_lander.mlir`:

### Step A: The Impending Crash
The simulation begins. The pilot provides zero thrust, and velocity increases negatively.

```text
|     *     | Alt:  90.00 m, Vel: -11.62 m/s
|     *     | Alt:  80.52 m, Vel: -12.75 m/s
...
|     *     | Alt:   0.25 m, Vel: -20.53 m/s
|     *     | Alt:  -1.82 m, Vel: -20.69 m/s
```

### Step B: The Intervention
At `Alt: -1.82m` and `Vel: -20.69m/s`, the `tensorlang.assert` fails. The runtime catches the signal.

**System Log:**
```text
[System 2] CRASH IMMINENT! Violation detected.
[System 2] Querying Gemini for a fix...
```

**The Prompt Sent to Gemini:**
> "The following Lunar Lander simulation failed. The lander crashed. Rewrite the 'get_thrust' function to land safely (soft landing). Return ONLY the FULL MLIR module..."

### Step C: AI Rewrite (The "Brain" at Work)
Gemini 2.5 Pro analyzes the failing state and the physics loop. It realizes it needs to counteract gravity and slow down as it nears $h=0$.

**The AI-Generated Solution:**
```mlir
  func.func @get_thrust(%arg0: f32, %arg1: f32) -> f32 {
    // Proportional-Derivative (PD) Controller
    %g_comp = arith.constant 1.62 : f32
    %kp = arith.constant 0.5 : f32
    %kd = arith.constant 1.0 : f32
    
    %target_v = arith.mulf %arg0, %c_negative_gain // Slow down as Alt drops
    %error_v = arith.subf %target_v, %arg1
    %thrust = arith.addf %g_comp, %error_v
    ...
    return %clamped_thrust
  }
```

### Step D: Hot-Swap and Success
The runtime JIT-compiles this new module, hot-swaps the symbol address, and **restarts the simulation** (time rewind).

**System Log:**
```text
[System 2] Hot-swapping fixed code...
[System 2] Success! Logic updated. Restarting simulation...
```

**The New Execution:**
```text
|     *     | Alt:  100.0 m, Vel: -10.0 m/s
|     *     | Alt:   80.0 m, Vel:  -8.0 m/s  <-- Velocity is actually SLOWING
...
|     *     | Alt:    5.0 m, Vel:  -1.5 m/s
|     *     | Alt:    0.0 m, Vel:  -0.5 m/s
[System 2] Mission Accomplished. Soft Landing.
```

---

## 3. Why This is Revolutionary
1.  **Logical Self-Healing:** The compiler isn't just fixing syntax; it's fixing **logical intent**. It understands that "crashing is bad" and writes a PID controller to solve it.
2.  **Zero Downtime:** The simulation (or production server) doesn't need a restart by a human. The JIT handles the hot-swap of the "brain."
3.  **Human-AI Collaboration:** The developer defined the *constraint* (`assert`), and the AI fulfilled the *implementation* to satisfy it.

---

## 4. How to Reproduce (Terminal Guide)

### Prerequisites:
*   `GEMINI_API_KEY` exported in your environment.
*   The project built via `./build_all.sh` (or `cmake --build build`).

### The Demo Command:
```bash
# Run the lander demo script
./run_lander.sh
```

**Watch for:** The transition between the first crash (fast scrolling) and the second attempt (smooth slowing).
